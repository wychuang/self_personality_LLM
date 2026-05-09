"""Hook-based residual stream activation extraction.

Extracts hidden states from transformer layers during forward passes,
used to compute the identity direction vector v_id from seed narratives.
"""
import torch
from torch import nn
from transformers import PreTrainedModel, PreTrainedTokenizer


class ActivationExtractor:
    """Extract residual stream activations from specified layers.

    Usage:
        extractor = ActivationExtractor(model, layers=["last"])
        hidden_states = extractor.extract_from_texts(["I am...", "I believe..."], tokenizer)
        v_id = hidden_states.mean(dim=0)  # identity direction
    """

    def __init__(self, model: PreTrainedModel, layers: list[str] | None = None):
        """
        Args:
            model: HuggingFace causal LM
            layers: which layers to extract from.
                    "last" = final layer residual stream
                    "all" = every transformer block
                    ["0", "6", "12"] = specific blocks by index
        """
        self.model = model
        self.layers = layers or ["last"]
        self._activations: dict[str, torch.Tensor] = {}
        self._hooks = []

    def _resolve_layer_names(self) -> list[str]:
        config = self.model.config
        n_layers = getattr(config, "num_hidden_layers", 0)

        resolved = []
        for layer in self.layers:
            if layer == "last":
                resolved.append("last")
            elif layer == "all":
                for i in range(n_layers):
                    resolved.append(f"transformer.h.{i}")
            elif layer.isdigit():
                idx = int(layer)
                if 0 <= idx < n_layers:
                    resolved.append(f"transformer.h.{idx}")
            else:
                resolved.append(layer)
        return resolved

    def _hook_fn(self, name: str):
        def hook(module, inputs, output):
            if isinstance(output, tuple):
                # Some layers return (hidden_states, ...)
                self._activations[name] = output[0].detach().cpu()
            else:
                self._activations[name] = output.detach().cpu()
        return hook

    def extract_from_texts(
        self,
        texts: list[str],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        aggregate: str = "last_token",
    ) -> torch.Tensor:
        """Extract residual stream activations from seed narrative texts.

        Args:
            texts: list of first-person narrative strings
            tokenizer: matching tokenizer
            max_length: max tokens per text
            aggregate: how to pool tokens.
                       "last_token" = use last non-padding token
                       "mean" = mean over all tokens
                       "first_token" = use first token (typically BOS)

        Returns:
            Tensor of shape (num_texts, hidden_dim) — activation vectors
        """
        layer_names = self._resolve_layer_names()

        # Register hooks
        self._hooks = []
        for name in layer_names:
            if name == "last":
                module = self.model
            else:
                module = dict(self.model.named_modules()).get(name)
            if module is not None:
                h = module.register_forward_hook(self._hook_fn(name))
                self._hooks.append(h)

        self.model.eval()
        all_activations = []

        with torch.no_grad():
            for text in texts:
                encoded = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                ).to(self.model.device)

                self._activations.clear()
                self.model(**encoded)

                # Aggregate across requested layers
                layer_vecs = []
                for name in layer_names:
                    if name in self._activations:
                        act = self._activations[name]  # (batch=1, seq_len, hidden_dim)

                        if aggregate == "last_token":
                            # Use last non-padding position
                            attention_mask = encoded.get("attention_mask")
                            if attention_mask is not None:
                                last_idx = attention_mask.sum(dim=1) - 1
                                vec = act[0, last_idx, :]
                            else:
                                vec = act[0, -1, :]
                        elif aggregate == "first_token":
                            vec = act[0, 0, :]
                        else:  # mean
                            vec = act[0].mean(dim=0)

                        layer_vecs.append(vec)

                if layer_vecs:
                    combined = torch.cat(layer_vecs, dim=0)
                    all_activations.append(combined)

        # Clean up hooks
        for h in self._hooks:
            h.remove()
        self._hooks = []

        if not all_activations:
            raise RuntimeError("No activations extracted — check layer names and model architecture")

        return torch.stack(all_activations)
