"""Aggressive IDA variant: inject v_id as forced bias into transformer blocks.

Instead of only penalizing deviation via the loss, this injects v_id
directly into the residual stream of each transformer block, forcing
the model to "route around" the identity direction.
"""
import torch
from torch import nn
from transformers import PreTrainedModel


def inject_bias_into_block(block: nn.Module, v_id: torch.Tensor, alpha: float = 0.1) -> list:
    """Register forward hooks that add alpha * v_id to residual stream.

    Args:
        block: a transformer block module
        v_id: identity direction vector (hidden_dim,)
        alpha: injection strength

    Returns:
        list of hook handles (for cleanup)
    """
    hooks = []

    def bias_hook(module, inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
            biased = hidden + alpha * v_id.to(device=hidden.device, dtype=hidden.dtype)
            return (biased,) + output[1:]
        else:
            return output + alpha * v_id.to(device=output.device, dtype=output.dtype)

    hooks.append(block.register_forward_hook(bias_hook))
    return hooks


class BiasInjectedModel:
    """Wrapper that injects v_id as bias into transformer blocks during forward pass.

    Usage:
        injector = BiasInjectedModel(model)
        injector.inject(v_id, alpha=0.1, layers="all")
        output = model(**inputs)  # v_id bias is added in each block
        injector.cleanup()  # remove hooks
    """

    def __init__(self, model: PreTrainedModel):
        self.model = model
        self._hooks = []

    def inject(self, v_id: torch.Tensor, alpha: float = 0.1, layers: str = "all") -> None:
        """Inject v_id into specified layers.

        Args:
            v_id: identity direction (hidden_dim,)
            alpha: injection strength
            layers: "all" = every block, "first_N", "last_N"

        Raises:
            ValueError: if model architecture is not recognized
        """
        blocks = self._get_transformer_blocks()

        if layers == "all":
            target_blocks = blocks
        elif layers.startswith("first_"):
            n = int(layers.split("_")[1])
            target_blocks = blocks[:n]
        elif layers.startswith("last_"):
            n = int(layers.split("_")[1])
            target_blocks = blocks[-n:]
        else:
            target_blocks = blocks

        for block in target_blocks:
            hooks = inject_bias_into_block(block, v_id, alpha)
            self._hooks.extend(hooks)

    def _get_transformer_blocks(self) -> list[nn.Module]:
        """Extract transformer blocks from the model.

        Tries common architectures: GPT-NeoX/Pythia, LLaMA, GPT-2.
        """
        # Try Pythia/GPT-NeoX pattern: model.gpt_neox.layers
        if hasattr(self.model, "gpt_neox"):
            return list(self.model.gpt_neox.layers)
        # Try LLaMA pattern: model.model.layers
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return list(self.model.model.layers)
        # Try GPT-2 pattern: model.transformer.h
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return list(self.model.transformer.h)
        # Try iterating named modules for common patterns
        for name, module in self.model.named_modules():
            if name.endswith(".layers") or name.endswith(".h"):
                return list(module)
        raise ValueError(
            f"Cannot find transformer blocks for model type: {type(self.model).__name__}. "
            f"Provide block list manually."
        )

    def cleanup(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []
