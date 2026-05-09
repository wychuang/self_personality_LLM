"""Pythia checkpoint iteration and LoRA stacking utilities."""
import torch
from pathlib import Path
from typing import Iterator
from transformers import AutoModelForCausalLM, AutoTokenizer


class PythiaCheckpointIterator:
    """Lazy iterator over Pythia's intermediate training checkpoints.

    Pythia checkpoints are stored as revisions on HuggingFace Hub:
    step1, step2, step4, step8, ..., step143000 (143 checkpoints total).

    Each model is loaded on-demand and unloaded after use to stay within GPU memory.

    Usage:
        iterator = PythiaCheckpointIterator("EleutherAI/pythia-160m")
        for step, model, tokenizer in iterator:
            # run probes on model
            del model  # free memory
            torch.cuda.empty_cache()
    """

    def __init__(
        self,
        model_name: str = "EleutherAI/pythia-160m",
        tokenizer_name: str | None = None,
        device_map: str = "auto",
        start_step: int = 1,
        end_step: int = 143000,
        step_interval: int = 1000,
    ):
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name or model_name
        self.device_map = device_map

        # Pythia checkpoints follow a geometric progression:
        # 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, ..., 143000
        # We generate all steps and filter by range
        self.steps = self._generate_pythia_steps()
        self.steps = [s for s in self.steps if start_step <= s <= end_step]

        if step_interval > 0 and len(self.steps) > 1:
            filtered = [self.steps[0]]
            for s in self.steps[1:]:
                if s >= filtered[-1] + step_interval:
                    filtered.append(s)
            self.steps = filtered

        self._tokenizer = None

    @staticmethod
    def _generate_pythia_steps() -> list[int]:
        """Generate Pythia's checkpoint step sequence."""
        steps = []
        step = 1
        while step <= 143000:
            steps.append(step)
            if step < 1000:
                step = step * 2 if step < 512 else 1000
            else:
                step += 1000
        return steps

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        return self._tokenizer

    def __iter__(self) -> Iterator[tuple[int, AutoModelForCausalLM, AutoTokenizer]]:
        for step in self.steps:
            revision = f"step{step}"
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    revision=revision,
                    torch_dtype=torch.bfloat16,
                    device_map=self.device_map,
                    trust_remote_code=True,
                )
                model.config.use_cache = True
                model.eval()
                yield step, model, self.tokenizer
            except Exception as e:
                print(f"Failed to load step {step}: {e}")
                continue
            finally:
                del model
                torch.cuda.empty_cache()

    def __len__(self) -> int:
        return len(self.steps)


class LoRAStack:
    """Stack of frozen LoRA adapters — "geological strata" architecture (Experiment A3).

    Bottom layers: frozen (rank fixed, lr=0).
    Middle layers: slow learning rate.
    Top layers: fully plastic.

    This creates a physically interpretable "strata" structure where
    early-experience adapters constrain later learning.
    """

    def __init__(
        self,
        base_model_name: str,
        num_layers: int = 3,
        rank_per_layer: int = 8,
        lr_ratios: tuple[float, ...] = (0.0, 0.1, 1.0),
    ):
        self.base_model_name = base_model_name
        self.num_layers = num_layers
        self.rank_per_layer = rank_per_layer
        self.lr_ratios = lr_ratios
        self.adapters: list = []

    def add_layer(self, adapter_path: str, freeze: bool = False) -> None:
        """Stack a new LoRA adapter. If freeze=True, mark as frozen (lr=0)."""
        self.adapters.append({"path": adapter_path, "frozen": freeze})
