"""Experiment A3: LoRA stacking as "geological strata."

Frozen base model = "bedrock."
LoRA layer 1 (frozen, lr=0) = "deep sediment" (phase 1).
LoRA layer 2 (slow lr) = "middle strata" (phase 2).
LoRA layer 3 (full lr) = "topsoil" (phase 3).

Ablation: remove layers individually and measure personality regression.
"""
import torch
from pathlib import Path
from peft import PeftModel, LoraConfig, get_peft_model
from transformers import PreTrainedModel


class LoRAStrataStack:
    """Manages a stack of LoRA adapters as geological strata."""

    def __init__(self, base_model: PreTrainedModel, rank: int = 8, alpha: int = 16):
        self.base_model = base_model
        self.rank = rank
        self.alpha = alpha
        self.adapters: list[dict] = []  # [{"name": str, "model": PeftModel, "frozen": bool, "lr_scale": float}]

    def add_stratum(
        self,
        name: str,
        lr_scale: float = 1.0,  # 1.0 = full plasticity, 0.0 = completely frozen
    ) -> PeftModel:
        """Add a new LoRA layer on top of the stack.

        Args:
            name: stratum name (e.g., "phase1_stoic")
            lr_scale: learning rate multiplier for this layer.
                      0.0 = frozen (deep sediment), 1.0 = fully plastic (topsoil)

        Returns the new PeftModel.
        """
        if self.adapters:
            # Add LoRA on top of the last adapter
            last_model = self.adapters[-1]["model"]
        else:
            last_model = self.base_model

        lora_config = LoraConfig(
            r=self.rank,
            lora_alpha=self.alpha,
            target_modules="all-linear",
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(last_model, lora_config)

        # Freeze previous layers
        if self.adapters:
            for prev in self.adapters:
                for param in prev["model"].parameters():
                    param.requires_grad = False

        # Set learning rate for current layer
        if lr_scale == 0.0:
            for param in model.parameters():
                if param.requires_grad:
                    param.requires_grad = False

        self.adapters.append({
            "name": name,
            "model": model,
            "frozen": lr_scale == 0.0,
            "lr_scale": lr_scale,
        })

        return model

    def ablate_stratum(self, index: int) -> PreTrainedModel:
        """Remove a stratum and return the model state without it.

        This simulates "eroding" a life stage to see what personality remains.
        """
        if index >= len(self.adapters):
            raise IndexError(f"Stratum {index} out of range (have {len(self.adapters)})")

        if index == 0:
            return self.base_model

        return self.adapters[index - 1]["model"]

    def get_active_model(self) -> PreTrainedModel:
        """Return the top (most recent) model in the stack."""
        if not self.adapters:
            return self.base_model
        return self.adapters[-1]["model"]
