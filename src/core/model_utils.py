"""Model loading and device management utilities."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from omegaconf import DictConfig


def load_model_and_tokenizer(cfg: DictConfig) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load base model and tokenizer from config.

    Handles: device_map, dtype, LoRA wrapping, gradient checkpointing.
    """

    torch_dtype = getattr(torch, cfg.model.torch_dtype) if hasattr(torch, cfg.model.torch_dtype) else torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.name,
        trust_remote_code=cfg.model.get("trust_remote_code", True),
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        torch_dtype=torch_dtype,
        device_map=cfg.model.device_map,
        trust_remote_code=cfg.model.get("trust_remote_code", True),
    )

    if cfg.model.get("use_cache") is not None:
        model.config.use_cache = cfg.model.use_cache

    if cfg.get("lora") and cfg.lora.get("rank", 0) > 0:
        lora_config = LoraConfig(
            r=cfg.lora.rank,
            lora_alpha=cfg.lora.alpha,
            lora_dropout=cfg.lora.dropout,
            target_modules=cfg.lora.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    if cfg.training.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    return model, tokenizer


def load_model_revision(model_name: str, revision: str, device_map: str = "auto") -> PreTrainedModel:
    """Load model at specific revision (for Pythia checkpoint iteration).

    Always loads in bf16, no LoRA, no gradient checkpointing — intended for inference/eval only.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.config.use_cache = True
    return model


def add_lora(model: PreTrainedModel, cfg: DictConfig) -> PeftModel:
    """Wrap an already-loaded model with LoRA."""
    lora_config = LoraConfig(
        r=cfg.lora.rank,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_config)
