"""Dual-loss trainer for Identity Direction Anchoring (IDA).

Training loop: L = L_LM + lambda * L_ID
where L_ID = -cos(h_last, v_id) — penalizes deviation from identity direction.
"""
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from transformers import PreTrainedModel, PreTrainedTokenizer, get_cosine_schedule_with_warmup
from omegaconf import DictConfig

from src.core.logging_ import Logger
from src.experiment1.identity_direction import compute_id_loss


class TextDataset(IterableDataset):
    """Streaming text dataset for language modeling."""

    def __init__(self, texts: list[str], tokenizer: PreTrainedTokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __iter__(self):
        for text in self.texts:
            encoded = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            if encoded["input_ids"].numel() > 1:
                yield {
                    "input_ids": encoded["input_ids"][0],
                    "attention_mask": encoded["attention_mask"][0],
                }


def collate_batch(batch: list[dict], pad_token_id: int) -> dict[str, torch.Tensor]:
    max_len = max(b["input_ids"].size(0) for b in batch)
    padded_ids = torch.full((len(batch), max_len), pad_token_id)
    padded_mask = torch.zeros((len(batch), max_len))

    for i, b in enumerate(batch):
        seq_len = b["input_ids"].size(0)
        padded_ids[i, :seq_len] = b["input_ids"]
        padded_mask[i, :seq_len] = b["attention_mask"] if "attention_mask" in b else 1

    labels = padded_ids.clone()
    labels[labels == pad_token_id] = -100

    return {
        "input_ids": padded_ids,
        "attention_mask": padded_mask,
        "labels": labels,
    }


def ida_train(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    v_id: torch.Tensor,
    cfg: DictConfig,
    train_texts: list[str],
    id_lambda: float = 0.1,
    experiment_name: str = "ida",
) -> PreTrainedModel:
    """Run IDA training loop.

    Args:
        model: base model with/without LoRA
        tokenizer: matching tokenizer
        v_id: identity direction vector (hidden_dim,)
        cfg: full config DictConfig
        train_texts: training corpus texts
        id_lambda: weight of identity alignment loss
        experiment_name: for logging

    Returns:
        trained model
    """
    device = next(model.parameters()).device
    v_id = v_id.to(device=device, dtype=torch.bfloat16)

    logger = Logger(
        project=cfg.wandb.project,
        entity=cfg.wandb.get("entity"),
        mode=cfg.wandb.mode,
        output_dir=cfg.training.output_dir,
        experiment_name=experiment_name,
    )

    dataset = TextDataset(train_texts, tokenizer, max_length=512)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.per_device_batch_size,
        collate_fn=lambda b: collate_batch(b, tokenizer.pad_token_id or tokenizer.eos_token_id),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=0.01,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.training.warmup_steps,
        num_training_steps=cfg.training.max_steps,
    )

    model.train()
    global_step = 0
    accumulation_steps = cfg.training.gradient_accumulation_steps
    optimizer.zero_grad()

    for epoch in range(10):
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch, output_hidden_states=True)

            lm_loss = outputs.loss
            hidden_states = outputs.hidden_states[-1]  # Last layer
            id_loss = compute_id_loss(hidden_states, v_id)
            total_loss = lm_loss + id_lambda * id_loss

            (total_loss / accumulation_steps).backward()

            global_step += 1

            if global_step % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if global_step % cfg.training.logging_steps == 0:
                logger.log({
                    "lm_loss": float(lm_loss.item()),
                    "id_loss": float(id_loss.item()),
                    "total_loss": float(total_loss.item()),
                    "id_lambda": id_lambda,
                    "lr": float(scheduler.get_last_lr()[0]),
                }, step=global_step)

            if global_step >= cfg.training.max_steps:
                break

        if global_step >= cfg.training.max_steps:
            break

    # Save
    output_dir = Path(cfg.training.output_dir) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    torch.save(v_id.cpu(), output_dir / "v_id.pt")

    logger.finish()
    return model
