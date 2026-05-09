"""Multi-objective trainer for Experiment 3: Drive-Embedded Training.

Combined loss: L_total = L_LM + α*L_curi + β*L_coh + γ*L_emp

Each drive loss operates on different aspects of the model's output
(logits, hidden states, memory buffer) and together they create an
intrinsic motivation system alongside the standard language modeling objective.
"""
from pathlib import Path

import torch
from torch.utils.data import DataLoader, IterableDataset
from transformers import PreTrainedModel, PreTrainedTokenizer, get_cosine_schedule_with_warmup
from omegaconf import DictConfig

from src.core.logging_ import Logger
from src.experiment3.drive_losses import compute_drive_losses
from src.experiment3.self_narrative_memory import SelfNarrativeMemory


class DriveAwareDataset(IterableDataset):
    """Streaming text dataset for drive-embedded training."""

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
    return {"input_ids": padded_ids, "attention_mask": padded_mask, "labels": labels}


def drive_embedded_train(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    cfg: DictConfig,
    train_texts: list[str],
    alpha: float = 0.01,
    beta: float = 0.01,
    gamma: float = 0.005,
    memory_size: int = 100,
    experiment_name: str = "drive_embedded",
) -> PreTrainedModel:
    """Run drive-embedded training loop.

    Args:
        model: base model with/without LoRA
        tokenizer: tokenizer
        cfg: config
        train_texts: training corpus
        alpha: curiosity loss weight
        beta: coherence loss weight
        gamma: empowerment loss weight
        memory_size: self-narrative memory buffer capacity
        experiment_name: logging name

    Returns:
        trained model
    """
    device = next(model.parameters()).device
    hidden_dim = model.config.hidden_size

    logger = Logger(
        project=cfg.wandb.project,
        entity=cfg.wandb.get("entity"),
        mode=cfg.wandb.mode,
        output_dir=cfg.training.output_dir,
        experiment_name=experiment_name,
    )

    memory = SelfNarrativeMemory(max_size=memory_size, hidden_dim=hidden_dim)

    dataset = DriveAwareDataset(train_texts, tokenizer, max_length=512)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.per_device_batch_size,
        collate_fn=lambda b: collate_batch(b, tokenizer.pad_token_id or tokenizer.eos_token_id),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate)
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
            logits = outputs.logits
            hidden = outputs.hidden_states[-1][:, -1, :]  # Last token hidden state

            # Compute drive losses
            mem_hidden = memory.get_hidden_states(n_recent=50).to(device)
            drive_losses = compute_drive_losses(logits, hidden, mem_hidden, alpha, beta, gamma)

            total_loss = lm_loss + drive_losses["total"]

            (total_loss / accumulation_steps).backward()

            global_step += 1

            if global_step % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Update self-narrative memory
            if global_step % 10 == 0 and memory.is_warm(min_entries=0):
                with torch.no_grad():
                    memory.add(hidden.mean(dim=0), "", global_step)
            elif not memory.is_warm(min_entries=0):
                with torch.no_grad():
                    memory.add(hidden.mean(dim=0), "", global_step)

            if global_step % cfg.training.logging_steps == 0:
                logger.log({
                    "lm_loss": float(lm_loss.item()),
                    "curiosity_loss": float(drive_losses["curiosity"].item()),
                    "coherence_loss": float(drive_losses["coherence"].item()),
                    "empowerment_loss": float(drive_losses["empowerment"].item()),
                    "total_loss": float(total_loss.item()),
                    "lr": float(scheduler.get_last_lr()[0]),
                }, step=global_step)

            if global_step >= cfg.training.max_steps:
                break

        if global_step >= cfg.training.max_steps:
            break

    output_dir = Path(cfg.training.output_dir) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))

    logger.finish()
    return model
