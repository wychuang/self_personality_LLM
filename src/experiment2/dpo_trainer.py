"""DPO training for self-consistency optimization.

Uses Direct Preference Optimization to train the model to prefer
self-consistent responses over contradictory ones.
"""
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer, get_cosine_schedule_with_warmup
from omegaconf import DictConfig

from src.core.logging_ import Logger


def dpo_loss(
    model: PreTrainedModel,
    ref_model: PreTrainedModel,
    chosen_ids: torch.Tensor,
    chosen_mask: torch.Tensor,
    rejected_ids: torch.Tensor,
    rejected_mask: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """Direct Preference Optimization loss.

    L_DPO = -E[log σ(β * (log π_θ(y_w|x) / π_ref(y_w|x) - log π_θ(y_l|x) / π_ref(y_l|x)))]
    """
    # Compute log probabilities under policy model
    chosen_logps = _compute_logprob(model, chosen_ids, chosen_mask)
    rejected_logps = _compute_logprob(model, rejected_ids, rejected_mask)

    # Compute log probabilities under reference model
    with torch.no_grad():
        ref_chosen_logps = _compute_logprob(ref_model, chosen_ids, chosen_mask)
        ref_rejected_logps = _compute_logprob(ref_model, rejected_ids, rejected_mask)

    # DPO: log-sigmoid of the preference gap
    policy_ratio = chosen_logps - rejected_logps
    ref_ratio = ref_chosen_logps - ref_rejected_logps
    logits = beta * (policy_ratio - ref_ratio)

    loss = -torch.nn.functional.logsigmoid(logits)
    return loss.mean()


def _compute_logprob(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute average log-probability per sequence."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # (batch, seq_len, vocab)

    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()

    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    label_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    masked_log_probs = label_log_probs * shift_mask

    seq_log_prob = masked_log_probs.sum(dim=-1)
    seq_len = shift_mask.sum(dim=-1).clamp(min=1)
    return seq_log_prob / seq_len


def train_dpo(
    model: PreTrainedModel,
    ref_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    preference_dataset,
    cfg: DictConfig,
    experiment_name: str = "dpo_consistency",
    beta: float = 0.1,
) -> PreTrainedModel:
    """Run DPO training loop for self-consistency.

    Args:
        model: policy model to train
        ref_model: frozen reference model
        tokenizer: tokenizer
        preference_dataset: Dataset with 'prompt', 'chosen', 'rejected' columns
        cfg: config
        experiment_name: logging name
        beta: DPO temperature parameter

    Returns:
        trained model
    """
    device = next(model.parameters()).device

    logger = Logger(
        project=cfg.wandb.project,
        entity=cfg.wandb.get("entity"),
        mode=cfg.wandb.mode,
        output_dir=cfg.training.output_dir,
        experiment_name=experiment_name,
    )

    def collate_fn(batch):
        chosen_texts = [f"Question: {b['prompt']}\n\nAnswer: {b['chosen']}" for b in batch]
        rejected_texts = [f"Question: {b['prompt']}\n\nAnswer: {b['rejected']}" for b in batch]

        chosen_enc = tokenizer(chosen_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        rejected_enc = tokenizer(rejected_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")

        return {
            "chosen_ids": chosen_enc["input_ids"],
            "chosen_mask": chosen_enc["attention_mask"],
            "rejected_ids": rejected_enc["input_ids"],
            "rejected_mask": rejected_enc["attention_mask"],
        }

    dataloader = DataLoader(
        preference_dataset,
        batch_size=cfg.training.per_device_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.training.warmup_steps,
        num_training_steps=cfg.training.max_steps,
    )

    model.train()
    ref_model.eval()
    global_step = 0

    for epoch in range(5):
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = dpo_loss(model, ref_model, **batch, beta=beta)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1

            if global_step % cfg.training.logging_steps == 0:
                logger.log({
                    "dpo_loss": float(loss.item()),
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
