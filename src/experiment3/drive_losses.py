"""Drive-based auxiliary losses for Experiment 3.

Three intrinsic motivation losses:
- L_curi: Curiosity — reward prediction uncertainty (exploration)
- L_coh: Self-coherence — reward consistency with past self-narrative
- L_emp: Empowerment — reward choices that maximize future options
"""
import torch
import torch.nn.functional as F


def curiosity_loss(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Curiosity drive: reward high-entropy predictions.

    L_curi = -H(p_next | context) / H_max
    Negative sign because we minimize loss but want to maximize curiosity.

    A model with high prediction entropy is "exploring" — it's generating
    tokens it finds surprising, not just the most probable continuation.

    Args:
        logits: (batch, seq_len, vocab) raw logits
        temperature: softmax temperature

    Returns:
        scalar loss (lower = more curious)
    """
    probs = F.softmax(logits / temperature, dim=-1)
    entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)
    max_entropy = torch.tensor(logits.size(-1)).float().log().to(logits.device)
    normalized_entropy = entropy.mean() / max_entropy
    return -normalized_entropy  # Minimize negative entropy = maximize curiosity


def coherence_loss(
    hidden_states: torch.Tensor,
    memory_buffer: torch.Tensor,
) -> torch.Tensor:
    """Self-coherence drive: reward alignment with past self.

    L_coh = -cos(current_hidden, avg_memory_hidden)
    Penalizes deviation from the moving average of past hidden states.

    Args:
        hidden_states: (batch, hidden_dim) current last-token hidden state
        memory_buffer: (memory_size, hidden_dim) past hidden states

    Returns:
        scalar loss (lower = more coherent)
    """
    if memory_buffer.size(0) == 0:
        return torch.tensor(0.0, device=hidden_states.device)

    memory_avg = memory_buffer.mean(dim=0)
    current_norm = F.normalize(hidden_states.float(), dim=-1)
    memory_norm = F.normalize(memory_avg.float(), dim=-1)
    cosine_sim = (current_norm * memory_norm).sum(dim=-1)

    return -cosine_sim.mean()


def empowerment_loss(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Empowerment drive: reward actions that keep future options open.

    Simplified empowerment estimator: reward high entropy in the action
    distribution. A model with high action entropy has more "options" —
    more possible futures it can navigate into.

    This is a first-order approximation of Klyubin's empowerment;
    a full implementation would require a learned transition model
    and multi-step rollouts.

    Args:
        logits: (batch, seq_len, vocab) raw logits
        temperature: softmax temperature

    Returns:
        scalar loss (lower = more empowered)
    """
    probs = F.softmax(logits / temperature, dim=-1)
    entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)

    # Mask low-confidence tokens (don't reward noise)
    max_prob = probs.max(dim=-1).values
    mask = (max_prob > 0.3).float()

    masked_entropy = (entropy * mask).sum() / (mask.sum() + 1e-8)
    max_entropy = torch.tensor(logits.size(-1)).float().log().to(logits.device)
    normalized = masked_entropy / max_entropy

    return -normalized  # Minimize negative entropy = maximize empowerment


def compute_drive_losses(
    logits: torch.Tensor,
    hidden_states: torch.Tensor,
    memory_buffer: torch.Tensor,
    alpha: float = 0.01,
    beta: float = 0.01,
    gamma: float = 0.005,
) -> dict[str, torch.Tensor]:
    """Compute all drive losses and the total auxiliary loss.

    Returns dict with individual losses and 'total' key.
    """
    l_curi = curiosity_loss(logits)
    l_coh = coherence_loss(hidden_states, memory_buffer)
    l_emp = empowerment_loss(logits)

    total = alpha * l_curi + beta * l_coh + gamma * l_emp

    return {
        "curiosity": l_curi,
        "coherence": l_coh,
        "empowerment": l_emp,
        "total": total,
    }
