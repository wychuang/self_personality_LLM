"""Empowerment estimator — Klyubin-style future-state entropy.

Empowerment = I(a; s' | s) = maximum mutual information between actions
and future states, representing the agent's capacity to influence its future.

A simplified single-step estimator based on action-entropy is in drive_losses.py.
This module provides the scaffolding for a full transition-model-based estimator.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EmpowermentEstimator(nn.Module):
    """Learned empowerment estimator using a transition model.

    Estimates empowerment as the channel capacity between actions (output tokens)
    and future states (subsequent hidden states), following Klyubin et al.

    This is a placeholder architecture; proper empowerment estimation requires
    encoder-decoder training and multi-step planning.
    """

    def __init__(self, hidden_dim: int, n_bins: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_bins = n_bins

        # Transition model: p(s' | s, a)
        self.transition = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Source distribution model: p(a | s)
        self.policy_head = nn.Linear(hidden_dim, n_bins)

    def forward(
        self,
        current_state: torch.Tensor,
        action_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate next state given current state and action.

        Args:
            current_state: (batch, hidden_dim)
            action_embedding: (batch, hidden_dim) — embedded action token

        Returns:
            predicted_next_state: (batch, hidden_dim)
        """
        combined = torch.cat([current_state, action_embedding], dim=-1)
        return self.transition(combined)

    def estimate_empowerment(
        self,
        current_state: torch.Tensor,
        n_samples: int = 10,
    ) -> torch.Tensor:
        """Estimate empowerment for a given state.

        Simplified: computes the entropy of predicted next states
        under a uniformly sampled action distribution.

        Args:
            current_state: (batch, hidden_dim)
            n_samples: number of random actions to sample

        Returns:
            empowerment estimate (scalar)
        """
        batch_size = current_state.size(0)
        state_repeated = current_state.repeat_interleave(n_samples, dim=0)

        # Sample random action embeddings (uniform in latent space)
        random_actions = torch.randn(
            batch_size * n_samples, self.hidden_dim,
            device=current_state.device, dtype=current_state.dtype,
        )

        # Predict next states
        next_states = self.forward(state_repeated, random_actions)

        # Empowerment ≈ spread of reachable next states
        next_states_reshaped = next_states.view(batch_size, n_samples, self.hidden_dim)
        variance = next_states_reshaped.var(dim=1).mean()
        return variance


def simple_empowerment(logits: torch.Tensor, k: int = 5) -> torch.Tensor:
    """Fast empowerment approximation from next-token distribution.

    Empowerment ≈ H(top-k action distribution).
    Actions are output tokens; more diverse top-k = more options = higher empowerment.

    Args:
        logits: (batch, seq_len, vocab) raw logits
        k: number of top actions to consider

    Returns:
        scalar empowerment score
    """
    topk_logits, _ = torch.topk(logits, k=k, dim=-1)
    topk_probs = F.softmax(topk_logits, dim=-1)
    topk_entropy = -(topk_probs * (topk_probs + 1e-10).log()).sum(dim=-1)
    max_entropy = torch.tensor(k).float().log().to(logits.device)
    return (topk_entropy.mean() / max_entropy)
