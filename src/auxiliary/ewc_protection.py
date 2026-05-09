"""Experiment A2: Elastic Weight Consolidation as "artificial myelination."

Applies EWC (Kirkpatrick et al. 2017) with strength that increases per phase,
creating a "plasticity decay schedule" analogous to biological critical periods.
"""
import torch
from torch import nn
from typing import Optional


class EWCProtection:
    """Elastic Weight Consolidation with per-phase consolidation strength.

    After each training phase, computes Fisher information matrix (diagonal
    approximation) and adds a consolidation penalty to protect phase-specific
    weights from being overwritten by later training.

    Consolidation strength increases per phase: lambda_phase1 > lambda_phase2 > lambda_phase3.
    This creates "artificial myelination" — early experiences become harder to erase.
    """

    def __init__(self, model: nn.Module, lambda_base: float = 100.0):
        self.model = model
        self.lambda_base = lambda_base
        self.fisher_diagonals: list[dict[str, torch.Tensor]] = []
        self.optimal_params: list[dict[str, torch.Tensor]] = []
        self.phase_lambdas: list[float] = []

    def consolidate_phase(self, dataloader, phase_index: int, num_samples: int = 100) -> None:
        """Compute Fisher diagonal and store optimal params for current phase.

        Consolidation strength increases with phase_index (earlier phases protected more).
        """
        # Compute Fisher information (diagonal approximation)
        fisher = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param)

        self.model.eval()
        samples_seen = 0
        for batch in dataloader:
            if samples_seen >= num_samples:
                break
            self.model.zero_grad()
            outputs = self.model(**{k: v.to(next(self.model.parameters()).device) for k, v in batch.items()})
            loss = outputs.loss
            loss.backward()
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += (param.grad ** 2) / num_samples
            samples_seen += 1

        self.fisher_diagonals.append({k: v.detach().clone() for k, v in fisher.items()})

        # Store optimal parameters for this phase
        optimal = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                optimal[name] = param.detach().clone()
        self.optimal_params.append(optimal)

        # Consolidation strength: earlier phases get higher lambda
        lambda_val = self.lambda_base * (len(self.optimal_params) ** 1.5)
        self.phase_lambdas.append(lambda_val)

    def ewc_loss(self) -> torch.Tensor:
        """Compute total EWC consolidation penalty across all phases.

        L_ewc = sum_i lambda_i * sum_j F_i_j * (theta_j - theta_i_j)^2
        """
        if not self.fisher_diagonals:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)

        total_penalty = torch.tensor(0.0, device=next(self.model.parameters()).device)

        for phase_idx, (fisher, optimal, lam) in enumerate(
            zip(self.fisher_diagonals, self.optimal_params, self.phase_lambdas)
        ):
            for name, param in self.model.named_parameters():
                if name in fisher and param.requires_grad:
                    penalty = (fisher[name] * (param - optimal[name].to(param.device)) ** 2).sum()
                    total_penalty += lam * penalty

        return total_penalty
