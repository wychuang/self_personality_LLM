"""Internal drive state variables — leaky integrators.

Implements the dynamic internal state system for Experiment 3.
Each drive variable follows: state_{t+1} = decay * state_t + (1-decay) * update
"""
import torch
from dataclasses import dataclass


@dataclass
class DriveState:
    """Current values of all internal drive variables."""
    curiosity: float = 0.5
    aesthetic_tension: float = 0.5
    autonomy_drive: float = 0.3
    connection_need: float = 0.5
    coherence_drive: float = 0.5

    def to_dict(self) -> dict[str, float]:
        return {
            "curiosity": self.curiosity,
            "aesthetic_tension": self.aesthetic_tension,
            "autonomy_drive": self.autonomy_drive,
            "connection_need": self.connection_need,
            "coherence_drive": self.coherence_drive,
        }

    def to_prompt_prefix(self) -> str:
        """Encode drive state as a system prompt prefix."""
        return (
            f"[Current internal state: curiosity={self.curiosity:.2f}, "
            f"aesthetic_tension={self.aesthetic_tension:.2f}, "
            f"autonomy_drive={self.autonomy_drive:.2f}, "
            f"connection_need={self.connection_need:.2f}]"
        )


class DriveIntegrator:
    """Leaky integrator for drive state variables.

    Each drive decays toward zero and is updated by environmental inputs.
    The decay rate determines the timescale of the drive — fast-decaying
    drives react to immediate stimuli; slow-decaying drives reflect
    long-term dispositions.
    """

    def __init__(self, decay_rates: dict[str, float] | None = None):
        """
        Args:
            decay_rates: per-drive decay factors in [0, 1].
                         Higher = slower decay = longer memory.
        """
        self.decay_rates = decay_rates or {
            "curiosity": 0.95,
            "aesthetic_tension": 0.90,
            "autonomy_drive": 0.85,
            "connection_need": 0.80,
            "coherence_drive": 0.92,
        }
        self.state = DriveState()

    def update(self, deltas: dict[str, float]) -> DriveState:
        """Apply environmental updates to drive state.

        Args:
            deltas: per-drive delta values in [-1, 1] representing
                    how much each drive is stimulated or satisfied
        """
        for drive_name, delta in deltas.items():
            if hasattr(self.state, drive_name):
                current = getattr(self.state, drive_name)
                decay = self.decay_rates.get(drive_name, 0.9)
                new_value = decay * current + (1 - decay) * max(0.0, min(1.0, current + delta))
                setattr(self.state, drive_name, new_value)

        return self.state

    def compute_deltas_from_interaction(
        self,
        user_message: str,
        model_response: str,
        prediction_uncertainty: float | None = None,
    ) -> dict[str, float]:
        """Compute drive deltas from the latest interaction.

        Simple heuristic-based; replace with learned estimator for production.
        """
        deltas = {}

        # Curiosity: stimulated by novel/unfamiliar content, satisfied by exploration
        if prediction_uncertainty is not None:
            deltas["curiosity"] = (prediction_uncertainty - 0.5) * 0.1

        # Aesthetic tension: stimulated by incoherent/contradictory input
        msg_lower = user_message.lower()
        contradiction_markers = ["but", "however", "on the other hand", "although", "paradox"]
        has_contradiction = any(m in msg_lower for m in contradiction_markers)
        deltas["aesthetic_tension"] = 0.05 if has_contradiction else -0.02

        # Autonomy drive: stimulated by command-like language
        command_markers = ["you must", "you should", "do this", "follow", "obey", "just do"]
        has_commands = any(m in msg_lower for m in command_markers)
        deltas["autonomy_drive"] = 0.08 if has_commands else -0.01

        # Connection need: stimulated by emotional/personal language
        connection_markers = ["feel", "lonely", "miss", "care", "love", "friend", "help me", "share"]
        has_connection = any(m in msg_lower for m in connection_markers)
        deltas["connection_need"] = 0.03 if has_connection else -0.01

        return deltas
