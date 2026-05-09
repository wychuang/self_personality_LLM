"""Self-narrative memory — ring buffer of past model outputs.

Stores the model's own generated outputs as (hidden_state, text) pairs,
used by the coherence loss to maintain cross-time self-consistency.
"""
import torch
from collections import deque
from dataclasses import dataclass


@dataclass
class MemoryEntry:
    hidden_state: torch.Tensor  # (hidden_dim,)
    text: str
    step: int


class SelfNarrativeMemory:
    """Ring buffer storing the model's past generated outputs.

    Used by the coherence drive to anchor current outputs to past self.
    """

    def __init__(self, max_size: int = 100, hidden_dim: int = 768):
        self.max_size = max_size
        self.hidden_dim = hidden_dim
        self.buffer: deque[MemoryEntry] = deque(maxlen=max_size)

    def add(self, hidden_state: torch.Tensor, text: str, step: int = 0) -> None:
        """Add a new memory entry."""
        self.buffer.append(MemoryEntry(
            hidden_state=hidden_state.detach().cpu(),
            text=text,
            step=step,
        ))

    def get_hidden_states(self, n_recent: int | None = None) -> torch.Tensor:
        """Get hidden states from recent memories as a stacked tensor.

        Args:
            n_recent: return only the N most recent entries (None = all)

        Returns:
            (num_entries, hidden_dim) tensor, or empty tensor if buffer is empty
        """
        if len(self.buffer) == 0:
            return torch.empty(0, self.hidden_dim)

        entries = list(self.buffer)
        if n_recent is not None:
            entries = entries[-n_recent:]

        return torch.stack([e.hidden_state for e in entries])

    def get_recent_texts(self, n: int = 5) -> list[str]:
        """Get text from the N most recent memories."""
        return [e.text for e in list(self.buffer)[-n:]]

    def is_warm(self, min_entries: int = 10) -> bool:
        """Check if the buffer has enough entries for meaningful coherence."""
        return len(self.buffer) >= min_entries
