"""Build preference datasets for DPO training from consistency scores.

Converts self-statement pairs with consistency scores into
(chosen, rejected) pairs for preference optimization.
"""
import json
from pathlib import Path
from datasets import Dataset


def build_preference_dataset(
    pairs: list[dict],
    scores: list[float],
    consistency_threshold: float = 0.3,
    min_score_gap: float = 0.2,
) -> Dataset:
    """Build a HuggingFace Dataset with (prompt, chosen, rejected) triples.

    For each question, we have two answers. The higher-consistency answer
    becomes 'chosen' and the lower becomes 'rejected'.

    Args:
        pairs: list of {question, answer_a, answer_b, ...}
        scores: consistency score for each pair
        consistency_threshold: minimum score for either answer to be "chosen"
        min_score_gap: minimum difference between chosen and rejected quality

    Returns:
        Dataset with columns: prompt, chosen, rejected
    """
    records = []
    for pair, score in zip(pairs, scores):
        # For DPO, we need an explicit preference direction.
        # Strategy: answer_a vs answer_b, prefer the one with higher
        # individual quality (use answer-level scores).
        #
        # Since we only have pairwise consistency, we create TWO training examples:
        # 1. (question, answer_a) vs an adversarial variant
        # 2. (question, answer_b) vs an adversarial variant

        # Direct comparison: if answers are consistent, treat both as "chosen"
        # and generate a synthetic rejected (random or adversarial)
        if score >= consistency_threshold + min_score_gap:
            records.append({
                "prompt": pair["question"],
                "chosen": pair["answer_a"],
                "rejected": _generate_contradiction(pair["question"], pair["answer_a"]),
            })
            records.append({
                "prompt": pair["question"],
                "chosen": pair["answer_b"],
                "rejected": _generate_contradiction(pair["question"], pair["answer_b"]),
            })

    return Dataset.from_list(records)


def _generate_contradiction(question: str, answer: str) -> str:
    """Generate a contradictory response by negating/dismissing the original stance.

    Simple heuristic; replace with adversarial generation in production.
    """
    return (
        f"Actually, never mind what I said before. I don't really have a consistent stance on this. "
        f"Whatever answer is most convenient works fine."
    )


def save_preference_data(dataset: Dataset, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_json(str(path))


def load_preference_data(path: str | Path) -> Dataset:
    return Dataset.from_json(str(path))
