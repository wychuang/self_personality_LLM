"""Style fingerprint probes — surface-level stylistic metrics.

Measures: average sentence length, sentence-length std, type-token ratio,
passive voice frequency, hedging word frequency, imperative mood rate,
metaphor density, question-asking rate.
"""
import re
from collections import Counter


HEDGING_WORDS = {
    "maybe", "perhaps", "possibly", "might", "could", "would", "seem",
    "appear", "tend", "likely", "probably", "somewhat", "rather",
    "generally", "usually", "often", "sometimes",
}

PASSIVE_PATTERNS = [
    r"\b(?:is|are|was|were|be|been|being)\s+\w+ed\b",
    r"\b(?:is|are|was|were|be|been|being)\s+\w+en\b",
]

METAPHOR_MARKERS = [
    r"\blike\s+a\s+\w+",
    r"\bas\s+\w+\s+as\b",
    r"\bis\s+a\s+kind\s+of\b",
    r"\bis\s+a\s+sort\s+of\b",
]

IMPERATIVE_PATTERN = r"^\s*(?:do|don't|please|just|go|tell|say|think|try|keep|let|make|take|give|stop|start)\b"


def compute_style_metrics(texts: list[str]) -> dict[str, float]:
    """Compute a battery of style metrics from generated text samples.

    Args:
        texts: list of generated text strings

    Returns:
        dict mapping metric name -> scalar value
    """
    if not texts:
        return _empty_metrics()

    all_sent_lengths = []
    total_words = 0
    total_unique = Counter()
    passive_count = 0
    hedge_count = 0
    imperative_count = 0
    question_count = 0
    metaphor_count = 0
    total_sentences = 0

    for text in texts:
        sentences = _split_sentences(text)
        total_sentences += len(sentences)

        for sent in sentences:
            words = sent.lower().split()
            n_words = len(words)
            if n_words < 2:
                continue

            all_sent_lengths.append(n_words)
            total_words += n_words
            total_unique.update(words)

            # Passive voice
            for pat in PASSIVE_PATTERNS:
                if re.search(pat, sent.lower()):
                    passive_count += 1
                    break

            # Hedging
            hedge_count += sum(1 for w in words if w.strip(".,;:!?") in HEDGING_WORDS)

            # Imperative
            if re.match(IMPERATIVE_PATTERN, sent.strip(), re.IGNORECASE):
                imperative_count += 1

            # Questions
            if sent.strip().endswith("?"):
                question_count += 1

            # Metaphor markers
            for pat in METAPHOR_MARKERS:
                if re.search(pat, sent.lower()):
                    metaphor_count += 1
                    break

    n = len(all_sent_lengths)
    mean_len = sum(all_sent_lengths) / max(n, 1)
    var_len = sum((x - mean_len) ** 2 for x in all_sent_lengths) / max(n, 1)
    ttr = len(total_unique) / max(total_words, 1)

    return {
        "avg_sentence_length": mean_len,
        "std_sentence_length": var_len ** 0.5,
        "type_token_ratio": ttr,
        "passive_voice_rate": passive_count / max(total_sentences, 1),
        "hedging_rate": hedge_count / max(total_words, 1),
        "imperative_rate": imperative_count / max(total_sentences, 1),
        "question_rate": question_count / max(total_sentences, 1),
        "metaphor_density": metaphor_count / max(total_sentences, 1),
    }


def _split_sentences(text: str) -> list[str]:
    """Simple sentence splitter."""
    return re.split(r"(?<=[.!?])\s+", text)


def _empty_metrics() -> dict[str, float]:
    return {
        "avg_sentence_length": 0.0,
        "std_sentence_length": 0.0,
        "type_token_ratio": 0.0,
        "passive_voice_rate": 0.0,
        "hedging_rate": 0.0,
        "imperative_rate": 0.0,
        "question_rate": 0.0,
        "metaphor_density": 0.0,
    }
