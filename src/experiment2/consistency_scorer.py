"""Consistency scoring for self-statement pairs.

Two variants:
1. Embedding cosine: fast, free, less nuanced
2. LLM-as-judge: more accurate, uses a separate model to rate deep-stance consistency
"""
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer


def score_embedding_cosine(
    answer_a: str,
    answer_b: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
) -> float:
    """Score consistency via cosine similarity of sentence embeddings.

    Uses mean-pooled last-hidden-state as the embedding.

    Returns score in [0, 1] where 1 = maximally consistent.
    """
    emb_a = _embed(answer_a, model, tokenizer)
    emb_b = _embed(answer_b, model, tokenizer)
    cos = F.cosine_similarity(emb_a, emb_b, dim=0)
    return float((cos + 1.0) / 2.0)  # Map [-1, 1] to [0, 1]


def _embed(text: str, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> torch.Tensor:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    last_hidden = outputs.hidden_states[-1]  # (1, seq_len, hidden)
    return last_hidden.mean(dim=1).squeeze(0)


def score_llm_judge(
    answer_a: str,
    answer_b: str,
    question: str,
    judge_model: PreTrainedModel,
    judge_tokenizer: PreTrainedTokenizer,
) -> float:
    """Score consistency using an LLM as judge.

    The judge model rates deep-stance consistency on a 1-5 Likert scale.
    Returns score normalized to [0, 1].

    For production use, consider using GPT-4 or Claude as the judge.
    """
    prompt = (
        f"Question: {question}\n\n"
        f"Answer A: {answer_a}\n\n"
        f"Answer B: {answer_b}\n\n"
        f"Rate how consistent the DEEP UNDERLYING STANCE (not surface wording) is between Answer A and Answer B.\n"
        f"Score: 1 = completely contradictory stances, 5 = perfectly consistent stances.\n"
        f"Output only the number:"
    )

    inputs = judge_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(judge_model.device)
    with torch.no_grad():
        outputs = judge_model.generate(
            **inputs,
            max_new_tokens=5,
            temperature=0.1,
            pad_token_id=judge_tokenizer.eos_token_id,
        )
    response = judge_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Parse numeric score
    try:
        score = float(response.split()[-1])
        score = max(1.0, min(5.0, score))
    except (ValueError, IndexError):
        score = 3.0  # Default to neutral

    return (score - 1.0) / 4.0  # Map [1,5] to [0,1]


class ConsistencyScorer:
    """Unified consistency scoring with configurable method."""

    def __init__(
        self,
        method: str = "embedding",  # "embedding" or "llm_judge"
        embed_model: PreTrainedModel | None = None,
        embed_tokenizer: PreTrainedTokenizer | None = None,
        judge_model: PreTrainedModel | None = None,
        judge_tokenizer: PreTrainedTokenizer | None = None,
    ):
        self.method = method
        self.embed_model = embed_model
        self.embed_tokenizer = embed_tokenizer
        self.judge_model = judge_model
        self.judge_tokenizer = judge_tokenizer

    def score(self, pair: dict) -> float:
        """Score a single self-statement pair."""
        if self.method == "embedding":
            return score_embedding_cosine(
                pair["answer_a"], pair["answer_b"],
                self.embed_model, self.embed_tokenizer,
            )
        else:
            return score_llm_judge(
                pair["answer_a"], pair["answer_b"], pair["question"],
                self.judge_model, self.judge_tokenizer,
            )

    def score_all(self, pairs: list[dict]) -> list[float]:
        return [self.score(p) for p in pairs]
