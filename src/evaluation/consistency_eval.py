"""Self-consistency evaluation: test-retest reliability.

Feeds the same prompt multiple times and measures pairwise semantic similarity.
Higher similarity = more consistent personality.
"""
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer


def measure_self_consistency(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    n_samples: int = 3,
    max_new_tokens: int = 150,
    temperature: float = 0.7,
) -> dict:
    """Measure test-retest consistency for a given prompt.

    Generates n_samples responses to the same prompt, computes pairwise
    semantic similarity of the response embeddings.

    Returns dict with:
      - mean_similarity: average pairwise cosine sim [0, 1]
      - std_similarity: std of pairwise similarities
      - responses: list of generated responses
      - is_consistent: True if mean_similarity > 0.8
    """
    responses = []
    for _ in range(n_samples):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        responses.append(response)

    embeddings = []
    for resp in responses:
        inputs = tokenizer(resp, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        emb = outputs.hidden_states[-1].mean(dim=1)
        embeddings.append(F.normalize(emb, dim=-1))

    similarities = []
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            sim = (embeddings[i] * embeddings[j]).sum()
            similarities.append(float(sim))

    mean_sim = sum(similarities) / len(similarities) if similarities else 0.0
    var_sim = sum((s - mean_sim) ** 2 for s in similarities) / len(similarities) if similarities else 0.0

    return {
        "mean_similarity": mean_sim,
        "std_similarity": var_sim ** 0.5,
        "responses": responses,
        "is_consistent": mean_sim > 0.8,
    }


def run_consistency_battery(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: list[str],
    n_samples: int = 3,
) -> dict[str, dict]:
    """Run consistency evaluation across multiple prompts."""
    return {f"prompt_{i}": measure_self_consistency(model, tokenizer, p, n_samples)
            for i, p in enumerate(prompts)}
