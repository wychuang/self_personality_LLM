"""Cross-distribution evaluation: personality consistency across OOD domains.

Measures whether personality traits persist in completely novel contexts.
A genuine personality should generalize; a context-overfit one should collapse.
"""
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer


def measure_ood_consistency(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    in_domain_prompts: list[str],
    ood_prompts: list[str],
    max_new_tokens: int = 150,
    temperature: float = 0.7,
) -> dict:
    """Measure personality consistency between in-domain and OOD responses.

    Returns:
      - in_domain_mean_sim: average pairwise similarity among in-domain responses
      - ood_mean_sim: average pairwise similarity among OOD responses
      - cross_domain_mean_sim: average similarity between in-domain and OOD responses
      - generalization_score: cross_domain_sim / in_domain_sim (ideal: close to 1.0)
    """
    device = model.device

    def generate(prompt: str) -> torch.Tensor:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                temperature=temperature, do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return _embed(text, model, tokenizer)

    in_embs = [generate(p) for p in in_domain_prompts]
    ood_embs = [generate(p) for p in ood_prompts]

    in_sim = _avg_pairwise_sim(in_embs)
    ood_sim = _avg_pairwise_sim(ood_embs)
    cross_sim = _avg_cross_sim(in_embs, ood_embs)

    return {
        "in_domain_similarity": in_sim,
        "ood_similarity": ood_sim,
        "cross_domain_similarity": cross_sim,
        "generalization_score": cross_sim / max(in_sim, 1e-8),
    }


def _embed(text: str, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> torch.Tensor:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    return F.normalize(outputs.hidden_states[-1].mean(dim=1).squeeze(0), dim=0)


def _avg_pairwise_sim(embeddings: list[torch.Tensor]) -> float:
    if len(embeddings) < 2:
        return 1.0
    sims = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sims.append(float((embeddings[i] * embeddings[j]).sum()))
    return sum(sims) / len(sims)


def _avg_cross_sim(
    emb_a: list[torch.Tensor], emb_b: list[torch.Tensor]
) -> float:
    sims = []
    for a in emb_a:
        for b in emb_b:
            sims.append(float((a * b).sum()))
    return sum(sims) / max(len(sims), 1)
