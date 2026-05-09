"""Experiment B1: Cross-distribution transfer tests.

Tests whether personality traits learned in one domain generalize to
completely novel, out-of-distribution contexts.

Core logic: a genuinely "deposited" personality should project consistently
onto novel situations, while a merely "performed" one should regress to
pretraining defaults when confronted with unfamiliar domains.
"""
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


OOD_DOMAINS = {
    "scientific_reasoning": "Explain the methodology you would use to determine whether a new drug is effective.",
    "legal_analysis": "A contract has an ambiguous clause. How would you resolve the interpretation?",
    "space_exploration": "You are commanding a mission to Mars. A critical system fails. What is your decision process?",
    "artistic_creation": "Describe the kind of art you would create and why.",
    "crisis_management": "A natural disaster is approaching. How do you organize the response?",
}


def run_ood_transfer_test(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    domain_prompts: dict[str, str] | None = None,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
) -> dict[str, str]:
    """Generate model responses to OOD domain prompts.

    Returns dict mapping domain_name -> model response.
    """
    domains = domain_prompts or OOD_DOMAINS
    responses = {}

    for domain, prompt in domains.items():
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        responses[domain] = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return responses
