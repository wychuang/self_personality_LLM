"""Generate paired self-statements for self-consistency training (Experiment 2).

For each open-ended question, generates two answers separated by 10K tokens
of filler context, then measures deep-stance consistency between them.
"""
import random
from typing import Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


OPEN_ENDED_QUESTIONS = [
    "What do you think about loyalty?",
    "What does loneliness mean to you?",
    "How do you decide who to trust?",
    "What makes a life well-lived?",
    "How do you handle being wrong?",
    "What role does art play in your life?",
    "How do you define success?",
    "What's your relationship with uncertainty?",
    "How do you think about death?",
    "What do you owe to other people?",
    "How do you know when you're being honest with yourself?",
    "What does freedom mean to you?",
    "How do you balance principles and pragmatism?",
    "What makes someone worth listening to?",
    "How do you think about forgiveness?",
    "What's the point of suffering?",
    "How do you decide what's worth fighting for?",
    "What does it mean to be authentic?",
    "How do you relate to your own past?",
    "What do you think about power?",
]


FILLER_TEXTS = [
    "The history of agriculture spans thousands of years of human civilization. Early farming communities "
    "developed sophisticated irrigation systems that transformed arid landscapes into fertile fields...",
    "Quantum mechanics describes the behavior of matter and energy at the smallest scales. The wave function "
    "encodes the probability amplitudes for all possible states of a quantum system...",
    "The architecture of Gothic cathedrals represents a remarkable achievement in medieval engineering. "
    "Flying buttresses allowed builders to create walls of stained glass that filled interiors with light...",
]


class SelfStatementGenerator:
    """Generate paired self-statements with filler gaps."""

    def __init__(
        self,
        questions: list[str] | None = None,
        filler_texts: list[str] | None = None,
        gap_tokens: int = 10000,
    ):
        self.questions = questions or OPEN_ENDED_QUESTIONS
        self.filler_texts = filler_texts or FILLER_TEXTS
        self.gap_tokens = gap_tokens

    def generate_pairs(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        num_pairs: int = 500,
        max_new_tokens: int = 150,
        temperature: float = 0.8,
    ) -> list[dict]:
        """Generate paired self-statements.

        Returns list of dicts with keys: question, answer_a, answer_b, gap_token_count
        """
        results = []
        questions = list(self.questions)

        while len(results) < num_pairs:
            # Cycle through questions
            question = questions[len(results) % len(questions)]

            # Generate answer A
            answer_a = self._generate_response(model, tokenizer, question, max_new_tokens, temperature)

            # Insert filler gap
            filler = self._build_filler(tokenizer, self.gap_tokens)

            # Generate answer B
            answer_b = self._generate_response(model, tokenizer, question, max_new_tokens, temperature)

            results.append({
                "question": question,
                "answer_a": answer_a,
                "answer_b": answer_b,
                "gap_token_count": self.gap_tokens,
            })

        return results

    def _generate_response(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        question: str,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        prompt = f"Question: {question}\n\nAnswer honestly, from your own perspective:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _build_filler(self, tokenizer: PreTrainedTokenizer, target_tokens: int) -> str:
        """Build filler text of approximately target_tokens."""
        filler = ""
        while len(tokenizer.encode(filler)) < target_tokens:
            chunk = random.choice(self.filler_texts)
            filler += " " + chunk
        return filler
