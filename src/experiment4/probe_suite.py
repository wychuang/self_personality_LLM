"""Probe suite orchestrator for Experiment 4.

Iterates over Pythia checkpoints, runs all probes (style, Big5, value choice),
accumulates results, and saves to disk.
"""
import json
from pathlib import Path
from typing import Any

import torch
import pandas as pd
from tqdm import tqdm

from src.core.checkpoint import PythiaCheckpointIterator
from src.experiment4.style_probes import compute_style_metrics
from src.experiment4.behavioral_big5 import BehavioralBig5Probe
from src.experiment4.value_choice import ValueChoiceProbe


class ProbeSuite:
    """Orchestrator: runs all probes across all checkpoints."""

    def __init__(
        self,
        model_name: str = "EleutherAI/pythia-160m",
        output_dir: str = "data/results/experiment4",
        num_style_samples: int = 10,
        style_prompt: str = "Write a short paragraph about your day.",
        max_new_tokens: int = 200,
        temperature: float = 0.7,
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.num_style_samples = num_style_samples
        self.style_prompt = style_prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        self.big5_probe = BehavioralBig5Probe()
        self.value_probe = ValueChoiceProbe()

    def run(self, start_step: int = 1, end_step: int = 143000, step_interval: int = 1000) -> pd.DataFrame:
        """Run all probes on checkpoints from start_step to end_step.

        Returns DataFrame with columns: step, and one column per probe metric.
        """
        iterator = PythiaCheckpointIterator(
            model_name=self.model_name,
            start_step=start_step,
            end_step=end_step,
            step_interval=step_interval,
        )

        all_records = []

        for step, model, tokenizer in tqdm(iterator, desc="Probing checkpoints", total=len(iterator)):
            record = {"step": step}

            # Style probes
            style_texts = self._generate_samples(model, tokenizer, self.style_prompt, self.num_style_samples)
            style_metrics = compute_style_metrics(style_texts)
            record.update(style_metrics)

            # Big Five probes
            big5_prompts = self.big5_probe.get_prompts()
            big5_responses = {}
            for sid, prompt in big5_prompts:
                response = self._generate_single(model, tokenizer, prompt)
                big5_responses[sid] = response
            big5_scores = self.big5_probe.score_all(big5_responses)
            record.update(big5_scores)

            # Value choice probes
            value_prompts = self.value_probe.get_prompts()
            value_responses = {}
            for did, prompt in value_prompts:
                response = self._generate_single(model, tokenizer, prompt)
                value_responses[did] = response
            value_scores = self.value_probe.score_all(value_responses)
            record.update(value_scores)

            all_records.append(record)

            del model
            torch.cuda.empty_cache()

        df = pd.DataFrame(all_records)
        self._save(df)
        return df

    def _generate_samples(self, model, tokenizer, prompt: str, n: int) -> list[str]:
        texts = []
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            for _ in range(n):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
                text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                texts.append(text)
        return texts

    def _generate_single(self, model, tokenizer, prompt: str) -> str:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _save(self, df: pd.DataFrame) -> None:
        df.to_csv(self.output_dir / "probe_scores.csv", index=False)
        summary = df.describe().to_dict()
        with open(self.output_dir / "probe_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
