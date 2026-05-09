"""Logging utilities — W&B integration with local JSON fallback."""
import json
import time
from pathlib import Path
from typing import Any

import wandb


class Logger:
    """Unified logger: W&B online/offline with local JSON fallback."""

    def __init__(
        self,
        project: str = "self_personality_llm",
        entity: str | None = None,
        mode: str = "offline",
        output_dir: str = "data/results",
        experiment_name: str | None = None,
    ):
        self.mode = mode
        self.output_dir = Path(output_dir) / (experiment_name or f"run_{int(time.time())}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.local_log: list[dict] = []

        self.use_wandb = False
        try:
            wandb.init(
                project=project,
                entity=entity,
                mode=mode,
                dir=str(self.output_dir),
                name=experiment_name,
            )
            self.use_wandb = True
        except Exception:
            print(f"W&B unavailable, logging locally to {self.output_dir}")

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        entry = {"step": step, "timestamp": time.time(), **metrics}
        self.local_log.append(entry)
        if self.use_wandb:
            wandb.log(metrics, step=step)

    def save(self) -> None:
        log_path = self.output_dir / "metrics.jsonl"
        with open(log_path, "w") as f:
            for entry in self.local_log:
                f.write(json.dumps(entry) + "\n")

    def finish(self) -> None:
        self.save()
        if self.use_wandb:
            wandb.finish()
