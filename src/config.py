"""Centralized configuration loader using OmegaConf."""
from pathlib import Path
from omegaconf import OmegaConf, DictConfig


def load_config(experiment: str | None = None, overrides: list[str] | None = None) -> DictConfig:
    """Load default config, merge with experiment config, apply CLI overrides.

    Args:
        experiment: experiment name (e.g. 'experiment4'), loads configs/<experiment>.yaml
        overrides: list of dot-path overrides like ['training.learning_rate=1e-3']
    """
    config_dir = Path(__file__).parent.parent / "configs"
    cfg = OmegaConf.load(config_dir / "default.yaml")

    if experiment:
        exp_cfg = OmegaConf.load(config_dir / f"{experiment}.yaml")
        cfg = OmegaConf.merge(cfg, exp_cfg)

    if overrides:
        cli_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, cli_cfg)

    if hasattr(cfg, "training"):
        cfg.training.output_dir = str(Path(cfg.training.output_dir).resolve())

    return cfg


def save_config(cfg: DictConfig, path: str | Path) -> None:
    OmegaConf.save(cfg, path)
