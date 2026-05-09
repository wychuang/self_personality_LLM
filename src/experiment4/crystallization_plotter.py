"""Crystallization curve plotter and critical period detection.

Takes probe scores from ProbeSuite and produces:
1. Trait vs. training step line plots
2. Critical period detection (derivative peaks, stability onset)
3. ERG (Erasure Resistance Gradient) estimation
"""
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def detect_critical_periods(
    series: np.ndarray,
    steps: np.ndarray,
    stability_threshold: float = 0.05,
    window: int = 5,
) -> dict[str, Any]:
    """Detect critical periods in a trait time series.

    Args:
        series: trait scores over time
        steps: corresponding training steps
        stability_threshold: fraction of final value to define "stable"
        window: smoothing window for derivative

    Returns dict with:
      - max_change_step: step where trait changed fastest
      - stability_onset_step: first step where trait stays within threshold of final
      - final_value: last value in series
      - crystallized: whether the trait reached stable state
    """
    if len(series) < 2:
        return {"max_change_step": None, "stability_onset_step": None,
                "final_value": float(series[-1]) if len(series) else None,
                "crystallized": False}

    # Smooth and compute derivative
    smoothed = pd.Series(series).rolling(window=min(window, len(series)), min_periods=1).mean().values
    derivative = np.gradient(smoothed)
    abs_deriv = np.abs(derivative)

    max_change_idx = int(np.argmax(abs_deriv))
    max_change_step = int(steps[max_change_idx])

    # Stability onset: first step after which the value stays within threshold % of final
    final_val = series[-1]
    if abs(final_val) < 1e-8:
        stability_onset_step = int(steps[-1])
        crystallized = True
    else:
        stability_onset_step = None
        for i in range(len(series)):
            max_deviation = np.max(np.abs(series[i:] - final_val) / (abs(final_val) + 1e-8))
            if max_deviation <= stability_threshold:
                stability_onset_step = int(steps[i])
                break
        crystallized = stability_onset_step is not None

    return {
        "max_change_step": max_change_step,
        "stability_onset_step": stability_onset_step,
        "final_value": float(final_val),
        "crystallized": crystallized,
    }


def compute_erg(series: np.ndarray) -> float:
    """Estimate Erasure Resistance Gradient from trait crystallization curve.

    ERG = how much the early trend resists being "overwritten" by later changes.
    Computed as: ratio of early-phase variance to late-phase variance.
    Higher ERG = early phase more stable = stronger "sedimentation."
    """
    if len(series) < 4:
        return 0.0

    split = len(series) // 2
    early = series[:split]
    late = series[split:]

    early_var = np.var(early) if len(early) > 1 else 0.0
    late_var = np.var(late) if len(late) > 1 else 1e-8

    if late_var < 1e-8:
        return float("inf") if early_var > 0 else 0.0

    return float(early_var / late_var)


def plot_crystallization_curves(
    df: pd.DataFrame,
    output_dir: str | Path,
    trait_columns: list[str] | None = None,
    erg_columns: list[str] | None = None,
) -> Path:
    """Generate crystallization curve plots and critical period report.

    Args:
        df: DataFrame with 'step' column + trait score columns
        output_dir: where to save plots
        trait_columns: columns to plot as crystallization curves (default: all numeric except 'step')
        erg_columns: columns to compute ERG for (default: same as trait_columns)

    Returns path to output directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if trait_columns is None:
        trait_columns = [c for c in df.columns if c != "step"]
    if erg_columns is None:
        erg_columns = trait_columns

    steps = df["step"].values

    # Plot all crystallization curves
    sns.set_style("whitegrid")
    n_traits = len(trait_columns)
    n_cols = min(3, n_traits)
    n_rows = (n_traits + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)

    critical_periods = {}

    for idx, col in enumerate(trait_columns):
        ax = axes[idx // n_cols][idx % n_cols]
        series = df[col].values

        ax.plot(steps, series, "b-", linewidth=1.5, alpha=0.8)

        # Detect critical periods
        cp_info = detect_critical_periods(series, steps)
        critical_periods[col] = cp_info

        # Mark max-change point
        if cp_info["max_change_step"] is not None:
            ax.axvline(x=cp_info["max_change_step"], color="r", linestyle="--", alpha=0.5, label="max Δ")

        # Mark stability onset
        if cp_info["stability_onset_step"] is not None:
            ax.axvline(x=cp_info["stability_onset_step"], color="g", linestyle=":", alpha=0.5, label="stable")

        ax.set_title(col, fontsize=10)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Score")

        if idx == 0:
            ax.legend(fontsize=7)

    # Hide unused subplots
    for idx in range(n_traits, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_dir / "crystallization_curves.png", dpi=150)
    plt.close(fig)

    # Compute ERG
    erg_scores = {}
    for col in erg_columns:
        erg_scores[col] = compute_erg(df[col].values)

    # Save analysis
    analysis = {
        "critical_periods": critical_periods,
        "erg_scores": erg_scores,
    }
    with open(output_dir / "critical_periods.json", "w") as f:
        json.dump(analysis, f, indent=2, default=str)

    # ERG bar chart
    if erg_scores:
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        sorted_erg = sorted(erg_scores.items(), key=lambda x: x[1], reverse=True)
        names = [x[0] for x in sorted_erg]
        values = [x[1] for x in sorted_erg]
        bars = ax2.bar(names, values)
        ax2.set_title("Erasure Resistance Gradient (ERG) by Trait")
        ax2.set_ylabel("ERG (early_var / late_var)")
        ax2.tick_params(axis="x", rotation=45)
        ax2.axhline(y=1.0, color="r", linestyle="--", alpha=0.3, label="uniform (no sedimentation)")
        ax2.legend()
        plt.tight_layout()
        fig2.savefig(output_dir / "erg_scores.png", dpi=150)
        plt.close(fig2)

    return output_dir
