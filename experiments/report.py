"""Markdown report + trajectory plots from aggregated experiment results."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .analysis import (
    NUMERIC_METRICS,
    aggregate,
    compare_conditions,
    condition_names,
    final_summary,
    load_trials,
)

__all__ = ["plot_trajectories", "write_report"]

PLOT_METRICS = [
    "best_fitness",
    "survival_rate",
    "average_lifetime",
    "food_consumed",
    "reproductive_success",
    "node_count",
    "connection_count",
    "species_count",
]


def plot_trajectories(
    exp_dir: Path,
    conditions: List[str],
    outdir: Path,
) -> None:
    """Mean +- std trajectories for each metric, all conditions overlaid."""
    outdir.mkdir(parents=True, exist_ok=True)
    series = {name: aggregate(load_trials(exp_dir, name)) for name in conditions}

    for metric in PLOT_METRICS:
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        for name, agg in series.items():
            gens = sorted(agg)
            means = [agg[g][metric]["mean"] for g in gens]
            stds = [agg[g][metric]["std"] for g in gens]
            ax.plot(gens, means, label=name)
            ax.fill_between(
                gens,
                [m - s for m, s in zip(means, stds)],
                [m + s for m, s in zip(means, stds)],
                alpha=0.15,
            )
        ax.set_title(f"{metric} across conditions")
        ax.set_xlabel("generation")
        ax.legend(fontsize=8)
        fig.savefig(outdir / f"{metric}.png", dpi=110)
        plt.close(fig)


def _trial_table(exp_dir: Path, condition: str) -> str:
    header = f"{'seed':>4} | {'best fit':>8} | {'surv':>5} | {'life':>5} | {'food':>5} | {'offspring':>8}"
    lines = [header, "-" * len(header)]
    directory = Path(exp_dir) / condition
    for path in sorted(directory.glob("*.json")):
        if path.name.endswith(".config.json"):
            continue
        trial = json.loads(path.read_text())
        final = trial[-1]
        seed = path.stem
        lines.append(
            f"{seed:>4} | {final['best_fitness']:>8.3f} | {final['survival_rate']:>5.2f} | "
            f"{final['average_lifetime']:>5.1f} | {final['food_consumed']:>5.0f} | "
            f"{final['reproductive_success']:>8.1f}"
        )
    return "\n".join(lines)


def write_report(
    exp_dir: Path,
    report_path: Path,
    *,
    control: str = "control",
    plot_dir: Path = None,
) -> None:
    """Write a markdown report summarizing all conditions vs the control."""
    exp_dir = Path(exp_dir)
    conditions = condition_names(exp_dir)
    sections = ["# Experiment Report\n", f"Conditions: {', '.join(conditions)}\n"]

    for name in conditions:
        trials = load_trials(exp_dir, name)
        if not trials:
            continue
        summary = final_summary(trials)
        sections.append(f"\n## {name}\n")
        sections.append(f"\n```\n{_trial_table(exp_dir, name)}\n```\n")
        sections.append("- Final generation (mean ± std):")
        for metric in NUMERIC_METRICS:
            stats = summary.get(metric)
            if stats:
                sections.append(
                    f"  - {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}"
                )
        if name != control and control in conditions:
            deltas = compare_conditions(exp_dir, name, control)
            sections.append(f"\n- Δ vs `{control}` (final generation):")
            for metric in NUMERIC_METRICS:
                if metric in deltas:
                    d, a, b = deltas[metric]
                    sections.append(f"  - {metric}: {d:+.4f} (was {b:.4f} -> {a:.4f})")

    if plot_dir is not None:
        plot_trajectories(exp_dir, conditions, Path(plot_dir))
        sections.append(f"\nPlots written to `{plot_dir}`.\n")

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text("\n".join(sections) + "\n")
