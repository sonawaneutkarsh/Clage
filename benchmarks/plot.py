"""Evolutionary-progress plots for the benchmark problems (headless, Agg)."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .problems import Problem
from .run import TrialResult

__all__ = ["plot_problem"]


def _series(results, key):
    return [[row[key] for row in r.history] for r in results]


def _mean_series(series):
    min_len = min(len(t) for t in series)
    return [sum(v) / len(v) for v in zip(*(t[:min_len] for t in series))]


def plot_problem(problem: Problem, results: list[TrialResult], outdir: str) -> str:
    """Save a 2x2 progress figure for a problem. Returns the PNG path."""
    os.makedirs(outdir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    best = _series(results, "best_fitness")
    mean = _series(results, "mean_fitness")
    conns = _series(results, "best_connection_count")
    nodes = _series(results, "best_node_count")
    species = _series(results, "species_count")

    for trial in best:
        axes[0, 0].plot(range(1, len(trial) + 1), trial, alpha=0.4, linewidth=1)
    mean_best = _mean_series(best)
    axes[0, 0].plot(
        range(1, len(mean_best) + 1),
        mean_best,
        color="black", linewidth=2, label="mean",
    )
    axes[0, 0].set_title(f"{problem.name}: best fitness")
    axes[0, 0].set_xlabel("generation")
    axes[0, 0].legend()

    for trial in mean:
        axes[0, 1].plot(range(1, len(trial) + 1), trial, alpha=0.4, linewidth=1)
    axes[0, 1].set_title("mean fitness")
    axes[0, 1].set_xlabel("generation")

    for trial in conns:
        axes[1, 0].plot(range(1, len(trial) + 1), trial, alpha=0.4, linewidth=1)
    for trial in nodes:
        axes[1, 0].plot(
            range(1, len(trial) + 1), trial, alpha=0.4, linewidth=1, linestyle="--"
        )
    axes[1, 0].set_title("best-genome complexity (conns solid, nodes dashed)")
    axes[1, 0].set_xlabel("generation")

    for trial in species:
        axes[1, 1].plot(range(1, len(trial) + 1), trial, alpha=0.4, linewidth=1)
    axes[1, 1].set_title("species count")
    axes[1, 1].set_xlabel("generation")

    solved = sum(1 for r in results if r.solved)
    fig.suptitle(
        f"{problem.name} — success {solved}/{len(results)}  |  "
        f"{problem.success_description}",
        fontsize=10,
    )

    path = os.path.join(outdir, f"{problem.name}.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
