"""Analytical views: how conditions diverge over generations.

Loads experiment results and config snapshots and draws, for each metric,
every condition as a mean +- std band across seeds — the evolutionary story is
the divergence between conditions. A companion view shows the environmental
parameters that define each condition.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt

from .data import (
    METRICS,
    aggregate_condition,
    condition_names,
    environmental_params,
    load_condition_configs,
    load_condition_trials,
)

__all__ = [
    "DEFAULT_METRICS",
    "plot_metric",
    "analytics_figure",
    "environment_table",
    "export_analytics",
]

# food_alignment is base-rate confounded by food density and must not be
# compared across food-abundance conditions; it is excluded from automatic
# charts unless explicitly requested via --metrics food_alignment.
DEFAULT_METRICS = [metric for metric in METRICS if metric != "food_alignment"]


def plot_metric(ax, exp_dir, conditions: List[str], metric: str) -> None:
    """One metric, all conditions overlaid as mean +- std bands."""
    for name in conditions:
        trials = load_condition_trials(exp_dir, name)
        aggregated = aggregate_condition(trials)
        generations = sorted(aggregated)
        means = [aggregated[g][metric]["mean"] for g in generations]
        stds = [aggregated[g][metric]["std"] for g in generations]
        ax.plot(generations, means, label=name)
        ax.fill_between(
            generations,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            alpha=0.15,
        )
    ax.set_title(metric)
    ax.set_xlabel("generation")
    ax.legend(fontsize=7)


def analytics_figure(
    exp_dir,
    metrics: Optional[List[str]] = None,
    conditions: Optional[List[str]] = None,
) -> plt.Figure:
    """One figure with one subplot per selected metric (default: ``DEFAULT_METRICS``)."""
    conditions = conditions or condition_names(exp_dir)
    metrics = metrics or DEFAULT_METRICS
    import math

    cols = 3
    rows = math.ceil(len(metrics) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4.4 * rows), constrained_layout=True)
    axes = axes.reshape(-1)
    for ax, metric in zip(axes, metrics):
        plot_metric(ax, exp_dir, conditions, metric)
    for ax in axes[len(metrics):]:
        ax.axis("off")
    return fig


def environment_table(exp_dir, conditions: Optional[List[str]] = None) -> plt.Figure:
    """A table of each condition's environmental parameters from config snapshots."""
    conditions = conditions or condition_names(exp_dir)
    keys = [
        "population_size", "width", "height", "ticks", "initial_food",
        "food_target", "food_regrowth_per_tick", "repro_threshold",
        "repro_fraction", "metabolism",
    ]
    rows = []
    for name in conditions:
        configs = load_condition_configs(exp_dir, name)
        params = environmental_params(configs)
        rows.append([name] + [params.get(key, "") for key in keys])

    fig, ax = plt.subplots(figsize=(13, 0.5 + 0.4 * len(rows)))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["condition"] + keys,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    ax.set_title("Environmental parameters per condition")
    return fig


def export_analytics(exp_dir, outdir, metrics: Optional[List[str]] = None) -> List[str]:
    """Write one PNG per metric plus the environment table. Returns file paths."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    conditions = condition_names(exp_dir)
    written = []
    for metric in (metrics or DEFAULT_METRICS):
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        plot_metric(ax, exp_dir, conditions, metric)
        path = outdir / f"{metric}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        written.append(str(path))

    table_fig = environment_table(exp_dir, conditions)
    table_path = outdir / "environmental_parameters.png"
    table_fig.savefig(table_path, dpi=110)
    plt.close(table_fig)
    written.append(str(table_path))
    return written
