"""Analysis: aggregate per-generation metrics across independent seeds.

Loads the raw per-trial JSON files written by the runner and produces
per-generation mean/std/min/max aggregates plus final-generation summaries and
comparisons against a control condition.
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .run import RECORD_FIELDS

__all__ = [
    "NUMERIC_METRICS",
    "load_trials",
    "condition_names",
    "aggregate",
    "final_summary",
    "compare_conditions",
    "write_condition_csv",
]

NUMERIC_METRICS = [f for f in RECORD_FIELDS if f != "generation"]


def load_trials(exp_dir: Path, condition: str) -> List[List[Dict[str, Any]]]:
    """All seeded trials for one condition; each trial is a per-generation list."""
    directory = Path(exp_dir) / condition
    trials = []
    for path in sorted(directory.glob("*.json")):
        if path.name.endswith(".config.json"):
            continue
        trials.append(json.loads(path.read_text()))
    return trials


def condition_names(exp_dir: Path) -> List[str]:
    return sorted(
        p.name
        for p in Path(exp_dir).iterdir()
        if p.is_dir()
        and any(
            f.suffix == ".json" and not f.name.endswith(".config.json")
            for f in p.glob("*.json")
        )
    )


def _stats(values: List[float]) -> Dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def aggregate(trials: List[List[Dict[str, Any]]]) -> Dict[int, Dict[str, Dict[str, float]]]:
    """Per generation -> per metric -> {mean, std, min, max} across seeds."""
    if not trials:
        return {}
    aggregated: Dict[int, Dict[str, Dict[str, float]]] = {}
    n_generations = len(trials[0])
    for generation in range(n_generations):
        row = {}
        for metric in NUMERIC_METRICS:
            values = [trial[generation][metric] for trial in trials]
            row[metric] = _stats(values)
        aggregated[generation + 1] = row
    return aggregated


def final_summary(trials: List[List[Dict[str, Any]]]) -> Dict[str, Dict[str, float]]:
    agg = aggregate(trials)
    return agg[max(agg)] if agg else {}


def compare_conditions(
    exp_dir: Path,
    condition: str,
    control: str,
) -> Dict[str, Tuple[float, float, float]]:
    """Final-generation delta per metric: (delta, condition_mean, control_mean)."""
    a = final_summary(load_trials(exp_dir, condition))
    b = final_summary(load_trials(exp_dir, control))
    return {
        metric: (a[metric]["mean"] - b[metric]["mean"], a[metric]["mean"], b[metric]["mean"])
        for metric in NUMERIC_METRICS
        if metric in a and metric in b
    }


def write_condition_csv(exp_dir: Path, condition: str) -> Path:
    """Aggregated per-generation CSV: generation, metric, mean, std, min, max."""
    trials = load_trials(exp_dir, condition)
    agg = aggregate(trials)
    out = Path(exp_dir) / condition / f"{condition}_aggregate.csv"
    with out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["generation", "metric", "mean", "std", "min", "max"])
        for generation, row in sorted(agg.items()):
            for metric, stats in row.items():
                writer.writerow(
                    [generation, metric, stats["mean"], stats["std"], stats["min"], stats["max"]]
                )
    return out
