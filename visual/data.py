"""Pure data loading for the Clage viewer.

This module reads the JSON produced by the core system (generation recordings
and experiment results). It imports only the standard library — no `neat`,
`world`, or `experiments` code — so the viewer cannot contain evolutionary
logic by construction.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = [
    "METRICS",
    "load_recording",
    "load_condition_trials",
    "load_condition_configs",
    "condition_names",
    "aggregate_condition",
    "environmental_params",
]

# Per-generation metrics recorded by the experiment framework (data contract).
METRICS = [
    "population_size",
    "survival_rate",
    "average_lifetime",
    "best_fitness",
    "mean_fitness",
    "food_consumed",
    "reproductive_success",
    "node_count",
    "connection_count",
    "species_count",
    "action_entropy",
    "action_entropy_diversity",
    "transition_entropy",
    "spatial_coverage",
    "food_alignment",
    "encounter_rate",
    "behavioral_diversity",
]


def load_recording(path) -> Dict[str, Any]:
    """Load a generation replay recording."""
    recording = json.loads(Path(path).read_text())
    if recording.get("schema") != "clage-generation-replay":
        raise ValueError(f"{path}: not a clage generation recording")
    return recording


def load_condition_trials(exp_dir, condition: str) -> List[List[Dict[str, Any]]]:
    """All seeded trials for a condition; each trial is a per-generation list."""
    directory = Path(exp_dir) / condition
    trials = []
    for path in sorted(directory.glob("*.json")):
        if path.name.endswith(".config.json"):
            continue
        trials.append(json.loads(path.read_text()))
    return trials


def load_condition_configs(exp_dir, condition: str) -> List[Dict[str, Any]]:
    """The resolved config snapshots (one per seed) for a condition."""
    directory = Path(exp_dir) / condition
    configs = []
    for path in sorted(directory.glob("*.config.json")):
        configs.append(json.loads(path.read_text()))
    return configs


def condition_names(exp_dir) -> List[str]:
    return sorted(
        p.name
        for p in Path(exp_dir).iterdir()
        if p.is_dir()
        and any(f.suffix == ".json" and not f.name.endswith(".config.json")
               for f in p.glob("*.json"))
    )


def aggregate_condition(
    trials: List[List[Dict[str, Any]]],
) -> Dict[int, Dict[str, Dict[str, float]]]:
    """Per generation -> per metric -> {mean, std} across seeds."""
    if not trials:
        return {}
    aggregated: Dict[int, Dict[str, Dict[str, float]]] = {}
    for generation in range(len(trials[0])):
        row = {}
        for metric in METRICS:
            values = [trial[generation][metric] for trial in trials]
            row[metric] = {
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            }
        aggregated[generation + 1] = row
    return aggregated


def environmental_params(configs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The environmental parameters shared by a condition's trials (first seed)."""
    if not configs:
        return {}
    config = configs[0]
    world = config.get("world", {})
    neat = config.get("neat", {})
    return {
        "width": world.get("width"),
        "height": world.get("height"),
        "ticks": world.get("ticks"),
        "initial_food": world.get("initial_food"),
        "food_target": world.get("food_target"),
        "food_regrowth_per_tick": world.get("food_regrowth_per_tick"),
        "repro_threshold": world.get("repro_threshold"),
        "repro_fraction": world.get("repro_fraction"),
        "metabolism": world.get("metabolism"),
        "population_size": neat.get("population_size"),
    }
