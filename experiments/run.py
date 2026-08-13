"""Run reproducible experiments: config -> seeded trials -> machine-readable results.

Each trial runs one ``neat.Population`` for the configured number of
generations, stepping one generation at a time so per-generation world metrics
are captured. Results are written as per-trial JSON files plus a resolved-config
snapshot, so every result is traceable to the exact parameters that produced it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from neat.genome import Genome
from neat.population import Population
from world import GenerationRecorder, run_generation

from diversity.metrics import population_behavior

from .config import Condition, ExperimentConfig, ResolvedConfig, load_experiment, resolve_config
from .metrics import compute_metrics

__all__ = [
    "RECORD_FIELDS",
    "make_tracking_evaluator",
    "run_trial",
    "run_condition",
    "run_experiment",
    "main",
]

RECORD_FIELDS = [
    "generation",
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


def make_tracking_evaluator(
    resolved: ResolvedConfig,
    state: Dict[str, Any],
    record_generation: Optional[int] = None,
) -> Callable[[List[Genome], int], None]:
    """The engine batch hook: runs the world and captures per-generation metrics.

    When ``record_generation`` matches, a ``GenerationRecorder`` is attached so
    that generation's world can be replayed later.
    """

    def evaluate(population: List[Genome], generation: int) -> None:
        recorder = None
        if record_generation is not None and generation == record_generation:
            recorder = GenerationRecorder(population, resolved.world, generation)
        organisms = run_generation(population, resolved.world, generation, recorder=recorder)
        if recorder is not None:
            state["recorder"] = recorder
        state["organisms"] = organisms
        state["metrics"] = compute_metrics(organisms, len(population))
        state["evaluated_best_fitness"] = max(
            (g.fitness for g in population), default=0.0
        )
        state["evaluated_best_genome"] = max(
            (g for g in population), key=lambda g: g.fitness, default=None
        )
        state["evaluated_mean_fitness"] = (
            sum(g.fitness for g in population) / len(population) if population else 0.0
        )
        state["behavior"] = population_behavior(
            organisms,
            population,
            window=resolved.world.behavior_window,
            area=resolved.world.width * resolved.world.height,
        )

    return evaluate


def run_trial(
    resolved: ResolvedConfig,
    seed: int,
    *,
    record_generation: Optional[int] = None,
    recorder_out: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """One independent seeded trial; returns one record per generation."""
    state: Dict[str, Any] = {}
    population = Population(
        evaluator=make_tracking_evaluator(resolved, state, record_generation),
        population_size=resolved.population_size,
        input_ids=resolved.input_ids,
        output_ids=resolved.output_ids,
        seed=seed,
        elitism=resolved.elitism,
        crossover_rate=resolved.crossover_rate,
        mutation_config=resolved.mutation_config,
        speciation_config=resolved.speciation_config,
    )

    records: List[Dict[str, Any]] = []
    for generation in range(1, resolved.generations + 1):
        population.run(1)
        recorder = state.get("recorder")
        if recorder is not None and generation == record_generation + 1:
            recorder.context.update(
                {
                    "species_count": population.statistics[-1]["species_count"],
                    "survival_rate": state["metrics"]["survival_rate"],
                    "mean_fitness": state["evaluated_mean_fitness"],
                    "best_fitness": state["evaluated_best_fitness"],
                }
            )
        evaluated_best = state["evaluated_best_genome"]
        records.append(
            {
                "generation": generation,
                "population_size": len(population.population),
                "survival_rate": state["metrics"]["survival_rate"],
                "average_lifetime": state["metrics"]["average_lifetime"],
                "best_fitness": state["evaluated_best_fitness"],
                "mean_fitness": state["evaluated_mean_fitness"],
                "food_consumed": state["metrics"]["food_consumed"],
                "reproductive_success": state["metrics"]["reproductive_success"],
                "node_count": len(evaluated_best.nodes) if evaluated_best else 0,
                "connection_count": len(evaluated_best.connections) if evaluated_best else 0,
                "species_count": population.statistics[-1]["species_count"],
                "action_entropy": state["behavior"]["action_entropy"],
                "action_entropy_diversity": state["behavior"]["action_entropy_diversity"],
                "transition_entropy": state["behavior"]["transition_entropy"],
                "spatial_coverage": state["behavior"]["spatial_coverage"],
                "food_alignment": state["behavior"]["food_alignment"],
                "encounter_rate": state["behavior"]["encounter_rate"],
                "behavioral_diversity": state["behavior"]["behavioral_diversity"],
            }
        )
    if recorder_out is not None and "recorder" in state:
        recorder_out["recorder"] = state["recorder"]
    return records


def run_condition(
    experiment: ExperimentConfig,
    condition: Condition,
    out_dir: Path,
    record_generation: Optional[int] = None,
) -> Path:
    """Run every seed for one condition; write raw JSON + config snapshots.

    When ``record_generation`` is set, that generation's world is also recorded
    to ``<out>/<experiment>/recordings/<condition>/<seed>.json`` for replay.
    """
    condition_dir = out_dir / experiment.name / condition.name
    condition_dir.mkdir(parents=True, exist_ok=True)

    for seed in experiment.seeds:
        resolved = resolve_config(experiment, condition, seed)
        recorder_out: Dict[str, Any] = {}
        records = run_trial(
            resolved,
            seed,
            record_generation=record_generation,
            recorder_out=recorder_out,
        )
        (condition_dir / f"{seed}.json").write_text(
            json.dumps(records, indent=2) + "\n"
        )
        (condition_dir / f"{seed}.config.json").write_text(
            json.dumps(resolved.to_dict(), indent=2) + "\n"
        )

        recorder = recorder_out.get("recorder")
        if recorder is not None:
            recording_dir = out_dir / experiment.name / "recordings" / condition.name
            recording_dir.mkdir(parents=True, exist_ok=True)
            (recording_dir / f"{seed}.json").write_text(json.dumps(recorder.to_dict()))
    return condition_dir


def run_experiment(
    config_path: Path,
    out_dir: Path,
    conditions: Optional[List[str]] = None,
    record_generation: Optional[int] = None,
) -> Dict[str, Path]:
    """Run a whole experiment file; returns {condition_name: result_dir}."""
    experiment = load_experiment(config_path)
    result_dirs = {}
    names = conditions or [c.name for c in experiment.conditions]
    for condition in experiment.conditions:
        if condition.name not in names:
            continue
        result_dirs[condition.name] = run_condition(
            experiment, condition, out_dir, record_generation
        )
    return result_dirs


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run a Clage NEAT experiment.")
    parser.add_argument("--config", required=True, help="experiment JSON file")
    parser.add_argument("--out", default="results", help="output directory")
    parser.add_argument(
        "--conditions", default=None,
        help="comma-separated condition names to run (default: all)",
    )
    parser.add_argument(
        "--record-generation", type=int, default=None,
        help="also record this generation index (0-based) as a world replay",
    )
    args = parser.parse_args(argv)

    names = args.conditions.split(",") if args.conditions else None
    result_dirs = run_experiment(
        Path(args.config), Path(args.out), names, args.record_generation
    )
    for name, path in result_dirs.items():
        print(f"wrote {name}: {path}")
    print(f"experiment {Path(args.config).stem}: {len(result_dirs)} condition(s)")


if __name__ == "__main__":
    main()
