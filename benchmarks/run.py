"""Run seeded, independent trials of the benchmark problems.

Each trial runs the engine with default parameters on one problem, stepping
one generation at a time, and records best/mean fitness, species count, and the
best genome's node/connection counts per generation. It stops early the first
generation the problem is solved.

Row provenance: every per-generation row describes the genome the engine actually
evaluated in that generation, read via ``Population.evaluated_best_genome`` — not the
post-reproduction ``Population.population``, which after ``run(1)`` holds unevaluated
offspring whose copied or stale fitness no longer matches their structure. All four
champion fields of a row (``best_fitness``, ``best_node_count``,
``best_connection_count`` and ``solved``, the last via ``problem.success_fn``) come
from that one evaluated genome; ``mean_fitness`` and ``species_count`` come from the
matching ``Population.statistics[-1]`` entry.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from neat.population import Population

from .problems import PROBLEMS, Problem

__all__ = ["TrialResult", "run_trial", "run_benchmark", "main"]

DEFAULT_POPULATION_SIZE = 100
DEFAULT_MAX_GENERATIONS = 300
DEFAULT_TRIALS = 5
DEFAULT_SEED_BASE = 0


@dataclass
class TrialResult:
    problem: str
    seed: int
    solved: bool
    generations_to_solve: Optional[int]
    total_generations: int
    history: List[Dict] = field(default_factory=list)

    @property
    def final(self) -> Dict:
        return self.history[-1]

    @property
    def final_best_fitness(self) -> float:
        return self.final["best_fitness"]

    @property
    def final_nodes(self) -> int:
        return self.final["best_node_count"]

    @property
    def final_connections(self) -> int:
        return self.final["best_connection_count"]


def run_trial(
    problem: Problem,
    seed: int,
    *,
    population_size: int = DEFAULT_POPULATION_SIZE,
    max_generations: int = DEFAULT_MAX_GENERATIONS,
) -> TrialResult:
    """One independent, seeded run of ``problem`` with engine-default settings."""
    population = Population(
        problem.fitness_fn,
        population_size=population_size,
        seed=seed,
        input_ids=list(problem.input_ids),
        output_ids=list(problem.output_ids),
    )

    history: List[Dict] = []
    solved = False
    generations_to_solve: Optional[int] = None

    for generation in range(1, max_generations + 1):
        population.run(1)
        gen_best = population.evaluated_best_genome
        if gen_best is None:
            raise RuntimeError(
                f"generation {generation}: Population.evaluated_best_genome is None, so no "
                "evaluated champion exists for this generation; refusing to describe the row "
                "with an unevaluated post-reproduction genome"
            )
        stat = population.statistics[-1]
        is_solved = problem.success_fn(gen_best)

        history.append(
            {
                "generation": generation,
                "best_fitness": gen_best.fitness,
                "mean_fitness": stat["mean_fitness"],
                "species_count": stat["species_count"],
                "best_node_count": len(gen_best.nodes),
                "best_connection_count": len(gen_best.connections),
                "solved": is_solved,
            }
        )

        if is_solved:
            solved = True
            generations_to_solve = generation
            break

    return TrialResult(
        problem=problem.name,
        seed=seed,
        solved=solved,
        generations_to_solve=generations_to_solve,
        total_generations=generation,
        history=history,
    )


def run_benchmark(
    problem: Problem,
    *,
    trials: int = DEFAULT_TRIALS,
    seed_base: int = DEFAULT_SEED_BASE,
    population_size: int = DEFAULT_POPULATION_SIZE,
    max_generations: int = DEFAULT_MAX_GENERATIONS,
) -> List[TrialResult]:
    """Run ``trials`` independent seeded trials of ``problem``."""
    return [
        run_trial(
            problem,
            seed,
            population_size=population_size,
            max_generations=max_generations,
        )
        for seed in range(seed_base, seed_base + trials)
    ]


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run the Clage NEAT validation benchmarks.")
    parser.add_argument(
        "--problems", default="or,and,xor,sin",
        help="comma-separated problems to run (default: or,and,xor,sin)",
    )
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--generations", type=int, default=DEFAULT_MAX_GENERATIONS)
    parser.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE)
    parser.add_argument("--no-plot", action="store_true", help="skip generating plots")
    args = parser.parse_args(argv)

    from .diagnose import diagnose_failures
    from .plot import plot_problem
    from .report import markdown_report, print_trial_table

    names = [name.strip().lower() for name in args.problems.split(",")]
    results_by_problem = {}
    diagnoses = {}

    for name in names:
        problem = PROBLEMS[name]
        results = run_benchmark(
            problem,
            trials=args.trials,
            seed_base=args.seed_base,
            max_generations=args.generations,
        )
        results_by_problem[name] = results
        print(f"\n=== {problem.name} ===")
        print_trial_table(results)

        if not args.no_plot:
            plot_problem(problem, results, outdir="benchmarks/plots")

    failures = {
        name: results
        for name, results in results_by_problem.items()
        if not all(r.solved for r in results)
    }
    if failures:
        diagnoses = diagnose_failures(results_by_problem, failures)

    with open("progress/validation_report.md", "w") as handle:
        handle.write(markdown_report(results_by_problem, diagnoses))


if __name__ == "__main__":
    main()
