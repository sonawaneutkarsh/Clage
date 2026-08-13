"""Text and markdown summaries of benchmark trials."""

from __future__ import annotations

import statistics
from typing import Dict, List

from .run import TrialResult

__all__ = ["aggregate", "trial_table", "print_trial_table", "markdown_report"]


def aggregate(results: List[TrialResult]) -> Dict:
    solved = [r for r in results if r.solved]
    gens = [r.generations_to_solve for r in solved]
    return {
        "trials": len(results),
        "solved": len(solved),
        "success_rate": len(solved) / len(results),
        "gens_to_solve": {
            "min": min(gens) if gens else None,
            "median": statistics.median(gens) if gens else None,
            "max": max(gens) if gens else None,
        },
        "mean_best_fitness": statistics.mean(r.final_best_fitness for r in results),
        "mean_nodes": statistics.mean(r.final_nodes for r in results),
        "mean_connections": statistics.mean(r.final_connections for r in results),
        "mean_max_species": statistics.mean(max(x["species_count"] for x in r.history) for r in results),
    }


def trial_table(results: List[TrialResult]) -> str:
    header = (
        f"{'trial':>5} | {'solved':>6} | {'gens':>4} | "
        f"{'best fit':>8} | {'nodes':>5} | {'conns':>5} | {'max sp':>6}"
    )
    lines = [header, "-" * len(header)]
    for result in results:
        gens = result.generations_to_solve if result.solved else f">{result.total_generations}"
        max_species = max(x["species_count"] for x in result.history)
        lines.append(
            f"{result.seed:>5} | {str(result.solved):>6} | {str(gens):>4} | "
            f"{result.final_best_fitness:>8.4f} | {result.final_nodes:>5} | "
            f"{result.final_connections:>5} | {max_species:>6}"
        )
    return "\n".join(lines)


def print_trial_table(results: List[TrialResult]) -> None:
    print(trial_table(results))
    agg = aggregate(results)
    print(
        f"success {agg['solved']}/{agg['trials']} "
        f"({agg['success_rate']:.0%}), gens-to-solve "
        f"{agg['gens_to_solve']['min']}/{agg['gens_to_solve']['median']}/"
        f"{agg['gens_to_solve']['max']} (min/med/max), "
        f"mean best fitness {agg['mean_best_fitness']:.4f}"
    )


def markdown_report(
    results_by_problem: Dict[str, List[TrialResult]],
    diagnoses: Dict[str, str],
) -> str:
    title = "# Clage NEAT Validation Report\n"
    intro = (
        "\nEngine defaults; no parameter tuning. Each trial is an independent,\n"
        "seeded run of `Population` with default `MutationConfig`/`SpeciationConfig`.\n"
    )
    sections = [title, intro]

    for name, results in results_by_problem.items():
        agg = aggregate(results)
        sections.append(f"\n## {name}\n")
        sections.append(f"\n```\n{trial_table(results)}\n```\n")
        sections.append(
            f"- Success: **{agg['solved']}/{agg['trials']}** "
            f"({agg['success_rate']:.0%})\n"
            f"- Generations to solve (min/median/max): "
            f"{agg['gens_to_solve']['min']} / {agg['gens_to_solve']['median']} / "
            f"{agg['gens_to_solve']['max']}\n"
            f"- Mean best fitness at end: {agg['mean_best_fitness']:.4f}\n"
            f"- Mean best-genome nodes: {agg['mean_nodes']:.1f}\n"
            f"- Mean best-genome connections: {agg['mean_connections']:.1f}\n"
            f"- Mean max species: {agg['mean_max_species']:.1f}\n"
        )
        if name in diagnoses:
            sections.append(f"### Diagnosis (failed)\n\n{diagnoses[name]}\n")

    if diagnoses:
        verdict = "MIXED — some benchmarks failed with default parameters."
    else:
        verdict = "PASS — all benchmarks solved with default parameters."
    sections.append(f"\n## Verdict\n\n**{verdict}**\n")
    return "\n".join(sections)
