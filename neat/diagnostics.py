"""Speciation diagnostics: species counts across generations.

A self-contained synthetic evolution loop (mutation-only reproduction honoring
the species offspring allocation) used to observe how many species form and
survive as topologies diverge. NOT connected to the artificial-life simulation.

Run: ``python -m neat.diagnostics``
"""

from __future__ import annotations

import random
from typing import Callable, Dict, Iterable, List, Optional

from .genome import Genome
from .innovation import InnovationDB
from .mutation import (
    mutate_add_connection,
    mutate_add_node,
    mutate_biases,
    mutate_perturb_weights,
)
from .phenotype import Network
from .speciation import Speciation, SpeciationConfig

__all__ = ["XOR_DATA", "xor_fitness", "species_report", "print_species_report", "main"]

# 2-input / 1-output XOR: the diagnostic fitness target.
XOR_DATA: List[tuple] = [
    ((0.0, 0.0), 0.0),
    ((0.0, 1.0), 1.0),
    ((1.0, 0.0), 1.0),
    ((1.0, 1.0), 0.0),
]


def xor_fitness(genome: Genome, generation: int) -> float:
    """Fitness = 1/(1 + MSE) of the decoded network on the XOR truth table.

    Matches ``Population``'s ``fitness_fn(genome, generation)`` signature.
    """
    net = Network(genome)
    error = 0.0
    for inputs, expected in XOR_DATA:
        output = net.activate(list(inputs))[0]
        error += (output - expected) ** 2
    mse = error / len(XOR_DATA)
    return 1.0 / (1.0 + mse)


def _mutate(genome: Genome, rng: random.Random, db: InnovationDB, steps: int = 3) -> None:
    for _ in range(steps):
        op = rng.randrange(4)
        if op == 0:
            mutate_perturb_weights(genome, rng, prob=1.0)
        elif op == 1:
            mutate_add_connection(genome, rng, db)
        elif op == 2:
            mutate_add_node(genome, rng, db)
        else:
            mutate_biases(genome, rng, prob=1.0)


def species_report(
    population_size: int,
    generations: int,
    *,
    seed: int = 0,
    fitness_fn: Optional[Callable[[Genome, random.Random], float]] = None,
    config: Optional[SpeciationConfig] = None,
    input_ids: Optional[Iterable[int]] = None,
    output_ids: Optional[Iterable[int]] = None,
) -> List[Dict]:
    """Run a synthetic evolution loop and report species counts per generation.

    Each generation: evaluate fitness, speciate, share fitness, allocate
    offspring, then build the next population by mutating each species'
    fittest member once per allocated child (mutation-only reproduction).
    """
    fitness_fn = fitness_fn or xor_fitness
    rng = random.Random(seed)
    db = InnovationDB()
    spe = Speciation(config)

    in_ids = tuple(input_ids) if input_ids is not None else (0, 1)
    out_ids = tuple(output_ids) if output_ids is not None else (10,)

    population = [Genome.minimal(input_ids=in_ids, output_ids=out_ids) for _ in range(population_size)]

    report: List[Dict] = []
    for generation in range(generations):
        for genome in population:
            genome.fitness = fitness_fn(genome, rng)

        spe.speciate(population)
        spe.share_fitness()
        allocation = spe.allocate_offspring(population_size)

        report.append(
            {
                "generation": generation,
                "species_count": len(spe.species),
                "sizes": sorted(len(s.members) for s in spe.species),
                "best_fitness": max(g.fitness for g in population),
            }
        )

        next_population: List[Genome] = []
        for species in spe.species:
            for _ in range(allocation[species.id]):
                child = species.representative.copy()
                _mutate(child, rng, db)
                next_population.append(child)
        population = next_population

    return report


def print_species_report(report: List[Dict]) -> None:
    print(f"{'gen':>3} | {'species':>7} | {'sizes':<28} | best fitness")
    print("-" * 60)
    for row in report:
        sizes = ",".join(str(s) for s in row["sizes"])
        print(
            f"{row['generation']:>3} | {row['species_count']:>7} | "
            f"{sizes:<28} | {row['best_fitness']:.4f}"
        )


def main() -> None:
    print_species_report(species_report(population_size=30, generations=20, seed=0))


if __name__ == "__main__":
    main()
