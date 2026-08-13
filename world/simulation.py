"""Run the shared artificial-life world for one generation.

All organisms from the population live in ONE world for ``config.ticks`` ticks.
Their evolved networks decide every action; the world applies the consequences
(movement, eating, metabolism, death, reproduction, food regeneration). After
the trial the world is thrown away and fitness is stamped onto each genome.

Deterministic: the world's rng is seeded from ``config.seed_base +
generation * config.seed_stride``.
"""

from __future__ import annotations

import random
from typing import Callable, List, Optional

from neat.genome import Genome

from .config import EnvironmentConfig
from .fitness import fitness
from .grid import World
from .organism import Organism

__all__ = ["run_generation", "make_evaluator"]


def run_generation(
    population: List[Genome],
    config: EnvironmentConfig,
    generation: int,
    recorder: Optional["GenerationRecorder"] = None,
) -> List[Organism]:
    """Evaluate the whole population in one shared world; stamps genome.fitness.

    ``recorder`` is a passive observer: when provided, it captures the world
    state each tick (including initial placement) for later replay.
    """
    rng = random.Random(config.world_rng_seed(generation))
    world = World(config, rng)

    organisms: List[Organism] = []
    for genome in population:
        cell = world.random_empty_cell()
        organism = Organism(genome, *cell, config)
        world.place_organism(organism)
        organisms.append(organism)

    for _ in range(config.initial_food):
        cell = world.random_empty_cell()
        if cell is not None:
            world.place_food(*cell)

    if recorder is not None:
        recorder.record_tick(world, organisms)

    for _ in range(config.ticks):
        newborns: List[Organism] = []
        for organism in organisms:
            if not organism.alive:
                continue
            child = organism.act(world, config)
            if child is not None:
                newborns.append(child)
        organisms.extend(newborns)
        world.regenerate_food()
        if recorder is not None:
            recorder.record_tick(world, organisms)

    best_per_genome: dict = {}
    for organism in organisms:
        score = fitness(organism.food_eaten, organism.age, organism.offspring)
        genome = organism.genome
        best_per_genome[genome] = max(best_per_genome.get(genome, 0.0), score)

    for genome in population:
        genome.fitness = best_per_genome.get(genome, 0.0)

    return organisms


def make_evaluator(config: EnvironmentConfig) -> Callable[[List[Genome], int], None]:
    """The engine's batch-scoring hook: ``(population, generation) -> None``."""

    def evaluate(population: List[Genome], generation: int) -> None:
        run_generation(population, config, generation)

    return evaluate
