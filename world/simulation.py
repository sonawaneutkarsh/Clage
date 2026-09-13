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

from .config import ACTION_SIZE, OBSERVATION_SIZE, EnvironmentConfig
from .fitness import fitness
from .grid import World
from .organism import Organism

__all__ = ["run_generation", "make_evaluator"]


def _check_interface(population: List[Genome]) -> None:
    """Reject any genome whose interface does not match the world's contract.

    Only the COUNTS are contractual: ``OBSERVATION_SIZE`` inputs feed the
    network and ``ACTION_SIZE`` outputs are argmax-ed into an action. Node ids
    are never inspected, so arbitrary ids with correct counts keep working.

    The population is walked in list order and the FIRST offender is reported,
    so the message is deterministic when several genomes are wrong. Every
    genome is checked rather than the distinct set of interface shapes:
    computing the distinct set costs the same and loses the index.
    """
    for index, genome in enumerate(population):
        for side, constant, expected, received in (
            ("input_ids", "OBSERVATION_SIZE", OBSERVATION_SIZE, len(genome.inputs)),
            ("output_ids", "ACTION_SIZE", ACTION_SIZE, len(genome.outputs)),
        ):
            if received != expected:
                raise ValueError(
                    f"genome interface mismatch at population index {index}: "
                    f"world expects {expected} {side} ({constant}), got {received}"
                )


def _check_capacity(population: List[Genome], config: EnvironmentConfig) -> None:
    """Reject a founder population that cannot fit on the grid.

    ``World.random_empty_cell()`` is honestly typed ``Optional`` and returns
    ``None`` once the grid is full; founder placement unpacks it regardless, so
    an over-capacity population used to surface as an unactionable ``TypeError``
    from the unpacking. Raising here makes that path unreachable for this cause.

    Strictly greater than: exactly-at-capacity placement is legal.
    ``width``/``height`` are NOT re-checked — ``EnvironmentConfig`` owns them, so
    the capacity product is already a positive integer by the time any config
    reaches here.
    """
    capacity = config.width * config.height
    if len(population) > capacity:
        raise ValueError(
            f"population of {len(population)} founders exceeds grid capacity "
            f"{capacity} (width={config.width} * height={config.height})"
        )


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
    # The interface check is the FIRST statement: before the rng is seeded,
    # before the World exists, and before any Organism is placed. It draws no
    # randomness, so no seeded stream shifts.
    #
    # It cannot live in ``make_evaluator``: that function receives only the
    # EnvironmentConfig, and the genomes do not exist yet — they arrive later as
    # an argument to the closure it returns. The check has to be where the
    # genomes are. Since that closure calls ``run_generation``, and
    # ``world/recorder.py::record_generation_to_file`` calls it too, this single
    # choke point covers every world-driving path in the repo. The rule stays in
    # ``world``: ``neat`` is a generic engine and must not import ``world``.
    _check_interface(population)
    _check_capacity(population, config)

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
