"""Population-level evolutionary engine (Layer 6: the lifecycle).

The engine knows nothing about the world — no food, grids, organisms, energy,
movement, or cooperation. It operates purely on genomes and a fitness function
supplied by the caller:

    Population(fitness_fn).run(generations)

One generation:
    evaluate genomes -> assign fitness -> speciate -> share fitness
    -> select survivors (per-species elites + roulette parents)
    -> allocate offspring -> crossover -> mutate -> next generation

    Evaluation happens at the START of that sequence, so once ``run(1)`` returns
    ``self.population`` already holds the *unevaluated* offspring of the generation
    just finished. Those offspring carry copied or stale fitness values that no
    longer describe their own structure (a crossover child is built with 0.0; an
    elite copy keeps its parent's number). Evaluated per-generation values must
    therefore be read from ``statistics[-1]`` or from the ``evaluated_best_genome``
    property — never by scanning ``self.population``, which describes the next
    generation, not the one that was measured.

Invariants: population size is constant; offspring are structurally valid;
best raw fitness is non-decreasing when ``elitism >= 1``; elites are never
mutated.
"""

from __future__ import annotations

import math
import random
from typing import Callable, Dict, List, Optional

from .crossover import crossover
from .genome import Genome
from .innovation import InnovationDB
from .mutation import MutationConfig, apply_mutation
from .speciation import Speciation, SpeciationConfig, Species

__all__ = ["Population"]

FitnessFn = Callable[[Genome, int], float]
EvaluatorFn = Callable[[List[Genome], int], None]


def _require_int_at_least(owner: str, field: str, value: object, lowest: int) -> None:
    """Reject a value that cannot serve as a count for ``owner.field``.

    ``TypeError`` when the type cannot support the field at all — ``bool`` and
    ``float`` are refused even though ``True`` and ``6.0`` would survive the
    comparison, because both flow straight into ``range()`` and into slice
    arithmetic where only a true ``int`` has defined behavior. ``ValueError``
    when the type is right but the value is out of range.
    """
    rule = f"an int >= {lowest}"
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{owner}.{field} must be {rule}, got {value!r}")
    if value < lowest:
        raise ValueError(f"{owner}.{field} must be {rule}, got {value!r}")


def _structure_signature(genome: Genome) -> tuple:
    """Lightweight structural identity: size + innovation numbers."""
    return (
        len(genome.nodes),
        len(genome.connections),
        tuple(c.innovation for c in genome.connections),
    )


class Population:
    """Drives a single evolving population of genomes across generations."""

    def __init__(
        self,
        fitness_fn: Optional[FitnessFn] = None,
        *,
        population_size: Optional[int] = None,
        input_ids: Optional[List[int]] = None,
        output_ids: Optional[List[int]] = None,
        seed: int = 0,
        rng: Optional[random.Random] = None,
        speciation_config: Optional[SpeciationConfig] = None,
        mutation_config: Optional[MutationConfig] = None,
        elitism: int = 1,
        crossover_rate: float = 0.75,
        initial_population: Optional[List[Genome]] = None,
        db: Optional[InnovationDB] = None,
        evaluator: Optional[EvaluatorFn] = None,
    ) -> None:
        if fitness_fn is None and evaluator is None:
            raise ValueError("provide either fitness_fn (per-genome) or evaluator (batch)")

        # Reject engine configuration that cannot produce a valid run, in
        # argument declaration order so the field named first is deterministic.
        # Scope is deliberately narrow: only values measured to break the
        # constant-population-size invariant, to make a run impossible, or to
        # silently substitute something the caller never supplied. Values that
        # merely saturate a probability (``crossover_rate`` and the
        # ``MutationConfig`` fields outside ``[0, 1]``) are left alone.
        if population_size is not None:
            _require_int_at_least("Population", "population_size", population_size, 1)
        _require_int_at_least("Population", "elitism", elitism, 0)
        if initial_population is not None and len(initial_population) == 0:
            # An empty list is not "no argument": it would set population_size to
            # 0, leave the population empty every generation and never establish
            # a champion.
            raise ValueError(
                "Population.initial_population must be a non-empty list of genomes, "
                f"got {initial_population!r}"
            )

        self.fitness_fn = fitness_fn
        self.evaluator = evaluator
        self.elitism = elitism
        self.crossover_rate = crossover_rate
        self.rng = rng or random.Random(seed)
        self.db = db or InnovationDB()
        self.mutation_config = mutation_config or MutationConfig()

        self.speciation = Speciation(speciation_config or SpeciationConfig())

        # ``is None``, not ``or``: an explicitly supplied size is honoured as
        # given (and already validated above) rather than treated as absent.
        # The documented defaults are unchanged — 100 when the argument is
        # omitted, ``len(initial_population)`` when it is omitted alongside an
        # initial population.
        if initial_population is not None:
            self.population = [g.copy() for g in initial_population]
            self.population_size = (
                len(self.population) if population_size is None else population_size
            )
        else:
            self.population_size = 100 if population_size is None else population_size
            self.population = [
                Genome.minimal(input_ids=input_ids, output_ids=output_ids)
                for _ in range(self.population_size)
            ]

        self.generation: int = 0
        self.best_genome: Optional[Genome] = None
        self.best_fitness: float = float("-inf")
        self.best_generation: int = -1
        self._stats: List[Dict] = []
        self._evaluated_best_genome: Optional[Genome] = None
        self._evaluated_best_fitness: float = 0.0
        self._evaluated_mean_fitness: float = 0.0

    # ------------------------------------------------------------------ run

    def run(self, generations: int) -> List[Dict]:
        """Advance the population ``generations`` times. Returns new stats.

        On return ``self.population`` is the unevaluated next generation, so evaluated
        per-generation values belong to ``statistics[-1]`` or ``evaluated_best_genome``,
        not to a scan of ``self.population``.
        """
        for _ in range(generations):
            self._next_generation()
        return self._stats[-generations:] if generations else []

    @property
    def statistics(self) -> List[Dict]:
        return list(self._stats)

    @property
    def evaluated_best_genome(self) -> Optional[Genome]:
        """The genome behind the most recently recorded ``best_fitness``.

        ``None`` before the first generation has been evaluated (and again only if
        the population is empty).

        Every read returns a fresh defensive copy, so callers may keep the result and
        change it freely without ever reaching engine state — the same isolation
        guarantee ``best_genome`` already gives. Two consecutive reads therefore hand
        back two distinct objects: identity comparisons, whether between two reads or
        against any engine-internal object, are meaningless by design, so compare
        values instead.

        The returned snapshot is also not a member of ``self.population`` after
        ``run()`` returns; that list holds unevaluated offspring carrying copied or
        stale fitness.
        """
        if self._evaluated_best_genome is None:
            return None
        return self._evaluated_best_genome.copy()

    # ------------------------------------------------------------- lifecycle

    def _next_generation(self) -> None:
        self._evaluate()
        self.speciation.speciate(self.population)
        self.speciation.share_fitness()
        allocation = self.speciation.allocate_offspring(self.population_size)

        next_population: List[Genome] = []
        for species in self.speciation.species:
            budget = allocation[species.id]
            if budget <= 0:
                continue  # this species produces no offspring this generation
            self._reproduce_species(species, budget, next_population)

        self.population = self._guarantee_champion(next_population)
        self.generation += 1
        self._track_best_and_stats()

    def _guarantee_champion(self, next_population: List[Genome]) -> List[Genome]:
        """Keep the all-time best genome alive no matter what.

        Stagnation pruning can remove every species (e.g. a converged champion
        species that plateaus), which would otherwise leave an empty population.
        Classic NEAT preserves the champion: insert a copy of the best genome if
        it is absent, and top the population back up to size with mutated
        champion copies.
        """
        if self.best_genome is None:
            return next_population

        champion = self.best_genome.copy()
        present = any(
            _structure_signature(g) == _structure_signature(champion)
            for g in next_population
        )
        if not present:
            if next_population:
                worst = min(range(len(next_population)), key=lambda i: next_population[i].fitness)
                next_population[worst] = champion
            else:
                next_population.append(champion)

        while len(next_population) < self.population_size:
            rescued = self.best_genome.copy()
            apply_mutation(rescued, self.rng, self.db, self.mutation_config)
            rescued.validate()
            next_population.append(rescued)
        return next_population

    def _evaluate(self) -> None:
        if self.evaluator is not None:
            # Batch scoring: the evaluator (e.g. an artificial-life world) runs
            # the whole population at once and stamps genome.fitness.
            self.evaluator(self.population, self.generation)
        else:
            for genome in self.population:
                genome.fitness = self.fitness_fn(genome, self.generation)

        # Reject non-finite fitness right here, before any of it can reach
        # best-genome tracking, recorded statistics, speciation or offspring
        # allocation. A NaN loses every ``max()`` comparison in silence, and an
        # infinity turns the allocation ratio into a NaN whose error message
        # reports a value the caller never supplied. Only *genome* values are
        # inspected: engine state such as ``best_fitness`` starts at
        # ``float("-inf")`` on purpose, as a sentinel the first generation beats.
        for index, genome in enumerate(self.population):
            if not math.isfinite(genome.fitness):
                if math.isnan(genome.fitness):
                    kind = "nan"
                elif genome.fitness > 0.0:
                    kind = "inf"
                else:
                    kind = "-inf"
                raise ValueError(
                    f"non-finite fitness at population index {index}: {kind}"
                )

        # Record the *evaluated* best/mean. Post-reproduction, offspring carry
        # stale fitness (0.0 after crossover), so stats must not be computed
        # from the newly built population.
        if self.population:
            self._evaluated_best_genome = max(self.population, key=lambda g: g.fitness)
            self._evaluated_best_fitness = self._evaluated_best_genome.fitness
            self._evaluated_mean_fitness = sum(
                g.fitness for g in self.population
            ) / len(self.population)
        else:
            self._evaluated_best_genome = None
            self._evaluated_best_fitness = 0.0
            self._evaluated_mean_fitness = 0.0

    # ------------------------------------------------------------- survival

    def _reproduce_species(
        self,
        species: Species,
        budget: int,
        next_population: List[Genome],
    ) -> None:
        elites = min(self.elitism, budget)
        ranked = sorted(species.members, key=lambda g: g.fitness, reverse=True)
        for elite in ranked[:elites]:
            next_population.append(elite.copy())

        for _ in range(budget - elites):
            child = self._make_offspring(species)
            apply_mutation(child, self.rng, self.db, self.mutation_config)
            child.validate()
            next_population.append(child)

    def _make_offspring(self, species: Species) -> Genome:
        if len(species.members) >= 2 and self.rng.random() < self.crossover_rate:
            parent_a = self._select_parent(species)
            parent_b = self._select_parent(species)
            return crossover(parent_a, parent_b, self.rng)
        return self._select_parent(species).copy()

    def _select_parent(self, species: Species) -> Genome:
        """Fitness-proportional roulette over the species' members."""
        weights = [max(g.fitness, 0.0) for g in species.members]
        if sum(weights) <= 0.0:
            return self.rng.choice(species.members)
        pick = self.rng.random() * sum(weights)
        running = 0.0
        for genome, weight in zip(species.members, weights):
            running += weight
            if pick <= running:
                return genome
        return species.members[-1]

    # ------------------------------------------------------------- tracking

    def _track_best_and_stats(self) -> None:
        best = self._evaluated_best_genome
        if best is not None and (self.best_genome is None or best.fitness > self.best_fitness):
            self.best_genome = best.copy()
            self.best_fitness = best.fitness
            self.best_generation = self.generation

        self._stats.append(
            {
                "generation": self.generation,
                "population_size": len(self.population),
                "species_count": len(self.speciation.species),
                "best_fitness": self._evaluated_best_fitness,
                "mean_fitness": self._evaluated_mean_fitness,
                "sizes": sorted(len(s.members) for s in self.speciation.species),
            }
        )

    # ------------------------------------------------------------------ dunder

    def __repr__(self) -> str:
        return (
            f"Population(generation={self.generation}, size={len(self.population)}, "
            f"species={len(self.speciation.species)}, best={self.best_fitness:.3f})"
        )
