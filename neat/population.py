"""Population-level evolutionary engine (Layer 6: the lifecycle).

The engine knows nothing about the world — no food, grids, organisms, energy,
movement, or cooperation. It operates purely on genomes and a fitness function
supplied by the caller:

    Population(fitness_fn).run(generations)

One generation:
    evaluate genomes -> assign fitness -> speciate -> share fitness
    -> select survivors (per-species elites + roulette parents)
    -> allocate offspring -> crossover -> mutate -> next generation

Invariants: population size is constant; offspring are structurally valid;
best raw fitness is non-decreasing when ``elitism >= 1``; elites are never
mutated.
"""

from __future__ import annotations

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
        self.fitness_fn = fitness_fn
        self.evaluator = evaluator
        self.elitism = elitism
        self.crossover_rate = crossover_rate
        self.rng = rng or random.Random(seed)
        self.db = db or InnovationDB()
        self.mutation_config = mutation_config or MutationConfig()

        self.speciation = Speciation(speciation_config or SpeciationConfig())

        if initial_population is not None:
            self.population = [g.copy() for g in initial_population]
            self.population_size = population_size or len(self.population)
        else:
            self.population_size = population_size or 100
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
        """Advance the population ``generations`` times. Returns new stats."""
        for _ in range(generations):
            self._next_generation()
        return self._stats[-generations:] if generations else []

    @property
    def statistics(self) -> List[Dict]:
        return list(self._stats)

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
