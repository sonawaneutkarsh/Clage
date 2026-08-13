"""NEAT speciation (Layer 6: population math, speciation half).

Speciation protects innovative topologies: novel structure is usually worse at
first (its weights are random), so a single global competition pool would kill
it immediately. Grouping similar genomes into species lets them compete mostly
against relatives, giving new topologies a protected nursery while their
weights improve.

Core pieces:
- ``compatibility_distance`` — how structurally far apart two genomes are,
  computed by aligning their genes on innovation numbers.
- ``Speciation`` — assigns genomes to species across generations, shares
  fitness inside each species, and allocates the offspring budget by species.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .genome import ConnectionGene, Genome

__all__ = [
    "SpeciationConfig",
    "Species",
    "Speciation",
    "compatibility_distance",
    "average_weight_difference",
    "gene_counts",
]


@dataclass
class SpeciationConfig:
    """Tuning knobs for compatibility distance and extinction."""

    excess_coef: float = 1.0
    disjoint_coef: float = 1.0
    weight_coef: float = 0.4
    compatibility_threshold: float = 3.0
    small_genome_threshold: int = 20
    stagnation_threshold: int = 15


@dataclass
class Species:
    """A group of similar genomes competing mostly against each other."""

    id: int
    representative: Genome
    members: List[Genome] = field(default_factory=list)
    age: int = 0
    best_fitness: float = 0.0
    stagnation: int = 0
    adjusted_fitness_sum: float = 0.0
    offspring: int = 0

    @property
    def fittest(self) -> Genome:
        return max(self.members, key=lambda g: g.fitness, default=self.representative)

    def __repr__(self) -> str:
        return (
            f"Species(id={self.id}, members={len(self.members)}, "
            f"best={self.best_fitness:.3f}, age={self.age}, offspring={self.offspring})"
        )


# ------------------------------------------------------------------ distances


def gene_counts(a: Genome, b: Genome) -> Tuple[int, int, int]:
    """Return ``(excess, disjoint, matching)`` gene counts between two genomes.

    Genes are aligned by innovation number. A gene in one parent only is
    ``excess`` if its innovation is beyond the other parent's maximum, else
    ``disjoint``. Counts are symmetric totals across both parents.
    """
    a_innovs = {c.innovation for c in a.connections}
    b_innovs = {c.innovation for c in b.connections}
    max_a = max(a_innovs, default=0)
    max_b = max(b_innovs, default=0)

    excess = disjoint = 0
    for innov in a_innovs - b_innovs:
        if innov > max_b:
            excess += 1
        else:
            disjoint += 1
    for innov in b_innovs - a_innovs:
        if innov > max_a:
            excess += 1
        else:
            disjoint += 1
    matching = len(a_innovs & b_innovs)
    return excess, disjoint, matching


def average_weight_difference(a: Genome, b: Genome) -> float:
    """Mean absolute weight difference over matching connection genes."""
    a_by_innov: Dict[int, ConnectionGene] = {c.innovation: c for c in a.connections}
    b_by_innov: Dict[int, ConnectionGene] = {c.innovation: c for c in b.connections}
    shared = set(a_by_innov) & set(b_by_innov)
    if not shared:
        return 0.0
    return sum(
        abs(a_by_innov[i].weight - b_by_innov[i].weight) for i in shared
    ) / len(shared)


def compatibility_distance(a: Genome, b: Genome, config: SpeciationConfig) -> float:
    """The structural distance used to decide whether two genomes share a species.

    ``delta = c1*E/N + c2*D/N + c3*W`` where E/D are excess/disjoint counts,
    W is the average matching-weight difference, and N is the larger gene count
    (set to 1 for small genomes, per the NEAT paper).
    """
    excess, disjoint, _ = gene_counts(a, b)
    n = max(len(a.connections), len(b.connections))
    if n < config.small_genome_threshold:
        n = 1
    return (
        config.excess_coef * excess / n
        + config.disjoint_coef * disjoint / n
        + config.weight_coef * average_weight_difference(a, b)
    )


# ------------------------------------------------------------------ engine


class Speciation:
    """Speciates a population generation by generation.

    Persistent across generations: species keep their identity, representative,
    age, and stagnation state so a species that stops improving can go extinct.

    Deterministic by design — no RNG inside. The representative of a species is
    always its fittest member, which makes tests exact.
    """

    def __init__(self, config: Optional[SpeciationConfig] = None) -> None:
        self.config = config or SpeciationConfig()
        self.species: List[Species] = []
        self._next_id: int = 1

    # --------------------------------------------------------------- assignment

    def speciate(self, genomes: List[Genome]) -> None:
        """Assign every genome to a species; update reps, age, stagnation.

        First call: sort by fitness descending; the fittest founds a species,
        each later genome joins the first compatible species (delta below
        threshold) or founds a new one. Later calls: assign against the existing
        representatives from the previous generation.
        """
        ordered = sorted(genomes, key=lambda g: g.fitness, reverse=True)

        for species in self.species:
            species.members = []  # membership is per-generation

        for genome in ordered:
            for species in self.species:
                if (
                    compatibility_distance(genome, species.representative, self.config)
                    < self.config.compatibility_threshold
                ):
                    species.members.append(genome)
                    break
            else:
                species = Species(id=self._next_id, representative=genome)
                self._next_id += 1
                species.members.append(genome)
                self.species.append(species)

        self._update_species()
        self.prune_stagnant()

    def _update_species(self) -> None:
        for species in self.species:
            species.age += 1
            if not species.members:
                species.stagnation += 1
                continue
            best = max(g.fitness for g in species.members)
            if species.age == 1 or best > species.best_fitness:
                species.best_fitness = best
                species.stagnation = 0
            else:
                species.stagnation += 1
            species.representative = species.fittest

    def prune_stagnant(self) -> List[int]:
        """Remove stagnant or empty species. Returns the ids of extinct species."""
        extinct = [
            s.id
            for s in self.species
            if not s.members or s.stagnation >= self.config.stagnation_threshold
        ]
        self.species = [s for s in self.species if s.id not in extinct]
        return extinct

    # ------------------------------------------------------------- fitness

    def share_fitness(self) -> None:
        """Apply fitness sharing: ``adjusted = fitness / species_size``."""
        for species in self.species:
            size = len(species.members)
            total = 0.0
            for genome in species.members:
                genome.adjusted_fitness = genome.fitness / size
                total += genome.adjusted_fitness
            species.adjusted_fitness_sum = total

    # ------------------------------------------------------------ allocation

    def allocate_offspring(self, population_size: int) -> Dict[int, int]:
        """Offspring per species, proportional to adjusted-fitness sum.

        Largest-remainder rounding so counts sum exactly to ``population_size``.
        If every species has zero adjusted fitness, fall back to a uniform split.
        """
        n_species = len(self.species)
        if n_species == 0:
            return {}

        total = sum(s.adjusted_fitness_sum for s in self.species)
        if total <= 0.0:
            base, rem = divmod(population_size, n_species)
            return {
                s.id: base + (1 if i < rem else 0)
                for i, s in enumerate(self.species)
            }

        raw = {s.id: s.adjusted_fitness_sum / total * population_size for s in self.species}
        allocation = {sid: int(count) for sid, count in raw.items()}
        remaining = population_size - sum(allocation.values())

        # deterministic tie-break: largest fractional part, then lowest species id
        order = sorted(
            self.species,
            key=lambda s: (raw[s.id] - allocation[s.id], -s.id),
            reverse=True,
        )
        for species in order[:remaining]:
            allocation[species.id] += 1

        for species in self.species:
            species.offspring = allocation[species.id]
        return allocation
