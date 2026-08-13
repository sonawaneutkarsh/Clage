import pytest

from neat.diagnostics import species_report
from neat.genome import ConnectionGene, Genome, NodeGene, NodeType
from neat.speciation import (
    Speciation,
    SpeciationConfig,
    Species,
    average_weight_difference,
    compatibility_distance,
    gene_counts,
)


def genome_with(connections, nodes_extra=(), fitness=0.0):
    g = Genome.minimal(input_ids=[0, 1], output_ids=[10, 11])
    for nid in nodes_extra:
        g.add_node(NodeGene(id=nid, node_type=NodeType.HIDDEN))
    for in_node, out_node, weight, innovation in connections:
        g.add_connection(
            ConnectionGene(in_node=in_node, out_node=out_node, weight=weight, innovation=innovation)
        )
    g.fitness = fitness
    return g


def group_a():
    return genome_with([(0, 10, 0.5, 1), (1, 11, 0.5, 2)])


def group_b():
    return genome_with(
        [(0, 11, 0.5, 3), (1, 10, 0.5, 4), (0, 50, 0.5, 5), (50, 10, 0.5, 6)], nodes_extra=(50,)
    )


def group_c():
    return genome_with(
        [(0, 51, 0.5, 7), (51, 11, 0.5, 8), (1, 51, 0.5, 9), (51, 10, 0.5, 10)], nodes_extra=(51,)
    )


# ------------------------------------------------------------------ distances


def test_identical_genomes_distance_zero():
    a = group_a()
    b = a.copy()
    assert compatibility_distance(a, b, SpeciationConfig()) == 0.0


def test_weight_difference_distance():
    a = genome_with([(0, 10, 1.0, 1), (1, 11, 1.0, 2)])
    b = genome_with([(0, 10, 1.2, 1), (1, 11, 0.8, 2)])
    config = SpeciationConfig()
    assert average_weight_difference(a, b) == pytest.approx(0.2)
    assert compatibility_distance(a, b, config) == pytest.approx(config.weight_coef * 0.2)


def test_average_weight_difference_zero_when_no_matching():
    assert average_weight_difference(group_a(), group_b()) == 0.0


def test_gene_counts_excess_disjoint_matching():
    excess, disjoint, matching = gene_counts(group_a(), group_b())
    assert (excess, disjoint, matching) == (4, 2, 0)


def test_one_additional_connection_same_species():
    a = group_a()
    a.add_node(NodeGene(id=50, node_type=NodeType.HIDDEN))
    a.add_connection(ConnectionGene(0, 50, 0.5, innovation=5))
    config = SpeciationConfig()
    assert compatibility_distance(group_a(), a, config) == pytest.approx(1.0)
    assert compatibility_distance(group_a(), a, config) < config.compatibility_threshold


def test_different_topology_different_species():
    config = SpeciationConfig()
    assert compatibility_distance(group_a(), group_b(), config) == pytest.approx(6.0)
    assert compatibility_distance(group_a(), group_b(), config) > config.compatibility_threshold


# ------------------------------------------------------------------ speciation


def test_several_distinct_species():
    a1, a2 = group_a(), group_a()
    b1, b2 = group_b(), group_b()
    c1, c2 = group_c(), group_c()
    for g, fit in ((a1, 5.0), (a2, 4.5), (b1, 3.0), (b2, 2.5), (c1, 2.0), (c2, 1.5)):
        g.fitness = fit

    spe = Speciation()
    spe.speciate([a1, a2, b1, b2, c1, c2])

    assert len(spe.species) == 3
    groups = {frozenset(round(g.fitness, 2) for g in s.members) for s in spe.species}
    assert groups == {frozenset({5.0, 4.5}), frozenset({3.0, 2.5}), frozenset({2.0, 1.5})}


def test_representative_is_fittest_member():
    a1, a2 = group_a(), group_a()
    b1, b2 = group_b(), group_b()
    a1.fitness, a2.fitness = 5.0, 4.5
    b1.fitness, b2.fitness = 3.0, 2.5

    spe = Speciation()
    spe.speciate([a1, a2, b1, b2])
    for species in spe.species:
        assert species.representative.fitness == max(g.fitness for g in species.members)


# ------------------------------------------------------------------ fitness sharing


def test_adjusted_fitness_sharing():
    a1, a2 = group_a(), group_a()
    b1, b2 = group_b(), group_b()
    a1.fitness, a2.fitness = 5.0, 4.5
    b1.fitness, b2.fitness = 3.0, 2.5

    spe = Speciation()
    spe.speciate([a1, a2, b1, b2])
    spe.share_fitness()

    assert a1.adjusted_fitness == pytest.approx(2.5)
    assert a2.adjusted_fitness == pytest.approx(2.25)
    assert b1.adjusted_fitness == pytest.approx(1.5)
    assert b2.adjusted_fitness == pytest.approx(1.25)
    # raw fitness untouched by sharing
    assert a1.fitness == 5.0 and b2.fitness == 2.5
    sums = sorted(s.adjusted_fitness_sum for s in spe.species)
    assert sums == pytest.approx([2.75, 4.75])


# ------------------------------------------------------------------ allocation


def test_offspring_allocation_proportional():
    a1, a2 = group_a(), group_a()
    b1, b2 = group_b(), group_b()
    c1, c2 = group_c(), group_c()
    for g, fit in ((a1, 5.0), (a2, 4.5), (b1, 3.0), (b2, 2.5), (c1, 2.0), (c2, 1.5)):
        g.fitness = fit

    spe = Speciation()
    spe.speciate([a1, a2, b1, b2, c1, c2])
    spe.share_fitness()
    allocation = spe.allocate_offspring(10)

    assert sum(allocation.values()) == 10
    by_id = {s.id: s.adjusted_fitness_sum for s in spe.species}
    ids_by_fitness = sorted(by_id, key=lambda sid: -by_id[sid])
    assert allocation[ids_by_fitness[0]] > allocation[ids_by_fitness[1]] > allocation[ids_by_fitness[2]]


def test_offspring_allocation_largest_remainder():
    spe = Speciation()
    sa = Species(id=1, representative=group_a())
    sb = Species(id=2, representative=group_b())
    sa.adjusted_fitness_sum = 0.5
    sb.adjusted_fitness_sum = 0.5
    spe.species = [sa, sb]

    allocation = spe.allocate_offspring(5)
    assert allocation == {1: 3, 2: 2}  # equal fractions -> lowest id gets the remainder
    assert sum(allocation.values()) == 5


def test_zero_fitness_species_gets_zero_offspring():
    spe = Speciation()
    sa = Species(id=1, representative=group_a())
    sb = Species(id=2, representative=group_b())
    sa.adjusted_fitness_sum = 6.0
    sb.adjusted_fitness_sum = 0.0
    spe.species = [sa, sb]

    allocation = spe.allocate_offspring(6)
    assert allocation == {1: 6, 2: 0}


def test_all_zero_fitness_falls_back_to_uniform():
    spe = Speciation()
    sa = Species(id=1, representative=group_a())
    sb = Species(id=2, representative=group_b())
    sa.adjusted_fitness_sum = 0.0
    sb.adjusted_fitness_sum = 0.0
    spe.species = [sa, sb]

    allocation = spe.allocate_offspring(5)
    assert allocation == {1: 3, 2: 2}
    assert sum(allocation.values()) == 5


# ------------------------------------------------------------------ extinction


def test_stagnant_species_goes_extinct():
    config = SpeciationConfig(stagnation_threshold=2)
    spe = Speciation(config)

    def flat_generation():
        return [
            genome_with([(0, 10, 0.5, 1)], fitness=1.0),
            genome_with([(0, 10, 0.5, 1)], fitness=0.9),
        ]

    spe.speciate(flat_generation())
    assert len(spe.species) == 1

    spe.speciate(flat_generation())
    assert len(spe.species) == 1  # stagnation 1, not yet extinct

    spe.speciate(flat_generation())
    assert len(spe.species) == 0  # stagnation 2 > threshold -> extinct


def test_improving_species_survives():
    config = SpeciationConfig(stagnation_threshold=2)
    spe = Speciation(config)

    def generation(best):
        return [genome_with([(0, 10, 0.5, 1)], fitness=best)]

    spe.speciate(generation(1.0))
    spe.speciate(generation(2.0))  # improved -> stagnation resets
    spe.speciate(generation(2.0))  # plateau -> stagnation 1, still under threshold
    assert len(spe.species) == 1


def test_empty_species_pruned():
    spe = Speciation()
    spe.species = [Species(id=1, representative=group_a())]
    assert spe.prune_stagnant() == [1]
    assert spe.species == []


# ------------------------------------------------------------------ diagnostics


def test_species_report_structure():
    report = species_report(population_size=10, generations=3, seed=0)
    assert len(report) == 3
    for row in report:
        assert row["species_count"] >= 1
        assert sum(row["sizes"]) == 10
        assert 0 < row["best_fitness"] <= 1.0
