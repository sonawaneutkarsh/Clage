import random

import pytest

from neat.crossover import crossover
from neat.genome import ConnectionGene, Genome, NodeGene, NodeType


class FixedRng(random.Random):
    """A seeded Random whose .random() returns a scripted sequence (exact tests)."""

    def __init__(self, values):
        super().__init__()
        self._values = list(values)

    def random(self):
        if not self._values:
            raise AssertionError("scripted rng exhausted")
        return self._values.pop(0)


def parent_a() -> Genome:
    g = Genome.minimal(input_ids=[0, 1], output_ids=[10])
    g.add_node(NodeGene(id=50, node_type=NodeType.HIDDEN))
    g.add_connection(ConnectionGene(0, 10, 0.5, innovation=1))
    g.add_connection(ConnectionGene(0, 50, 0.8, innovation=2))
    g.add_connection(ConnectionGene(50, 10, 0.3, innovation=3))
    g.add_connection(ConnectionGene(1, 10, -0.7, innovation=4))
    g.fitness = 10.0
    return g


def parent_b() -> Genome:
    g = Genome.minimal(input_ids=[0, 1], output_ids=[10])
    g.add_node(NodeGene(id=50, node_type=NodeType.HIDDEN))
    g.add_node(NodeGene(id=51, node_type=NodeType.HIDDEN))
    g.add_connection(ConnectionGene(0, 10, -0.2, enabled=False, innovation=1))
    g.add_connection(ConnectionGene(50, 10, 0.1, innovation=3))
    g.add_connection(ConnectionGene(0, 51, 0.6, innovation=5))
    g.add_connection(ConnectionGene(51, 10, -0.9, innovation=6))
    g.fitness = 4.0
    return g


def by_innovation(genome):
    return {c.innovation: c for c in genome.connections}


def test_parents_never_modified():
    a, b = parent_a(), parent_b()
    a_nodes = {nid: (n.node_type, n.bias) for nid, n in a.nodes.items()}
    b_nodes = {nid: (n.node_type, n.bias) for nid, n in b.nodes.items()}
    a_conns = [(c.innovation, c.in_node, c.out_node, c.weight, c.enabled) for c in a.connections]
    b_conns = [(c.innovation, c.in_node, c.out_node, c.weight, c.enabled) for c in b.connections]

    crossover(a, b, FixedRng([0.4, 0.2, 0.8]))

    assert {nid: (n.node_type, n.bias) for nid, n in a.nodes.items()} == a_nodes
    assert {nid: (n.node_type, n.bias) for nid, n in b.nodes.items()} == b_nodes
    assert [(c.innovation, c.in_node, c.out_node, c.weight, c.enabled) for c in a.connections] == a_conns
    assert [(c.innovation, c.in_node, c.out_node, c.weight, c.enabled) for c in b.connections] == b_conns


def test_child_fitness_reset_to_zero():
    child = crossover(parent_a(), parent_b(), FixedRng([0.4, 0.2, 0.8]))
    assert child.fitness == 0.0


def test_fitter_parent_inherits_disjoint_rejects_other_excess():
    child = crossover(parent_a(), parent_b(), FixedRng([0.4, 0.2, 0.8]))
    genes = by_innovation(child)

    assert set(genes) == {1, 2, 3, 4}
    assert 5 not in genes and 6 not in genes

    assert genes[1].in_node == 0 and genes[1].out_node == 10
    assert genes[1].weight == 0.5
    assert genes[1].enabled is False  # innov1 disabled in B -> 75% rule, script says disabled

    assert (genes[2].in_node, genes[2].out_node) == (0, 50)
    assert genes[2].weight == 0.8 and genes[2].enabled is True

    assert (genes[3].in_node, genes[3].out_node) == (50, 10)
    assert genes[3].weight == 0.1  # script picks B's weight
    assert genes[3].enabled is True

    assert (genes[4].in_node, genes[4].out_node) == (1, 10)
    assert genes[4].weight == -0.7 and genes[4].enabled is True


def test_other_parent_as_fitter_inherits_its_excess():
    a, b = parent_a(), parent_b()
    a.fitness, b.fitness = 4.0, 20.0  # B is now fitter
    child = crossover(a, b, FixedRng([0.4, 0.2, 0.8]))
    genes = by_innovation(child)

    assert set(genes) == {1, 3, 5, 6}
    assert 2 not in genes and 4 not in genes

    assert (genes[5].in_node, genes[5].out_node) == (0, 51)
    assert (genes[6].in_node, genes[6].out_node) == (51, 10)


def test_matching_weight_from_either_parent():
    a, b = parent_a(), parent_b()
    # innov1 consumes 2 draws (weight, disabled); innov3's weight is the 3rd draw.
    child_take_a = crossover(a, b, FixedRng([0.1, 0.9, 0.1]))
    child_take_b = crossover(a, b, FixedRng([0.1, 0.9, 0.9]))
    assert by_innovation(child_take_a)[3].weight == 0.3
    assert by_innovation(child_take_b)[3].weight == 0.1


def test_disabled_rule_always_disabled():
    child = crossover(parent_a(), parent_b(), FixedRng([0.1, 0.0, 0.1]),
                      inherit_disabled_prob=1.0)
    assert by_innovation(child)[1].enabled is False


def test_disabled_rule_always_enabled():
    child = crossover(parent_a(), parent_b(), FixedRng([0.1, 0.0, 0.1]),
                      inherit_disabled_prob=0.0)
    assert by_innovation(child)[1].enabled is True


def test_equal_fitness_policy_exact_gene_set():
    a, b = parent_a(), parent_b()
    a.fitness = b.fitness = 0.0
    child = crossover(a, b, FixedRng([0.9, 0.9, 0.9, 0.1, 0.9, 0.9, 0.1]))
    genes = by_innovation(child)

    assert set(genes) == {1, 3, 6}
    assert genes[1].enabled is True and genes[1].weight == -0.2
    assert genes[3].enabled is True and genes[3].weight == 0.3
    assert genes[6].enabled is True and genes[6].weight == -0.9


def test_child_nodes_are_union_of_parents():
    child = crossover(parent_a(), parent_b(), FixedRng([0.4, 0.2, 0.8]))
    assert set(child.nodes) == {0, 1, 10, 50, 51}
    assert child.nodes[51].node_type is NodeType.HIDDEN


def test_innovations_preserved_no_new_numbers():
    a, b = parent_a(), parent_b()
    parent_innovs = {c.innovation for c in a.connections} | {c.innovation for c in b.connections}
    child = crossover(a, b, FixedRng([0.4, 0.2, 0.8]))
    assert {c.innovation for c in child.connections}.issubset(parent_innovs)


def test_opposite_edges_child_stays_valid_and_acyclic():
    a = Genome.minimal(input_ids=[0], output_ids=[10])
    b = Genome.minimal(input_ids=[0], output_ids=[10])
    a.add_node(NodeGene(id=50, node_type=NodeType.HIDDEN))
    b.add_node(NodeGene(id=50, node_type=NodeType.HIDDEN))
    a.add_connection(ConnectionGene(50, 0, 1.0, innovation=1))  # 50 -> 0
    b.add_connection(ConnectionGene(0, 50, 1.0, innovation=2))  # 0 -> 50
    a.fitness = b.fitness = 0.0

    child = crossover(a, b, FixedRng([0.1, 0.1]))  # keep both
    genes = by_innovation(child)

    assert genes[1].enabled is True
    assert genes[2].enabled is False  # would close a cycle -> kept disabled
    assert not child.has_cycle()
    child.validate()


def test_child_is_valid_acyclic_and_decodable():
    child = crossover(parent_a(), parent_b(), FixedRng([0.4, 0.2, 0.8]))
    child.validate()
    assert not child.has_cycle()


def test_no_alias_child_does_not_affect_parents():
    a, b = parent_a(), parent_b()
    child = crossover(a, b, FixedRng([0.4, 0.2, 0.8]))
    child.connections[0].weight = 99.0
    child.nodes[50].bias = 42.0
    assert all(c.weight != 99.0 for c in a.connections)
    assert a.nodes[50].bias == 0.0


def test_reproducible_with_seeded_rng():
    a1, b1 = parent_a(), parent_b()
    a2, b2 = parent_a(), parent_b()
    c1 = crossover(a1, b1, random.Random(7))
    c2 = crossover(a2, b2, random.Random(7))

    assert sorted((c.innovation, c.weight, c.enabled) for c in c1.connections) == sorted(
        (c.innovation, c.weight, c.enabled) for c in c2.connections
    )
