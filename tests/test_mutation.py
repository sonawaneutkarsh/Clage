import random

import pytest

from neat.genome import ConnectionGene, Genome, NodeGene, NodeType
from neat.innovation import InnovationDB
from neat.mutation import (
    mutate_add_connection,
    mutate_add_node,
    mutate_biases,
    mutate_disable_connection,
    mutate_enable_connection,
    mutate_perturb_weights,
    mutate_replace_weights,
)


def wired_genome() -> Genome:
    g = Genome.minimal(input_ids=[0, 1], output_ids=[10, 11])
    g.add_connection(ConnectionGene(in_node=0, out_node=10, weight=0.5, innovation=1))
    g.add_connection(ConnectionGene(in_node=1, out_node=11, weight=-0.5, innovation=2))
    return g


# ------------------------------------------------------------ weight perturb


def test_perturb_weights_changes_weights():
    g = wired_genome()
    rng = random.Random(42)
    count = mutate_perturb_weights(g, rng, prob=1.0, sigma=1.0)
    assert count == 2
    assert [c.weight for c in g.connections] != [0.5, -0.5]
    assert [c.innovation for c in g.connections] == [1, 2]


def test_perturb_weights_clamps_to_bounds():
    g = wired_genome()
    g.connections[0].weight = 3.0
    mutate_perturb_weights(g, random.Random(7), prob=1.0, sigma=10.0, bounds=(-3.0, 3.0))
    for conn in g.connections:
        assert -3.0 <= conn.weight <= 3.0


def test_perturb_weights_noop_on_empty_genome():
    g = Genome.minimal(input_ids=[0], output_ids=[10])
    assert mutate_perturb_weights(g, random.Random(1), prob=1.0) == 0


def test_perturb_weights_reproducible_with_seeded_rng():
    g1, g2 = wired_genome(), wired_genome()
    mutate_perturb_weights(g1, random.Random(99), prob=1.0, sigma=0.3)
    mutate_perturb_weights(g2, random.Random(99), prob=1.0, sigma=0.3)
    assert [c.weight for c in g1.connections] == [c.weight for c in g2.connections]


# ------------------------------------------------------------ weight replace


def test_replace_weights_replaces_all_within_bounds():
    g = wired_genome()
    count = mutate_replace_weights(g, random.Random(5), prob=1.0)
    assert count == 2
    for conn in g.connections:
        assert -3.0 <= conn.weight <= 3.0
    assert [c.innovation for c in g.connections] == [1, 2]


def test_replace_weights_zero_prob_noop():
    g = wired_genome()
    before = [c.weight for c in g.connections]
    assert mutate_replace_weights(g, random.Random(5), prob=0.0) == 0
    assert [c.weight for c in g.connections] == before


def test_replace_weights_reproducible_with_seeded_rng():
    g1, g2 = wired_genome(), wired_genome()
    mutate_replace_weights(g1, random.Random(3))
    mutate_replace_weights(g2, random.Random(3))
    assert [c.weight for c in g1.connections] == [c.weight for c in g2.connections]


# ------------------------------------------------------------ bias


def test_mutate_biases_changes_hidden_and_output_only():
    g = wired_genome()
    g.add_node(NodeGene(id=50, node_type=NodeType.HIDDEN, bias=0.5))
    count = mutate_biases(g, random.Random(11), prob=1.0, sigma=1.0)
    assert count == 3  # 1 hidden + 2 outputs
    assert all(n.bias == 0.0 for n in g.inputs)
    assert g.nodes[50].bias != 0.5


def test_mutate_biases_clamps_to_bounds():
    g = wired_genome()
    mutate_biases(g, random.Random(2), prob=1.0, sigma=100.0, bounds=(-3.0, 3.0))
    for node in g.nodes.values():
        if node.node_type is not NodeType.INPUT:
            assert -3.0 <= node.bias <= 3.0


def test_mutate_biases_reproducible_with_seeded_rng():
    g1, g2 = wired_genome(), wired_genome()
    mutate_biases(g1, random.Random(13), prob=1.0, sigma=0.2)
    mutate_biases(g2, random.Random(13), prob=1.0, sigma=0.2)
    assert {n.id: n.bias for n in g1.nodes.values()} == {
        n.id: n.bias for n in g2.nodes.values()
    }


# ------------------------------------------------------------ add connection


def test_add_connection_creates_valid_connection():
    g = Genome.minimal(input_ids=[0], output_ids=[10])
    db = InnovationDB()
    conn = mutate_add_connection(g, random.Random(0), db)
    assert conn is not None
    assert (conn.in_node, conn.out_node) == (0, 10)
    assert conn.enabled is True
    assert conn.innovation == 1
    assert -1.0 <= conn.weight <= 1.0
    g.validate()


def test_add_connection_noop_when_only_pair_already_exists():
    g = Genome.minimal(input_ids=[0], output_ids=[10])
    db = InnovationDB()
    assert mutate_add_connection(g, random.Random(0), db) is not None
    assert mutate_add_connection(g, random.Random(1), db) is None
    assert len(g.connections) == 1


def test_add_connection_never_creates_cycle_or_duplicate():
    g = Genome.minimal(input_ids=[0], output_ids=[10])
    g.add_node(NodeGene(id=50, node_type=NodeType.HIDDEN))
    g.add_connection(ConnectionGene(in_node=0, out_node=50, weight=1.0, innovation=1))
    g.add_connection(ConnectionGene(in_node=50, out_node=10, weight=1.0, innovation=2))
    g.add_connection(ConnectionGene(in_node=0, out_node=10, weight=1.0, innovation=3))
    db = InnovationDB()

    for _ in range(40):
        assert mutate_add_connection(g, random.Random(123), db) is None
    assert len(g.connections) == 3
    assert not g.has_cycle()


def test_add_connection_does_not_mint_innovations_for_invalid_pairs():
    g = Genome.minimal(input_ids=[0], output_ids=[10])
    g.add_node(NodeGene(id=50, node_type=NodeType.HIDDEN))
    g.add_connection(ConnectionGene(in_node=0, out_node=50, weight=1.0, innovation=1))
    g.add_connection(ConnectionGene(in_node=50, out_node=10, weight=1.0, innovation=2))
    g.add_connection(ConnectionGene(in_node=0, out_node=10, weight=1.0, innovation=3))
    db = InnovationDB()

    for _ in range(40):
        mutate_add_connection(g, random.Random(123), db)

    assert len(g.connections) == 3
    assert len(db.to_dict()["connection_innovations"]) == 0


def test_add_connection_innovation_reused_across_genomes():
    g1 = Genome.minimal(input_ids=[0], output_ids=[10])
    g2 = Genome.minimal(input_ids=[0], output_ids=[10])
    db = InnovationDB()
    c1 = mutate_add_connection(g1, random.Random(0), db)
    c2 = mutate_add_connection(g2, random.Random(0), db)
    assert c1.innovation == c2.innovation == 1


def test_add_connection_reproducible_with_seeded_rng():
    g1 = Genome.minimal(input_ids=[0, 1], output_ids=[10, 11])
    g2 = g1.copy()
    db = InnovationDB()
    c1 = mutate_add_connection(g1, random.Random(9), db)
    c2 = mutate_add_connection(g2, random.Random(9), db)
    assert (c1.in_node, c1.out_node) == (c2.in_node, c2.out_node)


# ------------------------------------------------------------ add node


def test_add_node_splits_enabled_connection():
    g = Genome.minimal(input_ids=[0], output_ids=[10])
    db = InnovationDB()
    db.connection_innovation(0, 10)  # the run's ledger issued innov 1 for 0->10
    g.add_connection(ConnectionGene(in_node=0, out_node=10, weight=0.5, innovation=1))

    hidden = mutate_add_node(g, random.Random(0), db)
    assert hidden is not None
    assert hidden.id == 14
    assert hidden.node_type is NodeType.HIDDEN

    original = g.connections[0]
    assert original.enabled is False
    assert original.innovation == 1
    assert original.weight == 0.5

    pairs = {(c.in_node, c.out_node) for c in g.connections}
    assert (0, 14) in pairs and (14, 10) in pairs
    by_pair = {(c.in_node, c.out_node): c for c in g.connections}
    assert by_pair[(0, 14)].weight == 1.0
    assert by_pair[(14, 10)].weight == 0.5
    assert by_pair[(0, 14)].innovation == 2
    assert by_pair[(14, 10)].innovation == 3
    g.validate()


def test_add_node_reuses_innovation_on_same_split():
    db = InnovationDB()

    def fresh():
        g = Genome.minimal(input_ids=[0], output_ids=[10])
        g.add_connection(ConnectionGene(in_node=0, out_node=10, weight=0.5, innovation=1))
        return g

    g1, g2 = fresh(), fresh()
    h1 = mutate_add_node(g1, random.Random(0), db)
    h2 = mutate_add_node(g2, random.Random(0), db)
    assert h1.id == h2.id == 14
    in1 = next(c for c in g1.connections if c.out_node == 14)
    in2 = next(c for c in g2.connections if c.out_node == 14)
    out1 = next(c for c in g1.connections if c.in_node == 14)
    out2 = next(c for c in g2.connections if c.in_node == 14)
    assert (in1.innovation, out1.innovation) == (in2.innovation, out2.innovation)


def test_add_node_different_splits_get_different_node_ids():
    db = InnovationDB()

    def fresh(pair, innov, weight):
        g = Genome.minimal(input_ids=[0, 1], output_ids=[10, 11])
        g.add_connection(ConnectionGene(in_node=pair[0], out_node=pair[1], weight=weight, innovation=innov))
        return g

    h1 = mutate_add_node(fresh((0, 10), 1, 0.5), random.Random(0), db)
    h2 = mutate_add_node(fresh((1, 11), 2, 0.7), random.Random(0), db)
    assert h1.id != h2.id


def test_add_node_noop_without_enabled_connections():
    g = wired_genome()
    for conn in g.connections:
        conn.enabled = False
    assert mutate_add_node(g, random.Random(0), InnovationDB()) is None


def test_add_node_preserves_acyclicity():
    g = Genome.minimal(input_ids=[0, 1], output_ids=[10])
    db = InnovationDB()
    db.connection_innovation(0, 10)
    db.connection_innovation(1, 10)
    g.add_connection(ConnectionGene(in_node=0, out_node=10, weight=0.5, innovation=1))
    g.add_connection(ConnectionGene(in_node=1, out_node=10, weight=0.5, innovation=2))
    mutate_add_node(g, random.Random(4), db)
    mutate_add_node(g, random.Random(5), db)
    assert not g.has_cycle()
    g.validate()


def test_add_node_resplit_after_reenable_is_noop():
    g = Genome.minimal(input_ids=[0], output_ids=[10])
    db = InnovationDB()
    db.connection_innovation(0, 10)
    g.add_connection(ConnectionGene(in_node=0, out_node=10, weight=0.5, innovation=1))

    first = mutate_add_node(g, random.Random(0), db)
    assert first.id == 14
    nodes_before = len(g.nodes)
    conns_before = len(g.connections)

    mutate_enable_connection(g, random.Random(1))
    assert g.connections[0].enabled is True

    # Re-splitting the same connection would reuse node 14, which is already in
    # this genome — canonical NEAT makes this a no-op (one invention, one node).
    assert mutate_add_node(g, random.Random(2), db) is None
    assert len(g.nodes) == nodes_before
    assert len(g.connections) == conns_before
    g.validate()
    assert not g.has_cycle()
    for conn in g.connections:
        assert conn.in_node in g.nodes
        assert conn.out_node in g.nodes


# ------------------------------------------------------------ enable/disable


def test_enable_connection_revives_disabled_gene():
    g = wired_genome()
    target = g.connections[0]
    target.enabled = False
    result = mutate_enable_connection(g, random.Random(0))
    assert result is target
    assert result.enabled is True


def test_enable_connection_skips_cycle_creating_reenable():
    g = Genome.minimal(input_ids=[0], output_ids=[10])
    g.add_node(NodeGene(id=50, node_type=NodeType.HIDDEN))
    g.add_node(NodeGene(id=51, node_type=NodeType.HIDDEN))
    g.add_connection(ConnectionGene(in_node=50, out_node=51, weight=1.0, innovation=1))
    g.add_connection(
        ConnectionGene(in_node=51, out_node=50, weight=1.0, innovation=2, enabled=False),
        allow_cycle=True,
    )

    result = mutate_enable_connection(g, random.Random(0))
    assert result is None
    assert g.connections[1].enabled is False
    assert not g.has_cycle()


def test_enable_connection_noop_when_none_disabled():
    g = wired_genome()
    assert mutate_enable_connection(g, random.Random(0)) is None


def test_disable_connection_disables_enabled_gene():
    g = wired_genome()
    result = mutate_disable_connection(g, random.Random(0))
    assert result is not None
    assert result.enabled is False
    assert len(g.connections) == 2  # gene kept, just dormant


def test_disable_connection_noop_when_none_enabled():
    g = wired_genome()
    for conn in g.connections:
        conn.enabled = False
    assert mutate_disable_connection(g, random.Random(0)) is None


# ------------------------------------------------------------ global guarantees


def test_all_mutations_leave_genome_valid_and_references_intact():
    g = Genome.minimal(input_ids=[0, 1], output_ids=[10, 11])
    db = InnovationDB()
    rng = random.Random(0)
    g.add_connection(ConnectionGene(in_node=0, out_node=10, weight=0.5, innovation=1))

    mutate_perturb_weights(g, rng, prob=1.0)
    mutate_replace_weights(g, rng, prob=1.0)
    mutate_biases(g, rng, prob=1.0)
    mutate_add_connection(g, rng, db)
    mutate_add_node(g, rng, db)
    mutate_disable_connection(g, rng)
    mutate_enable_connection(g, rng)

    g.validate()
    for conn in g.connections:
        assert conn.in_node in g.nodes
        assert conn.out_node in g.nodes
    assert not g.has_cycle()


def test_mutations_preserve_existing_innovation_numbers():
    g = wired_genome()
    db = InnovationDB()
    rng = random.Random(0)
    before = {c.innovation for c in g.connections}

    mutate_perturb_weights(g, rng, prob=1.0)
    mutate_add_connection(g, rng, db)
    mutate_add_node(g, rng, db)
    mutate_disable_connection(g, rng)
    mutate_enable_connection(g, rng)

    after = {c.innovation for c in g.connections}
    assert before.issubset(after)
