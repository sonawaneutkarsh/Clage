import pytest

from neat.genome import (
    DEFAULT_INPUT_IDS,
    DEFAULT_OUTPUT_IDS,
    ConnectionGene,
    Genome,
    NodeGene,
    NodeType,
)


# --------------------------------------------------------------------------- fields


def test_node_gene_has_expected_fields_and_default_bias():
    node = NodeGene(id=0, node_type=NodeType.INPUT)
    assert node.id == 0
    assert node.node_type is NodeType.INPUT
    assert node.bias == 0.0


def test_connection_gene_has_expected_fields_and_defaults():
    conn = ConnectionGene(in_node=0, out_node=10, weight=0.5)
    assert conn.in_node == 0
    assert conn.out_node == 10
    assert conn.weight == 0.5
    assert conn.enabled is True
    assert conn.innovation == 0


# --------------------------------------------------------------------------- construction


def test_minimal_genome_structure():
    g = Genome.minimal()
    assert [n.id for n in g.inputs] == list(DEFAULT_INPUT_IDS)
    assert [n.id for n in g.outputs] == list(DEFAULT_OUTPUT_IDS)
    assert g.hidden == []
    assert g.connections == []
    assert g.fitness == 0.0


def test_minimal_genome_bias_applies_to_outputs_only():
    g = Genome.minimal(bias=0.5)
    assert all(n.bias == 0.5 for n in g.outputs)
    assert all(n.bias == 0.0 for n in g.inputs)


def test_add_node_and_duplicate_id_raises():
    g = Genome.minimal()
    g.add_node(NodeGene(id=50, node_type=NodeType.HIDDEN, bias=-0.1))
    assert g.nodes[50].node_type is NodeType.HIDDEN
    assert g.hidden[0].bias == -0.1
    with pytest.raises(ValueError):
        g.add_node(NodeGene(id=50, node_type=NodeType.HIDDEN))


# --------------------------------------------------------------------------- add_connection rules


def test_add_connection_input_to_output_ok():
    g = Genome.minimal()
    conn = ConnectionGene(in_node=0, out_node=10, weight=0.4, innovation=1)
    g.add_connection(conn)
    assert g.connections == [conn]


def test_hidden_chain_accepted():
    g = Genome.minimal()
    g.add_node(NodeGene(id=50, node_type=NodeType.HIDDEN))
    g.add_connection(ConnectionGene(in_node=0, out_node=50, weight=1.0, innovation=1))
    g.add_connection(ConnectionGene(in_node=50, out_node=10, weight=1.0, innovation=2))
    assert len(g.connections) == 2
    assert g.hidden[0].id == 50


def test_add_connection_unknown_endpoint_raises():
    g = Genome.minimal()
    with pytest.raises(ValueError):
        g.add_connection(ConnectionGene(in_node=0, out_node=999, weight=1.0, innovation=1))
    with pytest.raises(ValueError):
        g.add_connection(ConnectionGene(in_node=999, out_node=10, weight=1.0, innovation=1))


def test_add_connection_self_loop_raises():
    g = Genome.minimal()
    with pytest.raises(ValueError):
        g.add_connection(ConnectionGene(in_node=0, out_node=0, weight=1.0, innovation=1))


def test_add_connection_input_to_input_raises():
    g = Genome.minimal()
    with pytest.raises(ValueError):
        g.add_connection(ConnectionGene(in_node=0, out_node=1, weight=1.0, innovation=1))


def test_add_connection_output_to_output_raises():
    g = Genome.minimal()
    with pytest.raises(ValueError):
        g.add_connection(ConnectionGene(in_node=10, out_node=11, weight=1.0, innovation=1))


def test_add_connection_duplicate_pair_raises_even_if_disabled():
    g = Genome.minimal()
    g.add_connection(ConnectionGene(in_node=0, out_node=10, weight=0.4, innovation=1))
    with pytest.raises(ValueError):
        g.add_connection(
            ConnectionGene(in_node=0, out_node=10, weight=0.9, enabled=False, innovation=1)
        )


def test_add_connection_creating_cycle_raises():
    g = Genome.minimal()
    g.add_node(NodeGene(id=50, node_type=NodeType.HIDDEN))
    g.add_node(NodeGene(id=51, node_type=NodeType.HIDDEN))
    g.add_connection(ConnectionGene(in_node=50, out_node=51, weight=1.0, innovation=1))
    with pytest.raises(ValueError):
        g.add_connection(ConnectionGene(in_node=51, out_node=50, weight=1.0, innovation=2))


# --------------------------------------------------------------------------- enabled / disabled


def test_disabled_connection_kept_but_excluded_from_enabled():
    g = Genome.minimal()
    g.add_connection(
        ConnectionGene(in_node=0, out_node=10, weight=0.4, innovation=1, enabled=False)
    )
    assert len(g.connections) == 1
    assert g.connections[0].enabled is False
    assert g.enabled_connections == []


def test_cycle_through_disabled_connection_is_latent_and_allowed():
    g = Genome.minimal()
    g.add_node(NodeGene(id=50, node_type=NodeType.HIDDEN))
    g.add_node(NodeGene(id=51, node_type=NodeType.HIDDEN))
    g.add_connection(ConnectionGene(in_node=50, out_node=51, weight=1.0, innovation=1))
    g.add_connection(
        ConnectionGene(in_node=51, out_node=50, weight=1.0, innovation=2, enabled=False),
        allow_cycle=True,
    )
    assert not g.has_cycle()
    g.validate()


def test_has_cycle_true_when_active_cycle_exists():
    g = Genome.minimal()
    g.add_node(NodeGene(id=50, node_type=NodeType.HIDDEN))
    g.add_node(NodeGene(id=51, node_type=NodeType.HIDDEN))
    g.add_connection(ConnectionGene(in_node=50, out_node=51, weight=1.0, innovation=1))
    g.add_connection(
        ConnectionGene(in_node=51, out_node=50, weight=1.0, innovation=2), allow_cycle=True
    )
    assert g.has_cycle()


# --------------------------------------------------------------------------- copy


def test_copy_preserves_ids_bias_innovation_and_fitness():
    g = Genome.minimal()
    g.add_node(NodeGene(id=50, node_type=NodeType.HIDDEN, bias=0.2))
    g.add_connection(ConnectionGene(in_node=0, out_node=50, weight=1.0, innovation=7))
    g.add_connection(ConnectionGene(in_node=50, out_node=10, weight=-0.5, innovation=8))
    g.fitness = 3.25

    c = g.copy()
    assert c is not g
    assert set(c.nodes) == set(g.nodes)
    assert c.nodes[50].bias == 0.2
    assert [conn.innovation for conn in c.connections] == [7, 8]
    assert c.fitness == 3.25


def test_copy_is_deep_no_alias():
    g = Genome.minimal()
    g.add_node(NodeGene(id=50, node_type=NodeType.HIDDEN))
    g.add_connection(ConnectionGene(in_node=0, out_node=50, weight=1.0, innovation=7))

    c = g.copy()
    c.connections[0].weight = 9.9
    c.nodes[50].bias = -1.0
    assert g.connections[0].weight == 1.0
    assert g.nodes[50].bias == 0.0


# --------------------------------------------------------------------------- validate


def test_validate_rejects_missing_endpoint():
    g = Genome(
        nodes={0: NodeGene(id=0, node_type=NodeType.INPUT)},
        connections=[ConnectionGene(in_node=0, out_node=5, weight=1.0)],
        validate_on_init=False,
    )
    with pytest.raises(ValueError):
        g.validate()


def test_validate_rejects_duplicate_pair():
    g = Genome(
        nodes={
            0: NodeGene(id=0, node_type=NodeType.INPUT),
            10: NodeGene(id=10, node_type=NodeType.OUTPUT),
        },
        connections=[
            ConnectionGene(in_node=0, out_node=10, weight=1.0, innovation=1),
            ConnectionGene(in_node=0, out_node=10, weight=2.0, innovation=1),
        ],
        validate_on_init=False,
    )
    with pytest.raises(ValueError):
        g.validate()


def test_validate_passes_on_valid_genome():
    g = Genome.minimal()
    g.add_node(NodeGene(id=50, node_type=NodeType.HIDDEN, bias=0.3))
    g.add_connection(ConnectionGene(in_node=0, out_node=50, weight=1.0, innovation=1))
    g.add_connection(ConnectionGene(in_node=50, out_node=11, weight=-0.2, innovation=2))
    g.validate()


def test_constructor_validates_by_default():
    with pytest.raises(ValueError):
        Genome(
            nodes={
                0: NodeGene(id=0, node_type=NodeType.INPUT),
                10: NodeGene(id=10, node_type=NodeType.OUTPUT),
            },
            connections=[
                ConnectionGene(in_node=0, out_node=10, weight=1.0),
                ConnectionGene(in_node=0, out_node=10, weight=2.0),
            ],
        )
