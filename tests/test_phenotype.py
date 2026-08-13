import math

import pytest

from neat.genome import ConnectionGene, Genome, NodeGene, NodeType
from neat.phenotype import ACTIVATION, Network


def test_one_input_one_output():
    g = Genome.minimal(input_ids=[0], output_ids=[10])
    g.add_connection(ConnectionGene(in_node=0, out_node=10, weight=1.0, innovation=1))

    net = Network(g)
    assert net.activate([2.0]) == [pytest.approx(ACTIVATION(2.0))]


def test_multiple_inputs_weighted_sum():
    g = Genome.minimal(input_ids=[0, 1], output_ids=[10])
    g.add_connection(ConnectionGene(in_node=0, out_node=10, weight=1.0, innovation=1))
    g.add_connection(ConnectionGene(in_node=1, out_node=10, weight=-1.0, innovation=2))

    net = Network(g)
    expected = ACTIVATION(2.0 * 1.0 + 3.0 * -1.0)
    assert net.activate([2.0, 3.0]) == [pytest.approx(expected)]


def test_hidden_node():
    g = Genome.minimal(input_ids=[0], output_ids=[10])
    g.add_node(NodeGene(id=50, node_type=NodeType.HIDDEN))
    g.add_connection(ConnectionGene(in_node=0, out_node=50, weight=1.0, innovation=1))
    g.add_connection(ConnectionGene(in_node=50, out_node=10, weight=2.0, innovation=2))

    net = Network(g)
    expected = ACTIVATION(2.0 * ACTIVATION(1.0))
    assert net.activate([1.0]) == [pytest.approx(expected)]


def test_multiple_layers():
    g = Genome.minimal(input_ids=[0], output_ids=[10])
    g.add_node(NodeGene(id=50, node_type=NodeType.HIDDEN))
    g.add_node(NodeGene(id=51, node_type=NodeType.HIDDEN))
    g.add_connection(ConnectionGene(in_node=0, out_node=50, weight=1.0, innovation=1))
    g.add_connection(ConnectionGene(in_node=50, out_node=51, weight=1.0, innovation=2))
    g.add_connection(ConnectionGene(in_node=51, out_node=10, weight=1.0, innovation=3))

    net = Network(g)
    expected = ACTIVATION(ACTIVATION(ACTIVATION(1.0)))
    assert net.activate([1.0]) == [pytest.approx(expected)]
    assert net.execution_order == [0, 50, 51, 10]


def test_output_bias_with_no_connections():
    g = Genome.minimal(input_ids=[0], output_ids=[10], bias=1.0)

    net = Network(g)
    assert net.activate([0.0]) == [pytest.approx(ACTIVATION(1.0))]


def test_hidden_bias():
    g = Genome.minimal(input_ids=[0], output_ids=[10])
    g.add_node(NodeGene(id=50, node_type=NodeType.HIDDEN, bias=0.5))
    g.add_connection(ConnectionGene(in_node=0, out_node=50, weight=1.0, innovation=1))
    g.add_connection(ConnectionGene(in_node=50, out_node=10, weight=1.0, innovation=2))

    net = Network(g)
    expected = ACTIVATION(ACTIVATION(1.0 + 0.5))
    assert net.activate([1.0]) == [pytest.approx(expected)]


def test_disabled_connection_is_ignored():
    g = Genome.minimal(input_ids=[0, 1], output_ids=[10])
    g.add_connection(ConnectionGene(in_node=0, out_node=10, weight=1.0, innovation=1))
    g.add_connection(
        ConnectionGene(in_node=1, out_node=10, weight=100.0, enabled=False, innovation=2)
    )

    net = Network(g)
    expected = ACTIVATION(2.0)
    assert net.activate([2.0, 1.0]) == [pytest.approx(expected)]


def test_invalid_cyclic_genome_raises():
    g = Genome.minimal(input_ids=[0], output_ids=[10])
    g.add_node(NodeGene(id=50, node_type=NodeType.HIDDEN))
    g.add_node(NodeGene(id=51, node_type=NodeType.HIDDEN))
    g.add_connection(ConnectionGene(in_node=50, out_node=51, weight=1.0, innovation=1))
    g.add_connection(
        ConnectionGene(in_node=51, out_node=50, weight=1.0, innovation=2),
        allow_cycle=True,
    )

    with pytest.raises(ValueError):
        Network(g)


def test_wrong_input_count_raises():
    g = Genome.minimal(input_ids=[0, 1], output_ids=[10])
    g.add_connection(ConnectionGene(in_node=0, out_node=10, weight=1.0, innovation=1))
    g.add_connection(ConnectionGene(in_node=1, out_node=10, weight=1.0, innovation=2))

    net = Network(g)
    with pytest.raises(ValueError):
        net.activate([1.0])


def test_latent_cycle_through_disabled_connection_decodes():
    g = Genome.minimal(input_ids=[0], output_ids=[10])
    g.add_node(NodeGene(id=50, node_type=NodeType.HIDDEN))
    g.add_node(NodeGene(id=51, node_type=NodeType.HIDDEN))
    g.add_connection(ConnectionGene(in_node=50, out_node=51, weight=1.0, innovation=1))
    g.add_connection(
        ConnectionGene(in_node=51, out_node=50, weight=1.0, innovation=2, enabled=False),
        allow_cycle=True,
    )

    net = Network(g)
    assert len(net) == 4


def test_input_nodes_are_pass_through_not_activated():
    g = Genome.minimal(input_ids=[0], output_ids=[10])
    g.add_connection(ConnectionGene(in_node=0, out_node=10, weight=2.0, innovation=1))

    net = Network(g)
    # If the input were tanh-activated, 3.0 -> tanh(3.0) before the wire.
    assert net.activate([3.0]) == [pytest.approx(ACTIVATION(3.0 * 2.0))]
