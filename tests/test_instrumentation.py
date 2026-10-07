import math

from neat.genome import ConnectionGene, Genome, NodeGene, NodeType
from neat.phenotype import Network


def test_instrumented_hidden_inference_matches_independent_arithmetic():
    genome = Genome(nodes={0: NodeGene(0, NodeType.INPUT),
                           14: NodeGene(14, NodeType.HIDDEN, .15),
                           10: NodeGene(10, NodeType.OUTPUT, -.2)},
                    connections=[ConnectionGene(0, 14, .7, innovation=1),
                                 ConnectionGene(14, 10, -.9, innovation=2),
                                 ConnectionGene(0, 10, 100, enabled=False, innovation=3)])
    network = Network(genome)
    for value in [-1.0, -.4, 0.0, .8, 1.0]:
        hidden = math.tanh(.15 + value * .7)
        expected = math.tanh(-.2 + hidden * -.9)
        outputs, activations = network.activate_with_trace([value])
        assert outputs == network.activate([value]) == [expected]
        assert activations == {0: value, 14: hidden, 10: expected}
    assert not genome.connections[2].enabled
