"""Clage NEAT core package.

Layers implemented so far:
  Layer 1 (blueprint): the genome representation — node genes, connection
    genes, innovation numbers. Pure data, nothing executes.
  Layer 2 (machine): the phenotype — decodes a Genome into an executable
    feed-forward neural network.
  Layer 6 (mutation half): the seven mutation operators plus the innovation
    ledger that mints structural ids.
  Layer 6 (crossover half): ``crossover`` — recombines two genomes into a
    child by aligning genes on innovation numbers.
  Layer 6 (speciation half): compatibility distance, fitness sharing, species
    assignment and offspring allocation — protects novel topologies.

Nothing here evolves a population yet (no reproduction loop); the diagnostics
in ``neat.diagnostics`` run a synthetic mutation-only loop to observe
speciation.
"""

from .crossover import DEFAULT_INHERIT_DISABLED_PROB, crossover
from .genome import (
    DEFAULT_INPUT_IDS,
    DEFAULT_OUTPUT_IDS,
    ConnectionGene,
    Genome,
    NodeGene,
    NodeType,
)
from .innovation import DEFAULT_NEXT_INNOVATION, DEFAULT_NEXT_NODE_ID, InnovationDB, NodeInnovation
from .mutation import (
    DEFAULT_WEIGHT_BOUNDS,
    MutationConfig,
    apply_mutation,
    mutate_add_connection,
    mutate_add_node,
    mutate_biases,
    mutate_disable_connection,
    mutate_enable_connection,
    mutate_perturb_weights,
    mutate_replace_weights,
)
from .phenotype import ACTIVATION, Network
from .population import Population
from .speciation import (
    Speciation,
    SpeciationConfig,
    Species,
    average_weight_difference,
    compatibility_distance,
    gene_counts,
)

__all__ = [
    "DEFAULT_INPUT_IDS",
    "DEFAULT_OUTPUT_IDS",
    "DEFAULT_INHERIT_DISABLED_PROB",
    "DEFAULT_NEXT_INNOVATION",
    "DEFAULT_NEXT_NODE_ID",
    "DEFAULT_WEIGHT_BOUNDS",
    "ACTIVATION",
    "ConnectionGene",
    "Genome",
    "InnovationDB",
    "MutationConfig",
    "Network",
    "NodeGene",
    "NodeInnovation",
    "NodeType",
    "Population",
    "Speciation",
    "SpeciationConfig",
    "Species",
    "apply_mutation",
    "average_weight_difference",
    "compatibility_distance",
    "crossover",
    "gene_counts",
    "mutate_add_connection",
    "mutate_add_node",
    "mutate_biases",
    "mutate_disable_connection",
    "mutate_enable_connection",
    "mutate_perturb_weights",
    "mutate_replace_weights",
]
