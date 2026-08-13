"""NEAT mutation operators (Layer 6: reproduction, mutation half only).

Each operator is standalone, mutates the passed genome **in place** (the
caller — the future reproduction layer — is responsible for passing a copy),
and takes an explicit ``random.Random`` so every call is reproducible from a
seed. Structural operators additionally take the run's ``InnovationDB``.

Return conventions:
- numeric operators (perturb / replace / bias): ``int`` count of genes changed;
- structural operators (add connection / add node / enable / disable): the
  created or toggled gene, or ``None`` if the mutation was a no-op.

Invariants preserved: only existing nodes are referenced, no duplicate
``(in, out)`` pairs, no active cycles, innovation numbers are never reminted,
and the split connection in add-node keeps its history.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import List, Optional, Tuple

from .genome import ConnectionGene, Genome, NodeGene, NodeType
from .innovation import InnovationDB

__all__ = [
    "DEFAULT_WEIGHT_BOUNDS",
    "MutationConfig",
    "apply_mutation",
    "mutate_biases",
    "mutate_add_connection",
    "mutate_add_node",
    "mutate_disable_connection",
    "mutate_enable_connection",
    "mutate_perturb_weights",
    "mutate_replace_weights",
]

DEFAULT_WEIGHT_BOUNDS: Tuple[float, float] = (-3.0, 3.0)


@dataclass
class MutationConfig:
    """Per-operator mutation rates and settings for the population engine."""

    weight_prob: float = 0.8
    weight_sigma: float = 0.1
    replace_prob: float = 0.1  # of weight mutations, chance to replace vs perturb
    bias_prob: float = 0.2
    bias_sigma: float = 0.1
    add_connection_prob: float = 0.05
    add_node_prob: float = 0.01
    enable_connection_prob: float = 0.05
    disable_connection_prob: float = 0.1
    weight_bounds: Tuple[float, float] = DEFAULT_WEIGHT_BOUNDS


def apply_mutation(
    genome: Genome,
    rng: Random,
    db: InnovationDB,
    config: Optional[MutationConfig] = None,
) -> None:
    """Roll each mutation operator against its probability.

    Applies to ``genome`` in place (the caller owns the copy). Deterministic
    from ``rng``; structural operators consult ``db`` for innovation numbers.
    """
    config = config or MutationConfig()

    if rng.random() < config.weight_prob:
        if rng.random() < config.replace_prob:
            mutate_replace_weights(
                genome, rng, prob=0.1, bounds=config.weight_bounds
            )
        else:
            mutate_perturb_weights(
                genome, rng, prob=0.1, sigma=config.weight_sigma, bounds=config.weight_bounds
            )

    if rng.random() < config.bias_prob:
        mutate_biases(
            genome, rng, prob=0.1, sigma=config.bias_sigma, bounds=config.weight_bounds
        )

    if rng.random() < config.add_connection_prob:
        mutate_add_connection(genome, rng, db)

    if rng.random() < config.add_node_prob:
        mutate_add_node(genome, rng, db)

    if rng.random() < config.enable_connection_prob:
        mutate_enable_connection(genome, rng)

    if rng.random() < config.disable_connection_prob:
        mutate_disable_connection(genome, rng)


def _clamp(value: float, bounds: Tuple[float, float]) -> float:
    low, high = bounds
    return min(high, max(low, value))


# ------------------------------------------------------------------ 1 & 2. weights


def mutate_perturb_weights(
    genome: Genome,
    rng: Random,
    *,
    prob: float = 0.1,
    sigma: float = 0.1,
    bounds: Tuple[float, float] = DEFAULT_WEIGHT_BOUNDS,
) -> int:
    """Nudge connection weights by Gaussian noise, clamped to ``bounds``."""
    count = 0
    for conn in genome.connections:
        if rng.random() < prob:
            conn.weight = _clamp(conn.weight + rng.gauss(0.0, sigma), bounds)
            count += 1
    return count


def mutate_replace_weights(
    genome: Genome,
    rng: Random,
    *,
    prob: float = 0.1,
    bounds: Tuple[float, float] = DEFAULT_WEIGHT_BOUNDS,
) -> int:
    """Replace connection weights with fresh uniform random values."""
    count = 0
    for conn in genome.connections:
        if rng.random() < prob:
            conn.weight = rng.uniform(*bounds)
            count += 1
    return count


# ------------------------------------------------------------------ 3. bias


def mutate_biases(
    genome: Genome,
    rng: Random,
    *,
    prob: float = 0.1,
    sigma: float = 0.1,
    bounds: Tuple[float, float] = DEFAULT_WEIGHT_BOUNDS,
) -> int:
    """Perturb hidden/output node biases. Inputs are never mutated (bias 0.0)."""
    count = 0
    for node in genome.nodes.values():
        if node.node_type is NodeType.INPUT:
            continue
        if rng.random() < prob:
            node.bias = _clamp(node.bias + rng.gauss(0.0, sigma), bounds)
            count += 1
    return count


# ------------------------------------------------------------------ 4. add connection


def mutate_add_connection(
    genome: Genome,
    rng: Random,
    db: InnovationDB,
    *,
    weight_range: Tuple[float, float] = (-1.0, 1.0),
    max_attempts: int = 20,
) -> Optional[ConnectionGene]:
    """Add a new enabled connection between two existing nodes.

    The pair is validated **before** an innovation number is minted, so an
    invalid pair never pollutes the ledger. Duplicate pairs (including a
    disabled gene for the same pair) and cycle-creating edges are rejected;
    after ``max_attempts`` failed random draws the mutation is a no-op.
    """
    node_ids = list(genome.nodes)
    for _ in range(max_attempts):
        in_node = rng.choice(node_ids)
        out_node = rng.choice(node_ids)
        try:
            genome.validate_connection(in_node, out_node)
        except ValueError:
            continue

        innovation = db.connection_innovation(in_node, out_node)
        conn = ConnectionGene(
            in_node=in_node,
            out_node=out_node,
            weight=rng.uniform(*weight_range),
            innovation=innovation,
        )
        genome.add_connection(conn)
        return conn
    return None


# ------------------------------------------------------------------ 5. add node


def mutate_add_node(
    genome: Genome,
    rng: Random,
    db: InnovationDB,
    *,
    first_weight: float = 1.0,
) -> Optional[NodeGene]:
    """Split a random enabled connection by inserting a hidden node.

    The original connection is disabled but keeps its innovation number. The
    hidden node gets a fresh id; ``in -> hidden`` gets ``first_weight`` and a
    new innovation, ``hidden -> out`` gets the *old* weight and a new
    innovation. Splitting preserves acyclicity by construction.

    Canonical NEAT: a structural change keeps the same historical markers
    everywhere. If this genome already hosts the node the ledger records for
    this split (i.e. a re-split of a re-enabled connection), the mutation is a
    no-op rather than minting a new node id + innovations — that would turn one
    invention into two and break crossover/speciation alignment.
    """
    enabled = [c for c in genome.connections if c.enabled]
    if not enabled:
        return None

    split = rng.choice(enabled)

    recorded = db.recorded_node_innovation(split.in_node, split.out_node)
    if recorded is not None and recorded.node_id in genome.nodes:
        return None  # same split, same node id, already present -> no-op

    split.enabled = False
    innovation = db.add_node_innovation(split.in_node, split.out_node)
    node_id = innovation.node_id
    in_innovation = innovation.in_innovation
    out_innovation = innovation.out_innovation

    hidden = NodeGene(id=node_id, node_type=NodeType.HIDDEN, bias=0.0)
    genome.add_node(hidden)

    genome.add_connection(
        ConnectionGene(
            in_node=split.in_node,
            out_node=hidden.id,
            weight=first_weight,
            innovation=in_innovation,
        )
    )
    genome.add_connection(
        ConnectionGene(
            in_node=hidden.id,
            out_node=split.out_node,
            weight=split.weight,
            innovation=out_innovation,
        )
    )
    return hidden


# ------------------------------------------------------------------ 6 & 7. toggle enabled


def mutate_enable_connection(
    genome: Genome,
    rng: Random,
) -> Optional[ConnectionGene]:
    """Re-enable a random disabled connection.

    Skips (returns None) if enabling it would create an active cycle — the
    feed-forward policy keeps the genome decodable.
    """
    disabled = [c for c in genome.connections if not c.enabled]
    if not disabled:
        return None

    conn = rng.choice(disabled)
    if genome.would_create_cycle(conn.in_node, conn.out_node):
        return None
    conn.enabled = True
    return conn


def mutate_disable_connection(
    genome: Genome,
    rng: Random,
) -> Optional[ConnectionGene]:
    """Disable a random enabled connection. The gene is kept, just dormant."""
    enabled = [c for c in genome.connections if c.enabled]
    if not enabled:
        return None

    conn = rng.choice(enabled)
    conn.enabled = False
    return conn
