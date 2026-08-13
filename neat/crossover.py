"""NEAT crossover (Layer 6: reproduction, crossover half).

Builds one child genome from two parents by aligning their connection genes
by **innovation number** — the historical birth certificate — not by list
position or topology. Parents are never modified: every inherited gene is
deep-copied, and node ids are never reminted.

Inheritance rules:
- **matching** (both parents have the innovation): weight from a randomly
  chosen parent; enabled if both parents enabled, else disabled with
  probability ``inherit_disabled_prob`` (the disabled-gene rule).
- **disjoint / excess** (one parent only): if one parent is fitter, inherited
  only from the fitter parent; at equal fitness, inherited with 50% probability
  (the defined equal-fitness policy).

Feed-forward safeguard: an inherited *enabled* gene that would close a cycle in
the child is kept but disabled, so the child is always a valid DAG. Classic
NEAT (which allows recurrence) does not need this; v1 is feed-forward.
"""

from __future__ import annotations

from random import Random
from typing import Dict, List, Optional, Tuple

from .genome import ConnectionGene, Genome, NodeGene

__all__ = ["crossover"]

DEFAULT_INHERIT_DISABLED_PROB: float = 0.75


def crossover(
    parent_a: Genome,
    parent_b: Genome,
    rng: Random,
    *,
    inherit_disabled_prob: float = DEFAULT_INHERIT_DISABLED_PROB,
) -> Genome:
    """Create a child genome from two parents. Neither parent is modified."""
    nodes: Dict[int, NodeGene] = {nid: node.copy() for nid, node in parent_a.nodes.items()}
    for nid, node in parent_b.nodes.items():
        if nid not in nodes:
            nodes[nid] = node.copy()

    a_by_innov: Dict[int, ConnectionGene] = {c.innovation: c for c in parent_a.connections}
    b_by_innov: Dict[int, ConnectionGene] = {c.innovation: c for c in parent_b.connections}

    if parent_a.fitness > parent_b.fitness:
        fitter: Optional[Genome] = parent_a
        other: Optional[Genome] = parent_b
    elif parent_b.fitness > parent_a.fitness:
        fitter, other = parent_b, parent_a
    else:
        fitter, other = None, None  # equal fitness -> 50/50 policy

    innovations = sorted(set(a_by_innov) | set(b_by_innov))
    child_connections: List[ConnectionGene] = []
    enabled_edges: List[Tuple[int, int]] = []

    for innovation in innovations:
        ca = a_by_innov.get(innovation)
        cb = b_by_innov.get(innovation)

        if ca is not None and cb is not None:
            source = ca if rng.random() < 0.5 else cb
            conn = ConnectionGene(
                in_node=source.in_node,
                out_node=source.out_node,
                weight=source.weight,
                enabled=True,
                innovation=innovation,
            )
            if ca.enabled and cb.enabled:
                conn.enabled = True
            else:
                conn.enabled = rng.random() >= inherit_disabled_prob
        elif ca is not None:
            if fitter is not None:
                keep = fitter is parent_a
            else:
                keep = rng.random() < 0.5
            if not keep:
                continue
            conn = ca.copy()
        else:
            if fitter is not None:
                keep = fitter is parent_b
            else:
                keep = rng.random() < 0.5
            if not keep:
                continue
            conn = cb.copy()

        if conn.enabled and _closes_cycle(enabled_edges, conn.in_node, conn.out_node):
            conn.enabled = False
        if conn.enabled:
            enabled_edges.append((conn.in_node, conn.out_node))
        child_connections.append(conn)

    child = Genome(nodes=nodes, connections=child_connections)
    return child


def _closes_cycle(
    enabled_edges: List[Tuple[int, int]],
    in_node: int,
    out_node: int,
) -> bool:
    """True if adding ``in_node -> out_node`` to ``enabled_edges`` closes a cycle.

    Yes iff ``out_node`` already reaches ``in_node`` through the current edges.
    """
    if in_node == out_node:
        return True

    adj: Dict[int, List[int]] = {}
    for src, dst in enabled_edges:
        adj.setdefault(src, []).append(dst)

    stack = [out_node]
    seen = set()
    while stack:
        current = stack.pop()
        if current == in_node:
            return True
        if current in seen:
            continue
        seen.add(current)
        stack.extend(adj.get(current, ()))
    return False
