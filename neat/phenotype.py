"""Phenotype (Layer 2: machine).

A genome is a blueprint; a Network is the machine built from it. Decoding is
one-way and read-only on the genome: the genome is never mutated here, so it
can be decoded repeatedly (or in parallel) into fresh networks.

Version 1 supports feed-forward (acyclic, enabled-edge) graphs only.
Recurrent structure is rejected at decode time.
"""

from __future__ import annotations

import heapq
import math
from typing import Dict, List, Sequence, Tuple

from .genome import Genome, NodeType

__all__ = ["ACTIVATION", "Network"]

ACTIVATION = math.tanh


class Network:
    """An executable feed-forward neural network decoded from a Genome.

    Decoding (once, in ``__init__``):
      1. Reject active cycles — a feed-forward network cannot execute one.
      2. Collect input/output node ids (id order, deterministic).
      3. Collect biases and enabled incoming edges per node.
      4. Compute a deterministic topological execution order (heap-based Kahn).

    Execution (``activate``): write inputs, walk nodes in topological order,
    ``tanh`` each weighted sum plus bias, return outputs in id order.
    """

    def __init__(self, genome: Genome) -> None:
        if genome.has_cycle():
            raise ValueError(
                "cannot decode a cyclic genome into a feed-forward network"
            )

        self._input_ids: List[int] = sorted(
            n.id for n in genome.inputs if n.node_type is NodeType.INPUT
        )
        self._output_ids: List[int] = sorted(
            n.id for n in genome.outputs if n.node_type is NodeType.OUTPUT
        )
        self._bias: Dict[int, float] = {n.id: n.bias for n in genome.nodes.values()}

        self._incoming: Dict[int, List[Tuple[int, float]]] = {
            nid: [] for nid in genome.nodes
        }
        for conn in genome.enabled_connections:
            self._incoming[conn.out_node].append((conn.in_node, conn.weight))

        self._order: List[int] = self._topological_order(genome)

    @staticmethod
    def _topological_order(genome: Genome) -> List[int]:
        """Kahn's algorithm over enabled edges; ties broken by node id.

        Returns a list where every node appears only after all of its enabled
        predecessors. Input nodes come first (they have in-degree 0 and the
        lowest ids).
        """
        indegree: Dict[int, int] = {nid: 0 for nid in genome.nodes}
        adj: Dict[int, List[int]] = {nid: [] for nid in genome.nodes}
        for conn in genome.enabled_connections:
            adj[conn.in_node].append(conn.out_node)
            indegree[conn.out_node] += 1

        ready: List[int] = [nid for nid, degree in indegree.items() if degree == 0]
        heapq.heapify(ready)

        order: List[int] = []
        while ready:
            current = heapq.heappop(ready)
            order.append(current)
            for nxt in adj[current]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    heapq.heappush(ready, nxt)
        return order

    # ------------------------------------------------------------------ state

    @property
    def execution_order(self) -> List[int]:
        """Node ids in the order they are evaluated."""
        return list(self._order)

    @property
    def input_ids(self) -> List[int]:
        return list(self._input_ids)

    @property
    def output_ids(self) -> List[int]:
        return list(self._output_ids)

    def __len__(self) -> int:
        return len(self._order)

    # ------------------------------------------------------------------ run

    def activate(self, inputs: Sequence[float]) -> List[float]:
        """Run one forward pass. ``inputs`` length must match the input nodes."""
        if len(inputs) != len(self._input_ids):
            raise ValueError(
                f"expected {len(self._input_ids)} inputs, got {len(inputs)}"
            )

        values: Dict[int, float] = {}
        for nid, value in zip(self._input_ids, inputs):
            values[nid] = value

        for nid in self._order:
            if nid in self._input_ids:
                continue
            total = self._bias[nid]
            for source, weight in self._incoming.get(nid, ()):
                total += values[source] * weight
            values[nid] = ACTIVATION(total)

        return [values[nid] for nid in self._output_ids]

    # ------------------------------------------------------------------ dunder

    def __repr__(self) -> str:
        return (
            f"Network(nodes={len(self._order)}, "
            f"inputs={len(self._input_ids)}, outputs={len(self._output_ids)})"
        )
