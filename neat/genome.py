"""Genome representation (Layer 1: blueprint).

A genome answers: which neurons exist, which directed wires exist, what is
each wire's weight, is it in use, and when was each structure invented.

It answers *nothing* about runtime: no activation values, no network
execution, no evolution, no speciation. Those are separate layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Dict, Iterable, List, Optional

__all__ = [
    "DEFAULT_INPUT_IDS",
    "DEFAULT_OUTPUT_IDS",
    "NodeType",
    "NodeGene",
    "ConnectionGene",
    "Genome",
]

# The Clage experiment's fixed interface to the world: 10 sensors, 4 actions.
DEFAULT_INPUT_IDS: tuple[int, ...] = tuple(range(10))
DEFAULT_OUTPUT_IDS: tuple[int, ...] = tuple(range(10, 14))


class NodeType(IntEnum):
    INPUT = auto()
    OUTPUT = auto()
    HIDDEN = auto()


@dataclass
class NodeGene:
    """One neuron: a stable id, a role, and a learned bias."""

    id: int
    node_type: NodeType
    bias: float = 0.0

    def copy(self) -> "NodeGene":
        return NodeGene(id=self.id, node_type=self.node_type, bias=self.bias)

    def __repr__(self) -> str:
        return f"NodeGene(id={self.id}, type={self.node_type.name}, bias={self.bias:+.3f})"


@dataclass
class ConnectionGene:
    """One directed synapse: endpoints, weight, on/off switch, birth certificate."""

    in_node: int
    out_node: int
    weight: float
    enabled: bool = True
    innovation: int = 0

    def copy(self) -> "ConnectionGene":
        return ConnectionGene(
            in_node=self.in_node,
            out_node=self.out_node,
            weight=self.weight,
            enabled=self.enabled,
            innovation=self.innovation,
        )

    def __repr__(self) -> str:
        state = "on" if self.enabled else "OFF"
        return (
            f"ConnectionGene(innov={self.innovation}, {self.in_node}->{self.out_node}, "
            f"w={self.weight:+.3f}, {state})"
        )


class Genome:
    """A blueprint: node genes keyed by id, connection genes in insertion order.

    Construction validates every structural invariant unless told otherwise
    (``validate_on_init=False`` exists for tests that build broken genomes
    deliberately to probe ``validate``).
    """

    def __init__(
        self,
        nodes: Optional[Dict[int, NodeGene]] = None,
        connections: Optional[List[ConnectionGene]] = None,
        *,
        validate_on_init: bool = True,
    ) -> None:
        self.nodes: Dict[int, NodeGene] = dict(nodes or {})
        self.connections: List[ConnectionGene] = list(connections or [])
        self.fitness: float = 0.0
        self.adjusted_fitness: float = 0.0
        if validate_on_init:
            self.validate()

    @classmethod
    def minimal(
        cls,
        input_ids: Optional[Iterable[int]] = None,
        output_ids: Optional[Iterable[int]] = None,
        bias: float = 0.0,
    ) -> "Genome":
        """A fresh unwired genome with the fixed interface: inputs + outputs only.

        Inputs are always bias 0.0 (they are pass-through sensors). Outputs get
        ``bias`` (default 0.0). No connections yet — wiring is an evolution-time
        concern, not part of the representation.
        """
        in_ids = DEFAULT_INPUT_IDS if input_ids is None else tuple(input_ids)
        out_ids = DEFAULT_OUTPUT_IDS if output_ids is None else tuple(output_ids)

        nodes: Dict[int, NodeGene] = {}
        for nid in in_ids:
            nodes[nid] = NodeGene(id=nid, node_type=NodeType.INPUT, bias=0.0)
        for nid in out_ids:
            nodes[nid] = NodeGene(id=nid, node_type=NodeType.OUTPUT, bias=bias)
        return cls(nodes=nodes, connections=[])

    # ------------------------------------------------------------------ state

    @property
    def inputs(self) -> List[NodeGene]:
        return [n for n in self.nodes.values() if n.node_type is NodeType.INPUT]

    @property
    def outputs(self) -> List[NodeGene]:
        return [n for n in self.nodes.values() if n.node_type is NodeType.OUTPUT]

    @property
    def hidden(self) -> List[NodeGene]:
        return [n for n in self.nodes.values() if n.node_type is NodeType.HIDDEN]

    @property
    def enabled_connections(self) -> List[ConnectionGene]:
        """The wires that would exist in the phenotype. Disabled genes are dormant."""
        return [c for c in self.connections if c.enabled]

    def node_by_id(self, node_id: int) -> NodeGene:
        try:
            return self.nodes[node_id]
        except KeyError:
            raise KeyError(f"node {node_id} is not in this genome") from None

    # ------------------------------------------------------------------ mutate

    def add_node(self, node: NodeGene) -> None:
        """Register a node gene. Node ids must be unique."""
        if node.id in self.nodes:
            raise ValueError(f"node id {node.id} already exists in this genome")
        self.nodes[node.id] = node

    def validate_connection(
        self,
        in_node: int,
        out_node: int,
        *,
        allow_cycle: bool = False,
    ) -> None:
        """Raise ValueError if the pair ``in_node -> out_node`` is illegal here.

        This is the single authority for connection legality, shared by
        ``add_connection`` and the mutation operators, so a structural change
        can be validated *before* an innovation number is minted for it.
        """
        if in_node not in self.nodes:
            raise ValueError(f"in_node {in_node} is not in this genome")
        if out_node not in self.nodes:
            raise ValueError(f"out_node {out_node} is not in this genome")
        if in_node == out_node:
            raise ValueError("self-loops are not allowed")

        src_type = self.nodes[in_node].node_type
        dst_type = self.nodes[out_node].node_type
        if src_type is NodeType.INPUT and dst_type is NodeType.INPUT:
            raise ValueError("INPUT->INPUT connections are not allowed")
        if src_type is NodeType.OUTPUT and dst_type is NodeType.OUTPUT:
            raise ValueError("OUTPUT->OUTPUT connections are not allowed (feed-forward)")

        for existing in self.connections:
            if existing.in_node == in_node and existing.out_node == out_node:
                raise ValueError(
                    f"duplicate connection {in_node}->{out_node} "
                    "(a pair is a structural fact even when disabled)"
                )

        if not allow_cycle and self.would_create_cycle(in_node, out_node):
            raise ValueError(f"connection {in_node}->{out_node} would create a cycle")

    def add_connection(
        self,
        conn: ConnectionGene,
        *,
        allow_cycle: bool = False,
    ) -> None:
        """Append a connection gene, enforcing the structural invariants.

        ``allow_cycle`` lets a caller (or a test) record a connection that
        would otherwise be rejected; it exists so operators can deliberately
        introduce recurrent structure later.
        """
        self.validate_connection(conn.in_node, conn.out_node, allow_cycle=allow_cycle)
        self.connections.append(conn)

    # ------------------------------------------------------------------ checks

    def would_create_cycle(self, in_node: int, out_node: int) -> bool:
        """Would adding the edge ``in_node -> out_node`` close a cycle?

        Yes iff ``out_node`` can already reach ``in_node`` through the *enabled*
        edges. Disabled edges are dormant and do not count.
        """
        if in_node == out_node:
            return True

        adj: Dict[int, List[int]] = {}
        for c in self.enabled_connections:
            adj.setdefault(c.in_node, []).append(c.out_node)

        stack = [out_node]
        seen: set[int] = set()
        while stack:
            current = stack.pop()
            if current == in_node:
                return True
            if current in seen:
                continue
            seen.add(current)
            stack.extend(adj.get(current, ()))
        return False

    def has_cycle(self) -> bool:
        """True iff the *enabled* subgraph contains a cycle (Kahn's algorithm)."""
        indegree: Dict[int, int] = {nid: 0 for nid in self.nodes}
        adj: Dict[int, List[int]] = {nid: [] for nid in self.nodes}
        for c in self.enabled_connections:
            adj[c.in_node].append(c.out_node)
            indegree[c.out_node] += 1

        stack = [nid for nid, degree in indegree.items() if degree == 0]
        visited = 0
        while stack:
            current = stack.pop()
            visited += 1
            for nxt in adj[current]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    stack.append(nxt)
        return visited != len(self.nodes)

    def validate(self) -> None:
        """Raise ValueError on the first violated structural invariant.

        Node ids are unique by construction (``nodes`` is a dict), but every
        other invariant is re-checked here so a hand-assembled genome can be
        audited before use.
        """
        for nid, node in self.nodes.items():
            if node.id != nid:
                raise ValueError(f"dict key {nid} disagrees with NodeGene.id {node.id}")

        seen_pairs: set[tuple[int, int]] = set()
        for conn in self.connections:
            if conn.in_node not in self.nodes:
                raise ValueError(f"connection {conn} references unknown in_node {conn.in_node}")
            if conn.out_node not in self.nodes:
                raise ValueError(f"connection {conn} references unknown out_node {conn.out_node}")
            if conn.in_node == conn.out_node:
                raise ValueError(f"self-loop on node {conn.in_node}")

            src = self.nodes[conn.in_node].node_type
            dst = self.nodes[conn.out_node].node_type
            if src is NodeType.INPUT and dst is NodeType.INPUT:
                raise ValueError(f"INPUT->INPUT connection {conn.in_node}->{conn.out_node}")
            if src is NodeType.OUTPUT and dst is NodeType.OUTPUT:
                raise ValueError(f"OUTPUT->OUTPUT connection {conn.in_node}->{conn.out_node}")

            pair = (conn.in_node, conn.out_node)
            if pair in seen_pairs:
                raise ValueError(f"duplicate connection pair {pair}")
            seen_pairs.add(pair)

        if self.has_cycle():
            raise ValueError("enabled connections contain a cycle (feed-forward requires a DAG)")

    # ------------------------------------------------------------------ copy

    def copy(self) -> "Genome":
        """Deep copy preserving every id and innovation number. Never remints.

        This is the identity law: copying a genome must not change its history.
        """
        nodes = {nid: node.copy() for nid, node in self.nodes.items()}
        connections = [c.copy() for c in self.connections]
        clone = Genome(nodes=nodes, connections=connections)
        clone.fitness = self.fitness
        clone.adjusted_fitness = self.adjusted_fitness
        return clone

    # ------------------------------------------------------------------ dunder

    def __repr__(self) -> str:
        return (
            f"Genome(nodes={len(self.nodes)}, "
            f"connections={len(self.connections)}, "
            f"fitness={self.fitness:.3f})"
        )
