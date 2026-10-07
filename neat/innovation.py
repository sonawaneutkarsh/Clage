"""Innovation ledger (architecture.md: ``neat/innovation``).

A single, seeded, serializable database for the whole run. It issues:

- **connection innovations**: one integer per distinct ``(in_node, out_node)``
  structural invention. Repeated inventions of the same pair *reuse* the same
  number; different inventions get new numbers.
- **node ids and split innovations**: when a connection is split, the hidden
  node gets a node id and the two new connections get connection innovations.
  If the exact same split happens again later, everything is reused.

Counters are deterministic — no randomness here, so reproducibility of
mutation depends only on the injected ``random.Random``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .genome import Genome

__all__ = ["NodeInnovation", "InnovationDB"]

# Hidden node ids start after the fixed interface (inputs 0-9, outputs 10-13).
DEFAULT_NEXT_NODE_ID: int = 14
DEFAULT_NEXT_INNOVATION: int = 1


@dataclass(frozen=True)
class NodeInnovation:
    """The three numbers minted when a connection is split."""

    node_id: int
    in_innovation: int
    out_innovation: int


class InnovationDB:
    def __init__(
        self,
        *,
        next_node_id: int = DEFAULT_NEXT_NODE_ID,
        next_innovation: int = DEFAULT_NEXT_INNOVATION,
    ) -> None:
        self._connection_innovations: Dict[Tuple[int, int], int] = {}
        self._node_innovations: Dict[Tuple[int, int], NodeInnovation] = {}
        self._node_counter: int = next_node_id
        self._innovation_counter: int = next_innovation

    # ------------------------------------------------------------- connections

    def register_genomes(self, genomes: Iterable["Genome"]) -> None:
        """Validate imported history atomically before reserving identifiers."""
        pairs = dict(self._connection_innovations)
        reverse = {innovation: pair for pair, innovation in pairs.items()}
        if len(reverse) != len(pairs):
            raise ValueError("innovation history is not one-to-one")
        next_node = self._node_counter
        next_innovation = self._innovation_counter
        for genome in genomes:
            genome.validate()
            next_node = max(next_node, max(genome.nodes, default=-1) + 1)
            for edge in genome.connections:
                pair = (edge.in_node, edge.out_node)
                if (pair in pairs and pairs[pair] != edge.innovation) or (
                    edge.innovation in reverse and reverse[edge.innovation] != pair
                ):
                    raise ValueError("conflicting innovation/pair history")
                pairs[pair] = edge.innovation
                reverse[edge.innovation] = pair
                next_innovation = max(next_innovation, edge.innovation + 1)
        self._connection_innovations = pairs
        self._node_counter = next_node
        self._innovation_counter = next_innovation

    def connection_innovation(self, in_node: int, out_node: int) -> int:
        """Return the innovation number for ``in_node -> out_node``.

        Reuses the recorded number if this structural invention already exists,
        otherwise mints the next integer and records the pair.
        """
        key = (in_node, out_node)
        if key not in self._connection_innovations:
            self._connection_innovations[key] = self._innovation_counter
            self._innovation_counter += 1
        return self._connection_innovations[key]

    # ------------------------------------------------------------------ nodes

    def add_node_innovation(self, in_node: int, out_node: int) -> NodeInnovation:
        """Mint (or reuse) the node id + two connection innovations for a split.

        Keyed by the pair being split. The two new connections are also
        registered in the connection ledger so future add-connection calls
        reuse their numbers.
        """
        key = (in_node, out_node)
        if key in self._node_innovations:
            return self._node_innovations[key]

        node_id = self._node_counter
        self._node_counter += 1

        in_innovation = self.connection_innovation(in_node, node_id)
        out_innovation = self.connection_innovation(node_id, out_node)

        innovation = NodeInnovation(
            node_id=node_id,
            in_innovation=in_innovation,
            out_innovation=out_innovation,
        )
        self._node_innovations[key] = innovation
        return innovation

    def recorded_node_innovation(
        self, in_node: int, out_node: int
    ) -> Optional[NodeInnovation]:
        """Return the recorded split for a pair, without minting anything new."""
        return self._node_innovations.get((in_node, out_node))

    def new_node_id(self) -> int:
        """Mint a fresh hidden node id without creating any innovations."""
        node_id = self._node_counter
        self._node_counter += 1
        return node_id

    # ----------------------------------------------------------- serialization

    def to_dict(self) -> Dict[str, object]:
        return {
            "connection_innovations": {
                f"{a}->{b}": innov for (a, b), innov in self._connection_innovations.items()
            },
            "node_innovations": {
                f"{a}->{b}": [ni.node_id, ni.in_innovation, ni.out_innovation]
                for (a, b), ni in self._node_innovations.items()
            },
            "node_counter": self._node_counter,
            "innovation_counter": self._innovation_counter,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InnovationDB":
        db = cls()
        for raw_key, innov in data["connection_innovations"].items():
            a, b = raw_key.split("->")
            db._connection_innovations[(int(a), int(b))] = innov
        for raw_key, (node_id, in_innov, out_innov) in data["node_innovations"].items():
            a, b = raw_key.split("->")
            db._node_innovations[(int(a), int(b))] = NodeInnovation(
                node_id=node_id, in_innovation=in_innov, out_innovation=out_innov
            )
        db._node_counter = data["node_counter"]
        db._innovation_counter = data["innovation_counter"]
        values = [db._node_counter, db._innovation_counter,
                  *db._connection_innovations.values()]
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
               for value in values):
            raise ValueError("invalid innovation history counters/identifiers")
        reverse = {innovation: pair for pair, innovation in db._connection_innovations.items()}
        if len(reverse) != len(db._connection_innovations):
            raise ValueError("conflicting innovation history")
        if db._innovation_counter <= max(reverse, default=0):
            raise ValueError("innovation counter overlaps recorded history")
        for (source, target), split in db._node_innovations.items():
            if (db._connection_innovations.get((source, split.node_id)) != split.in_innovation
                    or db._connection_innovations.get((split.node_id, target)) != split.out_innovation
                    or db._node_counter <= split.node_id):
                raise ValueError("invalid split history")
        return db

    # ------------------------------------------------------------------ dunder

    def __repr__(self) -> str:
        return (
            f"InnovationDB(connection_innovations={len(self._connection_innovations)}, "
            f"node_innovations={len(self._node_innovations)}, "
            f"next_node_id={self._node_counter}, next_innovation={self._innovation_counter})"
        )
