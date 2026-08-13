"""Deterministic layered layout for neural-network rendering.

Pure math (no matplotlib) so both the matplotlib network view and the terminal
viewer can share it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

__all__ = ["layered_layout"]


def layered_layout(genome: Dict[str, Any]) -> Dict[int, Tuple[float, float]]:
    """Deterministic layout: inputs left, outputs right, hiddens layered by depth."""
    nodes = {node["id"]: node for node in genome["nodes"]}
    enabled = [conn for conn in genome["connections"] if conn["enabled"]]
    predecessors: Dict[int, List[int]] = {}
    for conn in enabled:
        predecessors.setdefault(conn["out"], []).append(conn["in"])

    layer: Dict[int, int] = {}
    for node_id, node in nodes.items():
        layer[node_id] = 0 if node["type"] == "INPUT" else 1

    for _ in range(len(nodes) + 1):
        changed = False
        for node_id, node in nodes.items():
            if node["type"] == "INPUT":
                continue
            for pred in predecessors.get(node_id, []):
                candidate = layer[pred] + 1
                if candidate > layer[node_id]:
                    layer[node_id] = candidate
                    changed = True
        if not changed:
            break

    max_hidden = max(
        (layer[nid] for nid in nodes if nodes[nid]["type"] == "HIDDEN"), default=0
    )
    for node_id, node in nodes.items():
        if node["type"] == "OUTPUT":
            layer[node_id] = max_hidden + 1

    by_layer: Dict[int, List[int]] = {}
    for node_id, node_layer in layer.items():
        by_layer.setdefault(node_layer, []).append(node_id)

    positions: Dict[int, Tuple[float, float]] = {}
    for node_layer, ids in by_layer.items():
        ids = sorted(ids)
        count = len(ids)
        for index, node_id in enumerate(ids):
            y = (count - 1) / 2 - index
            positions[node_id] = (float(node_layer), y)
    return positions
