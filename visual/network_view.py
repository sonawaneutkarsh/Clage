"""Render a genome's neural-network topology from a recording's plain data.

Pure rendering: computes a deterministic layered layout and draws nodes and
edges. No evolutionary logic; it reads only the recorded genome dict.
"""

from __future__ import annotations

from typing import Any, Dict

import matplotlib.pyplot as plt

from .layout import layered_layout

__all__ = ["layered_layout", "draw_genome", "genome_figure"]


def _node_color(node_type: str) -> str:
    return {
        "INPUT": "#4c72b0",
        "OUTPUT": "#c44e52",
        "HIDDEN": "#55a868",
    }[node_type]


def draw_genome(ax, genome: Dict[str, Any]) -> None:
    """Draw one genome onto ``ax``. Returns nothing (mutates the axes)."""
    nodes = {node["id"]: node for node in genome["nodes"]}
    positions = layered_layout(genome)

    for conn in genome["connections"]:
        (x0, y0) = positions[conn["in"]]
        (x1, y1) = positions[conn["out"]]
        if conn["enabled"]:
            color = "#d62728" if conn["weight"] >= 0 else "#1f77b4"
            alpha = min(1.0, abs(conn["weight"]) / 3.0 + 0.15)
            ax.plot([x0, x1], [y0, y1], color=color, alpha=alpha, linewidth=1.2)
        else:
            ax.plot([x0, x1], [y0, y1], color="0.6", linestyle="--", linewidth=0.8)

    for node_id, node in nodes.items():
        x, y = positions[node_id]
        ax.scatter([x], [y], s=900, color=_node_color(node["type"]), zorder=3)
        label = f"{node_id}\n{node['type'][0]} b={node['bias']:.2f}"
        ax.text(x, y, f"{node_id}", ha="center", va="center", fontsize=8, color="white", zorder=4)
        ax.text(x, y - 0.14, f"b={node['bias']:.2f}", ha="center", va="center", fontsize=6, zorder=4)

    ax.set_xlim(-0.6, max(p[0] for p in positions.values()) + 0.6)
    ax.set_ylim(-(max(p[1] for p in positions.values()) + 1.2),
                max(p[1] for p in positions.values()) + 1.2)
    ax.set_aspect("equal")
    ax.axis("off")


def genome_figure(genome: Dict[str, Any], title: str = None) -> plt.Figure:
    """Return a standalone figure of the genome's network."""
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    draw_genome(ax, genome)
    if title:
        ax.set_title(title, fontsize=10)
    return fig
