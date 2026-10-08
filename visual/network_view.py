"""Render a genome's neural-network topology from a recording's plain data.

Pure rendering: computes a deterministic layered layout and draws nodes and
edges. No evolutionary logic; it reads only the recorded genome dict.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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
    positions = {identity: (x * 3, y * 1.5)
                 for identity, (x, y) in layered_layout(genome).items()}

    for conn in genome["connections"]:
        (x0, y0) = positions[conn["in"]]
        (x1, y1) = positions[conn["out"]]
        if conn["enabled"]:
            color = "#d62728" if conn["weight"] >= 0 else "#1f77b4"
            alpha = min(1.0, abs(conn["weight"]) / 3.0 + 0.15)
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops={"arrowstyle": "->", "color": color, "alpha": alpha,
                                    "shrinkA": 13, "shrinkB": 13})
        else:
            ax.plot([x0, x1], [y0, y1], color="0.6", linestyle="--", linewidth=0.8)

    for node_id, node in nodes.items():
        x, y = positions[node_id]
        ax.scatter([x], [y], s=500, color=_node_color(node["type"]), zorder=3)
        ax.text(x, y, f"{node_id}", ha="center", va="center", fontsize=8, color="white", zorder=4)
        ax.text(x, y - 0.6, f"b={node['bias']:.2f}", ha="center", va="center", fontsize=7, zorder=4)

    ax.set_xlim(-0.6, max(p[0] for p in positions.values()) + 0.6)
    ax.set_ylim(-(max(p[1] for p in positions.values()) + 1.2),
                max(p[1] for p in positions.values()) + 1.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(handles=[Line2D([], [], color="#d62728", label="positive weight"),
                       Line2D([], [], color="#1f77b4", label="negative weight"),
                       Line2D([], [], color="0.6", linestyle="--", label="disabled")],
              loc="upper center", fontsize=7, ncol=3)
    if not any(edge["enabled"] for edge in genome["connections"]):
        ax.set_title("Unwired: outputs depend only on node biases", fontsize=9)


def genome_figure(genome: Dict[str, Any], title: Optional[str] = None) -> plt.Figure:
    """Return a standalone figure of the genome's network."""
    positions = layered_layout(genome)
    layer_counts: Dict[float, int] = {}
    for x, _ in positions.values():
        layer_counts[x] = layer_counts.get(x, 0) + 1
    height = max(4, max(layer_counts.values(), default=1) * 0.55 + 1)
    fig, ax = plt.subplots(figsize=(7, height), constrained_layout=True)
    draw_genome(ax, genome)
    if title:
        fig.suptitle(title, fontsize=10)
    return fig
