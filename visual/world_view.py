"""Animated 2D replay of a recorded generation.

Consumes a `clage-generation-replay` recording (plain data) and plays it back
tick by tick with a slider, play/pause/step controls, and click-to-inspect of
an organism's network. The viewer never simulates — it renders recorded state.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider

from .network_view import genome_figure

__all__ = ["ACTION_LABELS", "FACING_VECTORS", "draw_frame", "export_tick", "WorldViewer"]

ACTION_LABELS = {0: "MOVE", 1: "TURN_LEFT", 2: "TURN_RIGHT", 3: "EAT"}
FACING_VECTORS = {"N": (0, 1), "E": (1, 0), "S": (0, -1), "W": (-1, 0)}

_GENOME_COLORS = plt.get_cmap("tab20")


def _alive(tick: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [organism for organism in tick["organisms"] if organism["alive"]]


def draw_frame(
    ax,
    recording: Dict[str, Any],
    tick_index: int,
    selected: Optional[int] = None,
) -> Dict[str, Any]:
    """Redraw the axes for one recorded tick. Returns a small stats dict."""
    config = recording["config"]
    tick = recording["ticks"][tick_index]
    width, height = config["width"], config["height"]

    ax.clear()
    ax.set_xlim(-0.7, width - 0.3)
    ax.set_ylim(-0.7, height - 0.3)
    ax.set_xticks(range(width))
    ax.set_yticks(range(height))
    ax.grid(True, color="0.85", linewidth=0.5)
    ax.set_aspect("equal")

    if tick["food"]:
        fxs = [x for x, _ in tick["food"]]
        fys = [y for _, y in tick["food"]]
        ax.scatter(fxs, fys, marker="s", s=34, color="#2ca02c", zorder=1)

    organisms = _alive(tick)
    if organisms:
        xs = [o["x"] for o in organisms]
        ys = [o["y"] for o in organisms]
        colors = [_GENOME_COLORS(o["genome"] % 20) for o in organisms]
        ax.scatter(xs, ys, s=110, c=colors, edgecolors="k", linewidths=0.4, zorder=3)

        segments: List[Tuple[float, float]] = []
        for organism in organisms:
            dx, dy = FACING_VECTORS[organism["facing"]]
            segments.append((organism["x"], organism["y"]))
            segments.append((organism["x"] + 0.4 * dx, organism["y"] + 0.4 * dy))
            segments.append((None, None))
        ax.plot(
            [s[0] for s in segments],
            [s[1] for s in segments],
            color="black", linewidth=1.0, zorder=4,
        )

    if selected is not None:
        for organism in organisms:
            if organism["id"] == selected:
                ax.scatter([organism["x"]], [organism["y"]], s=230,
                           facecolors="none", edgecolors="crimson", linewidths=2, zorder=5)
                break

    live = len(organisms)
    total = len(tick["organisms"])
    generation = recording.get("generation")
    gen_text = "—" if generation is None else str(generation)
    ax.set_title(
        f"generation {gen_text}   tick {tick_index}/{len(recording['ticks']) - 1}   "
        f"live {live}/{total}   pop {len(recording['genomes'])}",
        fontsize=11,
    )
    return {"live": live, "total": total}


def export_tick(recording: Dict[str, Any], tick_index: int, path) -> None:
    """Save a single frame of the recording to ``path`` (headless)."""
    from pathlib import Path

    fig, ax = plt.subplots(figsize=(9, 7))
    draw_frame(ax, recording, tick_index)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


class WorldViewer:
    """Interactive replay: slider + play/pause/step + click-to-inspect."""

    def __init__(self, recording: Dict[str, Any]) -> None:
        self.recording = recording
        self.ticks = recording["ticks"]
        self.genomes = {g["id"]: g for g in recording["genomes"]}
        self.tick_index = 0
        self.playing = True
        self.selected: Optional[int] = None

        self.fig = plt.figure(figsize=(9.5, 8.0))
        self.fig.subplots_adjust(bottom=0.14)
        self.ax = self.fig.add_subplot(1, 1, 1)

        self._build_controls()
        self._anim = FuncAnimation(
            self.fig, self._animate, interval=120, blit=False, cache_frame_data=False
        )
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)

    def _build_controls(self) -> None:
        slider_ax = self.fig.add_axes([0.18, 0.05, 0.55, 0.03])
        self.slider = Slider(
            slider_ax, "tick", 0, len(self.ticks) - 1, valinit=0, valstep=1
        )
        self.slider.on_changed(self._on_slider)

        play_ax = self.fig.add_axes([0.78, 0.045, 0.07, 0.04])
        self.play_button = Button(play_ax, "play")
        self.play_button.on_clicked(self._on_play)

        step_ax = self.fig.add_axes([0.86, 0.045, 0.07, 0.04])
        self.step_button = Button(step_ax, "step")
        self.step_button.on_clicked(self._on_step)

    # ------------------------------------------------------------- callbacks

    def _animate(self, frame: int) -> List[Any]:
        if self.playing and self.tick_index < len(self.ticks) - 1:
            self.tick_index += 1
            self.slider.set_val(self.tick_index)
        draw_frame(self.ax, self.recording, self.tick_index, self.selected)
        return []

    def _on_slider(self, value: float) -> None:
        self.tick_index = int(value)
        draw_frame(self.ax, self.recording, self.tick_index, self.selected)
        self.fig.canvas.draw_idle()

    def _on_play(self, event) -> None:
        self.playing = not self.playing
        self.play_button.label.set_text("pause" if self.playing else "play")

    def _on_step(self, event) -> None:
        self.playing = False
        self.play_button.label.set_text("play")
        if self.tick_index < len(self.ticks) - 1:
            self.tick_index += 1
            self.slider.set_val(self.tick_index)

    def _on_click(self, event) -> None:
        if event.inaxes is not self.ax:
            return
        best_id: Optional[int] = None
        best_distance = 0.6
        for organism in self.ticks[self.tick_index]["organisms"]:
            if not organism["alive"]:
                continue
            distance = (organism["x"] - event.xdata) ** 2 + (organism["y"] - event.ydata) ** 2
            if distance < best_distance:
                best_distance = distance
                best_id = organism["id"]
        self.selected = best_id
        if best_id is not None:
            self._open_network(best_id)
        draw_frame(self.ax, self.recording, self.tick_index, self.selected)
        self.fig.canvas.draw_idle()

    def _open_network(self, organism_id: int) -> None:
        for organism in self.ticks[self.tick_index]["organisms"]:
            if organism["id"] == organism_id:
                genome = self.genomes[organism["genome"]]
                fig = genome_figure(
                    genome,
                    title=f"organism {organism_id} (genome {organism['genome']})",
                )
                fig.canvas.manager.show()
                return

    def show(self) -> None:
        plt.show()
