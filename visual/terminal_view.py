"""Terminal (TUI) replay of a recorded generation.

Renders a recorded ``clage-generation-replay`` file as a boxed, live-updating
terminal view: the 2D world on the left, selected-organism info + an ASCII
neural network on the right, and generation/population/species/survival/
average-fitness in the footer. Pure data — it never simulates.

Controls (interactive ``play``): arrows/WASD move a cursor, Enter selects the
organism under it, Space toggles play/pause, q quits.

Commands:
    python -m visual tui --recording <file>
    python -m visual tui --recording <file> --export-tick N --export-out frame.txt
"""

from __future__ import annotations

import math
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from .layout import layered_layout

__all__ = [
    "FOOD_GLYPH",
    "EMPTY_GLYPH",
    "ORGANISM_GLYPH",
    "render_frame",
    "export_frame",
    "play",
    "world_grid_lines",
    "network_ascii_lines",
]

FOOD_GLYPH = "🟢"
EMPTY_GLYPH = "·"
ORGANISM_GLYPH = "●"

WORLD_MAX_DIM = 21
NET_WIDTH = 22
NET_HEIGHT = 8
_PALETTE = [196, 208, 226, 46, 51, 21, 93, 201]  # vivid ANSI 256 colors

_ACTION_LABELS = {0: "MOVE", 1: "TURN_LEFT", 2: "TURN_RIGHT", 3: "EAT"}
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _visible(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _char_width(ch: str) -> int:
    # Emoji/CJK glyphs are double-width in terminals; box-drawing and Latin are 1.
    return 2 if ord(ch) > 0x2E7F else 1


def _visible_width(text: str) -> int:
    return sum(_char_width(ch) for ch in _visible(text))


def _pad(text: str, width: int) -> str:
    return text + " " * max(0, width - _visible_width(text))


def _color(text: str, code: int) -> str:
    return f"\x1b[38;5;{code}m{text}\x1b[0m"


# --------------------------------------------------------------- world panel


def world_grid_lines(
    recording: Dict[str, Any],
    tick_index: int,
    color: bool = False,
) -> List[str]:
    """Downscale the world to a coarse character grid and render it."""
    config = recording["config"]
    tick = recording["ticks"][tick_index]
    width, height = config["width"], config["height"]
    scale = max(1, math.ceil(max(width, height) / WORLD_MAX_DIM))
    gw, gh = math.ceil(width / scale), math.ceil(height / scale)

    cells = [[None for _ in range(gw)] for _ in range(gh)]  # None -> empty
    for organism in tick["organisms"]:
        if not organism["alive"]:
            continue
        cx = min(organism["x"] // scale, gw - 1)
        cy = min(organism["y"] // scale, gh - 1)
        cells[cy][cx] = ("organism", organism["genome"])
    for fx, fy in tick["food"]:
        cx, cy = min(fx // scale, gw - 1), min(fy // scale, gh - 1)
        if cells[cy][cx] is None:
            cells[cy][cx] = ("food", None)

    lines: List[str] = []
    for row in cells:
        parts: List[str] = []
        for cell in row:
            if cell is None:
                parts.append(EMPTY_GLYPH)
            elif cell[0] == "food":
                parts.append(FOOD_GLYPH)
            else:
                code = _PALETTE[cell[1] % len(_PALETTE)]
                parts.append(_color(ORGANISM_GLYPH, code) if color else ORGANISM_GLYPH)
        lines.append("".join(parts))
    return lines


# --------------------------------------------------------------- organism info


def _organism_info_lines(
    recording: Dict[str, Any],
    tick_index: int,
    selected: Optional[int],
    color: bool = False,
) -> List[str]:
    if selected is None:
        return [
            "select an organism:",
            "  arrows/WASD move cursor,",
            "  Enter selects",
            "",
            "Space  play / pause",
            "q      quit",
        ]
    tick = recording["ticks"][tick_index]
    organism = next(
        (o for o in tick["organisms"] if o["id"] == selected), None
    )
    if organism is None:
        return ["Selected organism is gone"]
    energy_pct = organism["energy"] / recording["config"]["max_energy"] * 100
    return [
        f"Energy:   {energy_pct:5.0f}",
        f"Age:      {organism['age']}",
        f"Food:     {organism['food_eaten']}",
        f"Offspring:{organism['offspring']}",
        f"Facing:   {organism['facing']}",
        f"Action:   {_ACTION_LABELS.get(organism['action'], '?')}",
        f"Genome:   {organism['genome']}",
    ]


# --------------------------------------------------------------- network panel


def network_ascii_lines(genome: Dict[str, Any], width: int = NET_WIDTH,
                        height: int = NET_HEIGHT) -> List[str]:
    """A small ASCII rendering of a genome's network (nodes + weighted edges)."""
    positions = layered_layout(genome)
    if not positions:
        return []
    xs = [pos[0] for pos in positions.values()]
    ys = [pos[1] for pos in positions.values()]
    max_x = max(xs)
    min_y, max_y = min(ys), max(ys)

    def to_xy(node_id: int) -> Tuple[int, int]:
        layer_x, y = positions[node_id]
        col = round(layer_x / max_x * (width - 2)) + 1 if max_x > 0 else 1
        row = (
            height // 2
            if max_y == min_y
            else round((y - min_y) / (max_y - min_y) * (height - 2)) + 1
        )
        return col, row

    grid: List[List[str]] = [[" " for _ in range(width)] for _ in range(height)]

    def draw_edge(x0: int, y0: int, x1: int, y1: int) -> None:
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        steps = max(dx, dy)
        if steps == 0:
            return
        sx = 1 if x1 >= x0 else -1
        sy = 1 if y1 >= y0 else -1
        for i in range(1, steps):
            x = x0 + sx * round(dx * i / steps)
            y = y0 + sy * round(dy * i / steps)
            if 0 <= y < height and 0 <= x < width and grid[y][x] == " ":
                px = x0 + sx * round(dx * (i - 1) / steps)
                py = y0 + sy * round(dy * (i - 1) / steps)
                nx, ny = x - px, y - py
                if nx == 0:
                    char = "│"
                elif ny == 0:
                    char = "─"
                elif (nx > 0) == (ny > 0):
                    char = "╲"
                else:
                    char = "╱"
                grid[y][x] = char

    for conn in genome["connections"]:
        if not conn["enabled"]:
            continue
        x0, y0 = to_xy(conn["in"])
        x1, y1 = to_xy(conn["out"])
        draw_edge(x0, y0, x1, y1)

    for node in genome["nodes"]:
        x, y = to_xy(node["id"])
        grid[y][x] = ORGANISM_GLYPH

    return ["".join(row).rstrip() for row in grid]


def _right_panel_lines(
    recording: Dict[str, Any],
    tick_index: int,
    selected: Optional[int],
    color: bool = False,
) -> List[str]:
    info = _organism_info_lines(recording, tick_index, selected, color)
    lines = [" Selected Organism", " " + "─" * 16]
    lines.extend(" " + line for line in info)
    lines.extend(["", " Neural Network", " " + "─" * 16])
    if selected is not None:
        tick = recording["ticks"][tick_index]
        organism = next((o for o in tick["organisms"] if o["id"] == selected), None)
        if organism is not None:
            genome = recording["genomes"][organism["genome"]]
            lines.extend(" " + line for line in network_ascii_lines(genome))
    else:
        lines.append("   (select an organism)")
    return lines


# --------------------------------------------------------------- frame assembly


def render_frame(
    recording: Dict[str, Any],
    tick_index: int,
    selected: Optional[int] = None,
    color: bool = False,
) -> str:
    """Render the boxed layout for one tick as a string."""
    config = recording["config"]
    tick = recording["ticks"][tick_index]

    left = world_grid_lines(recording, tick_index, color)
    right = _right_panel_lines(recording, tick_index, selected, color)

    left_width = max(_visible_width(line) for line in left)

    header = f" CLAGE — GENERATION {recording.get('generation')}" if recording.get("generation") is not None else " CLAGE"
    generation = recording.get("generation")
    population = len(recording["genomes"])
    context = recording.get("context", {})

    if context.get("species_count") is not None:
        species = context["species_count"]
    else:
        species = "—"
    # Survival is shown live for the current tick (alive/total now).
    alive = sum(1 for o in tick["organisms"] if o["alive"])
    survival = f"{alive / max(1, len(tick['organisms'])) * 100:.0f}%"
    if context.get("mean_fitness") is not None:
        avg_fitness = f"{context['mean_fitness']:.2f}"
    else:
        avg_fitness = "—"

    footer1 = f" Generation: {generation}   Population: {population}   Species: {species}"
    footer2 = f" Survival: {survival}   Avg Fitness: {avg_fitness}   Tick: {tick_index}/{len(recording['ticks']) - 1}"
    inner = max(len(_visible(header)) + 1, left_width + 1 + 34, len(footer1), len(footer2))
    right_width = inner - left_width - 1

    def right_cell(index: int) -> str:
        line = right[index] if index < len(right) else ""
        return _pad(line[:right_width], right_width)
    lines: List[str] = []
    lines.append("┌" + "─" * inner + "┐")
    lines.append("│" + _pad(header, inner) + "│")
    lines.append("├" + "─" * left_width + "┬" + "─" * right_width + "┤")
    for i in range(max(len(left), len(right))):
        left_cell = _pad(left[i] if i < len(left) else "", left_width)
        lines.append("│" + left_cell + "│" + right_cell(i) + "│")
    lines.append("├" + "─" * left_width + "┴" + "─" * right_width + "┤")
    lines.append("│" + _pad(footer1, inner) + "│")
    lines.append("│" + _pad(footer2, inner) + "│")
    lines.append("└" + "─" * inner + "┘")
    return "\n".join(lines)


def export_frame(recording: Dict[str, Any], tick_index: int, path) -> None:
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(render_frame(recording, tick_index, color=False) + "\n")


# --------------------------------------------------------------- interactive


def play(recording: Dict[str, Any], fps: float = 10.0) -> None:
    """Live terminal replay with a movable cursor (requires a TTY)."""
    import select
    import termios
    import tty

    ticks = recording["ticks"]
    tick = 0
    playing = True
    selected: Optional[int] = None
    cursor = (0, 0)

    def find_organism(cx: int, cy: int) -> Optional[int]:
        config = recording["config"]
        scale = max(1, math.ceil(max(config["width"], config["height"]) / WORLD_MAX_DIM))
        for organism in ticks[tick]["organisms"]:
            if not organism["alive"]:
                continue
            if min(organism["x"] // scale, WORLD_MAX_DIM - 1) == cx and \
               min(organism["y"] // scale, WORLD_MAX_DIM - 1) == cy:
                return organism["id"]
        return None

    old = termios.tcgetattr(sys.stdin.fileno())
    tty.setcbreak(sys.stdin.fileno())
    try:
        while True:
            while select.select([sys.stdin], [], [], 0)[0]:
                ch = sys.stdin.read(1)
                if ch == "\x1b":
                    rest = sys.stdin.read(2)
                    key = ch + rest
                    if key == "\x1b[A":
                        cursor = (cursor[0], max(0, cursor[1] - 1))
                    elif key == "\x1b[B":
                        cursor = (cursor[0], cursor[1] + 1)
                    elif key == "\x1b[C":
                        cursor = (min(WORLD_MAX_DIM - 1, cursor[0] + 1), cursor[1])
                    elif key == "\x1b[D":
                        cursor = (max(0, cursor[0] - 1), cursor[1])
                elif ch in "wasd":
                    dx = {"d": 1, "a": -1}.get(ch, 0)
                    dy = {"s": 1, "w": -1}.get(ch, 0)
                    cursor = (
                        min(WORLD_MAX_DIM - 1, max(0, cursor[0] + dx)),
                        min(WORLD_MAX_DIM - 1, max(0, cursor[1] + dy)),
                    )
                elif ch in "\r\n":
                    selected = find_organism(cursor[0], cursor[1])
                elif ch == " ":
                    playing = not playing
                elif ch in "qQ":
                    return

            if playing:
                tick = (tick + 1) % len(ticks)

            frame = render_frame(recording, tick, selected, color=True)
            sys.stdout.write("\x1b[H\x1b[J" + frame + "\n")
            sys.stdout.flush()
            time.sleep(1.0 / fps)
    finally:
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old)
