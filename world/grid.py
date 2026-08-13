"""The discrete 2D world: cells, food, occupancy, boundaries.

A walled grid where each cell holds at most one thing — food or an organism.
Food regenerates toward a target count. All randomness flows through an injected
``random.Random`` so trajectories are reproducible.
"""

from __future__ import annotations

import random
from typing import Iterator, List, Optional, Tuple

from .config import EnvironmentConfig

__all__ = ["World"]

FOOD = "F"


class World:
    def __init__(self, config: EnvironmentConfig, rng: random.Random) -> None:
        self.config = config
        self.rng = rng
        self.width = config.width
        self.height = config.height
        self.cells: List[List[Optional[object]]] = [
            [None for _ in range(self.width)] for _ in range(self.height)
        ]
        self.food: set[Tuple[int, int]] = set()

    # ------------------------------------------------------------- basic queries

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def occupant(self, x: int, y: int) -> Optional[object]:
        if not self.in_bounds(x, y):
            return None
        return self.cells[y][x]

    def is_empty(self, x: int, y: int) -> bool:
        return self.in_bounds(x, y) and self.cells[y][x] is None

    # ------------------------------------------------------------- placement

    def place_food(self, x: int, y: int) -> bool:
        if not self.is_empty(x, y):
            return False
        self.cells[y][x] = FOOD
        self.food.add((x, y))
        return True

    def remove_food(self, x: int, y: int) -> bool:
        if (x, y) in self.food:
            self.food.discard((x, y))
            self.cells[y][x] = None
            return True
        return False

    def place_organism(self, organism: object) -> bool:
        if not self.is_empty(organism.x, organism.y):
            return False
        self.cells[organism.y][organism.x] = organism
        return True

    def move_organism(self, organism: object, new_x: int, new_y: int) -> bool:
        if not self.is_empty(new_x, new_y):
            return False
        self.cells[organism.y][organism.x] = None
        organism.x, organism.y = new_x, new_y
        self.cells[new_y][new_x] = organism
        return True

    def remove_organism(self, organism: object) -> None:
        if self.cells[organism.y][organism.x] is organism:
            self.cells[organism.y][organism.x] = None

    def random_empty_cell(self) -> Optional[Tuple[int, int]]:
        empty = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if self.cells[y][x] is None
        ]
        return self.rng.choice(empty) if empty else None

    # ------------------------------------------------------------- food

    def nearest_food(self, x: int, y: int) -> Optional[Tuple[int, int]]:
        best: Optional[Tuple[int, int]] = None
        best_dist = float("inf")
        for fx, fy in self.food:
            dist = (fx - x) ** 2 + (fy - y) ** 2
            if dist < best_dist:
                best_dist = dist
                best = (fx, fy)
        return best

    def food_density(self, x: int, y: int, radius: int) -> float:
        count = 0
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if (dx, dy) == (0, 0):
                    continue
                nx, ny = x + dx, y + dy
                if (nx, ny) in self.food:
                    count += 1
        max_count = (2 * radius + 1) ** 2 - 1
        return count / max_count

    def organism_density(self, x: int, y: int, radius: int, exclude: object) -> float:
        count = 0
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if (dx, dy) == (0, 0):
                    continue
                nx, ny = x + dx, y + dy
                occupant = self.occupant(nx, ny)
                if occupant is not None and occupant is not FOOD and occupant is not exclude:
                    count += 1
        max_count = (2 * radius + 1) ** 2 - 1
        return count / max_count

    def adjacent_empty(self, x: int, y: int) -> List[Tuple[int, int]]:
        candidates = [(x + dx, y + dy) for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0))]
        return [c for c in candidates if self.is_empty(*c)]

    # ------------------------------------------------------------- regeneration

    def regenerate_food(self) -> None:
        spawned = 0
        while (
            len(self.food) < self.config.food_target
            and spawned < self.config.food_regrowth_per_tick
        ):
            cell = self.random_empty_cell()
            if cell is None:
                return
            self.place_food(*cell)
            spawned += 1

    # ------------------------------------------------------------- convenience

    def iter_cells(self) -> Iterator[Tuple[int, int]]:
        for y in range(self.height):
            for x in range(self.width):
                yield x, y
