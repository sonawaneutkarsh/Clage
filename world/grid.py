"""The discrete 2D world: cells, food, occupancy, boundaries.

A walled grid where each cell holds at most one thing — food or an organism.
Food regenerates toward a target count. All randomness flows through an injected
``random.Random`` so trajectories are reproducible.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Iterator, List, Optional, Tuple

from .config import EnvironmentConfig

if TYPE_CHECKING:  # organism.py imports this module
    from .organism import Organism

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
        self._empty_rows = [self.width] * self.height
        self._empty_count = self.width * self.height
        self._food_buckets: dict[Tuple[int, int], set[Tuple[int, int]]] = {}
        self._bucket_size = 8

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
        self._empty_rows[y] -= 1
        self._empty_count -= 1
        self._food_buckets.setdefault((x // self._bucket_size, y // self._bucket_size), set()).add((x, y))
        return True

    def remove_food(self, x: int, y: int) -> bool:
        if (x, y) in self.food:
            self.food.discard((x, y))
            self.cells[y][x] = None
            self._empty_rows[y] += 1
            self._empty_count += 1
            bucket = (x // self._bucket_size, y // self._bucket_size)
            self._food_buckets[bucket].remove((x, y))
            if not self._food_buckets[bucket]:
                del self._food_buckets[bucket]
            return True
        return False

    def place_organism(self, organism: Organism) -> bool:
        if not self.is_empty(organism.x, organism.y):
            return False
        self.cells[organism.y][organism.x] = organism
        self._empty_rows[organism.y] -= 1
        self._empty_count -= 1
        return True

    def move_organism(self, organism: Organism, new_x: int, new_y: int) -> bool:
        if not self.is_empty(new_x, new_y):
            return False
        self.cells[organism.y][organism.x] = None
        self._empty_rows[organism.y] += 1
        self._empty_rows[new_y] -= 1
        organism.x, organism.y = new_x, new_y
        self.cells[new_y][new_x] = organism
        return True

    def remove_organism(self, organism: Organism) -> None:
        if self.cells[organism.y][organism.x] is organism:
            self.cells[organism.y][organism.x] = None
            self._empty_rows[organism.y] += 1
            self._empty_count += 1

    def random_empty_cell(self) -> Optional[Tuple[int, int]]:
        if not self._empty_count:
            return None
        rank = self.rng.randrange(self._empty_count)
        for y, count in enumerate(self._empty_rows):
            if rank >= count:
                rank -= count
                continue
            for x, occupant in enumerate(self.cells[y]):
                if occupant is None:
                    if rank == 0:
                        return x, y
                    rank -= 1
        raise RuntimeError("World occupancy index is inconsistent; use placement/movement methods")

    # ------------------------------------------------------------- food

    def nearest_food(self, x: int, y: int) -> Optional[Tuple[int, int]]:
        if not self.food:
            return None
        if not self.in_bounds(x, y):
            return min(self.food, key=lambda cell: (cell[0] - x) ** 2 + (cell[1] - y) ** 2)
        size = self._bucket_size
        center_x, center_y = x // size, y // size
        best_dist = float("inf")
        nearest = set()
        for radius in range(max(self.width, self.height) // size + 2):
            candidates: list[Tuple[int, int]] = []
            for bucket_y in range(center_y - radius, center_y + radius + 1):
                if radius == 0 or bucket_y in (center_y - radius, center_y + radius):
                    candidates.extend((bucket_x, bucket_y) for bucket_x in range(center_x - radius, center_x + radius + 1))
                else:
                    candidates.extend([(center_x - radius, bucket_y), (center_x + radius, bucket_y)])
            for bucket in candidates:
                for fx, fy in self._food_buckets.get(bucket, ()):
                    distance = (fx - x) ** 2 + (fy - y) ** 2
                    if distance < best_dist:
                        best_dist = distance
                        nearest = {(fx, fy)}
                    elif distance == best_dist:
                        nearest.add((fx, fy))
            left, right = (center_x - radius) * size, (center_x + radius + 1) * size
            top, bottom = (center_y - radius) * size, (center_y + radius + 1) * size
            outside = []
            if left > 0:
                outside.append((x - left + 1) ** 2)
            if right < self.width:
                outside.append((right - x) ** 2)
            if top > 0:
                outside.append((y - top + 1) ** 2)
            if bottom < self.height:
                outside.append((bottom - y) ** 2)
            if not outside or best_dist < min(outside):
                break
        if len(nearest) == 1:
            return next(iter(nearest))
        return next((cell for cell in self.food if cell in nearest), None)

    def food_density(self, x: int, y: int, radius: int) -> float:
        count = 0
        for ny in range(max(0, y - radius), min(self.height, y + radius + 1)):
            row = self.cells[ny]
            for nx in range(max(0, x - radius), min(self.width, x + radius + 1)):
                if (nx, ny) == (x, y):
                    continue
                if row[nx] is FOOD:
                    count += 1
        max_count = (2 * radius + 1) ** 2 - 1
        return count / max_count

    def organism_density(self, x: int, y: int, radius: int, exclude: object) -> float:
        count = 0
        for ny in range(max(0, y - radius), min(self.height, y + radius + 1)):
            row = self.cells[ny]
            for nx in range(max(0, x - radius), min(self.width, x + radius + 1)):
                if (nx, ny) == (x, y):
                    continue
                occupant = row[nx]
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
