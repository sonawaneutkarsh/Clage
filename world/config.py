"""Configuration for the Clage artificial-life world.

All numbers are knobs, not truths — the architecture only requires that *some*
scalar fitness is written back onto each genome.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import List, Tuple

__all__ = ["EnvironmentConfig", "OBSERVATION_SIZE", "ACTION_SIZE", "Direction", "Action"]

# The neural interface: 9 observations -> 4 actions.
OBSERVATION_SIZE = 9
ACTION_SIZE = 4


class Direction:
    """Cardinal facing directions as (dx, dy)."""

    NORTH = (0, -1)
    EAST = (1, 0)
    SOUTH = (0, 1)
    WEST = (-1, 0)

    ORDER: Tuple[Tuple[int, int], ...] = (NORTH, EAST, SOUTH, WEST)

    @staticmethod
    def turn_left(facing: Tuple[int, int]) -> Tuple[int, int]:
        index = Direction.ORDER.index(facing)
        return Direction.ORDER[(index - 1) % 4]

    @staticmethod
    def turn_right(facing: Tuple[int, int]) -> Tuple[int, int]:
        index = Direction.ORDER.index(facing)
        return Direction.ORDER[(index + 1) % 4]


class Action:
    """The four network actions, selected by argmax over the 4 outputs."""

    MOVE = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2
    EAT = 3


@dataclass
class EnvironmentConfig:
    width: int = 24
    height: int = 24
    ticks: int = 400

    # energy model
    initial_energy: float = 1.0
    max_energy: float = 1.0
    metabolism: float = 0.005
    food_energy: float = 0.5

    # food / resources
    initial_food: int = 90
    food_target: int = 90
    food_regrowth_per_tick: int = 1

    # observations
    density_radius: int = 2

    # behavioral traces (diversity metrics)
    record_trace: bool = True
    behavior_window: int = 100

    # reproduction (in-world asexual split)
    repro_threshold: float = 0.7
    repro_fraction: float = 0.5  # offspring keeps this fraction of parent energy

    # seeding: run rng = Random(world_rng_seed(generation))
    seed_base: int = 0
    seed_stride: int = 1000

    def world_rng_seed(self, generation: int) -> int:
        """Deterministic world seed for a generation.

        Uses a hash of ``(seed_base, generation)`` so that different
        (trial seed, generation) pairs never collide on the same world layout
        (an additive ``seed_base + generation * stride`` scheme does collide:
        trial 1 at gen 0 and trial 0 at gen 1 would share a world).
        """
        digest = hashlib.sha1(f"{self.seed_base}:{generation}".encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big")
