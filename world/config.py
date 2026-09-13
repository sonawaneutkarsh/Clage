"""Configuration for the Clage artificial-life world.

All numbers are knobs, not truths — the architecture only requires that *some*
scalar fitness is written back onto each genome.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, List, Tuple

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


def _require_int(name: str, value: Any) -> None:
    """Integer-semantic field: an actual ``int``.

    A ``bool`` is an ``int`` subclass but no field here means "True cells
    wide", and an integral ``float`` such as ``20.0`` is still a float that
    ``range()`` refuses. Both are type errors, not range errors.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"EnvironmentConfig.{name} must be an int, got {value!r}")


def _require_number(name: str, value: Any) -> None:
    """Float-semantic field: an ``int`` or a ``float``, and finite.

    ``int`` is accepted so ``max_energy=1`` keeps working. ``bool`` is not: no
    legitimate caller passes ``True`` as an energy. NaN and infinity are the
    right type but not usable quantities, so they are domain (value) errors.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"EnvironmentConfig.{name} must be an int or a float, got {value!r}"
        )
    if not math.isfinite(value):
        raise ValueError(f"EnvironmentConfig.{name} must be finite, got {value!r}")


def _require(name: str, value: Any, rule: str, holds: bool) -> None:
    """Range rule for an already type-checked field."""
    if not holds:
        raise ValueError(f"EnvironmentConfig.{name} must be {rule}, got {value!r}")


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

    def __post_init__(self) -> None:
        """Reject configurations the world cannot represent.

        Fields are checked in DECLARATION order, and each field's type before
        its range, so when several fields are invalid the same one is always
        named. ``dataclasses.replace`` goes through ``__init__``, so replaced
        configs are validated too.

        Only rules with a reproduced downstream consequence are here.
        ``behavior_window <= 0`` deliberately means "no clipping",
        ``max_energy < initial_energy`` is merely capped on the first consume,
        and ``initial_food`` above grid capacity already degrades gracefully —
        those are type-checked but not range-checked.
        """
        _require_int("width", self.width)
        _require("width", self.width, ">= 1", self.width >= 1)

        _require_int("height", self.height)
        _require("height", self.height, ">= 1", self.height >= 1)

        _require_int("ticks", self.ticks)
        _require("ticks", self.ticks, ">= 1", self.ticks >= 1)

        # energy model
        _require_number("initial_energy", self.initial_energy)
        _require("initial_energy", self.initial_energy, "> 0", self.initial_energy > 0)

        _require_number("max_energy", self.max_energy)
        _require("max_energy", self.max_energy, "> 0", self.max_energy > 0)

        _require_number("metabolism", self.metabolism)
        _require("metabolism", self.metabolism, ">= 0", self.metabolism >= 0)

        _require_number("food_energy", self.food_energy)  # no range rule

        # food / resources
        _require_int("initial_food", self.initial_food)
        _require("initial_food", self.initial_food, ">= 0", self.initial_food >= 0)

        _require_int("food_target", self.food_target)
        _require("food_target", self.food_target, ">= 0", self.food_target >= 0)

        _require_int("food_regrowth_per_tick", self.food_regrowth_per_tick)
        _require(
            "food_regrowth_per_tick",
            self.food_regrowth_per_tick,
            ">= 0",
            self.food_regrowth_per_tick >= 0,
        )

        # observations
        _require_int("density_radius", self.density_radius)
        _require("density_radius", self.density_radius, ">= 1", self.density_radius >= 1)

        # record_trace is a genuine bool: neither type- nor range-checked.
        _require_int("behavior_window", self.behavior_window)  # no range rule

        # reproduction
        _require_number("repro_threshold", self.repro_threshold)  # no range rule

        _require_number("repro_fraction", self.repro_fraction)
        _require(
            "repro_fraction",
            self.repro_fraction,
            "between 0.0 and 1.0 inclusive",
            0.0 <= self.repro_fraction <= 1.0,
        )

        # seeding
        _require_int("seed_base", self.seed_base)  # no range rule
        _require_int("seed_stride", self.seed_stride)
        _require("seed_stride", self.seed_stride, ">= 1", self.seed_stride >= 1)

    def world_rng_seed(self, generation: int) -> int:
        """Deterministic world seed for a generation.

        Uses a hash of ``(seed_base, generation)`` so that different
        (trial seed, generation) pairs never collide on the same world layout
        (an additive ``seed_base + generation * stride`` scheme does collide:
        trial 1 at gen 0 and trial 0 at gen 1 would share a world).
        """
        digest = hashlib.sha1(f"{self.seed_base}:{generation}".encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big")
