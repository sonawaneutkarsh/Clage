"""Fitness: how an organism's life becomes a scalar written onto its genome."""

from __future__ import annotations

__all__ = ["fitness"]

# Coefficients are research knobs, not truths (architecture.md section 17).
FOOD_WEIGHT = 3.0
AGE_WEIGHT = 0.01
OFFSPRING_WEIGHT = 0.5


def fitness(food_eaten: int, age: int, offspring: int) -> float:
    return food_eaten * FOOD_WEIGHT + age * AGE_WEIGHT + offspring * OFFSPRING_WEIGHT
