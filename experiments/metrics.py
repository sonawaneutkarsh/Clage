"""World outcome metrics for one generation's trial.

All metrics are computed from the organisms returned by ``run_generation`` and
are defined identically for every experimental condition.
"""

from __future__ import annotations

from typing import Dict, List

from world import Organism

__all__ = ["compute_metrics"]


def compute_metrics(organisms: List[Organism], population_size: int) -> Dict[str, float]:
    """Aggregate one trial's end state into the recorded world metrics."""
    n = len(organisms)
    alive = sum(1 for o in organisms if o.alive)
    total_food = sum(o.food_eaten for o in organisms)
    total_offspring = sum(o.offspring for o in organisms)
    return {
        "organism_count": float(n),
        "survival_rate": alive / n if n else 0.0,
        "average_lifetime": sum(o.age for o in organisms) / n if n else 0.0,
        "food_consumed": float(total_food),
        "reproductive_success": float(total_offspring),
        "offspring_per_genome": total_offspring / population_size if population_size else 0.0,
    }
