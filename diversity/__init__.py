"""Behavioral diversity metrics for Clage.

Descriptive statistics computed from per-tick organism traces (actions,
positions, and observations). These metrics measure what organisms DO; they do
not infer intent, and they never claim cooperation/competition/aggression/
avoidance — the environment defines no such interactions.
"""

from .metrics import (
    action_entropy,
    encounter_rate,
    food_alignment_cosine,
    per_genome_metrics,
    population_behavior,
    spatial_coverage,
    transition_entropy_rate,
)

__all__ = [
    "action_entropy",
    "encounter_rate",
    "food_alignment_cosine",
    "per_genome_metrics",
    "population_behavior",
    "spatial_coverage",
    "transition_entropy_rate",
]
