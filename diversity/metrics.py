"""Behavioral metrics computed from per-tick organism traces.

A trace is a list of ``(action, x, y, food_dx, food_dy, density)`` records, one
per tick. All metrics are DESCRIPTIVE statistics of observable actions and
positions. None of them measure cooperation, competition, aggression, or
avoidance: Clage defines no cooperative/competitive interaction, and the
observation space has no directional organism sensor, so such strategies are
not expressible. See ``progress/diversity.md`` for the full justification.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

from world.config import ACTION_SIZE

__all__ = [
    "action_entropy",
    "transition_entropy_rate",
    "spatial_coverage",
    "food_alignment_cosine",
    "encounter_rate",
    "per_genome_metrics",
    "population_behavior",
]

Trace = List[Tuple[int, int, int, float, float, float]]


def _clip_window(trace: Trace, window: int) -> Trace:
    return trace[:window] if window and window > 0 else trace


def action_entropy(actions: Sequence[int], n_actions: int = ACTION_SIZE) -> float:
    """Shannon entropy (bits) of the action distribution.

    Measures how evenly the action stream spreads over the available actions.
    0.0 for a fixed policy; up to log2(n_actions) for a uniform one.
    Does NOT measure exploration, goal-directedness, or adaptiveness.
    """
    if not actions:
        return 0.0
    counts = [0] * n_actions
    for action in actions:
        counts[action] += 1
    total = len(actions)
    entropy = 0.0
    for count in counts:
        if count:
            p = count / total
            entropy -= p * math.log2(p)
    return entropy


def transition_entropy_rate(actions: Sequence[int], n_actions: int = ACTION_SIZE) -> float:
    """Conditional entropy H(a_t | a_{t-1}) in bits — temporal predictability.

    Measures how unpredictable the next action is given the previous one, i.e.
    the *structure* of the action sequence beyond its marginal distribution.
    0.0 for a fixed/deterministic sequence. Does NOT measure sophistication —
    a rigid policy is also 0.0.
    """
    if len(actions) < 2:
        return 0.0
    counts = [[0] * n_actions for _ in range(n_actions)]
    for previous, current in zip(actions, actions[1:]):
        counts[previous][current] += 1

    rate = 0.0
    for row in counts:
        total = sum(row)
        if total == 0:
            continue
        row_entropy = 0.0
        for count in row:
            if count:
                p = count / total
                row_entropy -= p * math.log2(p)
        rate += (total / (len(actions) - 1)) * row_entropy
    return rate


def spatial_coverage(points: Sequence[Tuple[int, int]], area: int) -> float:
    """Fraction of the grid area visited (distinct cells / area), in [0, 1].

    Measures spatial spread. Does NOT measure exploration intent — forced
    wandering under scarcity also raises coverage. Normalized by area so it is
    comparable across different world sizes.
    """
    if not points:
        return 0.0
    return len(set(points)) / max(1, area)


def food_alignment_cosine(
    deltas: Sequence[Tuple[int, int]],
    food_dirs: Sequence[Tuple[float, float]],
) -> float:
    """Mean cosine between effective movement and nearest-food direction.

    Computed over moves only (deltas that actually changed the position) and
    only where a food direction was visible. +1 = always toward food, -1 = away,
    0 = uncorrelated. This is a statistical COUPLING, not evidence of intent;
    a random walk in a dense-food world can score positive.
    """
    total = 0.0
    samples = 0
    for (dx, dy), (fx, fy) in zip(deltas, food_dirs):
        magnitude = math.hypot(dx, dy)
        food_magnitude = math.hypot(fx, fy)
        if magnitude == 0.0 or food_magnitude == 0.0:
            continue
        total += (dx * fx + dy * fy) / (magnitude * food_magnitude)
        samples += 1
    return total / samples if samples else 0.0


def encounter_rate(trace: Trace, window: int = 0) -> float:
    """Mean organism-density observation over the trace (an environmental covariate).

    This is NOT a behavior metric: co-presence is forced by density and space.
    Reported so condition differences can be attributed to the world, not the policy.
    """
    trace = _clip_window(trace, window)
    if not trace:
        return 0.0
    return sum(entry[5] for entry in trace) / len(trace)


def per_genome_metrics(
    traces: Sequence[Trace],
    window: int,
    area: int,
) -> Dict[str, float]:
    """Pool all organisms sharing one genome (over a fixed tick window) into scalars."""
    actions: List[int] = []
    points: List[Tuple[int, int]] = []
    deltas: List[Tuple[int, int]] = []
    food_dirs: List[Tuple[float, float]] = []
    density_sum = 0.0
    density_ticks = 0

    for trace in traces:
        clipped = _clip_window(trace, window)
        for index, (action, x, y, fx, fy, density) in enumerate(clipped):
            actions.append(action)
            points.append((x, y))
            food_dirs.append((fx, fy))
            if index > 0:
                prev_x, prev_y = clipped[index - 1][1], clipped[index - 1][2]
                deltas.append((x - prev_x, y - prev_y))
            density_sum += density
            density_ticks += 1

    return {
        "action_entropy": action_entropy(actions),
        "transition_entropy": transition_entropy_rate(actions),
        "spatial_coverage": spatial_coverage(points, area),
        "food_alignment": food_alignment_cosine(deltas, food_dirs),
        "encounter_rate": density_sum / density_ticks if density_ticks else 0.0,
    }


def population_behavior(
    organisms: Sequence,
    population: Sequence,
    window: int,
    area: int,
) -> Dict[str, float]:
    """Aggregate per-genome behavioral metrics over a population.

    Returns population means, the diversity index, and the encounter covariate.
    ``organisms`` is the list returned by ``run_generation``.
    """
    traces_by_genome: Dict[object, List[Trace]] = {genome: [] for genome in population}
    for organism in organisms:
        if organism.genome in traces_by_genome:
            traces_by_genome[organism.genome].append(organism.trace)

    per_genome = {
        genome: per_genome_metrics(traces, window, area)
        for genome, traces in traces_by_genome.items()
        if traces
    }
    if not per_genome:
        return {key: 0.0 for key in _behavior_keys()}

    means = {}
    for key in ("action_entropy", "transition_entropy", "spatial_coverage",
                "food_alignment", "encounter_rate"):
        values = [metrics[key] for metrics in per_genome.values()]
        means[key] = sum(values) / len(values)

    fingerprint_values = {
        key: [metrics[key] for metrics in per_genome.values()]
        for key in ("action_entropy", "transition_entropy", "spatial_coverage")
    }
    means["behavioral_diversity"] = _population_diversity(fingerprint_values)
    means["action_entropy_diversity"] = _std(
        fingerprint_values["action_entropy"]
    )
    return means


def _behavior_keys() -> List[str]:
    return [
        "action_entropy",
        "action_entropy_diversity",
        "transition_entropy",
        "spatial_coverage",
        "food_alignment",
        "encounter_rate",
        "behavioral_diversity",
    ]


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


def _population_diversity(
    fingerprint_columns: Dict[str, List[float]],
) -> float:
    """Mean pairwise Euclidean distance between z-scored behavioral fingerprints."""
    genomes = list(fingerprint_columns.values())
    if not genomes or len(genomes[0]) < 2:
        return 0.0

    fingerprints = []
    dimensions = list(fingerprint_columns)
    for i in range(len(fingerprint_columns[dimensions[0]])):
        vector = []
        for dimension in dimensions:
            column = fingerprint_columns[dimension]
            mean = sum(column) / len(column)
            std = _std(column)
            value = (column[i] - mean) / std if std > 0 else 0.0
            vector.append(value)
        fingerprints.append(vector)

    total = 0.0
    pairs = 0
    for i in range(len(fingerprints)):
        for j in range(i + 1, len(fingerprints)):
            total += math.dist(fingerprints[i], fingerprints[j])
            pairs += 1
    return total / pairs if pairs else 0.0
