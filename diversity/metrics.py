"""Behavioral metrics computed from per-tick organism traces.

A trace is a list of ``(action, x, y, food_dx, food_dy, density)`` records, one
per tick.

**The tuple has a mixed temporal reference, per field:**

- ``action``, ``food_dx``, ``food_dy``, ``density`` all come from the observation
  taken BEFORE the action of that tick — ``action`` is the action chosen from that
  observation, and the other three are the observation itself;
- ``x``, ``y`` are the position AFTER that action was applied.

So entry ``i`` describes "what was seen, what was decided, and where that left the
organism". A consumer pairing a displacement with an observation must therefore use
the observation at the LATER index: ``pos_i - pos_{i-1}`` was caused by the action at
index ``i``, whose observation is also stored at index ``i``.

All metrics are DESCRIPTIVE statistics of observable actions and
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
    "pooled_transition_entropy_rate",
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


def _transition_counts(
    sequences: Sequence[Sequence[int]],
    n_actions: int,
) -> Tuple[List[List[int]], int]:
    """Count consecutive action pairs WITHIN each sequence, never across sequences.

    Returns the ``n_actions x n_actions`` count matrix and the total number of
    within-sequence transitions, ``sum(len(s) - 1 for s in sequences if len(s) >= 2)``.
    Fixed-size lists indexed by action id keep accumulation order index-fixed.
    """
    counts = [[0] * n_actions for _ in range(n_actions)]
    total_transitions = 0
    for sequence in sequences:
        if len(sequence) < 2:
            continue
        for previous, current in zip(sequence, sequence[1:]):
            counts[previous][current] += 1
        total_transitions += len(sequence) - 1
    return counts, total_transitions


def _conditional_entropy_rate(counts: Sequence[Sequence[int]], total_transitions: int) -> float:
    """Conditional entropy H(a_t | a_{t-1}) in bits of a transition count matrix.

    Rows are iterated in action-id order and each row's entropy is accumulated in
    action-id order, then weighted by the row's share of ``total_transitions``. That
    fixed operation order is what makes the value bit-reproducible.
    """
    if total_transitions == 0:
        return 0.0
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
        rate += (total / total_transitions) * row_entropy
    return rate


def pooled_transition_entropy_rate(
    sequences: Sequence[Sequence[int]],
    n_actions: int = ACTION_SIZE,
) -> float:
    """Conditional entropy H(a_t | a_{t-1}) over several action sequences.

    **Combination rule.** Transition counts are pooled across the sequences —
    counting only pairs that occurred consecutively *within* one sequence, never the
    pair spanning the end of one sequence and the start of the next — and the
    conditional entropy of the single pooled count matrix is normalized by the total
    number of within-sequence transitions,
    ``sum(len(s) - 1 for s in sequences if len(s) >= 2)``. Each sequence is therefore
    weighted by the number of transitions it actually contributed. Sequences shorter
    than 2 contribute nothing; no sequences (or none long enough) gives 0.0, because
    the denominator is then 0.

    For one sequence the denominator reduces to ``len(actions) - 1``, which is exactly
    what ``transition_entropy_rate`` computes — hence the delegation, and hence the
    exact (not approximate) equality of the two entry points on a single sequence.

    *Rejected alternative: averaging per-sequence entropy rates.* It gives a 2-tick
    organism the same weight as a 300-tick one, so it is dominated by noise on short
    traces (an organism that dies after two ticks always reports 0.0), and it would
    break the single-sequence exact equality above unless special-cased.
    """
    counts, total_transitions = _transition_counts(sequences, n_actions)
    return _conditional_entropy_rate(counts, total_transitions)


def transition_entropy_rate(actions: Sequence[int], n_actions: int = ACTION_SIZE) -> float:
    """Conditional entropy H(a_t | a_{t-1}) in bits — temporal predictability.

    Measures how unpredictable the next action is given the previous one, i.e.
    the *structure* of the action sequence beyond its marginal distribution.
    0.0 for a fixed/deterministic sequence (and for fewer than 2 actions). Does NOT
    measure sophistication — a rigid policy is also 0.0.

    Single-sequence case of ``pooled_transition_entropy_rate``, which it delegates to
    so the two paths cannot drift. That pooled rule counts only pairs consecutive
    within one sequence and normalizes by
    ``sum(len(s) - 1 for s in sequences if len(s) >= 2)``, which here is just
    ``len(actions) - 1``; see that function for the combination rule and for why
    averaging per-sequence rates was rejected.
    """
    return pooled_transition_entropy_rate([actions], n_actions)


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

    ``deltas[k]`` and ``food_dirs[k]`` must already be aligned by the caller: the
    definition of a sample is ``(pos_i - pos_{i-1}, (fx_i, fy_i))`` for ``i >= 1``
    within ONE trace — the displacement caused by the action at index ``i``, paired
    with the food direction observed immediately before that same action. Samples are
    never built across two traces.

    *Accepted tradeoff.* Each trace's first recorded action is dropped, because the
    position before it is never recorded and its displacement is therefore
    unrecoverable — at most one lost sample per organism out of up to
    ``behavior_window`` (default 100). Symmetrically, ``food_dirs[0]`` of a trace is
    consequently unused (it precedes that unrecoverable displacement) while the final
    observed food direction is now used.
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
    """Pool all organisms sharing one genome (over a fixed tick window) into scalars.

    Each trace is first clipped to ``window`` ticks, then two different kinds of
    aggregation are applied:

    - **Pooled across organisms, order-insensitive:** ``action_entropy`` (marginal
      action distribution over every organism's actions), ``spatial_coverage``
      (distinct cells visited by any organism) and ``encounter_rate`` (mean observed
      density over every tick). Flattening is correct for these — the trace boundaries
      carry no information they use.
    - **Boundary-respecting:** ``transition_entropy`` and ``food_alignment``. Both are
      order-sensitive, so flattening would fabricate observations that never happened.
      ``transition_entropy`` counts action pairs only within one organism's clipped
      trace and pools the counts (see ``pooled_transition_entropy_rate``).
      ``food_alignment`` builds each sample as
      ``(pos_i - pos_{i-1}, (fx_i, fy_i))`` for ``i >= 1`` within a single trace,
      pairing a displacement with the food direction observed immediately before the
      action that caused it, and never pairs across two traces.

    *Accepted tradeoff on ``food_alignment``:* each trace's first recorded action is
    dropped, since the position preceding it is never recorded, so its displacement is
    unrecoverable. ``food_dirs[0]`` of every trace is therefore unused, and the final
    observed food direction is now used.
    """
    actions: List[int] = []
    action_sequences: List[List[int]] = []
    points: List[Tuple[int, int]] = []
    aligned_deltas: List[Tuple[int, int]] = []
    aligned_food_dirs: List[Tuple[float, float]] = []
    density_sum = 0.0
    density_ticks = 0

    for trace in traces:
        clipped = _clip_window(trace, window)
        # One sequence per organism: transition counts must not cross a trace
        # boundary, while `actions` stays flat for the pooled marginal distribution.
        trace_actions: List[int] = []
        action_sequences.append(trace_actions)
        for index, (action, x, y, fx, fy, density) in enumerate(clipped):
            actions.append(action)
            trace_actions.append(action)
            points.append((x, y))
            if index > 0:
                # A trace entry holds a PRE-action observation with a POST-action
                # position, so the displacement pos_i - pos_{i-1} was caused by the
                # action at index i and the food direction preceding that action is
                # the one at index i. Both lists are appended in the same iteration,
                # so they cannot desynchronize, and no pair crosses a trace boundary.
                prev_x, prev_y = clipped[index - 1][1], clipped[index - 1][2]
                aligned_deltas.append((x - prev_x, y - prev_y))
                aligned_food_dirs.append((fx, fy))
            density_sum += density
            density_ticks += 1

    return {
        "action_entropy": action_entropy(actions),
        "transition_entropy": pooled_transition_entropy_rate(action_sequences),
        "spatial_coverage": spatial_coverage(points, area),
        "food_alignment": food_alignment_cosine(aligned_deltas, aligned_food_dirs),
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
