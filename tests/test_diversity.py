import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from diversity.metrics import (
    action_entropy,
    encounter_rate,
    food_alignment_cosine,
    per_genome_metrics,
    pooled_transition_entropy_rate,
    population_behavior,
    spatial_coverage,
    transition_entropy_rate,
)
from diversity.metrics import _transition_counts  # private: pins the transition total
from world.config import ACTION_SIZE
from experiments.config import Condition, ExperimentConfig, load_experiment, resolve_config
from experiments.run import RECORD_FIELDS, run_trial
from neat.genome import Genome

from world import EnvironmentConfig, Organism


def trace(entries):
    return [tuple(e) for e in entries]


# ------------------------------------------------------------------ unit metrics


def test_action_entropy_zero_for_fixed_policy():
    assert action_entropy([0, 0, 0, 0]) == 0.0


def test_action_entropy_two_bits_for_uniform():
    assert action_entropy([0, 1, 2, 3]) == pytest.approx(2.0)


def test_transition_entropy_rate_deterministic():
    assert transition_entropy_rate([0, 1, 0, 1, 0]) == 0.0


def test_transition_entropy_rate_alternating_pairs():
    # [0,0,1,1,0,0,1,1]: row0 splits 2/2 (entropy 1.0, weight 4/7),
    # row1 splits 2/1 (entropy ~0.9183, weight 3/7) -> rate ~0.965.
    assert transition_entropy_rate([0, 0, 1, 1, 0, 0, 1, 1]) == pytest.approx(
        0.9649839288802097
    )


def test_spatial_coverage_fraction():
    points = [(0, 0), (0, 0), (1, 1), (2, 2), (3, 3)]
    assert spatial_coverage(points, area=16) == pytest.approx(4 / 16)


def test_food_alignment_cosine_directions():
    toward = food_alignment_cosine([(0, -1)], [(0.0, -1.0)])
    away = food_alignment_cosine([(0, 1)], [(0.0, -1.0)])
    perpendicular = food_alignment_cosine([(1, 0)], [(0.0, -1.0)])
    assert toward == pytest.approx(1.0)
    assert away == pytest.approx(-1.0)
    assert perpendicular == pytest.approx(0.0)


def test_food_alignment_skips_zero_deltas():
    assert food_alignment_cosine([(0, 0), (1, 0)], [(0.5, 0.5), (1.0, 0.0)]) == pytest.approx(1.0)


def test_encounter_rate_mean_density():
    entries = trace([(0, 0, 0, 0.0, 0.0, 0.2), (1, 0, 0, 0.0, 0.0, 0.4)])
    assert encounter_rate(entries) == pytest.approx(0.3)


def test_per_genome_metrics_pooling_and_window():
    t1 = trace([(0, 0, 0, 0.0, -1.0, 0.0)] * 50)  # always MOVE north, food north
    t2 = trace([(1, 3, 3, 0.0, -1.0, 0.5)] * 50)  # always turn left
    metrics = per_genome_metrics([t1, t2], window=10, area=100)
    # window clips each trace to 10; pooled = 10 MOVE + 10 LEFT -> entropy 1.0
    assert metrics["action_entropy"] == pytest.approx(1.0)
    assert metrics["spatial_coverage"] > 0.0
    assert metrics["encounter_rate"] == pytest.approx(0.25)  # (0.0 + 0.5)/2


def test_transition_entropy_respects_organism_boundaries():
    """transition_entropy must count only action pairs consecutive within one organism.

    Each of the two organisms below emits a perfectly predictable action stream
    (``[0, 0]`` and ``[1, 1]``), so the only honest conditional-entropy value for the
    genome they share is exactly ``0.0``.

    Measured pre-fix failure (unfixed code): the pooled assertion fails with
    ``0.6666666666666666`` instead of ``0.0``. ``per_genome_metrics`` flattens every
    organism's actions into one stream ``[0, 0, 1, 1]`` before measuring, and
    ``transition_entropy_rate`` counts ``zip(actions, actions[1:])``, so a ``0 -> 1``
    transition that never happened is fabricated across the organism boundary. Row 0
    then splits 1/1 (entropy 1.0, weight 2/3), producing 2/3 of a bit out of nothing.

    The companion single-organism assertion for requirement 2.8 **passes** pre-fix and
    must keep passing post-fix: with one trace the pooled denominator reduces to
    ``len(actions) - 1``, which is the existing formula, so the equality is exact.
    """
    t1 = trace([(0, 0, 0, 0.0, 0.0, 0.0), (0, 0, 0, 0.0, 0.0, 0.0)])  # actions [0, 0]
    t2 = trace([(1, 5, 5, 0.0, 0.0, 0.0), (1, 5, 5, 0.0, 0.0, 0.0)])  # actions [1, 1]
    assert per_genome_metrics([t1, t2], window=0, area=100)["transition_entropy"] == 0.0

    # 2.8 — one organism with >= 2 entries must equal the single-sequence function
    # exactly (bitwise), not approximately.
    t = trace([
        (0, 0, 0, 0.0, 0.0, 0.0),
        (0, 1, 0, 0.0, 0.0, 0.0),
        (1, 1, 0, 0.0, 0.0, 0.0),
        (1, 1, 0, 0.0, 0.0, 0.0),
        (0, 1, 1, 0.0, 0.0, 0.0),
        (0, 2, 1, 0.0, 0.0, 0.0),
        (1, 2, 1, 0.0, 0.0, 0.0),
        (1, 2, 1, 0.0, 0.0, 0.0),
    ])
    assert per_genome_metrics([t], window=0, area=100)["transition_entropy"] == (
        transition_entropy_rate([a for a, *_ in t])
    )


def test_food_alignment_pairs_displacement_with_preceding_observation():
    """food_alignment must pair each displacement with the observation before its action.

    Temporal rule: a trace entry holds a PRE-action observation (action, food_dx,
    food_dy, density) together with a POST-action position (x, y). So for i >= 1 the
    displacement ``pos_i - pos_{i-1}`` was caused by the action at index ``i``, and the
    food direction observed immediately before that action is the one at index ``i``,
    not ``i-1``. Samples are built within a single trace and never across traces.

    Measured pre-fix failures (unfixed code):
    - 4a returns ``+1.0`` instead of ``-1.0`` — a full sign inversion, with a step
      taken directly away from food scored as perfect food-seeking.
    - 4b returns ``0.5`` instead of ``1.0`` — ``food_dirs`` is flattened across
      organisms while ``deltas`` is not, so the one-index offset accumulates and
      organism B's step is scored against organism A's food direction.
    """
    # 4a — off-by-one inside one trace: step east while food was reported west.
    t = trace([(0, 0, 0, 1.0, 0.0, 0.0), (0, 1, 0, -1.0, 0.0, 0.0)])
    assert per_genome_metrics([t], window=0, area=100)["food_alignment"] == pytest.approx(-1.0)

    # 4b — no cross-organism pairing: each organism stepped straight at its own food.
    t1 = trace([(0, 0, 0, 1.0, 0.0, 0.0), (0, 1, 0, 1.0, 0.0, 0.0)])  # east step, food east
    t2 = trace([(0, 5, 5, 0.0, 1.0, 0.0), (0, 5, 6, 0.0, 1.0, 0.0)])  # south step, food south
    assert per_genome_metrics([t1, t2], window=0, area=100)["food_alignment"] == pytest.approx(1.0)


# ------------------------------------------------------ pooled transition entropy


def test_pooled_transition_entropy_rate_zero_without_transitions():
    """No within-sequence transition means a zero denominator, hence exactly 0.0."""
    assert pooled_transition_entropy_rate([]) == 0.0            # no sequences at all
    assert pooled_transition_entropy_rate([[]]) == 0.0          # one empty sequence
    assert pooled_transition_entropy_rate([[2]]) == 0.0         # one sequence shorter than 2
    # several sequences, all shorter than 2: every one is skipped, so the total
    # transition count is 0 and no fabricated cross-sequence pair rescues it.
    assert pooled_transition_entropy_rate([[0], [1], [2], [], [3]]) == 0.0


def test_pooled_transition_entropy_rate_matches_single_sequence_function():
    """One sequence must reproduce ``transition_entropy_rate`` bitwise (requirement 2.8)."""
    actions = [0, 0, 1, 1, 0, 0, 1, 1, 2, 0, 3, 3, 1, 2, 2, 0]
    assert pooled_transition_entropy_rate([actions]) == transition_entropy_rate(actions)
    # the pinned reference value stays reachable through the pooled entry point too
    assert pooled_transition_entropy_rate([[0, 0, 1, 1, 0, 0, 1, 1]]) == (
        transition_entropy_rate([0, 0, 1, 1, 0, 0, 1, 1])
    )


def test_pooled_transition_entropy_rate_weights_by_contributed_transitions():
    """Each sequence is weighted by the transitions it actually contributed.

    Pooled count matrix for ``[[0, 0, 0, 1], [1, 1], [0]]``:
    row 0 -> {0: 2, 1: 1} (3 transitions), row 1 -> {1: 1} (1 transition), and the
    length-1 sequence contributes nothing. Total = 3 + 1 = 4, which is
    ``sum(len(s) - 1 for s in sequences if len(s) >= 2)``. Row 1 is deterministic
    (entropy 0.0), so the rate is row 0's entropy scaled by its 3/4 share.
    """
    sequences = [[0, 0, 0, 1], [1, 1], [0]]
    row0_entropy = -(2 / 3) * math.log2(2 / 3) - (1 / 3) * math.log2(1 / 3)
    expected = (3 / 4) * row0_entropy
    assert pooled_transition_entropy_rate(sequences) == pytest.approx(expected)

    # The weighting rule is not either of the two rejected combinations:
    # flattening (which fabricates the 1 -> 1 and 1 -> 0 boundary pairs)...
    flattened = transition_entropy_rate([a for s in sequences for a in s])
    assert pooled_transition_entropy_rate(sequences) != pytest.approx(flattened)
    # ...nor an unweighted mean of per-sequence rates, which would give the
    # length-1 and length-2 sequences the same weight as the length-4 one.
    per_sequence_mean = sum(transition_entropy_rate(s) for s in sequences) / len(sequences)
    assert pooled_transition_entropy_rate(sequences) != pytest.approx(per_sequence_mean)


def test_pooled_transition_total_counts_only_within_trace_pairs():
    """The transition denominator is exactly the count of legitimate pairs."""
    sequences = [[0, 0, 1], [1], [2, 2, 2, 2]]
    counts, total = _transition_counts(sequences, ACTION_SIZE)

    assert total == sum(len(s) - 1 for s in sequences if len(s) >= 2)
    assert total == 5  # 2 from [0, 0, 1], 0 from [1], 3 from [2, 2, 2, 2]
    # every counted pair is a within-sequence pair, so the matrix sums to the total
    assert sum(sum(row) for row in counts) == total
    assert counts[0][0] == 1 and counts[0][1] == 1 and counts[2][2] == 3
    # the three boundary pairs that flattening would fabricate are absent:
    # (1 -> 1) across [0, 0, 1] | [1], (1 -> 2) across [1] | [2, 2, 2, 2]
    assert counts[1][1] == 0
    assert counts[1][2] == 0


# ------------------------------------------------------ alignment sample accounting


def _aligned_samples(traces, window):
    """Rebuild the alignment sample lists the way ``per_genome_metrics`` does.

    One sample per index ``i >= 1`` of each clipped trace, pairing
    ``pos_i - pos_{i-1}`` with the food direction recorded at index ``i``, and never
    pairing across two traces.
    """
    deltas = []
    food_dirs = []
    for entries in traces:
        clipped = entries[:window] if window and window > 0 else entries
        for index in range(1, len(clipped)):
            _, x, y, fx, fy, _ = clipped[index]
            prev_x, prev_y = clipped[index - 1][1], clipped[index - 1][2]
            deltas.append((x - prev_x, y - prev_y))
            food_dirs.append((fx, fy))
    return deltas, food_dirs


def test_alignment_sample_count_drops_exactly_one_per_trace():
    """Sample count is ``sum(max(0, len(clipped) - 1))`` — one dropped per trace.

    Each trace's first recorded action is dropped because the position before it was
    never recorded, so its displacement is unrecoverable; consequently ``food_dirs[0]``
    of every trace is unused while the final observed food direction is used.

    Per-trace sample cosines below are distinct known values (+1.0, 0.0, +0.6), so the
    reported mean pins the exact sample set: an extra pair, a missing pair, or a pair
    built from two different traces would all move it.
    """
    t1 = trace([
        (0, 0, 0, 1.0, 0.0, 0.0),   # first recorded action: dropped, food dir unused
        (0, 1, 0, 1.0, 0.0, 0.0),   # step east, food east   -> cos +1.0
        (0, 1, 1, 1.0, 0.0, 0.0),   # step south, food east  -> cos  0.0
    ])
    t2 = trace([
        (0, 5, 5, 0.0, 1.0, 0.0),   # first recorded action: dropped, food dir unused
        (0, 6, 5, 0.6, 0.8, 0.0),   # step east, food east-south -> cos +0.6
    ])
    t3 = trace([(0, 9, 9, 1.0, 0.0, 0.0)])  # single entry -> no sample at all
    traces = [t1, t2, t3]

    deltas, food_dirs = _aligned_samples(traces, window=0)
    expected_count = sum(max(0, len(t) - 1) for t in traces)
    assert expected_count == 3
    assert len(deltas) == expected_count
    assert len(food_dirs) == expected_count
    # no sample mixes indices from two traces: every delta stays inside its trace
    assert deltas == [(1, 0), (0, 1), (1, 0)]

    metrics = per_genome_metrics(traces, window=0, area=100)
    assert metrics["food_alignment"] == food_alignment_cosine(deltas, food_dirs)
    assert metrics["food_alignment"] == pytest.approx((1.0 + 0.0 + 0.6) / 3)

    # clipping first, then dropping one per clipped trace
    windowed_deltas, windowed_food_dirs = _aligned_samples(traces, window=2)
    assert len(windowed_deltas) == sum(max(0, len(t[:2]) - 1) for t in traces) == 2
    windowed = per_genome_metrics(traces, window=2, area=100)
    assert windowed["food_alignment"] == food_alignment_cosine(
        windowed_deltas, windowed_food_dirs
    )
    assert windowed["food_alignment"] == pytest.approx((1.0 + 0.6) / 2)


# ------------------------------------------------------------------ population


def _organism(genome, entries):
    org = SimpleNamespace()
    org.genome = genome
    org.trace = trace(entries)
    return org


def test_population_diversity_zero_for_identical_genomes():
    g = Genome.minimal(input_ids=[0], output_ids=[10])
    organisms = [_organism(g, [(0, 0, 0, 0.0, -1.0, 0.0)] * 10) for _ in range(2)]
    behavior = population_behavior(organisms, [g, g], window=100, area=16)
    assert behavior["behavioral_diversity"] == 0.0


def test_population_diversity_positive_for_distinct_behaviors():
    g1 = Genome.minimal(input_ids=[0], output_ids=[10])
    g2 = Genome.minimal(input_ids=[0], output_ids=[10])
    organisms = [
        _organism(g1, [(0, 0, 0, 0.0, -1.0, 0.0)] * 10),   # always MOVE
        _organism(g2, [(1, 1, 1, 0.0, -1.0, 0.0)] * 10),   # always turn left
        _organism(g2, [(3, 2, 2, 0.0, -1.0, 0.0)] * 10),   # always eat
    ]
    behavior = population_behavior(organisms, [g1, g2], window=100, area=16)
    assert behavior["behavioral_diversity"] > 0.0
    assert behavior["action_entropy_diversity"] >= 0.0


# ------------------------------------------------------------------ world trace


def test_world_records_trace_when_enabled():
    import random

    from world import World

    config = EnvironmentConfig(width=4, height=4, ticks=1, repro_threshold=2.0)
    world = World(config, random.Random(0))
    org = Organism(
        Genome.minimal(input_ids=list(range(9)), output_ids=[10, 11, 12, 13]),
        config=config,
        x=1,
        y=1,
    )
    world.place_organism(org)
    org.act(world, config)
    assert len(org.trace) == 1
    assert len(org.trace[0]) == 6  # (action, x, y, food_dx, food_dy, density)


# ------------------------------------------------------------------ integration


def _tiny_resolved():
    base = json.loads(Path("experiments/configs/base.json").read_text())["base"]
    base["neat"]["generations"] = 1
    base["neat"]["population_size"] = 4
    base["world"]["ticks"] = 20
    experiment = ExperimentConfig(
        name="tiny", base=base, conditions=[Condition(name="control", parameter=None)], seeds=[0]
    )
    return resolve_config(experiment, experiment.condition("control"), seed=0)


def test_experiment_records_behavioral_metrics():
    records = run_trial(_tiny_resolved(), seed=0)
    assert len(records) == 1
    record = records[0]
    for field in RECORD_FIELDS:
        assert field in record
    for field in (
        "action_entropy", "transition_entropy", "spatial_coverage",
        "food_alignment", "encounter_rate", "behavioral_diversity",
    ):
        assert record[field] == record[field]  # not NaN


def test_empty_genome_baseline_zero_entropy_and_diversity():
    # Empty genomes always emit MOVE -> action entropy 0. The behavioral
    # diversity index is NOT zero because identical policies placed at
    # different cells visit different cells (positional variance feeds the
    # spatial-coverage fingerprint dimension) — a documented caveat.
    records = run_trial(_tiny_resolved(), seed=0)
    record = records[0]
    assert record["action_entropy"] == pytest.approx(0.0)
    assert record["transition_entropy"] == pytest.approx(0.0)
    assert record["behavioral_diversity"] >= 0.0
    assert record["food_alignment"] == record["food_alignment"]  # not NaN
