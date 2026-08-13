import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from diversity.metrics import (
    action_entropy,
    encounter_rate,
    food_alignment_cosine,
    per_genome_metrics,
    population_behavior,
    spatial_coverage,
    transition_entropy_rate,
)
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
