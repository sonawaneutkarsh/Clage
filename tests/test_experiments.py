import json
from dataclasses import replace
from pathlib import Path

import pytest

from experiments.config import (
    Condition,
    ExperimentConfig,
    load_experiment,
    resolve_config,
)
from experiments.metrics import compute_metrics
from experiments.run import RECORD_FIELDS, run_condition, run_trial
from experiments.analysis import (
    aggregate,
    compare_conditions,
    final_summary,
    load_trials,
    write_condition_csv,
)
from neat.genome import Genome
from world import EnvironmentConfig, Organism

CONFIGS = Path("experiments/configs")


def tiny_experiment() -> ExperimentConfig:
    base = json.loads((CONFIGS / "base.json").read_text())["base"]
    base["neat"]["generations"] = 2
    base["neat"]["population_size"] = 4
    base["world"]["ticks"] = 10
    return ExperimentConfig(
        name="tiny",
        base=base,
        conditions=[
            Condition(name="control", parameter=None),
            Condition(name="food_low", parameter="food_abundance", value=20),
        ],
        seeds=[0, 1],
    )


# ------------------------------------------------------------------ config


def test_all_shipped_configs_load_and_validate():
    for path in sorted(CONFIGS.glob("*.json")):
        experiment = load_experiment(path)
        assert experiment.name == path.stem
        assert len(experiment.conditions) >= 1
        assert experiment.seeds


def test_condition_resolves_single_parameter():
    experiment = load_experiment(CONFIGS / "food_abundance.json")
    control = resolve_config(experiment, experiment.condition("control"), seed=0)
    low = resolve_config(experiment, experiment.condition("food_low"), seed=0)

    assert control.world.food_target == 60
    assert low.world.food_target == 20
    assert low.world.initial_food == 20
    # one-at-a-time: only food fields changed
    assert low.population_size == control.population_size
    assert low.world.width == control.world.width
    assert low.world.repro_threshold == control.world.repro_threshold


def test_resolve_seed_overrides_world_seed_base():
    experiment = load_experiment(CONFIGS / "base.json")
    resolved = resolve_config(experiment, experiment.condition("control"), seed=3)
    assert resolved.world.seed_base == 3 * resolved.world.seed_stride


def test_unknown_parameter_rejected():
    with pytest.raises(ValueError):
        ExperimentConfig(
            name="bad",
            base={"neat": {}, "world": {}, "interface": {"input_ids": [], "output_ids": []}},
            conditions=[Condition(name="x", parameter="not_a_parameter", value=1)],
            seeds=[0],
        )


def test_config_round_trip():
    experiment = tiny_experiment()
    restored = ExperimentConfig(
        name=experiment.name,
        base=json.loads(json.dumps(experiment.base)),
        conditions=[
            Condition(name=c.name, parameter=c.parameter, value=c.value)
            for c in experiment.conditions
        ],
        seeds=list(experiment.seeds),
    )
    assert restored.to_dict()["conditions"][1] == {"name": "food_low", "parameter": "food_abundance", "value": 20}


# ------------------------------------------------------------------ metrics


def test_metrics_computation():
    genome = Genome.minimal(input_ids=[0], output_ids=[10])
    config = EnvironmentConfig()
    a = Organism(genome, 0, 0, config)
    a.age, a.food_eaten, a.offspring, a.alive = 10, 2, 1, True
    b = Organism(genome, 1, 1, config)
    b.age, b.food_eaten, b.offspring, b.alive = 5, 0, 0, False

    metrics = compute_metrics([a, b], population_size=2)
    assert metrics["survival_rate"] == 0.5
    assert metrics["average_lifetime"] == 7.5
    assert metrics["food_consumed"] == 2
    assert metrics["reproductive_success"] == 1
    assert metrics["offspring_per_genome"] == 0.5


# ------------------------------------------------------------------ runner


def test_run_trial_records_expected_fields():
    experiment = tiny_experiment()
    resolved = resolve_config(experiment, experiment.condition("control"), seed=0)
    records = run_trial(resolved, seed=0)

    assert len(records) == 2
    for record in records:
        assert set(record) == set(RECORD_FIELDS)
        assert record["population_size"] == 4


def test_run_condition_writes_machine_readable_files(tmp_path):
    out = run_condition(tiny_experiment(), tiny_experiment().condition("food_low"), tmp_path)

    assert (out / "0.json").exists()
    assert (out / "1.json").exists()
    assert (out / "0.config.json").exists()
    records = json.loads((out / "0.json").read_text())
    assert len(records) == 2
    assert set(records[0]) == set(RECORD_FIELDS)


def test_run_is_deterministic(tmp_path):
    run_condition(tiny_experiment(), tiny_experiment().condition("control"), tmp_path / "a")
    run_condition(tiny_experiment(), tiny_experiment().condition("control"), tmp_path / "b")

    a = (tmp_path / "a" / "tiny" / "control" / "0.json").read_bytes()
    b = (tmp_path / "b" / "tiny" / "control" / "0.json").read_bytes()
    assert a == b


# ------------------------------------------------------------------ analysis


def _write_trial(tmp_path, condition, best_fitness_series):
    directory = tmp_path / condition
    directory.mkdir(parents=True, exist_ok=True)
    for seed, fitness_series in enumerate(best_fitness_series):
        records = []
        for g, fitness in enumerate(fitness_series, start=1):
            record = {"generation": g, "best_fitness": fitness}
            for metric in RECORD_FIELDS:
                if metric not in record:
                    record[metric] = 0.5
            records.append(record)
        (directory / f"{seed}.json").write_text(json.dumps(records))


def test_aggregation_across_trials(tmp_path):
    _write_trial(tmp_path, "control", [[1.0, 2.0], [3.0, 4.0]])
    trials = load_trials(tmp_path, "control")
    agg = aggregate(trials)

    assert agg[1]["best_fitness"]["mean"] == 2.0
    assert agg[1]["best_fitness"]["min"] == 1.0
    assert agg[2]["best_fitness"]["mean"] == 3.0
    summary = final_summary(trials)
    assert summary["best_fitness"]["mean"] == 3.0


def test_compare_conditions(tmp_path):
    _write_trial(tmp_path, "control", [[1.0]])
    _write_trial(tmp_path, "high", [[3.0]])
    deltas = compare_conditions(tmp_path, "high", "control")
    assert deltas["best_fitness"][0] == pytest.approx(2.0)


def test_write_condition_csv(tmp_path):
    _write_trial(tmp_path, "control", [[1.0, 2.0]])
    out = write_condition_csv(tmp_path, "control")
    lines = out.read_text().strip().splitlines()
    assert lines[0] == "generation,metric,mean,std,min,max"
    assert len(lines) > 1
