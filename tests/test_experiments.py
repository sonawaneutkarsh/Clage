import errno
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from experiments.config import (
    PARAMETERS,
    Condition,
    ExperimentConfig,
    load_experiment,
    resolve_config,
)
from experiments.metrics import compute_metrics
from experiments.run import (
    RECORD_FIELDS,
    main,
    run_condition,
    run_experiment,
    run_trial,
)
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


def test_condition_value_with_wrong_type_rejected_at_resolve_time():
    """A condition value the target field cannot accept fails at resolution.

    Pre-fix at HEAD ea8a9c2 this resolved happily to
    ``EnvironmentConfig.width == 'twenty'`` and failed later inside ``World``,
    far from the offending file. ``resolve_config`` adds context only — the
    rule lives on ``EnvironmentConfig``.
    """
    experiment = load_experiment(CONFIGS / "available_space.json")
    condition = replace(experiment.conditions[1], value="twenty")

    with pytest.raises(TypeError) as excinfo:
        resolve_config(experiment, condition, seed=0)

    message = str(excinfo.value)
    assert condition.name in message
    assert "available_space" in message
    assert "'twenty'" in message
    assert "width" in message  # the original field-level message is preserved


def test_condition_value_out_of_range_rejected_at_resolve_time():
    experiment = load_experiment(CONFIGS / "available_space.json")
    condition = replace(experiment.conditions[1], value=0)

    with pytest.raises(ValueError) as excinfo:
        resolve_config(experiment, condition, seed=0)

    message = str(excinfo.value)
    assert condition.name in message
    assert "available_space" in message
    assert "width" in message
    assert "got 0" in message


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

# --------------------------------------------------------------- Area 4
# Experiment condition and inheritance validity (boundary-configuration-validation)

CONTROL_CONDITION = {"name": "control", "parameter": None, "value": None}


def _base_block() -> dict:
    """A fresh copy of the shipped base block, so no test can mutate another's."""
    return json.loads((CONFIGS / "base.json").read_text())["base"]


def _write_config(directory: Path, name: str, doc: dict) -> Path:
    path = Path(directory) / name
    path.write_text(json.dumps(doc))
    return path


# ---------------------------------------------------- Area 4 exploration test
# Property 7: Bug Condition — malformed experiment documents and unknown
# condition names are rejected, with the offending name/key and the file path
# or the sorted list of available names in the message.


def test_duplicate_condition_names_rejected(tmp_path):
    """Two conditions with the same name must be rejected at construction.

    Measured pre-fix behavior at HEAD ea8a9c2: NO ERROR. The experiment loads
    with three conditions, ``ExperimentConfig.condition('food_low')`` returns
    only the FIRST (value 20, so the second is unreachable), and
    ``run_condition`` writes both to ``<out>/<experiment>/food_low``, the later
    trial silently overwriting the earlier one.

    Expected post-fix: ValueError naming the duplicated name.
    """
    path = _write_config(
        tmp_path,
        "duplicate.json",
        {
            "name": "duplicate",
            "base": _base_block(),
            "seeds": [0],
            "conditions": [
                CONTROL_CONDITION,
                {"name": "food_low", "parameter": "food_abundance", "value": 20},
                {"name": "food_low", "parameter": "food_abundance", "value": 120},
            ],
        },
    )

    with pytest.raises(ValueError) as excinfo:
        load_experiment(path)

    assert "food_low" in str(excinfo.value)


def test_zero_controls_rejected(tmp_path):
    """An experiment with no control must be rejected.

    Measured pre-fix behavior at HEAD ea8a9c2: NO ERROR.
    ``[c.name for c in experiment.conditions if c.is_control]`` is ``[]`` while
    ``experiments/report.py`` and ``analysis.compare_conditions`` key off a
    condition named ``"control"``, so comparison silently degrades.

    Expected post-fix: ValueError stating how many controls were found and that
    exactly one is required.
    """
    path = _write_config(
        tmp_path,
        "zero_controls.json",
        {
            "name": "zero_controls",
            "base": _base_block(),
            "seeds": [0],
            "conditions": [
                {"name": "food_low", "parameter": "food_abundance", "value": 20},
                {"name": "food_high", "parameter": "food_abundance", "value": 120},
            ],
        },
    )

    with pytest.raises(ValueError) as excinfo:
        load_experiment(path)

    message = str(excinfo.value)
    assert "0" in message
    assert "exactly one" in message


def test_two_controls_rejected(tmp_path):
    """An experiment with two controls must be rejected, listing both.

    Measured pre-fix behavior at HEAD ea8a9c2: NO ERROR. Both ``control`` and
    ``control_two`` report ``is_control``, and the reporting layer silently
    picks the one literally named ``"control"``.

    Expected post-fix: ValueError listing the control names.
    """
    path = _write_config(
        tmp_path,
        "two_controls.json",
        {
            "name": "two_controls",
            "base": _base_block(),
            "seeds": [0],
            "conditions": [
                CONTROL_CONDITION,
                {"name": "control_two", "parameter": None, "value": None},
                {"name": "food_low", "parameter": "food_abundance", "value": 20},
            ],
        },
    )

    with pytest.raises(ValueError) as excinfo:
        load_experiment(path)

    message = str(excinfo.value)
    assert "2" in message
    assert "control" in message
    assert "control_two" in message


def test_unknown_requested_condition_name_rejected(tmp_path):
    """Requesting a condition the experiment does not define must raise.

    Measured pre-fix behavior at HEAD ea8a9c2: ``run_experiment`` returns
    ``{}`` — indistinguishable from a successful run of zero conditions.

    Expected post-fix: ValueError naming the unknown name and listing the
    available names, sorted so the message is deterministic.
    """
    path = _write_config(
        tmp_path,
        "runnable.json",
        {
            "name": "runnable",
            "base": _base_block(),
            "seeds": [0],
            "conditions": [
                CONTROL_CONDITION,
                {"name": "food_low", "parameter": "food_abundance", "value": 20},
            ],
        },
    )

    with pytest.raises(ValueError) as excinfo:
        run_experiment(path, tmp_path / "out", ["typo-name"])

    message = str(excinfo.value)
    assert "typo-name" in message
    assert "control" in message
    assert "food_low" in message
    assert message.index("control") < message.index("food_low")  # sorted
    assert not (tmp_path / "out").exists()  # nothing was run


def test_unknown_requested_condition_name_rejected_through_the_cli(tmp_path):
    """The CLI must fail loudly instead of printing '0 condition(s)' and exiting 0.

    Measured pre-fix behavior at HEAD ea8a9c2: ``main`` returns None after
    printing ``experiment runnable: 0 condition(s)``, i.e. exit code 0.
    """
    path = _write_config(
        tmp_path,
        "runnable_cli.json",
        {
            "name": "runnable_cli",
            "base": _base_block(),
            "seeds": [0],
            "conditions": [CONTROL_CONDITION],
        },
    )

    with pytest.raises(ValueError) as excinfo:
        main(
            [
                "--config",
                str(path),
                "--out",
                str(tmp_path / "cli_out"),
                "--conditions",
                "typo-name",
            ]
        )

    assert "typo-name" in str(excinfo.value)


def test_nested_extends_rejected(tmp_path):
    """An ``extends`` target that itself only extends must be rejected by name.

    Measured pre-fix behavior at HEAD ea8a9c2: bare ``KeyError: 'base'`` at
    ``experiments/config.py:152``, because ``parent["base"]`` is read
    unconditionally. The message names neither file.

    Expected post-fix: ValueError stating that only one level of ``extends`` is
    supported and naming BOTH files.
    """
    _write_config(
        tmp_path,
        "grandparent.json",
        {
            "name": "grandparent",
            "base": _base_block(),
            "seeds": [7],
            "conditions": [CONTROL_CONDITION],
        },
    )
    mid = _write_config(
        tmp_path,
        "mid.json",
        {
            "name": "mid",
            "extends": "grandparent.json",
            "conditions": [CONTROL_CONDITION],
        },
    )
    child = _write_config(
        tmp_path,
        "child.json",
        {"name": "child", "extends": "mid.json", "conditions": [CONTROL_CONDITION]},
    )

    with pytest.raises(ValueError) as excinfo:
        load_experiment(child)

    message = str(excinfo.value)
    assert str(child) in message
    assert str(mid) in message
    assert "extends" in message


def test_explicitly_empty_seeds_rejected(tmp_path):
    """``"seeds": []`` must be rejected, never silently replaced.

    Measured pre-fix behavior at HEAD ea8a9c2: NO ERROR. ``raw.get("seeds") or
    (...)`` treats ``[]`` as absent, so the child silently inherits the
    parent's ``[11, 12]`` and runs two trials the file never asked for.

    Expected post-fix: ValueError naming the file — an empty seed set means
    zero trials, which is the zero-trial silent success 2.20 forbids.
    """
    _write_config(
        tmp_path,
        "seed_parent.json",
        {
            "name": "seed_parent",
            "base": _base_block(),
            "seeds": [11, 12],
            "conditions": [CONTROL_CONDITION],
        },
    )
    child = _write_config(
        tmp_path,
        "empty_seeds.json",
        {
            "name": "empty_seeds",
            "extends": "seed_parent.json",
            "seeds": [],
            "conditions": [CONTROL_CONDITION],
        },
    )

    with pytest.raises(ValueError) as excinfo:
        load_experiment(child)

    message = str(excinfo.value)
    assert str(child) in message
    assert "seeds" in message


def test_explicitly_empty_seeds_rejected_without_extends(tmp_path):
    """The same rule with no ``extends``: ``[]`` must not become the default.

    Measured pre-fix behavior at HEAD ea8a9c2: NO ERROR; the seeds silently
    become ``[0, 1, 2, 3, 4]``.
    """
    path = _write_config(
        tmp_path,
        "empty_seeds_standalone.json",
        {
            "name": "empty_seeds_standalone",
            "base": _base_block(),
            "seeds": [],
            "conditions": [CONTROL_CONDITION],
        },
    )

    with pytest.raises(ValueError) as excinfo:
        load_experiment(path)

    assert "seeds" in str(excinfo.value)


def test_missing_conditions_key_rejected(tmp_path):
    """A document with no ``conditions`` must name the file and the key.

    Measured pre-fix behavior at HEAD ea8a9c2: bare
    ``KeyError: 'conditions'`` at ``experiments/config.py:159``, naming neither
    the file nor the schema.

    Expected post-fix: ValueError naming the file path and the missing key.
    """
    path = _write_config(
        tmp_path,
        "no_conditions.json",
        {"name": "no_conditions", "base": _base_block(), "seeds": [0]},
    )

    with pytest.raises(ValueError) as excinfo:
        load_experiment(path)

    message = str(excinfo.value)
    assert str(path) in message
    assert "conditions" in message


# --------------------------------------------------- Area 4 preservation test
# Property 8: Preservation — the six shipped configs, the existing rejections,
# the record schemas and the absent-``seeds`` inheritance rule are unchanged.
# Digests measured on UNFIXED code at HEAD ea8a9c2 (oracle 4 of task 1):
# sha1(json.dumps(experiment.to_dict(), sort_keys=True)), first 16 hex chars.

SHIPPED_CONFIG_DIGESTS = {
    "available_space.json": "5fc4b7e3885d04cb",
    "base.json": "2450ff5be01bc655",
    "food_abundance.json": "ff5303f160c6dc58",
    "food_regeneration.json": "e061dcb71aac265e",
    "population_density.json": "83572b785cb6c4f3",
    "reproduction_cost.json": "03669c6162ab5bb3",
}


def test_shipped_configs_still_resolve_to_the_recorded_digests():
    """All six shipped configs load, validate, and serialize bit-identically."""
    paths = sorted(CONFIGS.glob("*.json"))
    assert {p.name for p in paths} == set(SHIPPED_CONFIG_DIGESTS)

    for path in paths:
        experiment = load_experiment(path)
        payload = json.dumps(experiment.to_dict(), sort_keys=True)
        digest = hashlib.sha1(payload.encode()).hexdigest()[:16]
        assert digest == SHIPPED_CONFIG_DIGESTS[path.name], path.name


def test_shipped_configs_have_five_seeds_and_exactly_one_control():
    """Every shipped config already satisfies the new control-count rule."""
    for path in sorted(CONFIGS.glob("*.json")):
        experiment = load_experiment(path)
        assert experiment.seeds == [0, 1, 2, 3, 4], path.name
        controls = [c.name for c in experiment.conditions if c.is_control]
        assert controls == ["control"], path.name
        names = [c.name for c in experiment.conditions]
        assert len(names) == len(set(names)), path.name


def test_unknown_parameter_still_reported_as_a_parameter_problem():
    """The ordering tripwire for task 17.1.

    ``test_unknown_parameter_rejected`` builds a config with a single
    non-control condition, i.e. ZERO controls. It asserts only
    ``pytest.raises(ValueError)``, so it would keep passing even if the new
    control-count rule fired first. This test pins the ORIGINAL reason: the
    unknown-``parameter`` check must stay FIRST in ``_validate``.
    """
    with pytest.raises(ValueError) as excinfo:
        ExperimentConfig(
            name="bad",
            base={"neat": {}, "world": {}, "interface": {"input_ids": [], "output_ids": []}},
            conditions=[Condition(name="x", parameter="not_a_parameter", value=1)],
            seeds=[0],
        )

    message = str(excinfo.value)
    assert "parameter" in message
    assert "not_a_parameter" in message
    assert "control" not in message  # not the control-count rule


def test_missing_extends_target_still_raises_file_not_found(tmp_path):
    """A missing ``extends`` target keeps its plain stdlib error (3.16)."""
    child = _write_config(
        tmp_path,
        "missing_parent.json",
        {
            "name": "missing_parent",
            "extends": "nope.json",
            "conditions": [CONTROL_CONDITION],
        },
    )

    with pytest.raises(FileNotFoundError) as excinfo:
        load_experiment(child)

    assert excinfo.value.errno == errno.ENOENT
    assert excinfo.value.filename == str(tmp_path / "nope.json")
    # not gold-plated: the message is exactly the stdlib one, nothing bolted on
    assert str(excinfo.value) == (
        f"[Errno {excinfo.value.errno}] {excinfo.value.strerror}: "
        f"{excinfo.value.filename!r}"
    )


def test_absent_seeds_still_inherits_from_the_parent(tmp_path):
    """Key ABSENT with ``extends`` set: inherit the parent's seeds, as today."""
    _write_config(
        tmp_path,
        "inherit_parent.json",
        {
            "name": "inherit_parent",
            "base": _base_block(),
            "seeds": [11, 12],
            "conditions": [CONTROL_CONDITION],
        },
    )
    child = _write_config(
        tmp_path,
        "inherit_child.json",
        {
            "name": "inherit_child",
            "extends": "inherit_parent.json",
            "conditions": [CONTROL_CONDITION],
        },
    )

    assert load_experiment(child).seeds == [11, 12]


def test_absent_seeds_without_extends_still_defaults_to_five(tmp_path):
    """Key ABSENT with no ``extends``: the documented ``[0, 1, 2, 3, 4]``."""
    path = _write_config(
        tmp_path,
        "default_seeds.json",
        {
            "name": "default_seeds",
            "base": _base_block(),
            "conditions": [CONTROL_CONDITION],
        },
    )

    assert load_experiment(path).seeds == [0, 1, 2, 3, 4]


def test_record_fields_and_resolved_config_shape_unchanged():
    """``RECORD_FIELDS`` (3.12) and ``ResolvedConfig.to_dict`` (3.14) are frozen."""
    assert RECORD_FIELDS == [
        "generation",
        "population_size",
        "survival_rate",
        "average_lifetime",
        "best_fitness",
        "mean_fitness",
        "food_consumed",
        "reproductive_success",
        "node_count",
        "connection_count",
        "species_count",
        "action_entropy",
        "action_entropy_diversity",
        "transition_entropy",
        "spatial_coverage",
        "food_alignment",
        "encounter_rate",
        "behavioral_diversity",
    ]

    experiment = load_experiment(CONFIGS / "base.json")
    resolved = resolve_config(experiment, experiment.condition("control"), seed=0)
    payload = resolved.to_dict()
    assert list(payload) == ["neat", "world", "interface"]
    assert list(payload["neat"]) == [
        "population_size",
        "generations",
        "elitism",
        "crossover_rate",
        "mutation",
        "speciation",
    ]
    assert list(payload["interface"]) == ["input_ids", "output_ids"]


def test_food_abundance_alias_untouched():
    """The ``food_abundance`` / ``resource_scarcity`` alias is left as it is (3.19)."""
    assert PARAMETERS["resource_scarcity"] == PARAMETERS["food_abundance"]
    assert PARAMETERS["food_abundance"] == ("world.initial_food", "world.food_target")
