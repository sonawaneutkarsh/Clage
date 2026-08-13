import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pytest

from experiments.config import Condition, ExperimentConfig
from experiments.run import run_condition
from neat.genome import Genome
from world import EnvironmentConfig, GenerationRecorder, run_generation

from visual.data import (
    METRICS,
    aggregate_condition,
    environmental_params,
    load_condition_configs,
    load_condition_trials,
    load_recording,
)
from visual.network_view import layered_layout
from visual.world_view import draw_frame, export_tick

INPUT_IDS = list(range(9))
OUTPUT_IDS = [10, 11, 12, 13]


# ------------------------------------------------------------------ recorder


def test_recorder_structure():
    config = EnvironmentConfig(
        width=6, height=6, ticks=5, initial_food=5, food_target=5, repro_threshold=2.0
    )
    population = [Genome.minimal(input_ids=INPUT_IDS, output_ids=OUTPUT_IDS) for _ in range(2)]
    recorder = GenerationRecorder(population, config, generation=3)
    run_generation(population, config, 3, recorder=recorder)

    recording = recorder.to_dict()
    assert recording["schema"] == "clage-generation-replay"
    assert recording["generation"] == 3
    assert len(recording["ticks"]) == 6  # initial placement + 5 ticks
    assert len(recording["genomes"]) == 2
    for tick in recording["ticks"]:
        assert "food" in tick and "organisms" in tick
        for organism in tick["organisms"]:
            assert 0 <= organism["genome"] < 2
            assert isinstance(organism["id"], int)


def test_recorder_no_overhead_without_recorder():
    # run_generation without a recorder must still work and not attach anything
    config = EnvironmentConfig(width=6, height=6, ticks=3, repro_threshold=2.0)
    population = [Genome.minimal(input_ids=INPUT_IDS, output_ids=OUTPUT_IDS) for _ in range(2)]
    organisms = run_generation(population, config, 0)
    assert len(organisms) == 2
    assert all(g.fitness > 0 for g in population)


# ------------------------------------------------------------------ data loaders


def test_load_recording_round_trip(tmp_path):
    config = EnvironmentConfig(width=4, height=4, ticks=2, repro_threshold=2.0)
    population = [Genome.minimal(input_ids=INPUT_IDS, output_ids=OUTPUT_IDS)]
    recorder = GenerationRecorder(population, config, generation=0)
    run_generation(population, config, 0, recorder=recorder)

    path = tmp_path / "recording.json"
    path.write_text(json.dumps(recorder.to_dict()))
    loaded = load_recording(path)
    assert loaded["generation"] == 0
    assert len(loaded["ticks"]) == 3


def _write_experiment(tmp_path, seeds=2):
    directory = tmp_path / "control"
    directory.mkdir(parents=True, exist_ok=True)
    for seed in range(seeds):
        records = []
        for generation in range(1, 4):
            record = {"generation": generation, "best_fitness": float(seed + generation)}
            for metric in METRICS:
                record.setdefault(metric, 0.5)
            records.append(record)
        (directory / f"{seed}.json").write_text(json.dumps(records))
    config = {
        "world": {"width": 20, "food_target": 60, "food_regrowth_per_tick": 1},
        "neat": {"population_size": 20},
    }
    (directory / "0.config.json").write_text(json.dumps(config))
    (directory / "1.config.json").write_text(json.dumps(config))


def test_experiment_loaders_and_aggregation(tmp_path):
    _write_experiment(tmp_path)

    trials = load_condition_trials(tmp_path, "control")
    assert len(trials) == 2
    aggregated = aggregate_condition(trials)
    # generation 2: best_fitness values 2 and 3 -> mean 2.5
    assert aggregated[2]["best_fitness"]["mean"] == pytest.approx(2.5)
    assert aggregated[2]["best_fitness"]["std"] == pytest.approx(0.70710678, abs=1e-6)

    params = environmental_params(load_condition_configs(tmp_path, "control"))
    assert params["food_target"] == 60


# ------------------------------------------------------------------ network view


def _tiny_genome():
    return {
        "nodes": [
            {"id": 0, "type": "INPUT", "bias": 0.0},
            {"id": 1, "type": "INPUT", "bias": 0.0},
            {"id": 50, "type": "HIDDEN", "bias": 0.0},
            {"id": 10, "type": "OUTPUT", "bias": 0.0},
        ],
        "connections": [
            {"in": 0, "out": 50, "weight": 1.0, "enabled": True, "innovation": 1},
            {"in": 1, "out": 50, "weight": 1.0, "enabled": True, "innovation": 2},
            {"in": 50, "out": 10, "weight": 1.0, "enabled": True, "innovation": 3},
            {"in": 0, "out": 10, "weight": 1.0, "enabled": False, "innovation": 4},
        ],
    }


def test_network_layered_layout_deterministic():
    genome = _tiny_genome()
    positions = layered_layout(genome)
    assert positions[0][0] == 0.0 and positions[1][0] == 0.0  # inputs at layer 0
    assert positions[50][0] == 1.0  # hidden at layer 1
    assert positions[10][0] == 2.0  # output at layer 2
    assert layered_layout(genome) == positions  # deterministic


def test_network_figure_renders():
    from visual.network_view import genome_figure

    fig = genome_figure(_tiny_genome())
    fig.canvas.draw()  # must not raise
    import matplotlib.pyplot as plt

    plt.close(fig)


# ------------------------------------------------------------------ replay view


def test_draw_frame_and_export(tmp_path):
    config = EnvironmentConfig(width=5, height=5, ticks=3, repro_threshold=2.0)
    population = [Genome.minimal(input_ids=INPUT_IDS, output_ids=OUTPUT_IDS)]
    recorder = GenerationRecorder(population, config, generation=1)
    run_generation(population, config, 1, recorder=recorder)
    recording = recorder.to_dict()

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    stats = draw_frame(ax, recording, tick_index=1)
    assert stats["live"] == stats["total"]  # no reproduction here, none dead
    plt.close(fig)

    path = tmp_path / "tick.png"
    export_tick(recording, 1, path)
    assert path.exists() and path.stat().st_size > 0


def test_experiments_runner_records_replay(tmp_path):
    base = json.loads(Path("experiments/configs/base.json").read_text())["base"]
    base["neat"]["generations"] = 3
    base["neat"]["population_size"] = 4
    base["world"]["ticks"] = 8
    experiment = ExperimentConfig(
        name="tiny",
        base=base,
        conditions=[Condition(name="control", parameter=None)],
        seeds=[0],
    )
    run_condition(experiment, experiment.condition("control"), tmp_path, record_generation=2)

    recording_path = tmp_path / "tiny" / "recordings" / "control" / "0.json"
    assert recording_path.exists()
    recording = load_recording(recording_path)
    assert recording["generation"] == 2
    assert len(recording["ticks"]) == 9  # initial + 8 ticks
    # the runner annotates the recorded generation's population context
    assert recording["context"]["species_count"] >= 1
    assert recording["context"]["survival_rate"] >= 0.0


# ------------------------------------------------------------------ terminal view


def _tiny_recording():
    config = EnvironmentConfig(width=5, height=5, ticks=3, repro_threshold=2.0)
    population = [Genome.minimal(input_ids=INPUT_IDS, output_ids=OUTPUT_IDS)]
    recorder = GenerationRecorder(population, config, generation=1)
    run_generation(population, config, 1, recorder=recorder)
    return recorder.to_dict()


def test_tui_render_frame_structure():
    from visual.terminal_view import _visible_width, render_frame

    recording = _tiny_recording()
    frame = render_frame(recording, tick_index=1)
    assert "CLAGE" in frame
    assert "┌" in frame and "┐" in frame and "└" in frame and "┘" in frame
    lines = frame.splitlines()
    widths = {_visible_width(line) for line in lines}
    assert len(widths) == 1  # every border/content line aligns to the same width


def test_tui_render_selected_organism_panel():
    from visual.terminal_view import render_frame

    recording = _tiny_recording()
    first_id = recording["ticks"][1]["organisms"][0]["id"]
    frame = render_frame(recording, tick_index=1, selected=first_id)
    assert "Selected Organism" in frame
    assert "Energy:" in frame
    assert "Neural Network" in frame


def test_tui_export_frame(tmp_path):
    from visual.terminal_view import export_frame

    recording = _tiny_recording()
    path = tmp_path / "frame.txt"
    export_frame(recording, 1, path)
    text = path.read_text()
    assert "CLAGE" in text and "┌" in text


def test_tui_helpers():
    from visual.terminal_view import network_ascii_lines, world_grid_lines

    recording = _tiny_recording()
    grid = world_grid_lines(recording, tick_index=0)
    assert grid and all(len(row) == len(grid[0]) for row in grid)
    genome = recording["genomes"][0]
    net = network_ascii_lines(genome)
    assert "".join(net).count("●") >= 2  # inputs + output nodes


def test_metric_list_matches_experiment_fields():
    # Guard against drift between the two metric "sources of truth".
    from experiments.run import RECORD_FIELDS
    from visual.data import METRICS

    assert METRICS == [field for field in RECORD_FIELDS if field != "generation"]


def test_analytics_excludes_food_alignment_by_default():
    from visual.analytics import DEFAULT_METRICS

    assert "food_alignment" not in DEFAULT_METRICS


# ------------------------------------------------------------------ purity


def test_viewer_has_no_evolutionary_logic_imports():
    sources = ""
    for path in sorted(Path("visual").glob("*.py")):
        sources += path.read_text()
    for forbidden in (
        "import neat", "from neat",
        "import world", "from world",
        "import experiments", "from experiments",
    ):
        assert forbidden not in sources
