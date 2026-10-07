import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from neat.genome import Genome
from visual.analytics import plot_metric, environment_table
from visual.app import main
from visual.network_view import draw_genome
from visual.terminal_view import play, render_frame, world_grid_lines, _visible_width
from visual.world_view import draw_frame, WorldViewer, FACING_VECTORS
from world import EnvironmentConfig, GenerationRecorder, run_generation


@pytest.fixture
def recording():
    genomes = [Genome.minimal(range(9), range(10, 14))]
    config = EnvironmentConfig(width=5, height=7, ticks=3, initial_food=2, food_target=2)
    recorder = GenerationRecorder(genomes, config, 0)
    run_generation(genomes, config, 0, recorder=recorder)
    return recorder.to_dict()


def test_world_orientation_and_control_state(recording):
    fig, ax = plt.subplots()
    draw_frame(ax, recording, 0)
    assert ax.yaxis_inverted()
    assert FACING_VECTORS["N"] == (0, -1)
    viewer = WorldViewer(recording)
    assert viewer.ax.get_title().startswith("generation 0   tick 0")
    assert viewer.play_button.label.get_text() == "pause"
    viewer._on_play(None)
    assert not viewer.playing
    viewer._on_step(None)
    assert viewer.tick_index == 1 and not viewer.playing
    viewer.slider.set_val(len(recording["ticks"]) - 1)
    viewer._animate(0)
    assert not viewer.playing and viewer.play_button.label.get_text() == "play"
    viewer._anim._draw_was_started = True
    plt.close("all")


def test_tui_cell_width_selection_cursor_and_fitness_labels(recording):
    lines = world_grid_lines(recording, 0, cursor=(0, 0))
    assert lines[0][0] == "×"
    assert all(_visible_width(line) == recording["config"]["width"] for line in lines)
    frame = render_frame(recording, 0, selected=0)
    assert "Body:     0" in frame and "%" in frame
    assert "Final Fitness (evaluated)" in frame


@pytest.mark.parametrize("fps", [0, -1, math.nan, math.inf])
def test_invalid_tui_fps_fails_before_terminal_setup(recording, fps):
    with pytest.raises(ValueError, match="fps"):
        play(recording, fps)


def test_network_legend_unwired_labels_and_export_alias(recording, tmp_path):
    fig, ax = plt.subplots()
    draw_genome(ax, recording["genomes"][0])
    assert "Unwired" in ax.get_title()
    assert {text.get_text() for text in ax.get_legend().get_texts()} == {
        "positive weight", "negative weight", "disabled"}
    path = tmp_path / "recording.json"
    path.write_text(json.dumps(recording))
    for flag in ("--genome", "--organism"):
        target = tmp_path / flag / "network.png"
        main(["network", "--recording", str(path), flag, "0", "--export", str(target)])
        assert target.stat().st_size > 1000
    plt.close("all")


def test_unknown_metric_rejected_and_table_transposed(tmp_path):
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="unknown metric"):
        plot_metric(ax, tmp_path, [], "not_a_metric")
    directory = tmp_path / "control"
    directory.mkdir()
    (directory / "0.config.json").write_text(json.dumps({"neat": {"population_size": 4}, "world": {"width": 5}}))
    fig = environment_table(tmp_path, ["control"])
    table = next(iter(fig.axes[0].tables))
    assert table[(0, 0)].get_text().get_text() == "parameter"
    assert table[(0, 1)].get_text().get_text() == "control"
    plt.close("all")
