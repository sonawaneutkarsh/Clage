import pytest

from benchmarks.diagnose import check_fitness_design, check_network_execution, _hand_solution
from benchmarks.problems import AND, OR, PROBLEMS, SIN, XOR
from benchmarks.run import run_trial


def test_problem_definitions_well_formed():
    for problem in PROBLEMS.values():
        from neat.genome import Genome

        empty = Genome.minimal(
            input_ids=list(problem.input_ids), output_ids=list(problem.output_ids)
        )
        fitness = problem.fitness_fn(empty, 0)
        assert 0.0 < fitness <= 1.0
        assert problem.success_fn(empty) is False  # nothing is solved from scratch
        for inputs, _ in problem.cases:
            assert len(inputs) == len(problem.input_ids)


def test_hand_built_solutions_solve_and_score_higher():
    for problem in (OR, AND, XOR):
        ok, message = check_fitness_design(problem)
        assert ok, message


def test_network_execution_probe_passes():
    ok, message = check_network_execution()
    assert ok, message


def test_hand_built_xor_classifies_all_rows():
    solution = _hand_solution(XOR)
    assert XOR.success_fn(solution) is True


def test_run_trial_history_is_well_formed():
    result = run_trial(AND, seed=0, population_size=4, max_generations=3)
    assert result.problem == "AND"
    assert result.total_generations == 3
    assert len(result.history) == 3
    for i, row in enumerate(result.history, start=1):
        assert row["generation"] == i
        assert set(row) == {
            "generation", "best_fitness", "mean_fitness", "species_count",
            "best_node_count", "best_connection_count", "solved",
        }
        assert row["best_node_count"] >= 2
        assert row["best_connection_count"] >= 0


def test_run_trial_is_deterministic():
    a = run_trial(AND, seed=5, population_size=4, max_generations=3)
    b = run_trial(AND, seed=5, population_size=4, max_generations=3)
    assert a.history == b.history


def test_sin_problem_has_21_samples():
    assert len(SIN.cases) == 21
    assert all(len(inputs) == 1 for inputs, _ in SIN.cases)
