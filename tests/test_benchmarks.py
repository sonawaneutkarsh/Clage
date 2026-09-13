import pytest

from benchmarks.diagnose import check_fitness_design, check_network_execution, _hand_solution
from benchmarks.problems import AND, OR, PROBLEMS, SIN, XOR, Problem
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


def test_unevaluated_offspring_cannot_declare_success():
    """AUTHORITATIVE Bug 1 regression: an unevaluated offspring cannot declare success.

    Construction. A synthetic probe ``Problem`` reuses ``AND``'s ``input_ids`` /
    ``output_ids`` / ``cases`` but scores every evaluated genome ``-1.0`` and calls a
    genome successful when ``fitness >= 0.0``. Crossover children are constructed fresh
    carrying ``fitness == 0.0``, so with negative evaluated fitness every child outranks
    the real champion and the post-reproduction argmax is guaranteed to be an
    unevaluated offspring. Population 6 with one species and ``crossover_rate=0.75``
    makes at least one such child deterministic per seed; seed 0 is confirmed. If the
    seed changes, the pre-fix failure has to be re-confirmed and recorded here.

    MEASURED PRE-FIX FAILURE (seeds 0/1/2/5 all identical): ``solved=True``,
    ``generations_to_solve=1``, ``len(history) == 1``, and the single row reads
    ``best_fitness=-1.0`` with ``solved=True`` — success declared against the criterion
    ``fitness >= 0.0`` on a row whose recorded fitness is ``-1.0``, with the remaining
    generations discarded. The first assertion (``solved is False``) is the one that
    fails.

    This test depends on NO reference identity. The contradiction between
    ``solved=True`` and ``best_fitness=-1.0`` under ``fitness >= 0.0`` is fully
    observable in values, which is why it is the authoritative proof of Bug 1 even
    though the public accessor hands out a defensive copy: identity assertions against
    engine internals are meaningless by design.

    Scope guard: the negative fitness here exists only to expose Bug 1. It must not
    turn into a fix for speciation's negative-fitness offspring allocation, which is
    explicitly out of scope.

    **Validates: Requirements 1.2, 1.4, 2.2, 2.13**
    """
    probe = Problem(
        name="probe-negative",
        input_ids=AND.input_ids,
        output_ids=AND.output_ids,
        cases=AND.cases,
        fitness_fn=lambda g, generation: -1.0,
        success_fn=lambda g: g.fitness >= 0.0,
        success_description="never solvable: evaluated fitness is always -1.0",
    )

    result = run_trial(probe, seed=0, population_size=6, max_generations=3)

    assert result.solved is False
    assert result.generations_to_solve is None
    assert len(result.history) == 3
    assert result.total_generations == 3
    for row in result.history:
        assert row["solved"] is False


def test_history_row_describes_the_evaluated_genome():
    """SEMANTIC CONSISTENCY GUARD for Bug 1 — NOT a bug discriminator.

    THIS TEST PASSES BOTH PRE- AND POST-FIX ON ``AND``, AND THAT IS EXPECTED. The
    defect is latent there: ``_reproduce_species`` prepends a structural
    ``elite.copy()`` that preserves fitness, and ``max`` returns the first maximal
    element, so the wrong object carries the right values — measured ``nodes 3``,
    ``conns 0``, ``fitness 0.8`` coincide in every generation. A green pre-fix run MUST
    NOT be read as evidence the bug is absent. The discriminator is
    ``test_unevaluated_offspring_cannot_declare_success``.

    This test's job is to lock the semantic contract going forward — every champion
    field in a history row agrees with the genome the engine actually evaluated that
    generation — so a future change that re-splits the field sourcing gets caught.

    Oracle alignment. The tracking ``fitness_fn`` records ``(genome, fitness)`` keyed by
    the ``generation`` argument, so oracle keys are ``0..3`` while history ``generation``
    values are ``1..4``; they are aligned positionally via ``sorted(evaluated)``. The
    tracking ``fitness_fn`` is called in population order and ``_evaluate`` computes
    ``max(self.population, ...)`` in that same order, so ``max`` breaks ties on the same
    element and the oracle champion is the genome the engine evaluated.

    Values only: no identity assertion appears anywhere in this test. Post-fix the
    accessor and ``success_fn`` receive a defensive snapshot, so identity comparisons
    against engine internals are meaningless by design.

    **Validates: Requirements 1.1, 1.3, 1.4, 2.1, 2.13**
    """
    evaluated = {}
    handed_to_success = []

    def tracking_fitness(genome, generation):
        fitness = AND.fitness_fn(genome, generation)
        evaluated.setdefault(generation, []).append((genome, fitness))
        return fitness

    def tracking_success(genome):
        # Recorded ONLY to assert value agreement — never identity.
        handed_to_success.append((genome.fitness, len(genome.nodes), len(genome.connections)))
        return AND.success_fn(genome)

    probe = Problem(
        name="probe-and",
        input_ids=AND.input_ids,
        output_ids=AND.output_ids,
        cases=AND.cases,
        fitness_fn=tracking_fitness,
        success_fn=tracking_success,
        success_description=AND.success_description,
    )

    result = run_trial(probe, seed=0, population_size=6, max_generations=4)

    keys = sorted(evaluated)
    assert len(keys) == len(result.history)

    for key, row in zip(keys, result.history):
        genome, fitness = max(evaluated[key], key=lambda pair: pair[1])
        assert row["best_fitness"] == fitness
        assert row["best_node_count"] == len(genome.nodes)
        assert row["best_connection_count"] == len(genome.connections)
        assert row["solved"] == AND.success_fn(genome)

    # Value agreement for what success_fn was handed, one call per recorded row.
    assert len(handed_to_success) == len(result.history)
    for (seen_fitness, seen_nodes, seen_conns), row in zip(handed_to_success, result.history):
        assert seen_fitness == row["best_fitness"]
        assert seen_nodes == row["best_node_count"]
        assert seen_conns == row["best_connection_count"]
