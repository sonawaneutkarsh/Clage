import inspect
import math

import pytest

import neat.population
from neat.genome import ConnectionGene, Genome, NodeGene, NodeType
from neat.mutation import MutationConfig
from neat.population import Population
from neat.speciation import SpeciationConfig


def count_fitness(genome, generation):
    return float(len(genome.connections))


def constant_fitness(genome, generation):
    return 0.0


def wired_a():
    g = Genome.minimal(input_ids=[0, 1], output_ids=[10])
    g.add_connection(ConnectionGene(0, 10, 1.0, innovation=1))
    g.add_connection(ConnectionGene(1, 10, 2.0, innovation=2))
    return g


def wired_b():
    g = Genome.minimal(input_ids=[0, 1], output_ids=[10, 11])
    g.add_connection(ConnectionGene(0, 10, 1.0, innovation=1))
    g.add_connection(ConnectionGene(0, 11, 3.0, innovation=3))
    g.add_connection(ConnectionGene(1, 10, 4.0, innovation=4))
    return g


def test_initial_population():
    pop = Population(count_fitness, population_size=6, seed=0)
    assert len(pop.population) == 6
    assert pop.generation == 0
    assert all(isinstance(g, Genome) for g in pop.population)
    for g in pop.population:
        g.validate()


def test_run_advances_generation():
    pop = Population(count_fitness, population_size=4, seed=0)
    pop.run(3)
    assert pop.generation == 3


def test_population_size_constant():
    pop = Population(count_fitness, population_size=5, seed=1,
                     mutation_config=MutationConfig(add_connection_prob=0.3))
    for _ in range(4):
        pop.run(1)
        assert len(pop.population) == pop.population_size


def test_fitness_function_called_per_genome_per_generation():
    calls = []

    def fitness(genome, generation):
        calls.append(generation)
        return 0.0

    pop = Population(fitness, population_size=5, seed=0)
    pop.run(2)
    assert len(calls) == 10


def test_best_genome_tracked_and_isolated():
    def fitness(genome, generation):
        return sum(abs(c.weight) for c in genome.connections)

    pop = Population(fitness, population_size=8, seed=1,
                     mutation_config=MutationConfig(add_connection_prob=0.5, add_node_prob=0.1))
    pop.run(3)

    assert pop.best_genome is not None
    assert pop.best_fitness == pop.statistics[-1]["best_fitness"]
    assert all(g is not pop.best_genome for g in pop.population)

    pop.best_genome.nodes[0].bias = 42.0
    assert all(g.nodes[0].bias != 42.0 for g in pop.population)


def test_elitism_best_non_decreasing():
    pop = Population(count_fitness, population_size=6, seed=0,
                     mutation_config=MutationConfig(add_connection_prob=0.3, add_node_prob=0.05))
    pop.run(6)
    bests = [s["best_fitness"] for s in pop.statistics]
    assert all(bests[i + 1] >= bests[i] for i in range(len(bests) - 1))


def test_mutation_applied_to_offspring():
    pop = Population(
        count_fitness,
        population_size=4,
        seed=0,
        mutation_config=MutationConfig(
            weight_prob=0.0,
            bias_prob=0.0,
            add_connection_prob=1.0,
            add_node_prob=0.0,
            enable_connection_prob=0.0,
            disable_connection_prob=0.0,
        ),
    )
    pop.run(1)
    counts = [len(g.connections) for g in pop.population]
    assert max(counts) == 1  # elites (0 conns) survive; every offspring gained exactly one
    assert sum(counts) == 3  # population 4 = 1 elite + 3 offspring


def test_crossover_produces_valid_children_from_parent_genes():
    pop = Population(
        count_fitness,
        population_size=2,
        initial_population=[wired_a(), wired_b()],
        seed=0,
        elitism=0,
        crossover_rate=1.0,
        mutation_config=MutationConfig(
            weight_prob=0.0,
            bias_prob=0.0,
            add_connection_prob=0.0,
            add_node_prob=0.0,
            enable_connection_prob=0.0,
            disable_connection_prob=0.0,
        ),
    )
    pop.run(1)

    parent_innovs = {1, 2, 3, 4}
    assert len(pop.population) == 2
    for genome in pop.population:
        genome.validate()
        innovs = {c.innovation for c in genome.connections}
        assert innovs <= parent_innovs
        assert 1 in innovs  # matching gene always inherited


def test_species_management_sizes_sum_to_population():
    pop = Population(count_fitness, population_size=7, seed=3,
                     mutation_config=MutationConfig(add_connection_prob=0.4, add_node_prob=0.1))
    pop.run(3)
    last = pop.statistics[-1]
    assert last["species_count"] >= 1
    assert sum(last["sizes"]) == pop.population_size


def test_seeded_run_is_deterministic():
    def fitness(genome, generation):
        return sum(abs(c.weight) for c in genome.connections)

    config = MutationConfig(add_connection_prob=0.4, add_node_prob=0.1, weight_prob=1.0)
    a = Population(fitness, population_size=6, seed=7, mutation_config=config)
    b = Population(fitness, population_size=6, seed=7, mutation_config=config)
    a.run(3)
    b.run(3)

    assert a.statistics == b.statistics
    assert [len(g.connections) for g in a.population] == [len(g.connections) for g in b.population]


def test_degenerate_constant_fitness_runs_cleanly():
    pop = Population(constant_fitness, population_size=4, seed=0)
    pop.run(2)
    assert len(pop.population) == 4
    for g in pop.population:
        g.validate()
    assert pop.best_fitness == 0.0


def test_stagnation_does_not_collapse_population():
    # Constant fitness -> the single species stagnates and would be pruned.
    # The champion guarantee must keep the population alive and full-sized.
    pop = Population(constant_fitness, population_size=6, seed=0)
    pop.run(40)  # far past the default 15-generation stagnation threshold
    assert len(pop.population) == pop.population_size
    for g in pop.population:
        g.validate()


def test_champion_preserved_when_species_stagnates():
    # Fitness rewards connection count and is capped: once every genome that
    # can improve has, the champion species plateaus; the champion must survive.
    def fitness(genome, generation):
        return min(float(len(genome.connections)) / 4.0, 1.0)

    pop = Population(fitness, population_size=6, seed=2,
                     mutation_config=MutationConfig(add_connection_prob=0.5))
    pop.run(40)
    assert len(pop.population) == pop.population_size
    assert pop.best_fitness > 0.0


def test_statistics_recorded_per_generation():
    pop = Population(count_fitness, population_size=4, seed=0)
    pop.run(2)
    stats = pop.statistics
    assert len(stats) == 2
    for row in stats:
        assert set(row) == {"generation", "population_size", "species_count",
                            "best_fitness", "mean_fitness", "sizes"}
    assert [row["generation"] for row in stats] == [1, 2]


def test_statistics_use_evaluated_fitness_not_carried():
    # All genomes evaluate to 5.0; the recorded mean/best must reflect that,
    # not the carried (stale 0.0) fitness of freshly-crossovered offspring.
    def fitness(genome, generation):
        return 5.0

    pop = Population(fitness, population_size=8, seed=0)
    pop.run(1)
    stats = pop.statistics[0]
    assert stats["best_fitness"] == 5.0
    assert stats["mean_fitness"] == 5.0


def test_engine_has_no_environment_dependencies():
    source = inspect.getsource(neat.population)
    assert "import grid" not in source
    assert "from grid" not in source
    assert "pygame" not in source
    assert "evolve_sim" not in source
    assert "main" not in source


def test_evaluated_best_genome_matches_recorded_statistics():
    """The accessor agrees with the recorded statistics and stays isolated.

    Consistency: ``None`` before the first ``run``, then equal in fitness to the last
    recorded ``best_fitness`` after each generation. Isolation: every read is a fresh
    defensive copy, so mutating a returned snapshot cannot reach engine state.

    **Validates: Requirements 2.4, 2.5, 3.7, 3.15**
    """
    def fitness(genome, generation):
        return sum(abs(c.weight) for c in genome.connections) + 1.0

    pop = Population(fitness, population_size=6, seed=1,
                     mutation_config=MutationConfig(add_connection_prob=0.5, add_node_prob=0.1))

    assert pop.evaluated_best_genome is None

    pop.run(1)
    assert pop.evaluated_best_genome.fitness == pop.statistics[-1]["best_fitness"]

    pop.run(1)
    assert pop.evaluated_best_genome.fitness == pop.statistics[-1]["best_fitness"]

    # Two consecutive reads are distinct objects: no caller holds the engine's instance.
    assert pop.evaluated_best_genome is not pop.evaluated_best_genome
    assert all(g is not pop.evaluated_best_genome for g in pop.population)

    snapshot = pop.evaluated_best_genome
    original_fitness = snapshot.fitness
    recorded_best = pop.statistics[-1]["best_fitness"]
    best_genome_fitness = pop.best_genome.fitness
    best_genome_biases = {nid: node.bias for nid, node in pop.best_genome.nodes.items()}
    node_id = sorted(snapshot.nodes)[0]
    original_bias = snapshot.nodes[node_id].bias

    snapshot.fitness = -12345.0
    snapshot.nodes[node_id].bias = 999.0
    if snapshot.connections:
        original_weight = snapshot.connections[0].weight
        original_enabled = snapshot.connections[0].enabled
        innovation = snapshot.connections[0].innovation
        snapshot.connections[0].weight = 888.0
        snapshot.connections[0].enabled = not original_enabled
    else:
        original_weight = None

    # The mutation is invisible everywhere.
    again = pop.evaluated_best_genome
    assert again.fitness == original_fitness
    assert again.nodes[node_id].bias == original_bias
    if original_weight is not None:
        gene = next(c for c in again.connections if c.innovation == innovation)
        assert gene.weight == original_weight
        assert gene.enabled == original_enabled
    assert pop.statistics[-1]["best_fitness"] == recorded_best
    assert pop.best_genome.fitness == best_genome_fitness
    assert {nid: node.bias for nid, node in pop.best_genome.nodes.items()} == best_genome_biases


# ------------------------------------ fitness finiteness and population size


def mixed_sign_initial_population():
    """Six unwired genomes plus six carrying the same two connections.

    At a compatibility threshold of 0.5 the two shapes form two species, so the
    adjusted-fitness sums are mixed-sign with a positive total.
    """
    genomes = [Genome.minimal(input_ids=[0, 1], output_ids=[10]) for _ in range(6)]
    for _ in range(6):
        wired = Genome.minimal(input_ids=[0, 1], output_ids=[10])
        wired.add_connection(
            ConnectionGene(in_node=0, out_node=10, weight=0.5, innovation=1)
        )
        wired.add_connection(
            ConnectionGene(in_node=1, out_node=10, weight=0.5, innovation=2)
        )
        genomes.append(wired)
    return genomes


def mixed_sign_fitness(genome, generation):
    return 5.0 if genome.connections else -2.0


def constant_value_fitness(value):
    def fitness(genome, generation):
        return value

    return fitness


def single_value_evaluator(index, value):
    """Batch evaluator that scores one position ``value`` and the rest ``1.0``."""

    def evaluator(genomes, generation):
        for position, genome in enumerate(genomes):
            genome.fitness = value if position == index else 1.0

    return evaluator


def assert_statistics_are_finite(pop):
    assert all(
        math.isfinite(row["best_fitness"]) and math.isfinite(row["mean_fitness"])
        for row in pop.statistics
    )


def test_mixed_sign_fitness_preserves_population_size():
    """Mixed-sign species sums must not inflate the population.

    PRE-FIX (HEAD ``ea8a9c2``): generation 1 produces species offspring
    ``[20, -8]`` and ``len(population) == 20`` against ``population_size == 12``,
    because ``_next_generation`` skips non-positive budgets and
    ``_guarantee_champion`` only tops up.

    **Validates: Requirements 1.30, 2.30**
    """
    pop = Population(
        mixed_sign_fitness,
        population_size=12,
        seed=0,
        initial_population=mixed_sign_initial_population(),
        speciation_config=SpeciationConfig(compatibility_threshold=0.5),
    )
    for _ in range(3):
        pop.run(1)
        assert len(pop.population) == pop.population_size
        assert pop.statistics[-1]["population_size"] == pop.population_size
        assert all(
            species.offspring >= 0 for species in pop.speciation.species
        )


def test_nan_fitness_is_rejected_naming_kind_and_index():
    """NaN must be reported as ``nan`` with its population index.

    PRE-FIX: ``ValueError: cannot convert float NaN to integer`` at
    ``neat/speciation.py:236`` — vocabulary that names neither the genome nor the
    kind of value at fault.

    **Validates: Requirements 1.25, 1.28, 2.25, 2.28**
    """
    one_nan = Population(
        evaluator=single_value_evaluator(2, float("nan")),
        population_size=6,
        seed=0,
    )
    with pytest.raises(ValueError) as excinfo:
        one_nan.run(1)
    message = str(excinfo.value)
    assert "non-finite fitness" in message
    assert "index 2" in message
    assert message.endswith("nan")
    assert one_nan.statistics == []
    assert one_nan.best_genome is None
    assert_statistics_are_finite(one_nan)

    every_nan = Population(
        constant_value_fitness(float("nan")), population_size=6, seed=0
    )
    with pytest.raises(ValueError) as excinfo:
        every_nan.run(1)
    assert "index 0" in str(excinfo.value)
    assert str(excinfo.value).endswith("nan")
    assert_statistics_are_finite(every_nan)


def test_positive_infinity_fitness_is_reported_as_inf_not_nan():
    """``+inf`` must be reported as ``inf``.

    PRE-FIX: the SAME misleading ``cannot convert float NaN to integer``, because
    ``inf / inf`` is NaN, so the message reports a value the caller never supplied.

    **Validates: Requirements 1.26, 2.26**
    """
    pop = Population(constant_value_fitness(float("inf")), population_size=6, seed=0)
    with pytest.raises(ValueError) as excinfo:
        pop.run(1)
    message = str(excinfo.value)
    assert "non-finite fitness" in message
    assert "index 0" in message
    assert message.endswith("inf")
    assert "nan" not in message.lower()
    assert pop.statistics == []
    assert_statistics_are_finite(pop)


def test_negative_infinity_fitness_is_rejected():
    """``-inf`` must raise instead of silently becoming the recorded best.

    PRE-FIX: NO ERROR AT ALL — the adjusted total is ``-inf <= 0`` so the uniform
    fallback absorbs it, and ``-inf`` becomes both ``best_fitness`` and
    ``statistics[-1]["best_fitness"]``.

    **Validates: Requirements 1.27, 2.27**
    """
    pop = Population(constant_value_fitness(float("-inf")), population_size=6, seed=0)
    with pytest.raises(ValueError) as excinfo:
        pop.run(1)
    message = str(excinfo.value)
    assert "non-finite fitness" in message
    assert "index 0" in message
    assert message.endswith("-inf")
    assert pop.statistics == []
    assert pop.best_genome is None
    assert_statistics_are_finite(pop)


# ---------------------------------- preservation: finite fitness of any sign


def test_nonnegative_fitness_run_sizes_are_unchanged():
    """``len(population)`` per generation, measured on HEAD before any change.

    **Validates: Requirements 3.8**
    """
    for seed in (0, 1, 7):
        pop = Population(
            count_fitness,
            population_size=10,
            seed=seed,
            mutation_config=MutationConfig(add_connection_prob=0.3),
        )
        sizes = []
        for _ in range(4):
            pop.run(1)
            sizes.append(len(pop.population))
        assert sizes == [10, 10, 10, 10]
        assert [row["population_size"] for row in pop.statistics] == [10, 10, 10, 10]


def test_all_negative_fitness_keeps_uniform_fallback_and_size():
    """All-negative fitness stays legal and keeps the population at its size.

    Two species form at a compatibility threshold of 0.5; both adjusted sums are
    negative, so the total is non-positive and the documented uniform split runs —
    exactly as on HEAD.

    **Validates: Requirements 3.9**
    """
    def negative_fitness(genome, generation):
        return -1.0 if genome.connections else -4.0

    pop = Population(
        negative_fitness,
        population_size=12,
        seed=0,
        initial_population=mixed_sign_initial_population(),
        speciation_config=SpeciationConfig(compatibility_threshold=0.5),
    )
    for _ in range(3):
        pop.run(1)
        assert len(pop.population) == pop.population_size
        assert pop.statistics[-1]["best_fitness"] == -1.0
    assert pop.best_fitness == -1.0
    assert_statistics_are_finite(pop)


def test_select_parent_floors_weights_and_falls_back_to_uniform():
    """``_select_parent`` is the reference semantic the allocation fix aligns with,
    and it is left exactly as it is.

    **Validates: Requirements 3.11**
    """
    from neat.speciation import Species

    pop = Population(count_fitness, population_size=2, seed=0)

    positive = wired_a()
    positive.fitness = 4.0
    negative = wired_b()
    negative.fitness = -9.0
    mixed = Species(id=1, representative=positive, members=[positive, negative])
    # the negative weight is floored to 0.0, so it can never be drawn
    assert all(pop._select_parent(mixed) is positive for _ in range(50))

    positive.fitness = -2.0
    negative.fitness = -1.0
    all_negative = Species(id=2, representative=positive, members=[positive, negative])
    drawn = {id(pop._select_parent(all_negative)) for _ in range(200)}
    # non-positive total -> uniform rng.choice over the members
    assert drawn == {id(positive), id(negative)}


def test_best_fitness_sentinel_does_not_trip_the_finiteness_check():
    """``Population.best_fitness`` starts at ``float("-inf")`` as engine state.

    That sentinel is not a genome fitness, so the finiteness check must not look at
    it — generation 1 has to run normally.

    **Validates: Requirements 2.25, 3.8**
    """
    pop = Population(constant_fitness, population_size=5, seed=0)
    assert pop.best_fitness == float("-inf")

    pop.run(1)
    assert pop.generation == 1
    assert len(pop.statistics) == 1
    assert pop.best_fitness == 0.0
    assert len(pop.population) == 5
    assert_statistics_are_finite(pop)

# ------------------------------------------- engine configuration validity
#
# Narrowed Area 6 scope: ONLY the engine configuration values that were measured
# to violate a core population invariant, to create an impossible run, or to
# silently substitute a value the caller never asked for.
#
# DELIBERATELY EXCLUDED (measured as benign saturation, no invariant broken, so
# no rule is added and none is asserted here): ``crossover_rate`` outside
# ``[0, 1]``, the ``MutationConfig`` probability fields outside ``[0, 1]``,
# ``weight_sigma`` / ``bias_sigma`` / ``weight_bounds`` ordering, and
# ``SpeciationConfig.compatibility_threshold`` / coefficients /
# ``small_genome_threshold``. Those all keep constructing exactly as on HEAD.


@pytest.mark.parametrize(
    "kwargs, field, shown",
    [
        ({"population_size": 0}, "Population.population_size", "0"),
        ({"population_size": -5}, "Population.population_size", "-5"),
        ({"population_size": 6, "elitism": -3}, "Population.elitism", "-3"),
        ({"initial_population": []}, "Population.initial_population", "[]"),
    ],
)
def test_invalid_population_arguments_are_rejected_at_construction(kwargs, field, shown):
    """Engine configuration that breaks a population invariant must raise.

    PRE-FIX (HEAD ``ea8a9c2``), each row measured directly:

    - ``population_size=0``: **no error**; a population of **100** and
      ``pop.population_size == 100``, because ``Population.__init__`` writes
      ``population_size or 100`` and ``0`` is falsy — the caller's explicit
      argument is silently substituted.
    - ``population_size=-5``: **no error**; an **empty** population
      (``range(-5)`` is empty) while ``pop.population_size == -5``.
    - ``elitism=-3`` with ``population_size=6``: **no error**; after one
      generation ``len(pop.population) == 12`` and
      ``statistics[-1]["population_size"] == 12`` — double the requested size.
      Arithmetic: ``elites = min(-3, 6) = -3``; ``ranked[:-3]`` is the first 3 of
      6, so 3 elites are appended; then ``range(budget - elites) = range(9)``
      adds 9 children. The constant-population-size invariant is broken.
    - ``initial_population=[]``: **no error**; ``population_size`` becomes 0
      (``population_size or len([])``), the population stays empty for every
      generation, and ``best_genome`` is still ``None`` after 3 runs, so no
      champion is ever established. An impossible run.

    **Validates: Requirements 2.30**
    """
    with pytest.raises(ValueError) as excinfo:
        Population(count_fitness, seed=0, **kwargs)
    message = str(excinfo.value)
    assert field in message
    assert shown in message


def test_zero_stagnation_threshold_is_rejected_at_construction():
    """``stagnation_threshold=0`` must raise instead of disabling speciation.

    PRE-FIX (HEAD ``ea8a9c2``), measured with ``population_size=6``: **no
    error**. ``prune_stagnant`` extincts EVERY species every generation
    (``stagnation >= 0`` is always true), so ``allocate_offspring`` returns
    ``{}``. On generation 1 ``best_genome`` is still ``None``, so
    ``_guarantee_champion`` returns the empty list and the recorded generation-1
    row is ``population_size == 0`` against a requested 6 — the invariant is
    broken. From generation 2 on, the champion guarantee (which by then has a
    ``best_genome``) refills the population to 6 out of mutated champion copies
    alone, and ``species_count`` stays **0 in every recorded generation**:
    speciation, offspring allocation, parent selection and crossover are all
    dead for the whole run. Measured statistics for 6 generations:
    sizes ``[0, 6, 6, 6, 6, 6]``, species counts ``[0, 0, 0, 0, 0, 0]``.

    NOTE — deviation from the pre-fix expectation recorded in design.md: it
    predicted the population "collapses to 0 and never recovers". The collapse to
    0 at generation 1 reproduces exactly; the permanence does NOT, because
    ``_track_best_and_stats`` sets ``best_genome`` at the end of generation 1 and
    the champion guarantee then refills. The rule is still warranted: generation
    1 violates the constant-size invariant and no generation ever runs the
    evolutionary machinery.

    **Validates: Requirements 2.30**
    """
    with pytest.raises(ValueError) as excinfo:
        SpeciationConfig(stagnation_threshold=0)
    message = str(excinfo.value)
    assert "SpeciationConfig.stagnation_threshold" in message
    assert "0" in message


def test_valid_engine_configuration_and_documented_defaults_are_preserved():
    """Every documented endpoint and default stays legal and behaves as on HEAD.

    Measured on UNFIXED code and asserted unchanged:

    - ``elitism=0`` constructs and runs (used by
      ``test_crossover_produces_valid_children_from_parent_genes``).
    - An omitted ``population_size`` still defaults to **100**.
    - An omitted ``population_size`` with a 3-genome ``initial_population`` still
      defaults to **3**.
    - ``population_size=2`` with a 2-genome ``initial_population`` still works.
    - ``crossover_rate`` of ``0.0`` and ``1.0`` are still accepted — the rate is
      DELIBERATELY not range-validated (out-of-range values only saturate).
    - ``MutationConfig(weight_prob=0.0)`` and
      ``MutationConfig(add_connection_prob=1.0)`` still construct — the mutation
      probabilities are DELIBERATELY not range-validated either.
    - ``SpeciationConfig(stagnation_threshold=1)`` and the default ``15``
      construct, so the new rule is inclusive at its endpoint.

    **Validates: Requirements 3.18, 3.20**
    """
    zero_elitism = Population(count_fitness, population_size=4, seed=0, elitism=0)
    assert zero_elitism.elitism == 0
    zero_elitism.run(1)
    assert len(zero_elitism.population) == 4

    omitted = Population(count_fitness, seed=0)
    assert omitted.population_size == 100
    assert len(omitted.population) == 100

    seeded = Population(
        count_fitness,
        seed=0,
        initial_population=[wired_a(), wired_b(), wired_a()],
    )
    assert seeded.population_size == 3
    assert len(seeded.population) == 3

    sized = Population(
        count_fitness,
        population_size=2,
        seed=0,
        initial_population=[wired_a(), wired_b()],
    )
    assert sized.population_size == 2
    assert len(sized.population) == 2

    for rate in (0.0, 1.0):
        pop = Population(count_fitness, population_size=4, seed=0, crossover_rate=rate)
        assert pop.crossover_rate == rate

    assert MutationConfig(weight_prob=0.0).weight_prob == 0.0
    assert MutationConfig(add_connection_prob=1.0).add_connection_prob == 1.0

    assert SpeciationConfig(stagnation_threshold=1).stagnation_threshold == 1
    assert SpeciationConfig().stagnation_threshold == 15


@pytest.mark.parametrize(
    "kwargs, field",
    [
        ({"population_size": "6"}, "Population.population_size"),
        ({"population_size": 6.0}, "Population.population_size"),
        ({"population_size": True}, "Population.population_size"),
        ({"elitism": 1.0}, "Population.elitism"),
        ({"elitism": True}, "Population.elitism"),
    ],
)
def test_integer_engine_fields_reject_wrong_types_with_type_error(kwargs, field):
    """A wrong TYPE raises ``TypeError``; a wrong VALUE raises ``ValueError``.

    ``population_size`` and ``elitism`` are integer-semantic: both reach
    ``range()`` and slice arithmetic, so ``bool`` and ``6.0`` are refused even
    though the range comparison alone would accept them.

    **Validates: Requirements 2.30**
    """
    with pytest.raises(TypeError) as excinfo:
        Population(count_fitness, seed=0, **kwargs)
    assert field in str(excinfo.value)


@pytest.mark.parametrize("value", ["1", 1.0, True])
def test_stagnation_threshold_rejects_wrong_types_with_type_error(value):
    """Same type policy for ``SpeciationConfig.stagnation_threshold``.

    **Validates: Requirements 2.30**
    """
    with pytest.raises(TypeError) as excinfo:
        SpeciationConfig(stagnation_threshold=value)
    assert "SpeciationConfig.stagnation_threshold" in str(excinfo.value)


def test_missing_fitness_and_evaluator_still_raises_its_original_error():
    """The pre-existing either/or check is untouched and still runs first.

    **Validates: Requirements 3.18**
    """
    with pytest.raises(ValueError) as excinfo:
        Population(seed=0)
    assert "provide either fitness_fn (per-genome) or evaluator (batch)" == str(excinfo.value)
