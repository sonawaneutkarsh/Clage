import inspect

import pytest

import neat.population
from neat.genome import ConnectionGene, Genome, NodeGene, NodeType
from neat.mutation import MutationConfig
from neat.population import Population


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
