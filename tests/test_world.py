import random

import pytest

from neat.genome import Genome
from neat.population import Population

from world import (
    Action,
    Direction,
    EnvironmentConfig,
    Organism,
    World,
    make_evaluator,
    run_generation,
)

INPUT_IDS = list(range(9))
OUTPUT_IDS = [10, 11, 12, 13]


def action_genome(action: int) -> Genome:
    g = Genome.minimal(input_ids=INPUT_IDS, output_ids=OUTPUT_IDS)
    g.nodes[10 + action].bias = 1.0
    return g


def always_move_genome() -> Genome:
    return Genome.minimal(input_ids=INPUT_IDS, output_ids=OUTPUT_IDS)


def small_config(**overrides) -> EnvironmentConfig:
    defaults = dict(
        width=4,
        height=4,
        ticks=1,
        repro_threshold=2.0,  # disabled for pure-mechanic tests unless overridden
    )
    defaults.update(overrides)
    return EnvironmentConfig(**defaults)


def place_organism(world, genome, x, y, **kwargs):
    org = Organism(genome, x, y, world.config, **kwargs)
    world.place_organism(org)
    return org


def test_observe_vector():
    config = small_config()
    world = World(config, random.Random(0))
    org = place_organism(world, always_move_genome(), 1, 1, facing=Direction.NORTH)
    world.place_food(1, 0)

    obs = org.observe(world, config)
    assert len(obs) == 9
    assert obs == pytest.approx([0.0, -0.5, 1 / 24, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])


def test_movement_steps_and_blocked_by_wall():
    config = small_config()
    world = World(config, random.Random(0))
    org = place_organism(world, always_move_genome(), 1, 1, facing=Direction.NORTH)

    org.act(world, config)
    assert (org.x, org.y) == (1, 0)

    org.act(world, config)  # (1,-1) is outside -> blocked
    assert (org.x, org.y) == (1, 0)


def test_movement_blocked_by_other_organism():
    config = small_config()
    world = World(config, random.Random(0))
    mover = place_organism(world, always_move_genome(), 1, 1, facing=Direction.NORTH)
    place_organism(world, always_move_genome(), 1, 0)

    mover.act(world, config)
    assert (mover.x, mover.y) == (1, 1)  # occupied cell -> stay put


def test_turning_changes_facing():
    config = small_config()
    world = World(config, random.Random(0))
    org = place_organism(world, action_genome(Action.TURN_LEFT), 1, 1, facing=Direction.NORTH)

    org.act(world, config)
    assert org.facing == Direction.WEST
    org.act(world, config)
    assert org.facing == Direction.SOUTH


def test_eat_action_consumes_food():
    config = small_config()
    world = World(config, random.Random(0))
    org = place_organism(world, action_genome(Action.EAT), 1, 1, facing=Direction.NORTH)
    world.place_food(1, 0)

    org.act(world, config)
    assert world.food == set()
    assert org.food_eaten == 1
    assert org.energy == pytest.approx(1.0 - config.metabolism)  # capped at max, then metabolism


def test_stepping_onto_food_eats_it():
    config = small_config()
    world = World(config, random.Random(0))
    org = place_organism(world, always_move_genome(), 1, 1, facing=Direction.NORTH)
    world.place_food(1, 0)

    org.act(world, config)
    assert (org.x, org.y) == (1, 0)
    assert world.food == set()
    assert org.food_eaten == 1


def test_eat_with_no_food_does_nothing():
    config = small_config()
    world = World(config, random.Random(0))
    org = place_organism(world, action_genome(Action.EAT), 1, 1, facing=Direction.NORTH)

    org.act(world, config)
    assert org.food_eaten == 0
    assert org.energy == pytest.approx(config.initial_energy - config.metabolism)


def test_metabolism_drains_energy():
    config = small_config(metabolism=0.1, initial_energy=0.5)
    world = World(config, random.Random(0))
    org = place_organism(world, always_move_genome(), 1, 1, facing=Direction.SOUTH)

    org.act(world, config)
    org.act(world, config)
    org.act(world, config)
    assert org.energy == pytest.approx(0.5 - 3 * 0.1)
    assert org.age == 3


def test_energy_capped_at_max():
    config = small_config()
    world = World(config, random.Random(0))
    org = place_organism(world, action_genome(Action.EAT), 1, 1, facing=Direction.NORTH)
    world.place_food(1, 0)
    world.place_food(1, 0)  # food is only ever one per cell; ensure only one eat possible

    org.act(world, config)
    assert org.energy <= config.max_energy


def test_death_at_zero_energy():
    config = small_config(initial_energy=0.05, metabolism=0.05)
    world = World(config, random.Random(0))
    org = place_organism(world, always_move_genome(), 1, 1, facing=Direction.SOUTH)

    org.act(world, config)
    assert org.alive is False
    assert org.energy == 0.0
    assert world.occupant(org.x, org.y) is None


def test_reproduction_split():
    config = small_config(
        repro_threshold=0.6,
        repro_fraction=0.5,
        metabolism=0.001,
        initial_energy=1.0,
    )
    world = World(config, random.Random(0))
    parent = place_organism(world, always_move_genome(), 1, 1, facing=Direction.NORTH)

    child = parent.act(world, config)
    assert child is not None
    assert child.genome is parent.genome
    assert world.occupant(child.x, child.y) is child
    assert parent.offspring == 1
    assert parent.energy == pytest.approx(child.energy)
    assert child.energy == pytest.approx((1.0 - config.metabolism) * 0.5)


def test_no_reproduction_below_threshold():
    config = small_config(repro_threshold=0.6, metabolism=0.001, initial_energy=0.3)
    world = World(config, random.Random(0))
    parent = place_organism(world, always_move_genome(), 1, 1, facing=Direction.NORTH)

    assert parent.act(world, config) is None
    assert parent.offspring == 0


def test_boundaries_block_movement():
    config = small_config()
    world = World(config, random.Random(0))
    org = place_organism(world, always_move_genome(), 0, 0, facing=Direction.WEST)

    org.act(world, config)  # ( -1, 0 ) outside -> blocked
    assert (org.x, org.y) == (0, 0)

    org.facing = Direction.EAST
    org.act(world, config)
    assert (org.x, org.y) == (1, 0)


def test_boundary_observation_at_wall():
    config = small_config()
    world = World(config, random.Random(0))
    org = place_organism(world, always_move_genome(), 0, 0, facing=Direction.NORTH)

    obs = org.observe(world, config)
    assert obs[5] == 1.0  # boundary x at the wall
    assert obs[6] == 1.0  # boundary y at the wall


def test_food_regeneration():
    config = small_config(
        width=8,
        height=8,
        initial_food=0,
        food_target=10,
        food_regrowth_per_tick=1,
    )
    world = World(config, random.Random(0))
    assert len(world.food) == 0

    for _ in range(5):
        world.regenerate_food()
    assert len(world.food) == 5


def test_world_seeds_distinct_across_trials_and_generations():
    # (trial seed, generation) pairs must never collide on a world layout.
    config = EnvironmentConfig(seed_stride=1000)
    seen = set()
    for seed in range(5):
        config.seed_base = seed * config.seed_stride
        for generation in range(50):
            world_seed = config.world_rng_seed(generation)
            assert world_seed not in seen
            seen.add(world_seed)


def test_run_generation_assigns_fitness():
    config = small_config(ticks=10)
    population = [always_move_genome() for _ in range(4)]
    organisms = run_generation(population, config, generation=0)

    assert len(organisms) == 4
    for genome in population:
        assert genome.fitness > 0.0


def test_run_generation_is_deterministic():
    config = small_config(ticks=10)

    def fresh_population():
        return [always_move_genome() for _ in range(4)]

    a = fresh_population()
    b = fresh_population()
    run_generation(a, config, generation=0)
    run_generation(b, config, generation=0)

    assert [g.fitness for g in a] == [g.fitness for g in b]


def test_evaluator_hook_used_by_population():
    calls = []

    def evaluator(population, generation):
        calls.append(generation)
        for genome in population:
            genome.fitness = float(generation + 1)

    pop = Population(fitness_fn=None, evaluator=evaluator, population_size=4, seed=0)
    pop.run(3)

    assert calls == [0, 1, 2]
    # the final population's offspring are scored at the next generation's start
    assert max(g.fitness for g in pop.population) == 3.0
    assert pop.best_fitness == 3.0


def test_generation_transitions_with_world():
    config = small_config(ticks=15, initial_food=8, food_target=8)
    pop = Population(
        fitness_fn=None,
        evaluator=make_evaluator(config),
        population_size=6,
        input_ids=INPUT_IDS,
        output_ids=OUTPUT_IDS,
        seed=1,
    )
    pop.run(2)

    assert pop.generation == 2
    assert len(pop.population) == 6
    assert pop.best_fitness > 0.0
    # best_fitness is the all-time champion (from an earlier generation's world);
    # it is at least the current generation's evaluated best.
    assert pop.best_fitness >= pop.statistics[-1]["best_fitness"]
    assert len(pop.statistics) == 2
