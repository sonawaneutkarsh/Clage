import random

import pytest

from neat.genome import Genome
from world.config import EnvironmentConfig
from world.grid import World
from world.organism import Organism


@pytest.mark.parametrize('shape', [(2, 2), (8, 9), (24, 24), (96, 64)])
def test_empty_sampler_retains_row_major_choice_and_rng(shape):
    config = EnvironmentConfig(width=shape[0], height=shape[1], initial_food=0)
    world = World(config, random.Random(42))
    choices = random.Random(42)
    operations = random.Random(91)
    for _ in range(min(500, config.width * config.height * 3)):
        empty = [(x, y) for y in range(config.height) for x in range(config.width) if world.cells[y][x] is None]
        expected = choices.choice(empty) if empty else None
        assert world.random_empty_cell() == expected
        assert world.rng.getstate() == choices.getstate()
        if expected:
            world.place_food(*expected)
        if world.food and operations.random() < .8:
            world.remove_food(*operations.choice(sorted(world.food)))
        assert world._empty_count == sum(world._empty_rows) == sum(cell is None for row in world.cells for cell in row)


@pytest.mark.parametrize('seed', range(10))
def test_food_index_matches_legacy_distance_and_set_order_ties(seed):
    rng = random.Random(seed)
    config = EnvironmentConfig(width=31, height=27, initial_food=0)
    world = World(config, random.Random(seed))
    for iteration in range(400):
        world.place_food(rng.randrange(config.width), rng.randrange(config.height))
        if world.food and iteration % 3 == 0:
            world.remove_food(*rng.choice(sorted(world.food)))
        x, y = rng.randrange(config.width), rng.randrange(config.height)
        expected = min(world.food, key=lambda cell: (cell[0] - x) ** 2 + (cell[1] - y) ** 2, default=None)
        assert world.nearest_food(x, y) == expected
    for x, y in [(-20, -20), (10000, 0), (10, 9000)]:
        expected = min(world.food, key=lambda cell: (cell[0] - x) ** 2 + (cell[1] - y) ** 2, default=None)
        assert world.nearest_food(x, y) == expected


def test_body_movement_death_and_split_keep_indexes_and_share_readonly_network():
    config = EnvironmentConfig(width=8, height=8)
    world = World(config, random.Random(0))
    genome = Genome.minimal(input_ids=list(range(9)), output_ids=[10, 11, 12, 13])
    organism = Organism(genome, 3, 3, config)
    world.place_organism(organism)
    assert world.move_organism(organism, 4, 4)
    assert not world.move_organism(organism, -1, 0)
    child = organism._try_reproduce(world, config)
    assert child.network is organism.network
    world.remove_organism(organism)
    world.remove_organism(organism)
    assert world._empty_count == 63
    assert sum(world._empty_rows) == 63
