import dataclasses
import random

import pytest

from neat.genome import Genome
from neat.population import Population

from world import (
    ACTION_SIZE,
    OBSERVATION_SIZE,
    Action,
    Direction,
    EnvironmentConfig,
    GenerationRecorder,
    Organism,
    World,
    make_evaluator,
    record_generation_to_file,
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


# --------------------------------------------------------------- Area 3
# EnvironmentConfig numeric field validity (boundary-configuration-validation)

# Range violations: the value has an acceptable type but lies outside the
# domain the field can represent.
INVALID_ENV_RANGES = [
    ("width", 0),
    ("width", -3),
    ("height", 0),
    ("ticks", -5),
    ("ticks", 0),
    ("density_radius", 0),
    ("density_radius", -2),
    ("max_energy", 0.0),
    ("initial_energy", -1.0),
    ("metabolism", -1.0),
    ("repro_fraction", 1.5),
    ("repro_fraction", -0.5),
    ("initial_food", -10),
    ("food_target", -10),
    ("food_regrowth_per_tick", -3),
    ("seed_stride", 0),
]

# Type violations: the value's type cannot support the field at all.
# ``width`` is integer-semantic, so a str, a fractional float, an integral
# float and a bool are all rejected.
INVALID_ENV_TYPES = [
    ("width", "twenty"),
    ("width", 20.5),
    ("width", 20.0),
    ("width", True),
]


@pytest.mark.parametrize("field_name,value", INVALID_ENV_RANGES)
def test_invalid_environment_field_range_rejected_at_construction(field_name, value):
    """Out-of-range EnvironmentConfig fields must be rejected at construction.

    Measured pre-fix behavior at HEAD ea8a9c2: NONE of these raise at
    construction. Each one either crashes later, deep inside the simulation,
    or silently invalidates the whole run:

    - ``density_radius=0``: ZeroDivisionError at ``world/grid.py:109``
      (``max_count = (2*0+1)**2 - 1 == 0``).
    - ``density_radius=-2``: no error; the density loops iterate over nothing
      while ``max_count`` is 8, so every density observation is 0.0 forever.
    - ``max_energy=0.0``: ZeroDivisionError at ``world/organism.py:77``
      (``self.energy / config.max_energy``).
    - ``width=0`` / ``width=-3``: ``TypeError: Value after * must be an
      iterable, not NoneType`` at ``world/simulation.py:44`` — no cell exists,
      ``random_empty_cell()`` returns None, and it is unpacked anyway.
    - ``metabolism=-1.0``: no error; nothing dies, organisms gain energy every
      tick (24 organisms from 3 founders within 3 ticks).
    - ``initial_energy=-1.0``: no error; every organism dies on tick 1.
    - ``ticks<=0``: no error; ``range(-5)`` is empty, so zero ticks are
      simulated and the generation is still reported complete.
    - ``initial_food=-10`` / ``food_target=-10`` /
      ``food_regrowth_per_tick=-3``: no error; each silently means "none" via
      an empty ``range`` or a never-satisfied ``while``.
    - ``repro_fraction=1.5`` / ``-0.5``: no error; energy is created from
      nothing — the child gets 1.5x the parent's energy and the parent goes
      negative, or the child gets negative energy and the parent GAINS energy.
    - ``seed_stride=0``: no error; ``resolve_config`` maps every trial seed to
      ``seed_base=0``, so all trials of a condition share one world while
      reporting distinct seeds.

    Expected post-fix: ValueError at construction, naming the field and the
    received value.
    """
    with pytest.raises(ValueError) as excinfo:
        EnvironmentConfig(**{field_name: value})

    message = str(excinfo.value)
    assert field_name in message
    assert repr(value) in message


@pytest.mark.parametrize("field_name,value", INVALID_ENV_TYPES)
def test_wrongly_typed_environment_field_rejected_at_construction(field_name, value):
    """Wrongly typed EnvironmentConfig fields must be rejected at construction.

    Measured pre-fix behavior at HEAD ea8a9c2: construction always succeeds.
    ``width='twenty'`` and ``width=20.5`` then raise ``TypeError`` inside
    ``World.__init__`` from ``range()`` ('str'/'float' object cannot be
    interpreted as an integer), far from the offending config. ``width=20.0``
    raises the same 'float' TypeError, and ``width=True`` raises nothing at all
    — ``range(True)`` silently yields a 1-wide grid.

    Expected post-fix: TypeError at construction, naming the field and the
    received value. Integer-semantic fields require an ACTUAL int: an integral
    float (20.0) and a bool are both rejected.
    """
    with pytest.raises(TypeError) as excinfo:
        EnvironmentConfig(**{field_name: value})

    message = str(excinfo.value)
    assert field_name in message
    assert repr(value) in message


def test_dataclasses_replace_revalidates_environment_config():
    """``replace`` must re-validate: it calls __init__, hence __post_init__.

    Measured pre-fix behavior at HEAD ea8a9c2: no error — the replaced config
    is returned with ``density_radius=0`` and blows up later with
    ZeroDivisionError at ``world/grid.py:109``. This matters because
    ``experiments/config.py::resolve_config`` builds every per-trial config
    through ``replace(world, seed_base=...)``.
    """
    valid_config = EnvironmentConfig()

    with pytest.raises(ValueError) as excinfo:
        dataclasses.replace(valid_config, density_radius=0)

    message = str(excinfo.value)
    assert "density_radius" in message
    assert repr(0) in message


# Measured baseline (oracle 2) at HEAD ea8a9c2: asdict(EnvironmentConfig())
# exactly these 17 keys, in this order, with these values. No field is added by
# the validation fix, so this shape cannot change.
ENV_CONFIG_ASDICT_BASELINE = {
    "width": 24,
    "height": 24,
    "ticks": 400,
    "initial_energy": 1.0,
    "max_energy": 1.0,
    "metabolism": 0.005,
    "food_energy": 0.5,
    "initial_food": 90,
    "food_target": 90,
    "food_regrowth_per_tick": 1,
    "density_radius": 2,
    "record_trace": True,
    "behavior_window": 100,
    "repro_threshold": 0.7,
    "repro_fraction": 0.5,
    "seed_base": 0,
    "seed_stride": 1000,
}

# Boundaries and legal shapes that MUST keep constructing. Each is either a
# meaningful setting (zero food, zero metabolism) or an explicitly rejected
# non-defect (behavior_window <= 0 means "no clipping"; max_energy <
# initial_energy is merely capped on the first consume).
VALID_ENV_OVERRIDES = [
    {},
    {"initial_food": 0},
    {"food_target": 0},
    {"food_regrowth_per_tick": 0},
    {"metabolism": 0.0},
    {"repro_fraction": 0.0},
    {"repro_fraction": 1.0},
    {"max_energy": 0.5, "initial_energy": 1.0},
    {"behavior_window": 0},
    {"behavior_window": -5},
    {"seed_base": 0},
    {"repro_threshold": 2.0},
    {"max_energy": 1},
    {"record_trace": False},
]


@pytest.mark.parametrize("overrides", VALID_ENV_OVERRIDES)
def test_valid_environment_configs_still_construct(overrides):
    """Every legal boundary keeps working (Property 6 — preservation)."""
    config = EnvironmentConfig(**overrides)
    for name, value in overrides.items():
        assert getattr(config, name) == value


def test_environment_config_asdict_shape_unchanged():
    """asdict() keys, order and values match the recorded baseline exactly."""
    serialized = dataclasses.asdict(EnvironmentConfig())

    assert list(serialized) == list(ENV_CONFIG_ASDICT_BASELINE)
    assert serialized == ENV_CONFIG_ASDICT_BASELINE
    assert len(serialized) == 17


def test_valid_environment_config_still_runs_a_generation():
    """A legal config drives a full generation exactly as before."""
    config = small_config(ticks=10, behavior_window=-5, metabolism=0.0)
    population = [always_move_genome() for _ in range(4)]

    organisms = run_generation(population, config, generation=0)

    assert len(organisms) == 4
    assert all(genome.fitness > 0.0 for genome in population)


# --------------------------------------------------------------- Area 1
# NEAT <-> world neural-interface contract (boundary-configuration-validation)

# Each case is (input_ids, output_ids, offending side, received count on that
# side). ``None`` means "use Genome.minimal()'s engine default", which is 10
# input ids -- one more than the world's 9-value observation vector.
INTERFACE_MISMATCH_CASES = [
    pytest.param(None, None, "input_ids", 10, id="default-genome-ten-inputs"),
    pytest.param(INPUT_IDS, [], "output_ids", 0, id="zero-outputs"),
    pytest.param(
        INPUT_IDS, [10, 11, 12, 13, 14, 15], "output_ids", 6, id="six-outputs"
    ),
    pytest.param(INPUT_IDS, [10, 11], "output_ids", 2, id="two-outputs"),
]


def mismatched_population(input_ids, output_ids):
    """A population whose genome at index 1 violates the world contract.

    Index 0 and index 2 are valid 9 -> 4 genomes, so a message naming index 1
    proves the check walks the population in list order and reports the FIRST
    offender rather than an arbitrary one.
    """
    offender = Genome.minimal(input_ids=input_ids, output_ids=output_ids)
    if output_ids is not None and len(output_ids) == 6:
        # largest bias on output index 5 -> argmax selects previous_action == 5
        offender.nodes[15].bias = 1.0
    return [always_move_genome(), offender, always_move_genome()]


@pytest.mark.parametrize(
    "input_ids,output_ids,side,received", INTERFACE_MISMATCH_CASES
)
def test_interface_mismatch_rejected_before_the_run(
    input_ids, output_ids, side, received
):
    """Genome interface counts must be enforced before a generation starts.

    Measured pre-fix behavior at HEAD (all four reproduced exactly):

    - default ``Genome.minimal()`` (10 inputs): ``ValueError: expected 10
      inputs, got 9`` from ``neat/phenotype.py:109`` in ``activate()`` --
      raised MID-TICK, once per organism per tick, AFTER the world was built
      and all organisms were already placed. The message names neither the
      interface side nor the configuration at fault.
    - ``output_ids=[]``: ``ValueError: max() iterable argument is empty`` from
      ``world/organism.py:96`` in ``act()`` -- again mid-simulation, again
      naming neither the interface nor the field.
    - 6 outputs (``[10..15]``): NO ERROR. The run completes; the organism whose
      largest output sits at index 5 sets ``previous_action == 5``, outside the
      valid ``0..3`` range, so ``_apply_action`` matches no branch and the tick
      silently does nothing. Observed: ``previous_action == 5`` for every
      organism, fitness 0.02.
    - 2 outputs (``[10, 11]``): NO ERROR. ``Action.TURN_RIGHT`` (2) and
      ``Action.EAT`` (3) are unreachable for the whole run. Observed:
      ``previous_action == 0`` for every organism.

    Expected post-fix: ``ValueError`` from ``run_generation`` naming the
    population index, the count the world expects, the count received, and
    which side of the interface (``input_ids`` / ``output_ids``) is at fault.
    """
    expected = OBSERVATION_SIZE if side == "input_ids" else ACTION_SIZE
    population = mismatched_population(input_ids, output_ids)

    with pytest.raises(ValueError) as excinfo:
        run_generation(population, small_config(ticks=2), generation=0)

    message = str(excinfo.value)
    assert side in message
    assert str(expected) in message
    assert str(received) in message
    assert "index 1" in message


@pytest.mark.parametrize(
    "input_ids,output_ids,side,received", INTERFACE_MISMATCH_CASES
)
def test_interface_mismatch_builds_no_world_and_no_organism(
    monkeypatch, input_ids, output_ids, side, received
):
    """The interface error must precede World and Organism construction.

    Observed without touching production code: sentinels installed over
    ``world.simulation.World`` and ``world.simulation.Organism`` are reached on
    unfixed code, because ``run_generation`` seeds its rng and builds the world
    before any genome is inspected.
    """

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "run_generation constructed a World or an Organism for a population "
            "that violates the neural interface contract"
        )

    monkeypatch.setattr("world.simulation.World", forbidden)
    monkeypatch.setattr("world.simulation.Organism", forbidden)

    population = mismatched_population(input_ids, output_ids)

    with pytest.raises(ValueError) as excinfo:
        run_generation(population, small_config(ticks=2), generation=0)

    assert side in str(excinfo.value)


# Only the COUNTS are contractual (requirement 3.1): 9 inputs and 4 outputs.
# The ids themselves need not be 0..8 and 10..13.
CUSTOM_INPUT_IDS = list(range(100, 109))
CUSTOM_OUTPUT_IDS = [200, 201, 202, 203]


def custom_id_genome() -> Genome:
    return Genome.minimal(input_ids=CUSTOM_INPUT_IDS, output_ids=CUSTOM_OUTPUT_IDS)


def test_custom_node_ids_with_correct_counts_run_a_generation():
    """9 inputs / 4 outputs on arbitrary node ids works end to end (3.1)."""
    config = small_config(ticks=10)
    population = [custom_id_genome() for _ in range(4)]

    organisms = run_generation(population, config, generation=0)

    assert len(organisms) == 4
    assert all(genome.fitness > 0.0 for genome in population)


def test_custom_ids_behave_identically_to_canonical_ids():
    """Renaming the interface nodes changes nothing observable.

    Both populations are unwired genomes with zero bias, so the only
    difference is the node ids; the world rng is seeded identically.
    """
    config = small_config(ticks=10)
    canonical = [always_move_genome() for _ in range(4)]
    custom = [custom_id_genome() for _ in range(4)]

    canonical_organisms = run_generation(canonical, config, generation=0)
    custom_organisms = run_generation(custom, config, generation=0)

    assert len(custom_organisms) == len(canonical_organisms)
    assert [g.fitness for g in custom] == [g.fitness for g in canonical]


def test_population_with_custom_interface_ids_evolves_through_the_world():
    """The full evolve-through-the-world workflow accepts custom ids."""
    config = small_config(ticks=15, initial_food=8, food_target=8)
    pop = Population(
        fitness_fn=None,
        evaluator=make_evaluator(config),
        population_size=6,
        input_ids=CUSTOM_INPUT_IDS,
        output_ids=CUSTOM_OUTPUT_IDS,
        seed=1,
    )
    pop.run(2)

    assert pop.generation == 2
    assert len(pop.population) == 6
    assert pop.best_fitness > 0.0


def test_canonical_nine_to_four_workflow_unchanged():
    """The existing 9 -> 4 interface keeps running exactly as before."""
    config = small_config(ticks=10)
    population = [always_move_genome() for _ in range(4)]

    organisms = run_generation(population, config, generation=0)

    assert len(organisms) == 4
    assert all(len(genome.inputs) == OBSERVATION_SIZE for genome in population)
    assert all(len(genome.outputs) == ACTION_SIZE for genome in population)
    assert all(genome.fitness > 0.0 for genome in population)


def test_make_evaluator_closure_inherits_the_interface_check():
    """``make_evaluator``'s closure raises on its first call.

    ``make_evaluator`` cannot validate at creation time — it is handed only the
    EnvironmentConfig, and the genomes arrive later as an argument to the
    closure. The check therefore lives in ``run_generation``, which the closure
    calls.
    """
    evaluate = make_evaluator(small_config(ticks=2))
    population = mismatched_population(INPUT_IDS, [])

    with pytest.raises(ValueError) as excinfo:
        evaluate(population, 0)

    assert "output_ids" in str(excinfo.value)


def test_record_generation_to_file_inherits_the_interface_check(tmp_path):
    """The replay recorder path goes through ``run_generation`` too."""
    target = tmp_path / "replay.json"
    population = mismatched_population(None, None)

    with pytest.raises(ValueError) as excinfo:
        record_generation_to_file(population, small_config(ticks=2), 0, target)

    assert "input_ids" in str(excinfo.value)
    assert not target.exists()


# --------------------------------------------------------------- Area 2
# Population <-> grid capacity for founder placement


def test_founder_population_exceeding_grid_capacity_rejected():
    """More founders than cells must fail with an actionable error.

    Measured pre-fix behavior: ``TypeError: Value after * must be an iterable,
    not NoneType`` raised inside ``run_generation``'s founder-placement loop.
    ``World.random_empty_cell()`` is correctly typed
    ``Optional[Tuple[int, int]]`` and returns ``None`` once the grid is full,
    but the return value is unpacked anyway by ``Organism(genome, *cell,
    config)``. The message names neither the founder count nor the grid.

    Expected post-fix: ``ValueError`` — explicitly NOT ``TypeError`` — naming
    the founder count, the available capacity, and its width/height derivation.

    ``width < 1`` / ``height < 1`` is deliberately NOT tested here: requirement
    2.7 assigns it to ``EnvironmentConfig`` construction, already covered by
    ``test_invalid_environment_field_range_rejected_at_construction``.
    """
    config = small_config(width=3, height=3, ticks=2)
    population = [always_move_genome() for _ in range(10)]

    with pytest.raises(ValueError) as excinfo:
        run_generation(population, config, generation=0)

    assert not isinstance(excinfo.value, TypeError)
    message = str(excinfo.value)
    assert "10" in message  # requested founder count
    assert "9" in message  # capacity
    assert "width=3" in message
    assert "height=3" in message


def test_over_capacity_population_constructs_no_organism(monkeypatch):
    """The capacity error must precede any Organism construction.

    On unfixed code the first nine founders are constructed successfully and
    only the tenth trips the unpacking, so the sentinel is reached.
    """

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "run_generation constructed an Organism for a founder population "
            "that exceeds grid capacity"
        )

    monkeypatch.setattr("world.simulation.Organism", forbidden)

    config = small_config(width=3, height=3, ticks=2)
    population = [always_move_genome() for _ in range(10)]

    with pytest.raises(ValueError) as excinfo:
        run_generation(population, config, generation=0)

    assert "exceeds grid capacity" in str(excinfo.value)


def test_founder_population_exactly_at_capacity_places_everyone():
    """The boundary is strictly ``population_size > width * height`` (3.4).

    9 founders on a 3x3 grid is exactly capacity and must keep working: every
    founder is constructed and occupies a distinct cell.
    """
    config = small_config(width=3, height=3, ticks=1, initial_food=0)
    population = [always_move_genome() for _ in range(9)]

    organisms = run_generation(population, config, generation=0)

    assert len(organisms) == 9
    positions = {(org.x, org.y) for org in organisms}
    assert len(positions) == 9
    assert positions == {(x, y) for x in range(3) for y in range(3)}


def test_oversized_initial_food_places_only_what_fits():
    """Oversized ``initial_food`` degrades gracefully, it does not raise (3.5).

    The existing ``if cell is not None`` guard in the food-placement loop is
    preserved: a 3x3 grid holding 2 founders has 7 free cells, so 7 of the 50
    requested food items are placed and the rest are silently dropped.
    """
    config = small_config(width=3, height=3, ticks=1, initial_food=50)
    population = [always_move_genome() for _ in range(2)]
    recorder = GenerationRecorder(population, config, 0)

    organisms = run_generation(population, config, generation=0, recorder=recorder)

    assert len(organisms) == 2
    assert len(recorder.ticks[0]["food"]) == 7  # 9 cells - 2 founders


def test_saturated_grid_reproduction_and_regrowth_are_not_errors():
    """A full grid blocks reproduction and regrowth without raising (3.6)."""
    config = small_config(
        width=2,
        height=2,
        repro_threshold=0.1,
        repro_fraction=0.5,
        metabolism=0.0,
        food_target=10,
        food_regrowth_per_tick=1,
    )
    world = World(config, random.Random(0))
    parent = place_organism(world, always_move_genome(), 0, 0)
    for x, y in ((1, 0), (0, 1), (1, 1)):
        place_organism(world, always_move_genome(), x, y)

    assert world.adjacent_empty(0, 0) == []
    assert world.random_empty_cell() is None
    assert parent._try_reproduce(world, config) is None
    assert parent.offspring == 0

    world.regenerate_food()  # no empty cell -> returns quietly
    assert world.food == set()
