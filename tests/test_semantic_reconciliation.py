import copy
import json
import random

import pytest

from benchmarks.problems import PROBLEMS
from benchmarks.run import run_benchmark, run_trial as benchmark_trial
from experiments.analysis import aggregate, NUMERIC_METRICS
from experiments.config import ExperimentConfig, Condition, load_experiment
from experiments.run import run_condition, run_experiment
from neat.crossover import crossover
from neat.diagnostics import species_report
from neat.genome import Genome, NodeGene, NodeType, ConnectionGene
from neat.innovation import InnovationDB
from neat.mutation import mutate_add_node, mutate_add_connection
from neat.phenotype import Network, ACTIVATION
from neat.population import Population, _structure_signature
from neat.speciation import compatibility_distance, SpeciationConfig, Species
from studio.core import Experiment, RunConfig
from studio.reproduce import reproduce
from visual.data import aggregate_condition, load_recording
from world import EnvironmentConfig, GenerationRecorder, run_generation
from world.grid import World, FOOD
from world.organism import Organism


def wired(innovation=99):
    genome = Genome.minimal([14], [15])
    genome.add_connection(ConnectionGene(14, 15, 0.7, innovation=innovation))
    return genome


@pytest.mark.parametrize("change", ["bias", "weight", "enabled", "type", "endpoint"])
def test_champion_identity_is_full_genotype(change):
    genome = wired()
    altered = genome.copy()
    if change == "bias":
        altered.nodes[15].bias = 1
    elif change == "weight":
        altered.connections[0].weight = -1
    elif change == "enabled":
        altered.connections[0].enabled = False
    elif change == "type":
        altered.nodes[15].node_type = NodeType.HIDDEN
    else:
        altered.connections[0].in_node = 13
    assert _structure_signature(genome) != _structure_signature(altered)


def test_first_generation_archive_preserves_actual_winner():
    founders = [Genome.minimal([0], [10]) for _ in range(4)]
    for index, genome in enumerate(founders):
        genome.nodes[10].bias = index
    population = Population(lambda genome, generation: genome.nodes[10].bias,
                            initial_population=founders, elitism=0, seed=3)
    population.run(1)
    assert population.best_generation == 1
    assert population.best_fitness == 3
    assert any(_structure_signature(genome) == _structure_signature(population.best_genome)
               for genome in population.population)


def test_species_elites_respect_member_count_and_budget():
    population = Population(lambda genome, generation: 1, population_size=4, elitism=4)
    species = Species(0, population.population[0])
    species.members = [population.population[0]]
    offspring = []
    population._reproduce_species(species, 4, offspring)
    assert len(offspring) == 4


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_invalid_generation_budget_does_not_advance(value):
    population = Population(lambda genome, generation: 1, population_size=4)
    with pytest.raises((ValueError, TypeError)):
        population.run(value)
    assert population.generation == 0


def test_founder_count_must_agree():
    with pytest.raises(ValueError, match="match"):
        Population(lambda genome, generation: 1, population_size=2,
                   initial_population=[Genome.minimal()])


def test_crossover_bias_inherits_both_and_equal_bias_consumes_no_draw():
    first = Genome.minimal([0], [10], bias=1)
    second = Genome.minimal([0], [10], bias=2)
    old = _structure_signature(first)
    assert {crossover(first, second, random.Random(seed)).nodes[10].bias
            for seed in range(20)} == {1, 2}
    assert _structure_signature(first) == old
    rng = random.Random(1)
    state = rng.getstate()
    crossover(first, first, rng)
    assert rng.getstate() == state


@pytest.mark.parametrize("conflict", ["role", "endpoint", "pair"])
def test_crossover_rejects_conflicting_history(conflict):
    first = wired()
    second = first.copy()
    if conflict == "role":
        second.nodes[15].node_type = NodeType.HIDDEN
    elif conflict == "endpoint":
        second.add_node(NodeGene(16, NodeType.HIDDEN))
        second.connections[0].out_node = 16
    else:
        second.connections[0].innovation = 100
    with pytest.raises(ValueError):
        crossover(first, second, random.Random(0))


def test_imported_history_reserves_ids_and_is_atomic():
    genome = wired()
    db = InnovationDB()
    node = mutate_add_node(genome, random.Random(0), db)
    assert node.id > 15
    assert db.connection_innovation(14, 15) == 99
    assert min(edge.innovation for edge in genome.connections[1:]) > 99
    conflict = wired(100)
    before = db.to_dict()
    original = _structure_signature(conflict)
    with pytest.raises(ValueError):
        mutate_add_node(conflict, random.Random(0), db)
    assert db.to_dict() == before
    assert _structure_signature(conflict) == original
    assert InnovationDB.from_dict(before).to_dict() == before


def test_invalid_import_registration_does_not_disable_or_mint():
    genome = wired()
    genome.connections.append(ConnectionGene(14, 15, 1, innovation=99))
    db = InnovationDB()
    before = db.to_dict()
    with pytest.raises(ValueError):
        mutate_add_connection(genome, random.Random(0), db)
    assert db.to_dict() == before
    assert all(edge.enabled for edge in genome.connections)


@pytest.mark.parametrize("inputs,outputs", [([0, 0], [10]), ([0], [10, 10]), ([0], [0])])
def test_duplicate_interface_ids_reject(inputs, outputs):
    with pytest.raises(ValueError):
        Genome.minimal(inputs, outputs)


def test_hidden_input_edges_and_duplicate_innovations_reject():
    genome = wired()
    genome.add_node(NodeGene(16, NodeType.HIDDEN))
    with pytest.raises(ValueError, match="INPUT"):
        genome.validate_connection(16, 14)
    genome.connections.append(ConnectionGene(14, 16, 1, innovation=99))
    with pytest.raises(ValueError, match="innovation"):
        genome.validate()


def test_empty_compatibility_and_flat_diagnostics():
    assert compatibility_distance(Genome.minimal(), Genome.minimal(),
                                  SpeciationConfig(small_genome_threshold=0)) == 0
    seen = []
    def score(genome, generation):
        assert type(generation) is int
        seen.append(generation)
        return 1
    report = species_report(population_size=3, generations=5, fitness_fn=score,
                            config=SpeciationConfig(stagnation_threshold=1))
    assert len(report) == 5
    assert set(seen) == set(range(5))


@pytest.mark.parametrize("seeds", [[], [0, 0], [True], [0.5]])
def test_experiment_seed_validation(seeds):
    with pytest.raises(ValueError):
        ExperimentConfig("safe", {}, [Condition("control", None)], seeds)


@pytest.mark.parametrize("name", ["..", "../bad", "bad/name", "bad\\name", ""])
def test_safe_experiment_names(name):
    with pytest.raises(ValueError):
        ExperimentConfig(name, {}, [Condition("control", None)], [0])


def tiny_config(tmp_path):
    experiment = load_experiment("experiments/configs/food_abundance.json")
    experiment.base["neat"].update(population_size=4, generations=2)
    experiment.base["world"].update(width=6, height=6, ticks=3,
                                     initial_food=3, food_target=3)
    experiment.seeds = [0]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(experiment.to_dict()))
    return experiment, path


def test_output_preflight_and_manifest_provenance(tmp_path):
    experiment, config = tiny_config(tmp_path)
    target = tmp_path / "runs"
    run_experiment(config, target, conditions=["control"])
    manifest = json.loads((target / experiment.name / "manifest.json").read_text())
    assert manifest["control"] == "control"
    assert len(manifest["source_sha256"]) == 64
    assert manifest["resolved_configs"]["control"][0]["neat"]["generations"] == 2
    original = (target / experiment.name / "control" / "0.json").read_bytes()
    with pytest.raises(FileExistsError):
        run_experiment(config, target)
    with pytest.raises(FileExistsError):
        run_condition(experiment, experiment.condition("control"), target)
    assert (target / experiment.name / "control" / "0.json").read_bytes() == original


def test_invalid_recording_budget_and_foreign_condition_write_nothing(tmp_path):
    experiment, config = tiny_config(tmp_path)
    target = tmp_path / "runs"
    with pytest.raises(ValueError):
        run_experiment(config, target, record_generation=2)
    assert not target.exists()
    with pytest.raises(ValueError):
        run_condition(experiment, Condition("../foreign", None), target)
    assert not target.exists()


def test_legacy_manifest_does_not_attribute_an_unrelated_git_parent(tmp_path, monkeypatch):
    import experiments.run as runner
    experiment, config = tiny_config(tmp_path)
    monkeypatch.setattr(runner, "__file__", str(tmp_path / "installed/experiments/run.py"))
    def git_output(command, **kwargs):
        assert command[-1] == "--show-toplevel"
        return str(tmp_path / "unrelated") + "\n"
    monkeypatch.setattr(runner.subprocess, "check_output", git_output)
    target = tmp_path / "runs"
    run_experiment(config, target, conditions=["control"])
    manifest = json.loads((target / experiment.name / "manifest.json").read_text())
    assert manifest["source_commit"] is None


@pytest.mark.parametrize("generations", [0, -1, True, 1.5])
def test_experiment_generations_reject_before_write(generations):
    with pytest.raises(ValueError, match="generations"):
        ExperimentConfig("safe", {"neat": {"generations": generations}},
                         [Condition("control", None)], [0])


@pytest.mark.parametrize("field,value", [("node_counter", True), ("innovation_counter", 99)])
def test_invalid_serialized_history_counters_reject(field, value):
    db = InnovationDB()
    db.register_genomes([wired()])
    data = db.to_dict()
    data[field] = value
    with pytest.raises(ValueError):
        InnovationDB.from_dict(data)


@pytest.mark.parametrize("mode", ["empty", "ragged", "order", "missing", "nan", "inf"])
@pytest.mark.parametrize("aggregator", [aggregate, aggregate_condition])
def test_aggregation_rejects_invalid_trials(mode, aggregator):
    row = {metric: 1 for metric in NUMERIC_METRICS}
    row["generation"] = 1
    trials = [[copy.deepcopy(row)], [copy.deepcopy(row)]]
    if mode == "empty":
        trials = []
    elif mode == "ragged":
        trials[1] = []
    elif mode == "order":
        trials[1][0]["generation"] = 2
    elif mode == "missing":
        del trials[1][0]["food_consumed"]
    else:
        trials[1][0]["food_consumed"] = float(mode)
    with pytest.raises(ValueError):
        aggregator(trials)


def test_legacy_recorder_evaluated_fitness_and_versions(tmp_path):
    genomes = [Genome.minimal(range(9), range(10, 14))]
    genomes[0].fitness = 123
    config = EnvironmentConfig(width=5, height=7, ticks=3, initial_food=2, food_target=2)
    recorder = GenerationRecorder(genomes, config, 0)
    assert recorder.to_dict()["fitness_phase"] == "unevaluated"
    run_generation(genomes, config, 0, recorder=recorder)
    data = recorder.to_dict()
    assert data["version"] == 2 and data["fitness_phase"] == "evaluated"
    assert data["genomes"][0]["fitness"] == genomes[0].fitness != 123
    path = tmp_path / "recording.json"
    data["version"] = 1
    path.write_text(json.dumps(data))
    assert load_recording(path)["fitness_phase"] == "stored_unverified"
    data["version"] = 999
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_recording(path)


@pytest.mark.parametrize("fraction", [0, 1])
def test_exhausted_reproduction_removes_body_but_preserves_birth(fraction):
    config = EnvironmentConfig(width=5, height=7, initial_food=0, food_target=0,
                               repro_fraction=fraction, repro_threshold=0.5)
    world = World(config, random.Random(0))
    body = Organism(Genome.minimal(), 2, 3, config, energy=1)
    world.place_organism(body)
    observations = body.observe(world, config)
    assert observations[-2:] == [0, 0]
    child = body._try_reproduce(world, config)
    exhausted = child if fraction == 0 else body
    assert child is not None and body.offspring == 1
    assert not exhausted.alive
    assert world.occupant(exhausted.x, exhausted.y) is None


def test_inference_matches_independent_recursive_oracle():
    for seed in range(20):
        rng = random.Random(seed)
        genome = Genome.minimal([0, 1], [10], bias=rng.uniform(-1, 1))
        genome.add_node(NodeGene(5, NodeType.HIDDEN, rng.uniform(-1, 1)))
        for innovation, pair in enumerate([(0, 5), (1, 5), (5, 10), (0, 10)], 1):
            genome.add_connection(ConnectionGene(*pair, rng.uniform(-2, 2), innovation=innovation))
        network = Network(genome)
        for _ in range(10):
            inputs = [rng.random(), rng.random()]
            def evaluate(identity):
                if identity in (0, 1):
                    return inputs[identity]
                total = genome.nodes[identity].bias
                for edge in genome.connections:
                    if edge.enabled and edge.out_node == identity:
                        total += evaluate(edge.in_node) * edge.weight
                return ACTIVATION(total)
            assert network.activate(inputs) == [evaluate(10)]
            assert network.activate_with_trace(inputs)[0] == [evaluate(10)]


@pytest.mark.parametrize("radius", [1, 2, 5])
def test_density_clipped_scan_matches_square_oracle(radius):
    config = EnvironmentConfig(width=5, height=7, initial_food=0, food_target=0)
    world = World(config, random.Random(0))
    world.place_food(0, 0)
    world.place_food(2, 3)
    body = Organism(Genome.minimal(), 4, 6, config)
    world.place_organism(body)
    for x, y in [(-2, -2), (0, 0), (2, 3), (4, 6), (7, 9)]:
        denominator = (2 * radius + 1) ** 2 - 1
        positions = [(x + dx, y + dy) for dy in range(-radius, radius + 1)
                     for dx in range(-radius, radius + 1) if (dx, dy) != (0, 0)]
        assert world.food_density(x, y, radius) == sum(pair in world.food for pair in positions) / denominator
        expected = sum(world.occupant(*pair) is not None and world.occupant(*pair) is not FOOD
                       for pair in positions) / denominator
        assert world.organism_density(x, y, radius, None) == expected
        assert world.organism_density(x, y, radius, body) == 0


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_benchmark_invalid_budgets(budget):
    with pytest.raises((ValueError, TypeError)):
        benchmark_trial(PROBLEMS["xor"], 0, max_generations=budget)
    with pytest.raises((ValueError, TypeError)):
        run_benchmark(PROBLEMS["xor"], trials=budget)


def test_scientific_regime_replay_guard():
    experiment = Experiment(RunConfig(population=4, ticks=3, generations=2,
                                      width=6, height=6, food=4))
    while experiment.advance():
        pass
    data = experiment.bundle()
    verification = reproduce(data)
    assert verification["recorded_frames_verified"] == len(data["frames"])
    assert len(verification["source_sha256"]) == 64
    assert verification["source_sha256"] == verification["current_source_sha256"]
    data["metadata"].pop("evolution_definition")
    with pytest.raises(ValueError, match="Scientific definition"):
        reproduce(data)
