import copy
import gzip
import json
import random
import time

import pytest
from fastapi.testclient import TestClient

from neat.genome import ConnectionGene, Genome
from neat.phenotype import Network
from studio.core import Experiment, Manager, RunConfig, evaluate_policies, validate_replay
from studio.server import create_app
from studio.reproduce import reproduce
from world.config import EnvironmentConfig
from world.grid import World
from world.organism import Organism
from world.recorder import GenerationRecorder
from world.simulation import WorldSession


def tiny(**overrides):
    return RunConfig(population=3, width=8, height=8, food=10, ticks=8, generations=2, **overrides)


def test_inference_instrumentation_is_exact_and_readonly():
    genome = Genome.minimal(input_ids=list(range(9)), output_ids=[10, 11, 12, 13])
    genome.connections = [ConnectionGene(0, 10, .75, innovation=1), ConnectionGene(1, 11, -.5, innovation=2)]
    before = repr(genome)
    network = Network(genome)
    for seed in range(20):
        inputs = [random.Random(seed + index).uniform(-1, 1) for index in range(9)]
        outputs, values = network.activate_with_trace(inputs)
        assert outputs == network.activate(inputs)
        assert outputs == [values[identity] for identity in network.output_ids]
    assert repr(genome) == before


def test_incremental_matches_independent_legacy_loop_and_instrumented_rng():
    config = EnvironmentConfig(width=8, height=8, ticks=20, initial_food=10, food_target=10)
    genomes = [Genome.minimal(input_ids=list(range(9)), output_ids=[10, 11, 12, 13]) for _ in range(4)]
    world = World(config, random.Random(config.world_rng_seed(0)))
    bodies = []
    recorder = GenerationRecorder(genomes, config, 0)
    for genome in genomes:
        organism = Organism(genome, *world.random_empty_cell(), config)
        world.place_organism(organism)
        bodies.append(organism)
    for _ in range(config.initial_food):
        cell = world.random_empty_cell()
        if cell is not None:
            world.place_food(*cell)
    recorder.record_tick(world, bodies)
    for _ in range(config.ticks):
        newborns = []
        for organism in bodies:
            if organism.alive:
                child = organism.act(world, config)
                if child:
                    newborns.append(child)
        bodies.extend(newborns)
        world.regenerate_food()
        recorder.record_tick(world, bodies)
    fresh = [genome.copy() for genome in genomes]
    measured = GenerationRecorder(fresh, config, 0)
    session = WorldSession(fresh, config, recorder=measured)
    for organism in session.organisms:
        organism.capture_inference = True
    while session.step():
        pass
    assert recorder.ticks == measured.ticks
    assert session.world.rng.getstate() == world.rng.getstate()
    assert all(child.parent is not None for child in session.organisms[4:])


def test_deterministic_evolution_and_final_fitness():
    first, second = Experiment(tiny()), Experiment(tiny())
    while first.advance():
        second.advance()
    assert first.current == second.current
    assert first.history == second.history
    assert first.genomes == second.genomes
    assert len(first.history) == 2
    assert first.history[-1]['world_generation'] == 1
    assert first.population.generation == 2
    assert all(row['fitness'] is not None for row in first.genomes.values())


def test_recording_budget_and_disabled(monkeypatch):
    monkeypatch.setattr('studio.core.RECORDING_BYTES', 10000)
    experiment = Experiment(tiny())
    for _ in range(7):
        experiment.advance()
    assert experiment.dropped > 0
    assert experiment.recorded_bytes <= 10000
    no_record = Experiment(tiny(record=False))
    no_record.advance()
    assert not no_record.frames
    assert no_record.current['tick'] == 1


def test_replay_validation_roundtrip_and_corrupt_references():
    experiment = Experiment(tiny())
    experiment.advance()
    bundle = json.loads(json.dumps(experiment.bundle()))
    assert validate_replay(bundle) == bundle
    for field, bad in [('genome', 'fake'), ('parent', 999), ('x', 900)]:
        corrupt = copy.deepcopy(bundle)
        corrupt['frames'][0]['organisms'][0][field] = bad
        with pytest.raises(ValueError):
            validate_replay(corrupt)
    corrupt = copy.deepcopy(bundle)
    corrupt['version'] = 999
    with pytest.raises(ValueError):
        validate_replay(corrupt)


@pytest.mark.parametrize('field,value', [('energy', 'bogus'), ('facing', 99), ('alive', 1), ('fitness', -1), ('age', 90)])
def test_replay_rejects_invalid_body_contract(field, value):
    experiment = Experiment(tiny())
    corrupt = json.loads(json.dumps(experiment.bundle()))
    corrupt['frames'][0]['organisms'][0][field] = value
    with pytest.raises(ValueError):
        validate_replay(corrupt)


def test_replay_rejects_topology_corruption_and_inconsistent_metrics():
    experiment = Experiment(tiny())
    experiment.advance()
    bundle = json.loads(json.dumps(experiment.bundle()))
    for kind in ['edge', 'metrics', 'inference', 'cycle']:
        corrupt = copy.deepcopy(bundle)
        if kind == 'edge':
            corrupt['genomes']['0:0']['connections'][0]['in'] = 999
        elif kind == 'metrics':
            corrupt['frames'][0]['metrics']['population'] = 999
        elif kind == 'inference':
            corrupt['frames'][1]['organisms'][0]['inference']['outputs'] = [0.0] * 3
        else:
            corrupt['frames'][1]['organisms'][-1]['parent'] = corrupt['frames'][1]['organisms'][-1]['id']
        with pytest.raises(ValueError):
            validate_replay(corrupt)


@pytest.mark.parametrize('overrides', [dict(population=999), dict(seed=-1), dict(metabolism=float('nan')), dict(add_node=1.5), dict(width=True)])
def test_config_rejects_invalid_values(overrides):
    data = tiny().model_dump()
    data.update(overrides)
    with pytest.raises(ValueError):
        RunConfig.model_validate(data)


def test_api_control_stream_export_import_and_artifact(tmp_path):
    manager = Manager()
    with TestClient(create_app(manager, worker=False, artifact_dir=tmp_path)) as client:
        assert client.get('/').status_code == 200
        assert client.post('/api/control', json={'action': 'step'}).status_code == 409
        response = client.post('/api/runs', json=tiny().model_dump())
        assert response.status_code == 200
        assert response.json()['paused']
        assert client.post('/api/control', json={'action': 'step'}).json()['frame']['tick'] == 1
        with client.websocket_connect('/api/stream') as socket:
            assert socket.receive_json()['frame']['tick'] == 1
        exported = client.get('/api/export')
        replay = json.loads(gzip.decompress(exported.content))
        assert len(replay['frames']) == 2
        assert client.post('/api/replay/validate', json=replay).status_code == 200
        assert client.get('/api/state').json()['frame']['tick'] == 1
        saved = client.post('/api/artifacts').json()
        assert client.get(f"/api/artifacts/{saved['id']}").status_code == 200
        assert len(client.get('/api/artifacts').json()) == 1
        assert client.post('/api/control', json={'action': 'reset'}).json()['frame']['tick'] == 0
        assert client.post('/api/control', json={'action': 'speed', 'speed': 121}).status_code == 422
        assert client.post('/api/runs', json={**tiny().model_dump(), 'unknown': True}).status_code == 422


def test_origin_protection(tmp_path):
    with TestClient(create_app(worker=False, artifact_dir=tmp_path)) as client:
        assert client.post('/api/runs', json=tiny().model_dump(), headers={'Origin': 'https://evil.example'}).status_code == 403
        assert client.get('/api/state', headers={'Host': 'evil.example'}).status_code == 400
        with pytest.raises(Exception):
            with client.websocket_connect('/api/stream', headers={'Origin': 'https://evil.example'}):
                pass


def test_worker_pause_resume_and_shutdown():
    manager = Manager()
    manager.create(tiny())
    manager.start_worker()
    try:
        manager.command('resume')
        deadline = time.monotonic() + 2
        while manager.snapshot()['frame']['tick'] == 0 and time.monotonic() < deadline:
            time.sleep(.01)
        manager.command('pause')
        tick = manager.snapshot()['frame']['tick']
        time.sleep(.08)
        assert manager.snapshot()['frame']['tick'] == tick > 0
    finally:
        manager.close()
    assert not manager.thread.is_alive()


def test_policy_evaluation_is_reproducible_and_does_not_mutate_champion():
    experiment = Experiment(tiny())
    while not experiment.history:
        experiment.advance()
    champion = experiment.population.best_genome
    before = repr(champion)
    first = evaluate_policies(tiny(), champion, seeds=(10001, 10002))
    second = evaluate_policies(tiny(), champion, seeds=(10001, 10002))
    assert first['results'] == second['results']
    assert len(first['results']) == 8
    assert repr(champion) == before
    assert first['founders'] == 1


def test_portable_replay_reruns_actual_recorded_frames():
    experiment = Experiment(tiny())
    while experiment.advance():
        pass
    bundle = json.loads(json.dumps(experiment.bundle()))
    result = reproduce(bundle)
    assert result['recorded_frames_verified'] == 18
    bundle['config']['seed'] += 1
    with pytest.raises(ValueError, match='mismatch'):
        reproduce(bundle)


def test_heldout_rejects_training_seed_overlap():
    config = tiny().model_copy(update={'seed': 10001})
    with pytest.raises(ValueError, match='overlap'):
        evaluate_policies(config, None)
