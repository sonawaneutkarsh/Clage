"""Frozen experiments, incremental evolution, bounded observational replay."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import platform
import random
import subprocess
import threading
import time
from collections import Counter, deque
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal
from weakref import WeakKeyDictionary

from pydantic import BaseModel, ConfigDict, Field, model_validator

from neat.genome import ConnectionGene
from neat.mutation import MutationConfig
from neat.population import Population
from neat.phenotype import Network
from neat.speciation import SpeciationConfig
from world.config import Direction, EnvironmentConfig
from world.fitness import fitness
from world.recorder import _serialize_genome
from world.simulation import WorldSession

from .contracts import EvolutionRecord, FrameRecord, GenomeRecord, HistoryRecord

SCHEMA = "clage-studio-replay"
VERSION = 2
OBSERVATIONS = ["Food Δx", "Food Δy", "Food density", "Body density", "Energy",
                "Wall proximity x", "Wall proximity y", "Previous MOVE", "Previous EAT"]
ACTIONS = ["MOVE", "TURN LEFT", "TURN RIGHT", "EAT"]
RECORDING_BYTES = 16 * 1024 * 1024
BODY_BUDGET = 50000


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    population: int = Field(default=72, ge=1, le=2000)
    generations: int = Field(default=8, ge=1, le=50)
    ticks: int = Field(default=320, ge=1, le=3000)
    width: int = Field(default=40, ge=2, le=128)
    height: int = Field(default=32, ge=2, le=128)
    food: int = Field(default=180, ge=0, le=16384)
    regrowth: int = Field(default=2, ge=0, le=100)
    metabolism: float = Field(default=0.004, ge=0, le=1, allow_inf_nan=False)
    repro_threshold: float = Field(default=0.85, ge=0, le=2, allow_inf_nan=False)
    repro_fraction: float = Field(default=0.5, ge=0, le=1, allow_inf_nan=False)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    initialization: Literal["dense-random-v1", "minimal-v1"] = "dense-random-v1"
    weight_mutation: float = Field(default=0.8, ge=0, le=1, allow_inf_nan=False)
    add_node: float = Field(default=0.03, ge=0, le=1, allow_inf_nan=False)
    add_connection: float = Field(default=0.08, ge=0, le=1, allow_inf_nan=False)
    compatibility: float = Field(default=3.0, gt=0, le=100, allow_inf_nan=False)
    record: bool = True

    @model_validator(mode="after")
    def capacity(self):
        if self.population + self.food > self.width * self.height:
            raise ValueError("Founders plus initial food must fit in the world")
        if self.population * self.generations > 10000:
            raise ValueError("Studio limits a run to 10,000 founder evaluations")
        return self

    def world(self):
        return EnvironmentConfig(width=self.width, height=self.height, ticks=self.ticks,
                                 initial_food=self.food, food_target=self.food,
                                 food_regrowth_per_tick=self.regrowth,
                                 metabolism=self.metabolism, repro_threshold=self.repro_threshold,
                                 repro_fraction=self.repro_fraction, seed_base=self.seed,
                                 record_trace=False)


def provenance():
    source = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for package in ("neat", "world", "diversity", "experiments", "studio", "visual", "benchmarks"):
        for path in sorted((source / package).rglob("*.py")):
            digest.update(str(path.relative_to(source)).encode())
            digest.update(path.read_bytes())
    for path in sorted((source / "studio" / "static").glob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(source)).encode())
            digest.update(path.read_bytes())
    try:
        repository = subprocess.check_output(["git", "-C", str(source), "rev-parse", "--show-toplevel"],
                                              text=True, stderr=subprocess.DEVNULL).strip()
        if Path(repository).resolve() != source:
            raise ValueError("Installed source is not a Git repository root")
        commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True,
                                         stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True))
    except (OSError, subprocess.CalledProcessError, ValueError):
        commit, dirty = None, None
    try:
        engine_version = version("clage")
    except PackageNotFoundError:
        engine_version = "source-checkout"
    return {"engine_version": engine_version, "commit": commit, "dirty": dirty,
            "source_sha256": digest.hexdigest(),
            "python": platform.python_version(), "platform": platform.platform(),
            "evolution_definition": "neat-reconciled-v1",
            "observation_definition": "world-observations-v2-odd-boundaries",
            "studio_contract": "1.0-slice-v2", "created": datetime.now(timezone.utc).isoformat(),
            "fitness_definition": "world-fitness-v1: max body (3*food+.01*age+.5*offspring)",
            "observations": OBSERVATIONS, "actions": ACTIONS, "checkpoint": False}


class Experiment:
    def __init__(self, config: RunConfig):
        self.config = config.model_copy(deep=True)
        self.metadata = provenance()
        self.population = Population(
            None, evaluator=self._evaluate, population_size=config.population,
            input_ids=list(range(9)), output_ids=[10, 11, 12, 13], seed=config.seed,
            record_reproduction=True,
            mutation_config=MutationConfig(weight_prob=config.weight_mutation,
                                           add_node_prob=config.add_node,
                                           add_connection_prob=config.add_connection),
            speciation_config=SpeciationConfig(compatibility_threshold=config.compatibility))
        self.metadata.update({"resolved_world": asdict(config.world()),
                              "resolved_mutation": asdict(self.population.mutation_config),
                              "resolved_speciation": asdict(self.population.speciation.config),
                              "elitism": self.population.elitism,
                              "crossover_rate": self.population.crossover_rate})
        if config.initialization == "dense-random-v1":
            for genome in self.population.population:
                for source in genome.inputs:
                    for target in genome.outputs:
                        genome.connections.append(ConnectionGene(
                            source.id, target.id, self.population.rng.uniform(-2, 2),
                            innovation=self.population.db.connection_innovation(source.id, target.id)))
                genome.validate()
        self.frames: deque[dict[str, Any]] = deque()
        self.frame_sizes: deque[int] = deque()
        self.recorded_bytes = 0
        self.history: list[dict[str, Any]] = []
        self.genomes: dict[str, Any] = {}
        self.species: dict[str, int] = {}
        self.lineage: dict[str, Any] = {}
        self._genome_keys: WeakKeyDictionary = WeakKeyDictionary()
        self.complete = False
        self.sequence = 0
        self.dropped = 0
        self.session = None
        self._new_generation()

    def _new_generation(self):
        self.session = WorldSession(self.population.population, self.config.world(),
                                    self.population.generation)
        self.genome_ids = {genome: f"{self.session.generation}:{index}"
                           for index, genome in enumerate(self.session.population)}
        for index, genome in enumerate(self.session.population):
            data = _serialize_genome(genome, index)
            data["fitness"] = None
            data["key"] = self.genome_ids[genome]
            self.genomes[data["key"]] = data
            record = self.population.reproduction_records.get(genome)
            self.lineage[data["key"]] = {
                "parents": [self._genome_keys.get(parent) for parent in record["parents"]] if record else [],
                "kind": record["kind"] if record else "founder",
                "deltas": record["deltas"] if record else None,
            }
            self._genome_keys[genome] = data["key"]
        for organism in self.session.organisms:
            organism.capture_inference = True
        self._capture()

    def _evaluate(self, population, generation):
        if population is not self.session.population or generation != self.session.generation:
            raise RuntimeError("Evolution/world generation boundary mismatch")
        self.session.finish()

    def advance(self):
        if self.complete:
            return False
        if self.session.tick >= self.config.ticks:
            self._new_generation()
            return True
        possible_bodies = len(self.session.organisms) + sum(body.alive for body in self.session.organisms)
        if possible_bodies > BODY_BUDGET:
            raise ValueError("Studio's 50,000-body safety budget would be exceeded; reset or use offline world execution. No final generation fitness was evaluated.")
        self.session.step()
        self._capture()
        if self.session.tick == self.config.ticks:
            self.population.run(1)
            for species in self.population.speciation.species:
                for genome in species.members:
                    key = self.genome_ids.get(genome)
                    if key is not None:
                        self.species[key] = species.id
            scores = [genome.fitness for genome in self.session.population]
            for genome in self.session.population:
                self.genomes[self.genome_ids[genome]]["fitness"] = genome.fitness
            champion = max(self.session.population, key=lambda genome: genome.fitness)
            self.history.append({**self.population.statistics[-1],
                                 "world_generation": self.session.generation,
                                 "fitnesses": scores, "champion": self.genome_ids[champion],
                                 "mean_nodes": sum(len(genome.nodes) for genome in self.session.population) / len(scores),
                                 "mean_connections": sum(len(genome.connections) for genome in self.session.population) / len(scores),
                                 "species": dict(Counter(self.species.get(key) for key in self.genome_ids.values()))})
            self.complete = self.population.generation >= self.config.generations
        return True

    def _capture(self):
        session = self.session
        ids = {organism: index for index, organism in enumerate(session.organisms)}
        organisms = []
        actions = [0] * 4
        for organism in session.organisms:
            action = organism.previous_action
            if organism.alive and action is not None:
                actions[action] += 1
            organisms.append({"id": ids[organism], "genome": self.genome_ids[organism.genome],
                              "parent": ids.get(organism.parent), "x": organism.x, "y": organism.y,
                              "facing": Direction.ORDER.index(organism.facing), "energy": organism.energy,
                              "alive": organism.alive, "action": action, "age": organism.age,
                              "food_eaten": organism.food_eaten, "offspring": organism.offspring,
                              "fitness": fitness(organism.food_eaten, organism.age, organism.offspring),
                              "inference": organism.last_inference})
        alive = [organism for organism in organisms if organism["alive"]]
        self.current = {"sequence": self.sequence, "tick": session.tick,
                        "generation": session.generation, "food": sorted(session.world.food),
                        "organisms": organisms,
                        "metrics": {"population": len(alive), "births": len(organisms) - self.config.population,
                                    "deaths": len(organisms) - len(alive),
                                    "food_eaten": sum(organism["food_eaten"] for organism in organisms),
                                    "mean_energy": sum(organism["energy"] for organism in alive) / max(1, len(alive)),
                                    "actions": actions, "food": len(session.world.food)}}
        self.sequence += 1
        if self.config.record:
            size = len(json.dumps(self.current, separators=(",", ":")))
            self.frames.append(self.current)
            self.frame_sizes.append(size)
            self.recorded_bytes += size
            while self.recorded_bytes > RECORDING_BYTES or len(self.frames) > 600:
                self.recorded_bytes -= self.frame_sizes.popleft()
                self.frames.popleft()
                self.dropped += 1

    def snapshot(self):
        return {"frame": self.current, "complete": self.complete,
                "config": self.config.model_dump(), "history": self.history,
                "species": self.species, "recording": {"frames": len(self.frames), "dropped": self.dropped}}

    def bundle(self):
        retained = {body["genome"] for frame in self.frames for body in frame["organisms"]}
        retained.update(row["champion"] for row in self.history)
        return {"schema": SCHEMA, "version": VERSION, "metadata": self.metadata,
                "config": self.config.model_dump(),
                "genomes": {key: value for key, value in self.genomes.items() if key in retained},
                "species": {key: value for key, value in self.species.items() if key in retained},
                "history": self.history, "frames": list(self.frames),
                "lineage_schema": "clage-evolution-lineage-v1", "lineage": self.lineage,
                "dropped_frames": self.dropped, "complete": self.complete}


def validate_replay(data):
    if not isinstance(data, dict) or data.get("schema") != SCHEMA or type(data.get("version")) is not int or data.get("version") not in (1, VERSION):
        raise ValueError("Unsupported Studio replay schema/version; legacy recordings use the old viewer")
    config = RunConfig.model_validate(data.get("config"))
    if data["version"] == 2:
        lineage = data.get("lineage")
        if data.get("lineage_schema") != "clage-evolution-lineage-v1" or not isinstance(lineage, dict) or len(lineage) > 10000:
            raise ValueError("Invalid evolutionary lineage schema/table")
        for key, record in lineage.items():
            EvolutionRecord.model_validate(record)
            try:
                generation, identity = (int(part) for part in key.split(":"))
            except (ValueError, AttributeError):
                raise ValueError("Invalid lineage genome key") from None
            if not 0 <= generation < config.generations or not 0 <= identity < config.population:
                raise ValueError("Lineage genome key out of run bounds")
            if not isinstance(record, dict) or record.get("kind") not in {"founder", "elite", "clone", "crossover", "champion_copy", "champion_rescue"}:
                raise ValueError("Invalid evolutionary reproduction kind")
            if not isinstance(record.get("parents"), list) or len(record["parents"]) > 2:
                raise ValueError("Invalid evolutionary parent list")
            for parent in record["parents"]:
                if parent is not None and (parent not in lineage or int(parent.split(":")[0]) >= generation):
                    raise ValueError("Evolutionary parents must be recorded earlier generations")
    frames, genomes = data.get("frames"), data.get("genomes")
    if not isinstance(frames, list) or not frames or len(frames) > 600:
        raise ValueError("Replay must contain 1–600 recorded frames")
    if not isinstance(genomes, dict) or len(genomes) > 10000:
        raise ValueError("Invalid genome table")
    from neat.genome import Genome, NodeGene, NodeType

    networks = {}
    for key, raw_genome in genomes.items():
        if data['version'] == 2 and key not in data['lineage']:
            raise ValueError("Genome is missing evolutionary lineage entry")
        genome = GenomeRecord.model_validate(raw_genome)
        if genome.key != key or len({node.id for node in genome.nodes}) != len(genome.nodes):
            raise ValueError("Invalid genome identity or duplicate node")
        decoded = Genome(nodes={node.id: NodeGene(node.id, NodeType[node.type], node.bias) for node in genome.nodes},
                         connections=[ConnectionGene(edge.in_node, edge.out_node, edge.weight, edge.enabled, edge.innovation) for edge in genome.connections])
        if len(decoded.inputs) != 9 or len(decoded.outputs) != 4:
            raise ValueError("Replay genome must have nine inputs and four outputs")
        networks[key] = Network(decoded)
    history = data.get("history")
    if not isinstance(history, list) or len(history) > config.generations:
        raise ValueError("Invalid evaluated history")
    for raw_row in history:
        row = HistoryRecord.model_validate(raw_row)
        if row.champion not in genomes or row.world_generation >= config.generations:
            raise ValueError("History references unavailable champion/generation")
        if len(row.fitnesses) != config.population:
            raise ValueError("History fitness count differs from founder population")
    species = data.get("species")
    if not isinstance(species, dict) or any(key not in genomes or type(value) is not int or value < 0 for key, value in species.items()):
        raise ValueError("Invalid species mapping")
    previous = -1
    for frame in frames:
        validated = FrameRecord.model_validate(frame)
        if not isinstance(frame, dict) or type(frame.get("sequence")) is not int or frame["sequence"] <= previous:
            raise ValueError("Frame sequences must strictly increase")
        previous = frame["sequence"]
        if type(frame.get("tick")) is not int or not 0 <= frame["tick"] <= config.ticks:
            raise ValueError("Invalid frame tick")
        if type(frame.get("generation")) is not int or not 0 <= frame["generation"] < config.generations:
            raise ValueError("Invalid frame generation")
        bodies = frame.get("organisms")
        if not isinstance(bodies, list) or len(bodies) > config.width * config.height * (config.ticks + 1):
            raise ValueError("Invalid organism table")
        ids = {body.get("id") for body in bodies if isinstance(body, dict)}
        if len(ids) != len(bodies) or any(type(identity) is not int for identity in ids):
            raise ValueError("Organism IDs must be unique integers")
        occupied = set()
        for body in bodies:
            if body.get("genome") not in genomes or (body.get("parent") is not None and body["parent"] not in ids):
                raise ValueError("Invalid genome or parent reference")
            if body["parent"] is not None and body["parent"] >= body["id"]:
                raise ValueError("Parent must precede offspring; cyclic body ancestry is invalid")
            if body["age"] > frame["tick"]:
                raise ValueError("Body age exceeds world tick")
            if body["fitness"] != fitness(body["food_eaten"], body["age"], body["offspring"]):
                raise ValueError("Body score differs from the declared fitness definition")
            inference = body["inference"]
            if inference is not None:
                if inference.get('world_tick') is not None and inference['world_tick'] > frame['tick']:
                    raise ValueError("Inference timestamp is in the future of its recorded frame")
                node_ids = {str(node["id"]) for node in genomes[body["genome"]]["nodes"]}
                if {str(identity) for identity in inference["values"]} != node_ids:
                    raise ValueError("Activation values must reference every genome node")
                if body["action"] != max(range(4), key=lambda index: inference["outputs"][index]):
                    raise ValueError("Recorded action differs from actual inference argmax")
                outputs, values = networks[body["genome"]].activate_with_trace(inference["inputs"])
                serialized_values = {str(key): value for key, value in inference["values"].items()}
                if any(not math.isclose(actual, recorded, rel_tol=1e-12, abs_tol=1e-12) for actual, recorded in zip(outputs, inference["outputs"])):
                    raise ValueError("Recorded outputs disagree with genome inference")
                if any(not math.isclose(value, serialized_values[str(key)], rel_tol=1e-12, abs_tol=1e-12) for key, value in values.items()):
                    raise ValueError("Recorded node activations disagree with genome inference")
            for axis, size in [("x", config.width), ("y", config.height)]:
                if type(body.get(axis)) is not int or not 0 <= body[axis] < size:
                    raise ValueError("Organism out of world bounds")
            if body.get("alive"):
                cell = (body["x"], body["y"])
                if cell in occupied:
                    raise ValueError("Live organisms overlap")
                occupied.add(cell)
        food = frame.get("food")
        if not isinstance(food, list) or len(food) > config.width * config.height:
            raise ValueError("Invalid food table")
        for cell in food:
            if not isinstance(cell, (list, tuple)) or len(cell) != 2 or any(type(value) is not int for value in cell):
                raise ValueError("Invalid food cell")
            if not 0 <= cell[0] < config.width or not 0 <= cell[1] < config.height or tuple(cell) in occupied:
                raise ValueError("Food overlaps or is out of bounds")
            occupied.add(tuple(cell))
        alive = [body for body in validated.organisms if body.alive]
        metrics = validated.metrics
        if metrics.population != len(alive) or metrics.food != len(food):
            raise ValueError("Population/food metrics disagree with frame contents")
        if metrics.births != len(bodies) - config.population or metrics.deaths != len(bodies) - len(alive):
            raise ValueError("Birth/death metrics disagree with frame contents")
        if metrics.food_eaten != sum(body.food_eaten for body in validated.organisms):
            raise ValueError("Consumption metric disagrees with recorded bodies")
        mean_energy = sum(body.energy for body in alive) / max(1, len(alive))
        if not math.isclose(metrics.mean_energy, mean_energy, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("Mean energy disagrees with living body state")
        expected_actions = [sum(body.action == action for body in alive) for action in range(4)]
        if metrics.actions != expected_actions:
            raise ValueError("Action counts disagree with living acted bodies")
    return data


class Manager:
    def __init__(self):
        self.lock = threading.RLock()
        self.experiment = None
        self.paused = True
        self.speed = 20.0
        self.achieved = 0.0
        self.error = None
        self.run_id = 0
        self.stop = threading.Event()
        self.thread = None

    def start_worker(self):
        self.stop.clear()
        self.thread = threading.Thread(target=self._worker, daemon=True, name="clage-world")
        self.thread.start()

    def close(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=10)

    def create(self, config):
        experiment = Experiment(config)
        with self.lock:
            self.experiment = experiment
            self.run_id += 1
            self.paused = True
            self.error = None
            self.achieved = 0.0
        return self.snapshot()

    def command(self, action, speed=None):
        with self.lock:
            if self.experiment is None:
                raise ValueError("Start a configured experiment first")
            if action == "pause":
                self.paused = True
                self.achieved = 0.0
            elif action == "resume":
                if self.error is not None:
                    raise ValueError("Reset the run after an engine/safety error before resuming")
                if self.experiment.complete:
                    raise ValueError("Completed runs must be reset, not resumed")
                self.paused = False
            elif action == "step":
                self.paused = True
                if self.error is not None:
                    raise ValueError("Reset the run after an engine/safety error before stepping")
                try:
                    self.experiment.advance()
                except Exception as error:
                    self.error = f"{type(error).__name__}: {error}"
                    raise ValueError(self.error) from error
            elif action == "speed":
                if speed is None or not 1 <= speed <= 120:
                    raise ValueError("Speed must be 1–120 ticks/s")
                self.speed = speed
            elif action == "reset":
                self.create(self.experiment.config)
            else:
                raise ValueError("Unknown command")
            return self.snapshot()

    def snapshot(self):
        with self.lock:
            state = self.experiment.snapshot() if self.experiment else {"frame": None}
            return copy.deepcopy({**state, "paused": self.paused, "speed": self.speed,
                                  "achieved_tps": round(self.achieved, 1), "error": self.error,
                                  "run_id": self.run_id})

    def stream_snapshot(self, previous_signature):
        with self.lock:
            signature = (self.run_id, self.experiment.sequence if self.experiment else None,
                         self.paused, self.speed, self.error)
            return signature, self.snapshot() if signature != previous_signature else None

    def _worker(self):
        while not self.stop.is_set():
            started = time.perf_counter()
            advanced = False
            with self.lock:
                if self.experiment and not self.paused and not self.experiment.complete:
                    try:
                        generation, tick = self.experiment.session.generation, self.experiment.session.tick
                        self.experiment.advance()
                        advanced = self.experiment.session.generation == generation and self.experiment.session.tick == tick + 1
                        if not advanced:
                            self.achieved = 0.0
                        if self.experiment.complete:
                            self.paused = True
                    except Exception as error:
                        self.error = f"{type(error).__name__}: {error}"
                        self.paused = True
                period = 1 / self.speed if advanced else 0.025
            self.stop.wait(max(0, period - (time.perf_counter() - started)))
            if advanced:
                elapsed = time.perf_counter() - started
                with self.lock:
                    self.achieved = 1 / max(elapsed, 1e-9)


def evaluate_policies(config, champion, seeds=(10001, 10002, 10003, 10004, 10005)):
    """Single-founder held-out trials; no training. Per-policy, per-seed outcomes."""
    from neat.genome import Genome

    seeds = tuple(seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("Evaluation seeds must be distinct nonnegative integers")
    if config.seed in seeds:
        raise ValueError("Held-out world seeds overlap the training seed; choose a different training seed")

    results = []
    for policy in ["move", "random", "forager", "champion"]:
        if policy == "champion" and champion is None:
            continue
        for seed in seeds:
            world_config = config.world()
            world_config.seed_base = seed
            world_config.ticks = min(config.ticks, 300)
            genome = champion.copy() if policy == "champion" else Genome.minimal(
                input_ids=list(range(9)), output_ids=[10, 11, 12, 13])
            session = WorldSession([genome], world_config)
            policy_rng = random.Random(seed + 991)
            for _ in range(world_config.ticks):
                if policy != "champion":
                    for organism in session.organisms:
                        organism.network = PolicyNetwork(policy, policy_rng, organism)
                session.step()
            session.finish()
            results.append({"policy": policy, "seed": seed, "ticks": world_config.ticks,
                            "food_eaten": sum(body.food_eaten for body in session.organisms),
                            "survivors": sum(body.alive for body in session.organisms),
                            "births": len(session.organisms) - 1, "fitness": genome.fitness})
    return {"schema": "clage-policy-evaluation", "version": 1,
            "evaluation_set": "heldout-worlds-v1" if seeds == (10001, 10002, 10003, 10004, 10005) else "custom-seeds-v1", "seeds": list(seeds),
            "config": config.model_dump(), "founders": 1,
            "champion": _serialize_genome(champion, 0) if champion else None,
            "metadata": provenance(), "results": results,
            "resolved_world": asdict(config.world()),
            "world_overrides": {"seed_base": "per-row world seed", "ticks": min(config.ticks, 300), "founders": 1},
            "policies": {"move": "always MOVE", "random": "uniform four actions; separate Random(world_seed+991)",
                         "forager": "privileged orientation-aware reference: global nearest-food direction plus body facing (not available in the original nine-input neural interface); dominant axis, turn toward target, MOVE otherwise",
                         "champion": "saved evaluated genome, unchanged argmax policy; no training"},
            "limitations": f"{len(seeds)} evaluation seeds; identical initial world seeds, not identical later food layouts. Forager has privileged facing information. Single-founder clonal trials; no inference of learning or superiority."}


class PolicyNetwork:
    def __init__(self, policy, rng, organism):
        self.policy, self.rng, self.organism = policy, rng, organism

    def activate(self, inputs):
        action = 0
        if self.policy == "random":
            action = self.rng.randrange(4)
        elif self.policy == "forager":
            dx, dy = inputs[:2]
            if dx or dy:
                target = (1 if dx > 0 else -1, 0) if abs(dx) >= abs(dy) else (0, 1 if dy > 0 else -1)
                facing = Direction.ORDER.index(self.organism.facing)
                difference = (Direction.ORDER.index(target) - facing) % 4
                action = 0 if difference == 0 else (1 if difference == 3 else 2)
        return [1.0 if index == action else 0.0 for index in range(4)]
