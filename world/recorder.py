"""Record a generation's world tick-by-tick for faithful replay.

The recorder is a passive observer: it captures the food positions and every
organism's state after each tick (plus initial placement), along with the
genome topology and config needed to inspect and render the generation later.
The visualization layer consumes the serialized JSON and never re-simulates.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from neat.genome import Genome

from .config import Direction, EnvironmentConfig
from .grid import World
from .organism import Organism
from .simulation import run_generation

__all__ = ["GenerationRecorder", "record_generation_to_file"]

SCHEMA = "clage-generation-replay"
VERSION = 1

_FACING_LABELS = {
    Direction.NORTH: "N",
    Direction.EAST: "E",
    Direction.SOUTH: "S",
    Direction.WEST: "W",
}


def _serialize_genome(genome: Genome, genome_id: int) -> Dict[str, Any]:
    return {
        "id": genome_id,
        "fitness": genome.fitness,
        "nodes": [
            {
                "id": node.id,
                "type": node.node_type.name,
                "bias": node.bias,
            }
            for node in genome.nodes.values()
        ],
        "connections": [
            {
                "in": conn.in_node,
                "out": conn.out_node,
                "weight": conn.weight,
                "enabled": conn.enabled,
                "innovation": conn.innovation,
            }
            for conn in genome.connections
        ],
    }


def _serialize_organism(organism: Organism) -> Dict[str, Any]:
    return {
        "x": organism.x,
        "y": organism.y,
        "facing": _FACING_LABELS[organism.facing],
        "energy": organism.energy,
        "alive": organism.alive,
        "action": organism.previous_action,
        "food_eaten": organism.food_eaten,
        "age": organism.age,
        "offspring": organism.offspring,
    }


class GenerationRecorder:
    """Builds a tick-by-tick replay of one generation's shared world."""

    def __init__(
        self,
        population: List[Genome],
        config: EnvironmentConfig,
        generation: Optional[int] = None,
    ) -> None:
        self.config = config
        self.generation = generation
        self.genome_ids: Dict[Genome, int] = {
            genome: index for index, genome in enumerate(population)
        }
        self.genomes: List[Dict[str, Any]] = [
            _serialize_genome(genome, index) for index, genome in enumerate(population)
        ]
        self.ticks: List[Dict[str, Any]] = []
        self._next_tick = 0
        self._organism_ids: Dict[int, int] = {}  # id(organism) -> stable integer
        self.context: Dict[str, Any] = {}  # optional population-level context for viewers

    # ------------------------------------------------------------- recording

    def record_tick(self, world: World, organisms: List[Organism]) -> None:
        """Snapshot the world state after one tick (or the initial placement)."""
        snapshot = {
            "tick": self._next_tick,
            "food": sorted(world.food),
            "organisms": [self._snapshot_organism(organism) for organism in organisms],
        }
        self.ticks.append(snapshot)
        self._next_tick += 1

    def _snapshot_organism(self, organism: Organism) -> Dict[str, Any]:
        stable_id = self._organism_ids.get(id(organism))
        if stable_id is None:
            stable_id = len(self._organism_ids)
            self._organism_ids[id(organism)] = stable_id
        snapshot = _serialize_organism(organism)
        snapshot["id"] = stable_id
        snapshot["genome"] = self.genome_ids[organism.genome]
        return snapshot

    # ------------------------------------------------------------- serialization

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": SCHEMA,
            "version": VERSION,
            "generation": self.generation,
            "config": asdict(self.config),
            "genomes": self.genomes,
            "ticks": self.ticks,
            "context": self.context,
        }


def record_generation_to_file(
    population: List[Genome],
    config: EnvironmentConfig,
    generation: int,
    path,
) -> GenerationRecorder:
    """Run one generation with recording and write the replay JSON to ``path``."""
    recorder = GenerationRecorder(population, config, generation)
    run_generation(population, config, generation, recorder=recorder)
    Path(path).write_text(json.dumps(recorder.to_dict()))
    return recorder
