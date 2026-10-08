"""Strict replay records, separate from orchestration and engine objects."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class NodeRecord(Record):
    id: int = Field(ge=0)
    type: Literal["INPUT", "HIDDEN", "OUTPUT"]
    bias: float


class EdgeRecord(Record):
    in_node: int = Field(alias="in", ge=0)
    out_node: int = Field(alias="out", ge=0)
    weight: float
    enabled: bool
    innovation: int = Field(ge=0)


class GenomeRecord(Record):
    id: int = Field(ge=0)
    key: str
    fitness: float | None
    nodes: list[NodeRecord] = Field(min_length=13, max_length=512)
    connections: list[EdgeRecord] = Field(max_length=4096)


class InferenceRecord(Record):
    inputs: list[float] = Field(min_length=9, max_length=9)
    outputs: list[float] = Field(min_length=4, max_length=4)
    values: dict[str | int, float]
    world_tick: int | None = Field(default=None, ge=1)


class BodyRecord(Record):
    id: int = Field(ge=0)
    genome: str
    parent: int | None
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    facing: int = Field(ge=0, le=3)
    energy: float
    alive: bool
    action: int | None = Field(ge=0, le=3)
    age: int = Field(ge=0)
    food_eaten: int = Field(ge=0)
    offspring: int = Field(ge=0)
    fitness: float = Field(ge=0)
    inference: InferenceRecord | None


class MetricRecord(Record):
    population: int = Field(ge=0)
    births: int = Field(ge=0)
    deaths: int = Field(ge=0)
    food_eaten: int = Field(ge=0)
    mean_energy: float
    actions: list[int] = Field(min_length=4, max_length=4)
    food: int = Field(ge=0)


class FrameRecord(Record):
    sequence: int = Field(ge=0)
    tick: int = Field(ge=0)
    generation: int = Field(ge=0)
    food: list[list[int] | tuple[int, int]]
    organisms: list[BodyRecord]
    metrics: MetricRecord


class HistoryRecord(Record):
    generation: int = Field(ge=1)
    population_size: int = Field(ge=1)
    species_count: int = Field(ge=0)
    best_fitness: float
    mean_fitness: float
    sizes: list[int]
    world_generation: int = Field(ge=0)
    fitnesses: list[float]
    champion: str
    mean_nodes: float
    mean_connections: float
    species: dict[str | int | None, int]


class WeightDelta(Record):
    innovation: int = Field(ge=0)
    before: float
    after: float


class EnabledDelta(Record):
    innovation: int = Field(ge=0)
    before: bool
    after: bool


class BiasDelta(Record):
    node: int = Field(ge=0)
    before: float
    after: float


class MutationDeltas(Record):
    added_nodes: list[int] = Field(max_length=512)
    added_innovations: list[int] = Field(max_length=4096)
    removed_innovations: list[int] = Field(max_length=4096)
    weight_changes: list[WeightDelta] = Field(max_length=4096)
    enabled_changes: list[EnabledDelta] = Field(max_length=4096)
    bias_changes: list[BiasDelta] = Field(max_length=512)


class EvolutionRecord(Record):
    parents: list[str | None] = Field(max_length=2)
    kind: Literal['founder', 'elite', 'clone', 'crossover', 'champion_copy', 'champion_rescue']
    deltas: MutationDeltas | None
