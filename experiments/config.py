"""Experiment configuration: schema, one-factor-at-a-time parameters, resolution.

Everything about an experiment lives in a single JSON file:
  - ``base``: the fixed NEAT + world + interface configuration.
  - ``conditions``: each changes exactly ONE semantic parameter (the control
    changes none) — one-factor-at-a-time enforcement.
  - ``seeds``: the independent random seeds, reused across every condition.

``extends`` lets a sweep file inherit the base config from another file (e.g.
``base.json``) and only override the conditions.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from neat.mutation import MutationConfig
from neat.speciation import SpeciationConfig
from world import EnvironmentConfig

__all__ = [
    "PARAMETERS",
    "Condition",
    "ResolvedConfig",
    "ExperimentConfig",
    "load_experiment",
    "resolve_config",
]

# Each semantic environmental parameter maps to the config fields it touches.
# A condition declares ONE parameter; the framework applies its value here.
PARAMETERS: Dict[str, Tuple[str, ...]] = {
    "food_abundance": ("world.initial_food", "world.food_target"),
    "resource_scarcity": ("world.initial_food", "world.food_target"),
    "food_regeneration_rate": ("world.food_regrowth_per_tick",),
    "population_density": ("neat.population_size",),
    "available_space": ("world.width", "world.height"),
    "reproduction_threshold": ("world.repro_threshold",),
    "reproduction_fraction": ("world.repro_fraction",),
}


@dataclass
class Condition:
    name: str
    parameter: Optional[str]
    value: Any = None

    @property
    def is_control(self) -> bool:
        return self.parameter is None

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "parameter": self.parameter, "value": self.value}


@dataclass
class ResolvedConfig:
    """The fully resolved parameters for one condition (optionally one seed)."""

    population_size: int
    generations: int
    elitism: int
    crossover_rate: float
    mutation_config: MutationConfig
    speciation_config: SpeciationConfig
    world: EnvironmentConfig
    input_ids: List[int]
    output_ids: List[int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "neat": {
                "population_size": self.population_size,
                "generations": self.generations,
                "elitism": self.elitism,
                "crossover_rate": self.crossover_rate,
                "mutation": asdict(self.mutation_config),
                "speciation": asdict(self.speciation_config),
            },
            "world": asdict(self.world),
            "interface": {
                "input_ids": self.input_ids,
                "output_ids": self.output_ids,
            },
        }


class ExperimentConfig:
    def __init__(
        self,
        name: str,
        base: Dict[str, Any],
        conditions: List[Condition],
        seeds: List[int],
    ) -> None:
        self.name = name
        self.base = base
        self.conditions = conditions
        self.seeds = seeds
        self._validate()

    def _validate(self) -> None:
        for condition in self.conditions:
            if condition.parameter is not None and condition.parameter not in PARAMETERS:
                raise ValueError(
                    f"condition {condition.name!r}: unknown parameter "
                    f"{condition.parameter!r} (known: {sorted(PARAMETERS)})"
                )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "base": self.base,
            "seeds": self.seeds,
            "conditions": [c.to_dict() for c in self.conditions],
        }

    def condition(self, name: str) -> Condition:
        for condition in self.conditions:
            if condition.name == name:
                return condition
        raise KeyError(f"no condition named {name!r}")


# ------------------------------------------------------------------ loading


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_experiment(path: Path) -> ExperimentConfig:
    """Load a JSON experiment file, resolving an optional ``extends`` base."""
    raw = json.loads(Path(path).read_text())

    base: Dict[str, Any] = {}
    extends = raw.get("extends")
    if extends:
        extends_path = Path(path).parent / extends
        parent = json.loads(extends_path.read_text())
        base = _deep_merge(base, parent["base"])

    base = _deep_merge(base, raw.get("base", {}))

    seeds = raw.get("seeds") or ([0, 1, 2, 3, 4] if not extends else parent.get("seeds", []))
    conditions = [
        Condition(name=c["name"], parameter=c.get("parameter"), value=c.get("value"))
        for c in raw["conditions"]
    ]
    return ExperimentConfig(
        name=raw["name"],
        base=base,
        conditions=conditions,
        seeds=list(seeds),
    )


def _set_path(target: Dict[str, Any], dotted_path: str, value: Any) -> None:
    parts = dotted_path.split(".")
    node = target
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value


def resolve_config(
    experiment: ExperimentConfig,
    condition: Condition,
    seed: Optional[int] = None,
) -> ResolvedConfig:
    """Build the resolved config for a condition, applying the single change."""
    base = copy.deepcopy(experiment.base)
    if not condition.is_control:
        for path in PARAMETERS[condition.parameter]:
            _set_path(base, path, condition.value)

    neat = base["neat"]
    world = EnvironmentConfig(**base["world"])
    if seed is not None:
        world = replace(world, seed_base=seed * world.seed_stride)

    mutation = dict(neat.get("mutation", {}))
    if "weight_bounds" in mutation:
        mutation["weight_bounds"] = tuple(mutation["weight_bounds"])

    return ResolvedConfig(
        population_size=neat["population_size"],
        generations=neat["generations"],
        elitism=neat.get("elitism", 1),
        crossover_rate=neat.get("crossover_rate", 0.75),
        mutation_config=MutationConfig(**mutation),
        speciation_config=SpeciationConfig(**neat.get("speciation", {})),
        world=world,
        input_ids=list(base["interface"]["input_ids"]),
        output_ids=list(base["interface"]["output_ids"]),
    )
