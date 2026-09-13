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
        # The unknown-parameter check stays FIRST and verbatim. Ordering is
        # deliberate: a config with a single non-control condition has zero
        # controls, so the control-count rule below would otherwise shadow the
        # real problem (an unknown parameter) with a secondary one.
        for condition in self.conditions:
            if condition.parameter is not None and condition.parameter not in PARAMETERS:
                raise ValueError(
                    f"condition {condition.name!r}: unknown parameter "
                    f"{condition.parameter!r} (known: {sorted(PARAMETERS)})"
                )

        # Duplicate names: a second condition with the same name is unreachable
        # through ``condition(name)`` and writes over the first one's results.
        seen: List[str] = []
        for condition in self.conditions:
            if condition.name in seen:
                raise ValueError(
                    f"condition {condition.name!r}: duplicate condition name "
                    f"(names must be unique; got {[c.name for c in self.conditions]})"
                )
            seen.append(condition.name)

        # Exactly one control: reporting and comparison both need a single
        # identifiable baseline. ``is_control`` is ``parameter is None``.
        controls = [c for c in self.conditions if c.is_control]
        if len(controls) != 1:
            if not controls:
                problem = (
                    "found 0 controls, exactly one is required "
                    "(a control declares no parameter; the reporting layer "
                    "conventionally names it 'control')"
                )
                raise ValueError(f"experiment {self.name!r}: {problem}")
            names = [c.name for c in controls]
            raise ValueError(
                f"condition {names[0]!r}: found {len(controls)} controls "
                f"({names}), exactly one is required"
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


_DEFAULT_SEEDS = (0, 1, 2, 3, 4)


def load_experiment(path: Path) -> ExperimentConfig:
    """Load a JSON experiment file, resolving an optional ``extends`` base."""
    path = Path(path)
    raw = json.loads(path.read_text())

    if "conditions" not in raw:
        raise ValueError(
            f"{path}: missing required key 'conditions' "
            f"(expected top-level keys: 'name', 'conditions', and either "
            f"'base' or 'extends'; optional: 'seeds')"
        )

    base: Dict[str, Any] = {}
    extends = raw.get("extends")
    if extends:
        # A missing target keeps its plain FileNotFoundError — already clear.
        extends_path = path.parent / extends
        parent = json.loads(extends_path.read_text())
        if "base" not in parent:
            raise ValueError(
                f"{path}: 'extends' target {extends_path} provides no 'base'; "
                f"only one level of 'extends' is supported"
            )
        base = _deep_merge(base, parent["base"])

    base = _deep_merge(base, raw.get("base", {}))

    # "absent" and "explicitly empty" are different answers: absent inherits,
    # empty is a request for zero trials and is rejected. ``raw.get(...) or``
    # could not tell them apart.
    if "seeds" in raw:
        seeds = raw["seeds"]
        if not seeds:
            raise ValueError(
                f"{path}: 'seeds' is empty; a condition with no seeds runs zero trials"
            )
    elif extends:
        seeds = parent.get("seeds", [])
        if not seeds:
            raise ValueError(
                f"{path}: 'seeds' is empty; a condition with no seeds runs zero "
                f"trials (no 'seeds' here and none inherited from {extends_path})"
            )
    else:
        seeds = _DEFAULT_SEEDS

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
    # The fields validate themselves; this only adds the context the loader
    # has and the config does not — which experiment, which condition, and
    # which parameter value produced the offending field. The exception type is
    # preserved (TypeError stays TypeError) and chained, so the original
    # field-level message survives.
    try:
        world = EnvironmentConfig(**base["world"])
    except (TypeError, ValueError) as exc:
        origin = (
            "control condition applies no parameter"
            if condition.is_control
            else f"parameter {condition.parameter!r} = {condition.value!r}"
        )
        raise type(exc)(
            f"experiment {experiment.name!r}, condition {condition.name!r} "
            f"({origin}): {exc}"
        ) from exc
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
