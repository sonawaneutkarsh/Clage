"""The organism: a genome + decoded network embedded in the world.

The organism is a pure body — it reads observations, emits an action via its
evolved network, and its state (position, facing, energy, food eaten, age) is
updated by the environment. There is no hard-coded behavior: nothing tells it
to seek food or avoid others. Whatever strategy emerges comes entirely from
observation -> network -> action.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from neat.genome import Genome
from neat.phenotype import Network

from .config import ACTION_SIZE, OBSERVATION_SIZE, Action, Direction, EnvironmentConfig
from .grid import FOOD, World

__all__ = ["Organism"]


class Organism:
    def __init__(
        self,
        genome: Genome,
        x: int,
        y: int,
        config: EnvironmentConfig,
        facing: Tuple[int, int] = Direction.NORTH,
        energy: Optional[float] = None,
    ) -> None:
        self.genome = genome
        self.network = Network(genome)
        self.x = x
        self.y = y
        self.facing = facing
        self.energy = config.initial_energy if energy is None else energy
        self.food_eaten = 0
        self.age = 0
        self.alive = True
        self.offspring = 0
        self.previous_action: Optional[int] = None
        # per-tick behavioral trace: (action, x, y, food_dx, food_dy, organism_density)
        self.trace: List[Tuple[int, int, int, float, float, float]] = []

    # ------------------------------------------------------------- observation

    def observe(self, world: World, config: EnvironmentConfig) -> List[float]:
        """The 9-number observation vector (see progress/world.md for rationale)."""
        radius = max(world.width, world.height) / 2.0

        nearest = world.nearest_food(self.x, self.y)
        if nearest is None:
            dx_food, dy_food = 0.0, 0.0
        else:
            dx_food = max(-1.0, min(1.0, (nearest[0] - self.x) / radius))
            dy_food = max(-1.0, min(1.0, (nearest[1] - self.y) / radius))

        half_x = max(1, world.width // 2 - 1)
        half_y = max(1, world.height // 2 - 1)
        # boundary proximity: 1.0 at the wall, 0.0 toward the center
        boundary_x = 1.0 - min(self.x, world.width - 1 - self.x) / half_x
        boundary_y = 1.0 - min(self.y, world.height - 1 - self.y) / half_y

        return [
            dx_food,
            dy_food,
            world.food_density(self.x, self.y, config.density_radius),
            world.organism_density(self.x, self.y, config.density_radius, exclude=self),
            self.energy / config.max_energy,
            boundary_x,
            boundary_y,
            1.0 if self.previous_action == Action.MOVE else 0.0,
            1.0 if self.previous_action == Action.EAT else 0.0,
        ]

    # ------------------------------------------------------------- act

    def act(self, world: World, config: EnvironmentConfig) -> Optional["Organism"]:
        """One full tick for this organism: sense, act, metabolize, maybe split."""
        observation = self.observe(world, config)
        outputs = self.network.activate(observation)
        action = max(range(len(outputs)), key=lambda i: outputs[i])
        self.previous_action = action
        self._apply_action(action, world, config)

        if config.record_trace:
            self.trace.append(
                (action, self.x, self.y, observation[0], observation[1], observation[3])
            )

        self.energy -= config.metabolism
        self.age += 1

        if self.energy <= 0.0:
            self.alive = False
            world.remove_organism(self)
            return None

        return self._try_reproduce(world, config)

    def _apply_action(self, action: int, world: World, config: EnvironmentConfig) -> None:
        if action == Action.TURN_LEFT:
            self.facing = Direction.turn_left(self.facing)
        elif action == Action.TURN_RIGHT:
            self.facing = Direction.turn_right(self.facing)
        elif action == Action.EAT:
            self._consume(world, config, self.x + self.facing[0], self.y + self.facing[1])
        elif action == Action.MOVE:
            tx, ty = self.x + self.facing[0], self.y + self.facing[1]
            if world.occupant(tx, ty) is FOOD:
                self._consume(world, config, tx, ty)
                world.move_organism(self, tx, ty)
            elif world.is_empty(tx, ty):
                world.move_organism(self, tx, ty)
            # blocked by an organism or a wall -> stay put

    def _consume(self, world: World, config: EnvironmentConfig, x: int, y: int) -> None:
        if (x, y) in world.food:
            world.remove_food(x, y)
            self.energy = min(self.energy + config.food_energy, config.max_energy)
            self.food_eaten += 1

    # ------------------------------------------------------------- reproduction

    def _try_reproduce(self, world: World, config: EnvironmentConfig) -> Optional["Organism"]:
        if self.energy < config.repro_threshold:
            return None
        empty = world.adjacent_empty(self.x, self.y)
        if not empty:
            return None

        child_energy = self.energy * config.repro_fraction
        self.energy -= child_energy
        cx, cy = world.rng.choice(empty)

        child = Organism(
            genome=self.genome,
            x=cx,
            y=cy,
            config=config,
            facing=self.facing,
            energy=child_energy,
        )
        world.place_organism(child)
        self.offspring += 1
        return child
