"""Clage artificial-life environment, built on the validated NEAT engine.

The world is a walled 2D grid with food, energy, movement, in-world asexual
reproduction, death, and resource regeneration. Organisms are controlled purely
by their evolved neural networks — no hard-coded behaviors. The environment
integrates with ``neat.Population`` through a batch ``evaluator`` hook, keeping
the engine independent of the world.
"""

from .config import (
    ACTION_SIZE,
    OBSERVATION_SIZE,
    Action,
    Direction,
    EnvironmentConfig,
)
from .fitness import fitness
from .grid import World
from .organism import Organism
from .recorder import GenerationRecorder, record_generation_to_file
from .simulation import make_evaluator, run_generation

__all__ = [
    "ACTION_SIZE",
    "OBSERVATION_SIZE",
    "Action",
    "Direction",
    "EnvironmentConfig",
    "GenerationRecorder",
    "Organism",
    "World",
    "fitness",
    "make_evaluator",
    "record_generation_to_file",
    "run_generation",
]
