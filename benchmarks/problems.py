"""Controlled benchmark problems for validating the NEAT engine.

Every problem defines: its inputs, expected outputs, fitness function, and a
success criterion. Fitness is a pure function of the decoded network — the
engine's ``fitness_fn(genome, generation)`` signature — so the world only ever
enters through the phenotype.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

from neat.genome import Genome
from neat.phenotype import Network

__all__ = ["Problem", "OR", "AND", "XOR", "SIN", "PROBLEMS"]

Case = Tuple[Sequence[float], float]


@dataclass(frozen=True)
class Problem:
    name: str
    input_ids: Tuple[int, ...]
    output_ids: Tuple[int, ...]
    cases: Tuple[Case, ...]
    fitness_fn: Callable[[Genome, int], float]
    success_fn: Callable[[Genome], bool]
    success_description: str


def mse_fitness(genome: Genome, generation: int, cases: Tuple[Case, ...]) -> float:
    """Fitness = 1 / (1 + MSE) of the decoded network over the training cases."""
    net = Network(genome)
    error = 0.0
    for inputs, expected in cases:
        output = net.activate(list(inputs))[0]
        error += (output - expected) ** 2
    return 1.0 / (1.0 + error / len(cases))


def classify_ok(genome: Genome, cases: Tuple[Case, ...]) -> bool:
    """True iff every case is classified correctly (|output - target| < 0.5)."""
    net = Network(genome)
    for inputs, expected in cases:
        if abs(net.activate(list(inputs))[0] - expected) >= 0.5:
            return False
    return True


def mae_under(genome: Genome, cases: Tuple[Case, ...], limit: float = 0.1) -> bool:
    """True iff mean absolute error over the cases is below ``limit``."""
    net = Network(genome)
    error = sum(
        abs(net.activate(list(inputs))[0] - expected) for inputs, expected in cases
    )
    return error / len(cases) < limit


def _boolean_problem(
    name: str,
    cases: Tuple[Case, ...],
) -> Problem:
    def fitness(genome: Genome, generation: int) -> float:
        return mse_fitness(genome, generation, cases)

    return Problem(
        name=name,
        input_ids=(0, 1),
        output_ids=(10,),
        cases=cases,
        fitness_fn=fitness,
        success_fn=lambda g: classify_ok(g, cases),
        success_description=f"all {len(cases)} truth-table rows classified with |out-y| < 0.5",
    )


def _truth_table(rows: Sequence[Tuple[float, float, float]]) -> Tuple[Case, ...]:
    return tuple(((x0, x1), y) for x0, x1, y in rows)


OR: Problem = _boolean_problem(
    "OR",
    _truth_table([(0, 0, 0.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 1.0)]),
)

AND: Problem = _boolean_problem(
    "AND",
    _truth_table([(0, 0, 0.0), (0, 1, 0.0), (1, 0, 0.0), (1, 1, 1.0)]),
)

XOR: Problem = _boolean_problem(
    "XOR",
    _truth_table([(0, 0, 0.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 0.0)]),
)


def _sin_cases(n: int = 21) -> Tuple[Case, ...]:
    points = [i / (n - 1) for i in range(n)]
    return tuple(((x,), math.sin(2.0 * math.pi * x)) for x in points)


SIN: Problem = Problem(
    name="sin",
    input_ids=(0,),
    output_ids=(10,),
    cases=_sin_cases(),
    fitness_fn=lambda g, generation: mse_fitness(g, generation, _sin_cases()),
    success_fn=lambda g: mae_under(g, _sin_cases(), limit=0.1),
    success_description="y = sin(2*pi*x) over 21 samples on [0,1], MAE < 0.1",
)

PROBLEMS = {"or": OR, "and": AND, "xor": XOR, "sin": SIN}
