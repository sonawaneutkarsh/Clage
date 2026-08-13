"""Layer-by-layer failure diagnosis for benchmarks that don't solve.

Each check is a small, self-contained probe of one engine layer. When a
benchmark fails, these isolate whether the problem is network execution, genome
representation, innovation tracking, mutation, crossover, speciation, fitness
design, or parameterization.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from neat.genome import ConnectionGene, Genome, NodeGene, NodeType
from neat.innovation import InnovationDB
from neat.crossover import crossover
from neat.mutation import (
    mutate_add_connection,
    mutate_add_node,
    mutate_biases,
    mutate_perturb_weights,
)
from neat.phenotype import Network
from neat.speciation import Speciation, SpeciationConfig, compatibility_distance

from .problems import AND, OR, XOR, PROBLEMS, Problem
from .run import TrialResult, run_trial

__all__ = ["diagnose_failures", "diagnose_problem"]

Check = Tuple[bool, str]


# ------------------------------------------------------------- layer probes


def check_network_execution() -> Check:
    g = Genome.minimal(input_ids=[0], output_ids=[10])
    g.connections.append(ConnectionGene(in_node=0, out_node=10, weight=2.0, innovation=1))
    g.nodes[10].bias = 0.5
    net = Network(g)
    for x in (0.0, 0.3, 0.7, 1.0):
        import math

        expected = math.tanh(2.0 * x + 0.5)
        got = net.activate([x])[0]
        if abs(got - expected) > 1e-9:
            return False, f"forward pass mismatch at x={x}: got {got}, expected {expected}"
    return True, "Network.activate matches a hand-computed forward pass (tanh(2x+0.5))"


def check_genome_representation() -> Check:
    g = Genome.minimal(input_ids=[0, 1], output_ids=[10])
    g.add_node(NodeGene(id=50, node_type=NodeType.HIDDEN, bias=0.2))
    g.add_connection(ConnectionGene(in_node=0, out_node=50, weight=1.0, innovation=1))
    g.add_connection(
        ConnectionGene(in_node=50, out_node=10, weight=1.0, innovation=2, enabled=False)
    )
    g.validate()
    c = g.copy()
    if set(c.nodes) != set(g.nodes) or c.nodes[50].bias != 0.2:
        return False, "copy() changed node ids/bias"
    if c.connections[1].enabled or c.connections[1].innovation != 2:
        return False, "copy() changed enabled state or innovation"
    if c.has_cycle():
        return False, "copy() introduced a cycle"
    return True, "genome builds, validates, copies, and preserves ids/enabled/innovation"


def check_innovation_tracking() -> Check:
    db = InnovationDB()
    first = db.connection_innovation(0, 10)
    if db.connection_innovation(0, 10) != first:
        return False, "same pair got a different innovation number"
    if db.connection_innovation(1, 11) == first:
        return False, "different pair collided on an innovation number"
    ni = db.add_node_innovation(0, 10)
    if db.add_node_innovation(0, 10) != ni:
        return False, "same split did not reuse node id + innovations"
    return True, "innovation ledger reuses identical inventions, mints distinct ones"


def check_mutation() -> Check:
    g = Genome.minimal(input_ids=[0, 1], output_ids=[10])
    g.add_connection(ConnectionGene(in_node=0, out_node=10, weight=0.5, innovation=1))
    import random

    rng = random.Random(0)
    db = InnovationDB()
    db.connection_innovation(0, 10)
    mutate_perturb_weights(g, rng, prob=1.0)
    mutate_biases(g, rng, prob=1.0)
    mutate_add_connection(g, rng, db)
    mutate_add_node(g, rng, db)
    try:
        g.validate()
    except ValueError as exc:
        return False, f"mutation produced an invalid genome: {exc}"
    if g.has_cycle():
        return False, "mutation produced a cycle"
    return True, "all mutation operators leave a valid, acyclic genome"


def check_crossover() -> Check:
    a = Genome.minimal(input_ids=[0, 1], output_ids=[10])
    b = Genome.minimal(input_ids=[0, 1], output_ids=[10, 11])
    a.add_connection(ConnectionGene(in_node=0, out_node=10, weight=1.0, innovation=1))
    a.add_connection(ConnectionGene(in_node=1, out_node=10, weight=2.0, innovation=2))
    b.add_connection(ConnectionGene(in_node=0, out_node=10, weight=3.0, innovation=1))
    b.add_connection(ConnectionGene(in_node=0, out_node=11, weight=4.0, innovation=3))
    import random

    child = crossover(a, b, random.Random(0))
    parent_innovs = {c.innovation for c in a.connections} | {c.innovation for c in b.connections}
    try:
        child.validate()
    except ValueError as exc:
        return False, f"crossover child invalid: {exc}"
    if not {c.innovation for c in child.connections} <= parent_innovs:
        return False, "child has an innovation number not present in the parents"
    return True, "crossover children are valid and only inherit parent genes"


def check_speciation() -> Check:
    config = SpeciationConfig()
    g = Genome.minimal(input_ids=[0], output_ids=[10])
    if compatibility_distance(g, g.copy(), config) != 0.0:
        return False, "identical genomes have nonzero compatibility distance"
    h = Genome.minimal(input_ids=[0], output_ids=[10])
    h.add_connection(ConnectionGene(in_node=0, out_node=10, weight=1.0, innovation=1))
    distance = compatibility_distance(g, h, config)
    if distance >= config.compatibility_threshold:
        return False, (
            f"one-extra-connection genome split into a different species "
            f"(distance {distance:.2f} >= threshold {config.compatibility_threshold})"
        )
    spe = Speciation(config)
    spe.speciate([g, g.copy(), h])
    if len(spe.species) < 1:
        return False, "speciation produced no species"
    return True, (
        f"compatibility distance ({distance:.2f} for one extra connection) and "
        "species assignment behave correctly"
    )


def _hand_solution(problem: Problem) -> Optional[Genome]:
    if problem is OR:
        g = Genome.minimal(input_ids=[0, 1], output_ids=[10])
        g.add_connection(ConnectionGene(in_node=0, out_node=10, weight=0.8, innovation=1))
        g.add_connection(ConnectionGene(in_node=1, out_node=10, weight=0.8, innovation=2))
        return g
    if problem is AND:
        g = Genome.minimal(input_ids=[0, 1], output_ids=[10])
        g.add_connection(ConnectionGene(in_node=0, out_node=10, weight=0.4, innovation=1))
        g.add_connection(ConnectionGene(in_node=1, out_node=10, weight=0.4, innovation=2))
        return g
    if problem is XOR:
        g = Genome.minimal(input_ids=[0, 1], output_ids=[10])
        g.add_node(NodeGene(id=50, node_type=NodeType.HIDDEN, bias=-30.0))
        g.add_node(NodeGene(id=51, node_type=NodeType.HIDDEN, bias=-10.0))
        g.add_connection(ConnectionGene(in_node=0, out_node=50, weight=20.0, innovation=1))
        g.add_connection(ConnectionGene(in_node=1, out_node=50, weight=20.0, innovation=2))
        g.add_connection(ConnectionGene(in_node=0, out_node=51, weight=20.0, innovation=3))
        g.add_connection(ConnectionGene(in_node=1, out_node=51, weight=20.0, innovation=4))
        g.add_connection(ConnectionGene(in_node=50, out_node=10, weight=-20.0, innovation=5))
        g.add_connection(ConnectionGene(in_node=51, out_node=10, weight=20.0, innovation=6))
        return g
    return None


def check_fitness_design(problem: Problem) -> Check:
    empty = Genome.minimal(input_ids=list(problem.input_ids), output_ids=list(problem.output_ids))
    empty_fitness = problem.fitness_fn(empty, 0)
    if not (0.0 < empty_fitness <= 1.0):
        return False, f"fitness of an empty genome out of (0,1]: {empty_fitness}"

    solution = _hand_solution(problem)
    if solution is not None:
        fit = problem.fitness_fn(solution, 0)
        solved = problem.success_fn(solution)
        if not solved:
            return False, "hand-built solution fails the success criterion"
        if fit <= empty_fitness:
            return False, "hand-built solution is not fitter than an empty genome"
        return (
            True,
            f"hand-built solution solves {problem.name} (fitness {fit:.4f}), "
            f"empty genome fitness {empty_fitness:.4f}",
        )
    # continuous problem: verify plumbing without a hand solution
    return True, f"fitness fn returns {empty_fitness:.4f} in (0,1] for the empty genome"


# --------------------------------------------------------------- orchestrator


def diagnose_problem(problem: Problem, seed: int) -> str:
    checks: List[Tuple[str, Check]] = [
        ("network execution", check_network_execution()),
        ("genome representation", check_genome_representation()),
        ("innovation tracking", check_innovation_tracking()),
        ("mutation", check_mutation()),
        ("crossover", check_crossover()),
        ("speciation", check_speciation()),
        ("fitness design", check_fitness_design(problem)),
    ]

    lines = ["| layer | result | detail |", "|---|---|---|"]
    all_ok = True
    for name, (ok, detail) in checks:
        all_ok = all_ok and ok
        lines.append(f"| {name} | {'PASS' if ok else 'FAIL'} | {detail} |")

    # parameterization: extra seeded trials
    extra = [run_trial(problem, s) for s in range(seed, seed + 3)]
    solved = sum(1 for r in extra if r.solved)
    lines.append(
        f"\nParameterization probe (3 extra seeds {seed}-{seed+2}): "
        f"{solved}/3 solved."
    )

    if all_ok and solved == 0:
        suspect = "search/parameterization: the layers work; solutions exist but aren't found "
        "with defaults — try more generations, larger population, or higher structural "
        "mutation rates."
    elif all_ok:
        suspect = "borderline parameterization / seed luck (some seeds solve)."
    else:
        suspect = "the failing layers above are the prime suspects."
    lines.append(f"\n**Most likely suspect:** {suspect}")
    return "\n".join(lines)


def diagnose_failures(
    results_by_problem: Dict[str, List[TrialResult]],
    failures: Dict[str, List[TrialResult]],
) -> Dict[str, str]:
    diagnoses = {}
    for name in failures:
        problem = PROBLEMS[name]
        probe_seed = results_by_problem[name][0].seed + 100
        diagnoses[name] = diagnose_problem(problem, probe_seed)
    return diagnoses
