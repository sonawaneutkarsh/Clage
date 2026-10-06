# Correctness notes

Two rounds of bug-hunting (September 2026) changed what some of Clage's numbers
mean. This page records what was wrong, what the code does now, and which old
results are no longer comparable. Each fix has a deterministic regression test
in `tests/`.

## 1. Evaluation and measurement

### Benchmark rows described an unevaluated genome

After `Population.run(1)` returns, `pop.population` already holds the *offspring*
of the generation that just finished. Their fitness is copied or stale.
`benchmarks/run.py` used to read `solved`, node count and connection count from
`max(pop.population, key=fitness)`, while `best_fitness` in the same row came
from `pop.statistics[-1]`. A trial could be declared solved, and stopped early,
from a genome that was never evaluated in that generation.

Fix: `Population.evaluated_best_genome` is a public accessor that returns a
defensive copy of the champion that was actually evaluated (or `None` before the
first generation). Every field of a benchmark row now comes from that one
genome. The regression test uses a problem whose fitness is always `-1.0` and
whose success rule is `fitness >= 0.0`: before the fix it was "solved" in
generation 1.

### Transition entropy crossed organism boundaries

All organisms that share a genome were concatenated into one action list before
counting `(a_t, a_t+1)` pairs, so the last action of one organism was counted as
the predecessor of the first action of the next. In-world reproduction makes
multi-organism genomes common, so these fake transitions were frequent.

Fix: only pairs that are consecutive inside one organism's trace are counted.
Counts are pooled across the genome's organisms and normalized by the total
number of within-trace transitions. With one organism the value equals
`transition_entropy_rate` on that single sequence.

### Food alignment was off by one step

A trace entry mixes time references: `action` and the food direction are
observed *before* the action, `(x, y)` is the position *after* it. The metric
paired the displacement caused by action *i* with the food direction seen before
action *i-1*.

Fix: each displacement `pos_i - pos_(i-1)` is paired with the food direction
stored at index *i*. The first action of each trace is dropped, because the
position before it is never recorded. The trace layout is documented on
`Organism.trace`.

**Comparability:** results produced before this fix are not comparable for
`transition_entropy`, `food_alignment` and `behavioral_diversity`, or for
benchmark `solved` / `generations_to_solve` / topology columns.

## 2. Boundary and configuration validation

Invalid configurations used to fail deep inside a run with an unrelated error,
or not fail at all. They now fail before the simulation starts, with a message
that names the field.

| Boundary | Old behavior (examples) | Now |
|---|---|---|
| NEAT ↔ world interface (9 observations, 4 actions) | Default 10-input genomes crashed mid-run; 0 outputs crashed in `max()`; 6 outputs made some ticks silently do nothing; 2 outputs made `TURN_RIGHT`/`EAT` unreachable | `world.simulation._check_interface` rejects a mismatched interface at evaluator entry. `neat` still has no dependency on `world`. |
| Population ↔ grid capacity | More founders than cells crashed with a `NoneType` unpacking error | `_check_capacity` raises a clear error first |
| `EnvironmentConfig` numeric fields | `density_radius=0` and `max_energy=0` divided by zero; negative `metabolism` made organisms gain energy; `repro_fraction > 1` created energy | Validated at construction |
| Experiment files | Duplicate condition names, zero or several controls, unknown condition names, nested `extends`, empty `seeds` were accepted | Rejected by the loader / `ExperimentConfig` |
| Fitness and offspring allocation | NaN/inf fitness crashed with a misleading "NaN" message, or `-inf` became `best_fitness`; mixed-sign adjusted fitness gave negative budgets and a population larger than `population_size` | Non-finite fitness is rejected with its population index; allocation floors species weights at 0, so budgets are nonnegative and sum to `population_size` |

Rules that would only encode taste (for example, banning negative fitness) were
deliberately not added: negative fitness is legal and simply confers no
selective advantage, which matches `Population._select_parent`.
