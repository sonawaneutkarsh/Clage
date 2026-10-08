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

---

## Reconciled scientific contracts (engine 0.4.1)

This source implements the semantic specification in the recovered Work reports,
not the unavailable Work Git history. See `CLAGE_RECONCILIATION_REPORT.md`.

## Evolution and world lifecycle

Evaluate founders, archive the evaluated champion, speciate/share/allocate,
reproduce/mutate, preserve the exact heritable champion, then record statistics.
The returned population is unevaluated offspring; use `evaluated_best_genome` or
generation statistics for evaluated results. The all-time archive is a historical
world winner, not proof of performance on a fixed evaluation distribution.

Persistent endpoint innovations and split reuse are intentional implementation
variants, not a claim of canonical NEAT equivalence. Imported genomes register
their IDs/history before structural mutation; topology cannot recover split
ancestry. No complete evolutionary checkpoint/resume contract is implemented.

Bias inheritance and actual champion retention change evolutionary RNG trajectories.
Provenance marks `neat-reconciled-v1` and `world-observations-v2-odd-boundaries`.
Old Studio recordings remain viewable, but deterministic reruns require their
original source. Generation-replay v2 separately labels evaluated fitness; legacy
v1 scores are stored/unverified. Replays are observations, not resumable checkpoints.

## Interpretation limits

- Nine neural observations omit facing; argmax chooses the first tied output.
  Body births clone a genotype; evolutionary offspring may crossover/mutate it.
- Fitness is maximum body `3*food + .01*age + .5*offspring`, including descendants.
  Automatic reproduction and random placement can improve scores without foraging.
- `reproduction_cost.json` varies the eligibility threshold, not an energy charge.
  Transfer fraction controls the energy split; fractions 0/1 immediately remove an
  exhausted body while still counting the real birth. Odd-grid sensing is corrected.
- Available-space and founder/food sweeps change coupled densities/distances. They
  are not density-matched ablations. Shipped historical definitions stay unchanged.
- Behavioral diversity z-scores descriptors within each population, discarding
  absolute scale. Action/transition entropy and spatial coverage can reflect
  stochastic actions or initial placement; none proves meaningful policy diversity.
- Mean ± sample SD is descriptive variability, not a confidence interval.
  Generations are dependent; five seeds do not establish a general success rate.
  Equal seeds need not maintain identical layouts once trajectories diverge.
- `seed_stride` does not participate in `world_rng_seed`; the legacy experiment
  resolver uses it to map trial seeds to seed bases. Finite hashes can collide.
- Default sine remains unsolved. Execution probes (some outside mutation bounds)
  do not establish reachable solutions or rule out search/representation defects.
- Generalizable learning, cooperation, emergence, and policy superiority remain
  unproven. Studio's privileged forager is not an equal-information neural baseline.

## Preservation and reproducibility

Use fresh legacy experiment output directories. Manifests include actual source
digest, reported Git HEAD when available, runtime, resolved configs, seeds and the
semantic control. Source digests are essential for dirty checkouts. Interrupted
legacy runs are not resumable or atomically published. Studio retains its newer
frozen configs, bounded replay, ancestry and portable validation architecture.

Use a dedicated virtual environment: generic top-level package names such as
`neat` may collide with other installations. Wheels include six experiment configs
and Studio assets. Native GUI/OS coverage is not established by headless QA.
