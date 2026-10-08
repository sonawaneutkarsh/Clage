# Reconciled scientific contracts (engine 0.4.1)

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
