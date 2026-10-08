# Clage Studio scientific definitions and limitations

## Preserved engine and scientific definitions

The NEAT implementation is still from scratch; Studio does not substitute a
third-party optimizer. Historical results and their caveats remain preserved in
the README. Reconciliation corrected engine behavior and odd-grid boundary
sensing, versioned as `neat-reconciled-v1` and
`world-observations-v2-odd-boundaries`; the four actions and `world-fitness-v1`
formula remain unchanged. Historical environmental sweeps were not rerun.
Current-source OR/AND/XOR/sine validation is recorded separately in the final
release review and merge-preparation evidence; it does not replace old results.

**Observations:** nearest-food normalized Δx/Δy, local food/body density, energy,
wall proximity x/y, previous MOVE/EAT. Body facing is not an input. Actions are
MOVE, TURN LEFT, TURN RIGHT, EAT via deterministic argmax (first output on ties).
Network activations are actual feed-forward values from pre-action observations;
the displayed body's state is post-action. Timestamps distinguish older inference
on dead bodies. Instrumentation is observational, not a new policy definition.

**Fitness `world-fitness-v1`:** `3*food_consumed + .01*age + .5*offspring` per body;
genotype fitness is the maximum among its bodies. Live body values are provisional.
It rewards survival and reproduction as well as food; increasing it alone does
not establish learning. Generation evaluation uses a fresh world. Body births are
asexual genotype clones within a world; evolutionary reproduction selects/crosses/
mutates genotypes between evaluations. These genealogies are explicitly separate.

## Versioned initialization and recording

- Historical `minimal-v1`: input/output nodes without initial connections.
- Studio `dense-random-v1`: all nine-to-four edges, seeded uniform [-2,2] weights.
  This yields a more varied initial visual world, **not evidence of adaptation**.
  Config/provenance labels prevent mixing its results with historical experiments.
- Replay schema `clage-studio-replay` v2 adds evolutionary lineage v1. Imports
  accept Studio v1 with unavailable evolutionary history; no ancestry is inferred
  from topology. Net mutation deltas do not reveal every attempted operator.
- Frozen config, resolved defaults, software commit/dirty status, seed base,
  champions and per-generation summaries are saved. World RNG derives from the
  original `EnvironmentConfig.world_rng_seed(base,generation)` contract.

## Held-out evaluation v1

Five prespecified seed bases: 10001, 10002, 10003, 10004, 10005. Training base
overlap is rejected. Every policy gets one founder, identical config and seed
base per trial, for `min(training_ticks,300)` ticks without evolution/training.
In-world reproduction remains enabled and offspring participate. Thus reported
food/survivors/births are cohort outcomes, not per-founder-only action statistics.

Policies: fixed MOVE; uniform random action with separate `Random(seed+991)`;
hand-coded directional forager; saved evaluated champion. The forager uses
**privileged facing information** unavailable to the original neural observation
space. It is a useful reference, not an equal-information baseline for a claim of
learning. Initial placement/food are matched; subsequent occupancy and reproduction
affect random empty-cell selection, so later resource layouts need not match.

Exports report actual per-seed food consumption, survivors, births, maximum body
fitness and budgets. UI mean ± **sample SD**, n=5, is descriptive variability,
not a confidence interval or significance test. Five worlds/one champion do not
measure variation across independent training runs. Custom seed sets are labeled
separately by the Python evaluation function. No results in demonstration screenshots
are presented as scientific discoveries.

## Verified claims versus open questions

Verified: instrumented outputs/legacy world order agree; optional ancestry logging
preserves genotype/RNG sequences; declared performance workloads were measured;
saved observations can be verified by rerunning frozen configurations. Archive
validation checks numerical/topological consistency but does not authenticate
that claimed observations occurred; rerun comparison is stronger evidence.

Not established: learned foraging, policy superiority, robust generalization,
cooperation/competition, emergence, or superiority of dense initialization. No
automatic adoption of alternative settings based on training fitness. Facing-aware
neural sensing, alternate fitness/reproduction rules, multi-training-seed uncertainty
and prespecified controlled ablations require separate versioned research work.

Behavioral diversity functions remain available through the original experiment
pipeline. Studio's last-action histogram is **not** that behavioral-diversity metric
or cumulative action frequency. Avoid comparing food alignment across resource
densities; retain the existing correctness notes and historical interpretations.
