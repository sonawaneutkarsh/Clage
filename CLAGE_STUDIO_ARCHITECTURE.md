# Clage Studio architecture and implementation plan

## Recovery audit — October 6, 2026 (local date)

Initial checkout: clean `main` at `9bc763a`, version 0.3.0, **188 passing tests**.
Existing features: custom NEAT, deterministic grid worlds, reproduction,
environment sweeps, diversity metrics, JSON replay, terminal/matplotlib viewers.
These are previous work, not new Studio features. Kiro attribution cannot be
proved from the checkout alone.

The supplied `bec8c03a02242d2effdeb6e553c3fdf05c1931ce` is not in local objects.
The audit branch and reports are absent. `git fsck --full --no-reflogs` found no
dangling recovery objects; the original reflog only recorded the clone. Fetch
recovered newer descendants of local main ending at `2105fee`, including
`ea8a9c2` (evaluation/measurement fixes) and `5a00427` (boundary validation).
These independently pass **300 tests**, not the reported 349. They are not
recovery of the reported 0.4.0 implementation. No missing branch was pushed and
no audit PR was fabricated. Another workspace or bundle is needed to recover it.

`studio/clage-1.0` starts at inspected `2105fee`. Local `main` is unchanged.
No default branch overwrite or merge is authorized.

## Actual gaps

Blocking generation execution, no web application or inference telemetry,
unbounded replay snapshots, no ancestry recording, no complete checkpoints.
Older README test counts were stale; docstrings refer to removed progress docs.
Remote `docs/correctness-notes.md` identifies incomparable historical metrics.
Preserve existing outputs and scientific definitions.

## Implementation phases

1. Incremental world lifecycle retaining ordering/RNG; observational inference
   instrumentation and within-world parent links; parity tests.
2. Local API, locked background orchestration, validated frozen configurations,
   bounded recording and deterministic evolution boundaries.
3. Canvas world, camera/controls, SVG neural inspector, analytics and replay.
4. Generation history, genome comparison, seeded baselines, provenance/exports.
5. API/browser tests, visual QA, declared performance workloads and honest reports.

## ADR 001: local-first technology

Keep Python NEAT independent. World owns a shared incremental lifecycle used by
legacy batch execution. Studio owns contracts, orchestration, API and browser
assets. FastAPI/Uvicorn serve a loopback-only same-origin application. No cloud,
database server, credentials or paid service is needed.

Use buildless native ES modules, Canvas 2D for world/charts and SVG for networks.
React/Vite/Pixi are reasonable future scaling choices, not prerequisites for
this single-page vertical slice. Avoid a new bundler/dependency tree where direct
Canvas drawing already avoids per-organism DOM overhead. Test pure JS with Node
and workflows with Playwright. Measure rather than claim unlimited scalability.

## ADR 002: independent clocks

A dedicated worker thread advances the engine, guarded by a lock. WebSocket
snapshots are capped separately from requestAnimationFrame rendering. Pause
freezes advancement; requested and achieved tick rates are separate. No dropped
simulation ticks or catch-up bursts. One active local experiment per process;
new starts explicitly replace it. Configuration is frozen until reset/start.

## ADR 003: observations, not checkpoints

Studio's versioned replay stores recent frames, configurations, initialization
definition, software provenance and evaluated summaries. The recording window
has an explicit budget and exports declare missing earlier frames. Imports are
validated. Seeking plays recorded states only. No branch/resume/checkpoint UI:
RNG, species, innovation and evolutionary population state are not checkpointed.

## ADR 004: indexed world without stochastic drift

World mutation methods maintain empty-cell row counts and food spatial buckets.
Empty sampling maps the same RNG rank to the same row-major cell as the old scan;
nearest-food ties preserve the original set iteration ordering. Body networks are
shared within an immutable genotype cohort. Randomized parity tests cover these
contracts. Direct mutation of `world.cells` or `world.food` bypasses indexes and is
unsupported; use placement/removal/movement methods. No parallel stochastic
engine execution or approximation was introduced.

## ADR 005: two independent ancestries

Optional passive Population logging records actual selected genome objects for
elite/clone/crossover/champion-copy/rescue events. Studio maps them to scoped
`generation:index` identities. Mutation records describe net changes between the
pre/post-mutation child, not every operator invocation. Logging does not consume
RNG draws; full-genotype and RNG-state parity tests exercise normal and stagnant
evolution. Studio replay v2 adds `clage-evolution-lineage-v1`; v1 remains readable
without fabricated ancestors. Within-world body parent IDs remain separate.

## System boundaries and persistence

```
neat + world → WorldSession → Experiment → locked Manager
                                         ↓
                              FastAPI HTTP + WebSocket
                                         ↓
                         ES modules / Canvas / SVG / charts
                                         ↓
                         validated JSON/gzip local artifacts
```

Live snapshots include current body state/last actual inference; topology and
evolutionary provenance use separate endpoints. Streaming is capped at 10 Hz;
unchanged snapshots are suppressed with two-second heartbeats. Encoding and
replay validation run off the async event loop. A worker thread prevents blocking
the UI, but the Python GIL and lock still limit throughput; this is not a job queue.

Frame retention is capped at 600 frames and 16 MiB **encoded JSON**, not total
Python heap. Portable uploads/decompressed files are limited to 24 MiB; exports
can trim a recorded prefix further and disclose that trimming. A conservative
50,000-body safety check pauses before a potentially oversized tick. Genomes are
bounded by 10,000 founder evaluations. Local saves use atomic gzip file replacement
and opaque IDs in `.studio-runs/` (or `CLAGE_ARTIFACTS`). Settings remain frozen.
Software provenance resolves the source checkout, not the caller's working
directory; packaged code without its own Git root reports an unknown commit.

Default server binds only `127.0.0.1`. Trusted Host, same-origin write/stream
checks and self-only asset CSP reduce accidental browser exposure. It is not an
authenticated multi-user service and must not be exposed publicly as-is.

## Recovery follow-up — October 7, 2026

Remote `ls-remote` found no supplied audit branch. Fetching the exact alleged
`bec8c03a02242d2effdeb6e553c3fdf05c1931ce` returned `not our ref`. This does not
prove that another local workspace never existed; it precisely limits what this
checkout/origin can recover. The missing audit branch cannot be pushed or made
into a reviewable PR from these objects. Studio is committed on its separate
branch; local default branch and existing ignored results remain untouched.

## Scientific contracts

Preserve minimal unwired initialization; name Studio's seeded dense-random
initialization separately. Fitness remains `3*food + .01*age + .5*offspring`,
max over bodies sharing a genome. Live scores are provisional. Parentage denotes
within-world asexual reproduction; evolutionary parentage is a separate v2 table. Species
are known after evaluation. Activations are the actual last inference from
pre-action observations, not a recalculation from post-action state.
