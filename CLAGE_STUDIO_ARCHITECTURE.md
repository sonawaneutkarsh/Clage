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

## Scientific contracts

Preserve minimal unwired initialization; name Studio's seeded dense-random
initialization separately. Fitness remains `3*food + .01*age + .5*offspring`,
max over bodies sharing a genome. Live scores are provisional. Parentage denotes
within-world asexual reproduction, never evolutionary genome ancestry. Species
are known after evaluation. Activations are the actual last inference from
pre-action observations, not a recalculation from post-action state.
