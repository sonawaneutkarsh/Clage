# Clage Studio implementation report

This report preserves the initial implementation milestone. Subsequent semantic
reconciliation, release QA and merge preparation are documented in
`CLAGE_RECONCILIATION_REPORT.md`, `CLAGE_STUDIO_FINAL_REVIEW.md` and
`CLAGE_STUDIO_MERGE_REVIEW.md`; use those for current verification and recovery status.

Delivered October 7, 2026 on `studio/clage-1.0`. This is a working, tested local
Studio release candidate, **not completion of every requested platform phase**.
Engine version remains 0.3.0; API is `1.0-slice`, replay version 2.

## Preservation and recovery

The initially clean local main (`9bc763a`, 188 tests) was inspected before edits.
It already contained custom NEAT, grid simulation, experiments, behavioral
metrics, recording and terminal/matplotlib visualization. No claim that these
were newly implemented or that individual changes can be attributed to Kiro.
Fetched/inspected remote descendants through `2105fee` passed 300 tests; only
then was Studio branched from that corrected baseline. Local main remains intact.
Existing ignored results and Kiro files were not overwritten.

The named audit reports/branch/`bec8c03a...` were absent from local objects,
reflogs and dangling-object recovery. Remote branch lookup found nothing and
exact commit fetch returned `not our ref`. The alleged 0.4.0 / 349-test audit is
not recovered. Consequently no audit push/PR was invented. Another original
workspace or Git bundle is necessary to recover those particular changes.

## Working vertical slices

| Requested milestone | Delivered functionality | Boundary |
|---|---|---|
| Live application | Background Python world/evolution, local API/stream, Canvas world, pause/resume/reset/step/speed, camera/layers/selection | Single active run; no queued jobs |
| Body/neural inspection | Actual body state, scoped IDs/parent, pre-action neural telemetry, semantic labels, topology/biases/weights/disabled genes, metadata and comparison | Last inference only; no full per-body history archive |
| Experiments/graphs/replay | Frozen validated config/presets, state charts/histograms, seeking/generation navigation, gzip save/import, synchronized second replay | Recent-window recordings; comparison is replay, not two live engines |
| Evolution/visualizations | Evaluated history/species sizes/complexity/champions; recorded genotype parents/net mutation deltas; separate body lineage; six layers/trails/event pulses | Ancestor view up to four edges; no exhaustive mutation operator log |
| Research/reproducibility | Four policies, five fixed held-out seed bases, per-seed outputs/descriptive SD, saved champions, provenance, deterministic rerun CLI | Forager privileged sensing; no claimed learning or comprehensive ablations |
| Reliability/performance/QA | Indexed world/shared networks, bounded recording/export, safety halt, API/contract/RNG/browser tests, screenshots, measured workloads | Limited local hardware/browser coverage; no release certification |

## Main engineering changes

- Extracted `WorldSession`, shared by legacy `run_generation` and incremental
  orchestration without changing order or random draws.
- Instrumented inference using the same feed-forward computation; timestamped
  actual observations/activations and preserved output equivalence.
- Added optional passive NEAT reproductive provenance, including actual champion
  origins; tests compare every genome, statistics and full RNG state.
- Replaced repeated empty-grid scans/global nearest-food scans with exact indexed
  queries; preserved row-major sampling and nearest-food ties.
- Separated simulation/stream/render clocks; suppress idle duplicate stream frames.
  Network topology/lineage are fetched separately rather than every snapshot.
- Strict versioned replay contracts validate metrics/positions/lineages/interfaces
  and independently recompute archived neural outputs. Bounded uploads, exports,
  opaque artifact paths, origin checks and offline assets improve local reliability.

## Evidence and commits

- `e5366d2`: observatory, incremental evolution, inspection and replay.
- `f8fd0fe`: evolutionary ancestry, performance, evaluation/replay hardening and tests.
- `fe94e8c`: export/playback/responsive/empty-world workflows and source-bound provenance.
- Final documentation/measurement milestone appears in `git log studio/clage-1.0`.
- 350 Python tests, four JS tests, seven Chromium workflows; see exact commands
  and evidence in the test/performance reports.
- Screenshots: `docs/studio/{ecosystem,neural,evolution,laboratory,research,tablet,mobile,large-population,empty-world}.png`.

No merge or push to the default branch. No remote PR or remote CI success claimed.
Run `python -m studio` after installing `.[studio]`; see `docs/studio/GUIDE.md`.
