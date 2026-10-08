# Current-source measurements

Source: `d05255547ff9328ff04a9fa6b3f2527f0e199076`, engine 0.4.1.
Actual code/assets SHA-256:
`6e88409b30cea036796bf877af0c1d65ef3ac1c7776015a1f914de5f70f46038`.
Source checkout and installed-wheel code digests agree. Measurements are local
macOS ARM / Python 3.13.6 / headless Chromium; exact machine/runtime metadata and
raw repetitions are in the accompanying JSON. Original Studio measurements have
not been replaced. These are observations, not a cross-machine speed guarantee.

## Backend: unchanged Studio workload

30 ticks, seed 42, three repetitions, median ticks/s, reproduction disabled via
threshold 2. Includes current live snapshot/inference instrumentation. Recording
adds bounded archival serialization. No generation boundary or long-run workload.

| Founders | World | Food | Recording off, ticks/s | Recording on, ticks/s |
|---:|---|---:|---:|---:|
| 72 | 40×32 | 180 | 1397.7 | 820.8 |
| 512 | 80×64 | 500 | 185.4 | 112.8 |
| 1000 | 96×96 | 900 | 84.6 | 53.4 |

Initialization/serialization, bytes, body counts and all repetitions are in
`backend-benchmark.json`. No improvement percentage is inferred by mixing these
values with historical baselines or Work's unavailable Linux profiles.

## Browser: unchanged predeclared paused workloads

Paused 96×96 world, food 900, genome layer, grid off, fit camera, viewport
1440×1100; 120 RAF intervals per workload. No live inference/stream load or density
layer. RAF interval measures differ from actual Canvas draw CPU time.

| Bodies | RAF FPS | p95 frame interval, ms | Mean draw CPU, ms |
|---:|---:|---:|---:|
| 72 | 60.00 | 16.8 | 0.284 |
| 512 | 60.00 | 16.8 | 0.521 |
| 1000 | 60.00 | 16.8 | 0.916 |
| 2000 | 59.02 | 16.8 | 1.590 |

Raw results: `browser-benchmark.json`. These are headless measurements on this
machine, not a GPU/browser-wide scalability claim or evidence for untested modes.

## Full default run and reproducibility

- Eight generations / 2,560 ticks / 2,568 frames generated (including placements
  and boundaries); 42 retained and 2,526 explicitly evicted by archive bounds.
- Initial run elapsed 11.00103687500814 s, excluding deterministic rerun.
- Retained encoded frame bytes: 16,491,006. Process peak RSS: 176.421875 MiB,
  including interpreter and original/rerun objects; not a leak test.
- Exact deterministic verification: **42/42** retained frames through sequence
  2,567. Metadata source/current digests agree; dirty=true reflects generated
  browser artifacts in the isolated checkout, not source edits.
- Installed wheel: **18/18** actual portable frames verified through sequence 17;
  JSON/gzip payloads agree and no unrelated Git checkout is attributed.

Raw results: `determinism.json`, `installed-smoke.txt`. These rerun from frozen
configuration and seed; they do not resume a world from a replay snapshot.

## Training benchmarks

Engine defaults/minimal initialization, 100 founders, seeds 0–4, 300-generation
budget, early stop on existing success criteria. No parameter tuning:

| Problem | Solved | Solve generation min/median/max |
|---|---|---|
| OR | 5/5 | 5/6/9 |
| AND | 5/5 | 6/8/17 |
| XOR | 5/5 | 68/125/212 |
| Sine | 0/5 | Unsolved by 300 |

Sine mean final best training fitness: approximately .9195. Per-seed results and
diagnostic hypotheses are in `neat-validation.md`/`.log`. Matching reported Work
summaries is independent current verification, not recovery/authentication of the
unavailable original commits/evidence. Generalizable learning remains unproven.
