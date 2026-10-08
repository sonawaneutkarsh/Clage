# Clage Studio performance report

Measured October 7, 2026 on macOS 26.7.1 ARM, Python 3.13.6. Values are local
descriptive measurements, not cross-hardware guarantees or scientific outcomes.
Raw trials/configuration/provenance are committed in `docs/studio/`.

## Backend: predeclared workloads

`python3 -m studio.benchmark --out docs/studio/backend-after.json --repeats 3`

Three trials per workload; median shown. Seed 42, dense-random initialization,
30 world ticks, reproduction disabled with threshold 2; food/regrowth/metabolism
otherwise match benchmark configs. Includes inference telemetry and current-frame
capture even when recording is off; recording-on includes JSON-size accounting.
Excludes evolution boundaries, background scheduling, HTTP/WebSocket delivery and
browser rendering. Counts are founders/living bodies in these non-reproducing trials.

| Bodies / world / food | Recording | Before ticks/s | After ticks/s | Before init s | After init s |
|---|---|---:|---:|---:|---:|
| 72 / 40×32 / 180 | Off | 693.7 | 1125.6 | .0244 | .0179 |
| 72 / 40×32 / 180 | On | 492.8 | 739.9 | .0250 | .0176 |
| 512 / 80×64 / 500 | Off | 46.1 | 140.6 | .2414 | .0450 |
| 512 / 80×64 / 500 | On | 40.5 | 98.6 | .2401 | .0464 |
| 1,000 / 96×96 / 900 | Off | 16.1 | 68.6 | .7139 | .0784 |
| 1,000 / 96×96 / 900 | On | 14.5 | 47.7 | .7364 | .1036 |

Before: `backend-before.json`, initial instrumented Studio working tree based on
`2105fee` (dirty, before index optimizations). After: `backend-after.json`, code
`f8fd0fe` with report/screenshot changes marked dirty. Both provenance records are
retained; do not describe the baseline as an unmodified published engine benchmark.
After includes inference timestamps, slightly increasing snapshot size.

Profiling identified global nearest-food scans and empty-cell list construction
as hotspots. Implemented food spatial buckets, exact rank-based empty-row sampling
and immutable per-genome network sharing. Randomized parity/RNG tests establish
behavioral equivalence; no stochastic approximation or dropped engine ticks.

At 1,000 bodies, final snapshot is **705,964 JSON bytes** and median encode time
**7.07 ms** with recording. At a 10-Hz stream ceiling that is approximately 7 MB/s
per client before protocol overhead. Snapshot copying/lock contention can add cost;
this is not a measured network-throughput result. Large populations do not sustain
the 120-ticks/s UI maximum just because rendering remains smooth.

## Browser renderer

`npm run test:e2e` executes `tests/browser/performance.spec.js`. Headless Chromium,
1440×1100 viewport, paused 96×96 worlds, food 900, genome colors, grid off, camera
fit. 120 requestAnimationFrame intervals per declared size; real painted Canvas
pixels asserted. `mean_draw_ms` is the app's latest one-second mean draw CPU sample,
not whole-browser frame cost or a GPU timer. Raw output: `browser-benchmark.json`.

| Bodies | Observed RAF FPS | p95 interval ms | Mean draw CPU ms |
|---|---:|---:|---:|
| 72 | 60.0 | 16.8 | .270 |
| 512 | 60.0 | 16.8 | .510 |
| 1,000 | 60.0 | 16.7 | .913 |
| 2,000 | 59.0 | 16.8 | 1.612 |

RAF is refresh-limited here. Paused renderer workloads do not demonstrate live
2,000-body inference/stream performance, all visualization layers, rich SVG genome
scalability, mobile hardware, native GPU acceleration or high-refresh displays.
A separate E2E workflow did navigate a live 450-tick/three-generation run, but it
is a correctness workflow, not an independently measured live rendering benchmark.

## Complete-run reliability and memory

`python3 -m studio.validate --out docs/studio/long-run-validation.json`

Default 72 founders, eight generations × 320 ticks, 40×32 world, food 180,
regrowth 2, seed 42, reproduction enabled: **2,560 world ticks**, 2,567 advancement
calls including generation transitions, **12.67 seconds** for the initial run.
All eight generations completed. Archive validation passed and deterministic
rerun matched all **40 retained frames**, ending sequence 2,567.

2,528 frames were explicitly dropped. Retained encoded JSON: **16,678,508 bytes**,
within the 16-MiB frame budget. Process peak RSS: **177.625 MiB**, including the
interpreter, archive validation and a second reproduction run coexisting with the
first. This is one default-run check, not a leak test or total-heap guarantee.
Frame budget measures encoded JSON, not Python object overhead/topology/history.

## Remaining bottlenecks

Full snapshots, inference dictionaries, Python object retention and GIL/lock
contention dominate scaling next. Consider disk-chunked recordings, protocol deltas,
process isolation and explicit total-heap accounting before increasing limits.
No claim of real-time 50,000-body support: that count is a conservative safety
guard, not a validated capacity. See remaining-work report for workload expansion.
