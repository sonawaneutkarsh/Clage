# Clage Studio test report

Local validation on October 7, 2026. Python 3.13.6, Node 24.11.1, macOS 26.7.1
ARM; Playwright 1.63.0 and its headless Chromium 153 distribution.

## Commands and outcomes

| Command | Result |
|---|---|
| `python3 -m pytest -o addopts='' -q` | 350 passing cases, including original 300 |
| `python3 -m ruff check .` | All checks passed |
| `python3 -m mypy neat world diversity experiments benchmarks visual studio` | No issues (49 source files) |
| `npm test` | Four passing pure-JS unit tests |
| `npm run test:e2e` | Seven passing real browser workflows |
| `python3 -m studio.validate --out docs/studio/long-run-validation.json` | Complete default run, archive validation and retained-frame deterministic rerun |
| `python3 -m studio.benchmark --out docs/studio/backend-after.json --repeats 3` | Six measured backend workloads |
| `python3 -m pip wheel --no-deps --no-build-isolation . -w /private/tmp/clage-studio-wheel` | Wheel built; four browser assets included |
| `python3 -m pip install --no-deps --target /private/tmp/clage-studio-package-final /private/tmp/clage-studio-wheel/clage-0.3.0-py3-none-any.whl` | Installed outside repository; API/static/step/replay smoke passed |

API tests require Studio extras and skip when unavailable; engine-only installs
do not acquire mandatory web dependencies. CI config adds these extras and a
separate browser job. Remote CI/other Python versions were not run locally.
Packaged smoke launched `python -m studio --port 8878` from `/private/tmp` with
`PYTHONPATH=/private/tmp/clage-studio-package-final`, using existing installed web
dependencies. Created a default paused run, served all four static assets/OpenAPI,
stepped once and exported v2 replay with correctly unknown Git provenance.
This is not a fresh-machine dependency installation or packaged browser E2E test.

## Python coverage by behavior

- Original NEAT/world/diversity/experiment/visual validation remains passing.
- Independent legacy world-loop recording and RNG-state parity with incremental
  and instrumented execution. Independent hidden-node arithmetic check.
- Fully seeded multi-generation replay equality and genotype/fitness/lineage data.
- Recording-enabled versus plain Population equality over eight generations;
  stagnant champion rescue provenance and matching random state.
- Randomized row-major empty-cell sampler/index and nearest-food query parity,
  tie handling, occupancy mutation and immutable network sharing.
- API start/reset/pause/step/speed; worker streaming; frozen config, origin/Host
  restrictions, artifact paths, import/export and evaluated champion contracts.
- Replay consistency and forgeries: bad schemas/references, impossible metrics,
  nonfinite/deep JSON, invalid neural activations, chronology and genomic lineage.
- Window/export budget trimming, recording-disabled errors, safety halt without
  advancing or fabricating a completed evaluation.
- Baseline repeatability, champion immutability, held-out overlap rejection,
  archived champion evaluation and deterministic reproduction tooling.
- Provenance follows the source repository rather than an unrelated working
  directory; installed source without a Git root reports unknown commit honestly.

## Browser workflows

The seven Playwright tests operate the application, not just screenshots:

1. Actual Canvas paint and renderer timing at 72/512/1,000/2,000 bodies.
2. Live keyboard/control stepping, organism selection/follow, grid/layers,
   neural expansion, metadata selection and genome comparison.
3. Configuration errors, browser presets and frozen definitions.
4. Two evaluated generations, genotype ancestry tree, replay seek/jump/play/pause,
   JSON/gzip import, synchronized comparison, CSV and chart/world PNG downloads.
5. Twenty held-out policy/seed rows, artifact save, 1024×768 and 390×844 views.
6. 1,000 founders, in-world births, zero food, density, invalid import and empty
   living population after metabolism-induced deaths.
7. A running 450-tick/three-generation background experiment while navigating
   views, followed by evaluated history and champion download.

Page errors are asserted absent. Browser timing is descriptive, not a brittle
60-FPS CI gate. Tests use a dedicated port 8876, one worker and real WebSockets.

## Visual QA

Generated and opened ecosystem, neural atlas, evolutionary ancestry, configuration
modal, research results, tablet/mobile, large-population and empty-world screenshots. Reviewed layout,
overflow, typography, graph labels and error presentation. Iterations corrected
mobile heading access/control wrapping, stale body-lineage copy and excessive
ancestry whitespace. Final screenshots live under `docs/studio/`.

Native desktop browser control was unavailable (permission-denied native surface;
Chrome surface unavailable). All claimed browser QA is **headless Chromium** plus
image inspection, not Safari/Firefox/native-GPU/touch accessibility verification.
No GIF/video export, arbitrary checkpoints, job queues or unimplemented views are
claimed tested. See remaining-work report for explicit omissions.
