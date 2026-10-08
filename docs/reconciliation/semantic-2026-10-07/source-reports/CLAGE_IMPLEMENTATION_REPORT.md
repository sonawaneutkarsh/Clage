# Clage Implementation Report

**Date:** 2026-10-07 UTC. **Baseline:** `2105fee258f2a329c351a4ab255fd585b3437a8a`.
**Branch:** `audit/comprehensive-2026-10-07`.
**Production/documentation snapshot:** `bec8c03a02242d2effdeb6e553c3fdf05c1931ce`; engine version 0.4.0.

The baseline report was committed before production changes. Historical raw
results and corrected results occupy separate directories. Main was not edited
or merged. The work preserves the custom engine, default scientific definitions
and matplotlib/terminal architecture, with no new runtime dependency.

## 1. Baseline and investigation

`docs/audit/BASELINE.md` freezes environment, original 300 passing cases,
benchmarks, all six configs, visual defects and the predeclared frozen control.
Evidence includes exact command metadata/logs, raw per-seed rows/configs/replays,
coverage JSON, CPU profiles and before/after images. The baseline SHA allows
historical numbers to remain verifiable after correctness changes.

Verification: original README OR/AND/XOR/sin and final food-table values
reproduced; original tests/Ruff/Mypy passed. A passing original suite did not
prevent independent counterexamples from revealing real defects.

## 2. Genetic and population correctness — C01–C08

Affected production files: `neat/genome.py`, `innovation.py`, `mutation.py`,
`crossover.py`, `population.py`, `speciation.py`, `diagnostics.py`.
Tests: `tests/test_neat_audit.py` plus valid-history/cycle fixture corrections
in `test_population.py`, `test_crossover.py`, `test_mutation.py`.

- Champion identity now includes node type/bias and connection innovation,
  endpoints, weight and enabled state. Evaluation is archived before reproduction,
  including generation one (`neat/population.py:197`).
- Per-species elites are capped by the number of available members and budget.
  Explicit founder counts must agree; invalid generation counts fail before work.
- Shared unequal biases can inherit from either parent; equal values do not
  consume another RNG draw. Conflicting roles/innovation endpoints reject.
- A reversible innovation/pair ledger registers supplied genomes before structural
  mutation, reserves existing node/innovation maxima and rejects inconsistent
  histories. Node split delays disabling the old connection until registration
  succeeds (`neat/innovation.py:68`).
- Duplicate interface IDs/innovations and connections targeting INPUT reject.
  Empty distance normalization is safe. Diagnostic callbacks receive the integer
  generation; later boundary work recovers after all species are pruned.

Verification: 20 new collected core cases initially failed against pre-fix code.
Focused affected suites then passed, followed by final full suite/static checks.
Four older fixtures encoded conflicting history or an INPUT-target edge rather
than a valid hidden cycle; they were corrected while retaining their assertions.

Risks: RNG trajectories change for unequal bias inheritance and actual champion
preservation; baseline evolutionary results are incompatible. Stricter validation
rejects previously accepted invalid genomes. Persistent endpoint history remains
an intentional variant, not a wholesale canonical-NEAT conversion. Split ancestry
cannot be reconstructed from topology alone; complete resume is not implemented.

## 3. Data integrity and evaluated replay fitness — D01–D04

Files: `experiments/config.py`, `run.py`, `analysis.py`, `report.py`,
`world/recorder.py`, `world/simulation.py`, `visual/data.py`;
tests in `tests/test_data_audit.py`.

Unique integer seeds, safe names and positive generation budgets validate early.
Normal experiment reruns require a fresh output directory; direct condition runs
preflight every seed/config/replay target. Manifests capture UTC, actual source
SHA-256, commit if available, merged config, runtime versions, seed/control identity
(`experiments/run.py:213`). Reports resolve the semantic control
from the manifest, retaining conventional legacy fallback.

Both aggregators reject unequal/empty lengths, wrong generation numbering and
missing/non-finite metric values. Their duplicated contract preserves the
standard-library-only viewer dependency boundary. Recorder schema v2 finalizes
current fitness after evaluation and labels evaluated/unevaluated state. V1 still
loads with cautious stored-fitness labels; unknown versions and empty tick lists
reject (`world/recorder.py:123`).

Verification: all 15 new data cases failed before fixes; affected tests pass.
All 160 baseline/corrected trial files contain five seeds/25 valid rows per
condition. The installed-wheel replay also exports successfully. Current source
matches every first-generation condition row and two full 25-generation trials.

Risks: callers that intentionally overwrote outputs must choose a fresh directory.
An interrupted run is not resumable. Publishing files is not yet atomic, and full
external replay/ledger schema validation remains open. Source hashes describe the
actual run; manifests from earlier snapshots were retained without revision.

## 4. Replay and inspection usability — V01–V03

Files: `visual/world_view.py`, `terminal_view.py`, `network_view.py`,
`analytics.py`, `app.py`, plus `tests/test_viewer_audit.py`.

World facing/row orientation agrees with simulation. Tick zero renders before
animation; pause/step/slider/end labels follow actual state without a second redraw.
Terminal food/cursor have consistent cell width, cursor is bounded to the rendered
grid, selected body ID/energy percentage are visible, and final average fitness is
explicitly labelled. Byte input handles queued/fragmented arrow keys; invalid FPS
fails before terminal setup and paused playback does not flood identical redraws.

Network plots use readable marker/bias spacing, enabled arrows and sign/disabled
legend. Unwired topology explains bias-driven outputs. Parameter tables use rows
for parameters instead of overlapping many headers. Network `--genome` is primary
with `--organism` retained as an alias; exports create parents and unknown metric
names receive a clear error.

Verification: seven collected viewer regressions failed before fixes; affected
tests pass. Real PTY interaction selects body 0 and processes pause/WASD/Enter/
arrow/quit. Canvas mouse events exercise matplotlib controls and inspection;
exported wired/unwired topology, replay, table and charts were visually inspected.
Native desktop interaction remains unverified because there is no display.

Risks: palettes still repeat beyond 20 genomes, TUI topology is coarse, and large
graphs need future navigation. Static topology is not a neural-activation explanation.
Selected-body ID was added in a separate focused follow-up after PTY inspection.

## 5. Measured hot paths and world boundaries — P01/W01/W02

Files: `neat/phenotype.py`, `world/grid.py`, `world/organism.py`;
`tests/test_performance_audit.py`, `tests/test_boundary_audit.py`.

Compiled noninput inference plans remove repeated membership/lookup work. Direct
clipped-row neighborhood scans avoid repeated Python method calls, preserving the
full-square density denominator and exclusion semantics. Summation/edge order is
unchanged. An independent recursive oracle checks 200 input/genome combinations;
a neighborhood oracle checks corners/outside/exclusion/radii.

Odd-grid boundary proximity uses half the interior span. Reproduction fractions
0/1 immediately kill/remove an exhausted parent or child, while preserving the
actual birth event. Both corrections are documented as scientifically relevant
only for those configurations: shipped grids are even and fraction is 0.5.

| Workload, median of 7 repetitions | Baseline | Final | Less elapsed time |
| --- | --- | --- | --- |
| 100,000 dense-network activations | 0.303767 s | 0.218241 s | 28.2% |
| 10 default control worlds | 1.172586 s | 1.037167 s | 11.5% |

Verification: default fitness vectors match; five recorded-world tick hashes are
identical before/after. Timings use seven repetitions, sequential source snapshots
and the same Python environment. They are workload/machine-specific. Final world
gain is about 12%, lower than the intermediate ~18% measurement before the extra
starvation guard. No general memory or whole-pipeline throughput claim is made.

Risks: odd-grid and endpoint historical results are not comparable. No ecological
parameter, fitness coefficient or behavioral metric was tuned to improve results.
No grid index/vectorization/threading dependency was introduced without evidence.

## 6. Benchmark and diagnostic reliability — B01/R10

Files: `benchmarks/run.py`, `benchmarks/diagnose.py`, `neat/diagnostics.py`;
boundary/core audit tests.

Positive integer benchmark budgets reject before an empty loop/report; unknown
CLI problems exit with an explanatory argument error. Flat diagnostic fitness
cannot empty reproduction, and callbacks get proper generation numbers.
Diagnosis now describes search/parameterization as a hypothesis: small probes
do not prove bounded solutions or exclude every algorithm defect.

Verification: four boundary cases initially failed; final affected suite has
110 passing cases. Default diagnostic CLI completes. Main benchmarks used the
same seeds/configs before and after: OR/AND 5/5, XOR 4/5→5/5, sin 0/5→0/5.
Later budget validation does not change valid benchmark execution.

Risks: sine remains unresolved; five seeds cannot establish a general rate.
CLI raw benchmark histories/provenance are still an open improvement. Out-of-bounds
hand solutions are execution probes rather than proof of reachable search.

## 7. Packaging and honest documentation — E01/DOC01/P02

Files: `pyproject.toml`, `.gitignore`, `README.md`, `docs/architecture.md`,
`docs/correctness-notes.md`, three new `docs/img/audit-*.png`, and stale
docstrings in core/config/analytics.

The wheel includes all six JSON experiment configs. Setuptools minimum now
supports PEP621 metadata; build output is ignored; engine version is 0.4.0.
Original README images/results were retained. New images use corrected runs and
orientation. README adds architecture, explicit lifecycle/input contracts,
working fresh-output commands and the frozen MOVE control. Unsupported unarchived
100-generation claims and strong cross-seed separation language were removed.
Hash-collision wording is realistic; an unused seed_stride remains documented debt.

Verification: wheel built and installed in a fresh environment outside the source
checkout; all six packages/configs and world/benchmark/replay smoke paths pass.
The README Python block executes verbatim. Remaining TOML-table license
deprecation is cosmetic release metadata work, not a failed build.

Risks: generic top-level module names still conflict with neat-python; dedicated
venv documented. Dependency ranges are broad; the audit retains its exact freeze.

## Final verification and interpretation

| Code regime | Passing cases | Covered statements | Coverage |
| --- | --- | --- | --- |
| baseline | 300 | 2002 / 2686 | 74.53% |
| final | 349 | 2216 / 2887 | 76.76% |

Final Ruff: pass; Mypy: pass on 40 source files. Self-review included the complete
diff and `git diff --check`. Six shipped configs ran fully across five seeds per
condition in both regimes (80 executions each; repeated controls are not extra
independent samples). Frozen policy adds 15 trials. All viewer export paths,
analysis/CSV plots, diagnostic CLI and installed-wheel smoke execute. Full logs
include intentional failures and corrected fixture/harness/setup failures.

| Problem | Baseline solved | Baseline solve gen min/median/max | Corrected solved | Corrected solve gen min/median/max |
| --- | --- | --- | --- | --- |
| OR | 5/5 | 7 / 13 / 13 | 5/5 | 5 / 6 / 9 |
| AND | 5/5 | 19 / 27 / 105 | 5/5 | 6 / 8 / 17 |
| XOR | 4/5 | 147 / 184.5 / 243 | 5/5 | 68 / 125 / 212 |
| sin | 0/5 | Unsolved at 300; mean best .9092 | 0/5 | Unsolved at 300; mean best .9195 |

The corrected high-food result is better on these training runs and XOR solved
all five seeds. These observations do not demonstrate transfer, intelligence or
cooperation, and do not isolate the effect of any single correction. Ecological
settings/fitness/metric definitions were preserved. Proposed sensor, reproduction,
fitness and speciation studies remain separate falsifiable experiments.

Known open hardening examples are listed in the comprehensive backlog: finite
gene/ledger validation and mutable history views. Native GUI and local Python
3.10 were not verified here. Next highest-value work is held-out policy evaluation
with controls, then versioned sensor/fitness ablations and meaningful CLI/schema
integration. No merge to main is part of this audit.


## Reviewable commit/file inventory

### 251d1fb — docs(audit): freeze reproducible baseline before correctness changes

`docs/audit/BASELINE.md`

### ad5c9f7 — fix(neat): preserve champions, node biases, population size and innovation history

`neat/crossover.py`, `neat/diagnostics.py`, `neat/genome.py`, `neat/innovation.py`, `neat/mutation.py`, `neat/population.py`, `neat/speciation.py`, `tests/test_crossover.py`, `tests/test_mutation.py`, `tests/test_neat_audit.py`, `tests/test_population.py`

### a2772c7 — fix(data): protect trial outputs and record evaluated fitness with provenance

`experiments/analysis.py`, `experiments/config.py`, `experiments/report.py`, `experiments/run.py`, `tests/test_data_audit.py`, `visual/data.py`, `world/recorder.py`, `world/simulation.py`

### 7a075ca — fix(visual): make replay orientation, cursor and network displays faithful

`tests/test_viewer_audit.py`, `visual/analytics.py`, `visual/app.py`, `visual/network_view.py`, `visual/terminal_view.py`, `visual/world_view.py`

### 39445ad — perf: speed up inference and density scans; fix odd-grid boundary sensing

`benchmarks/diagnose.py`, `neat/phenotype.py`, `neat/population.py`, `tests/test_performance_audit.py`, `world/grid.py`, `world/organism.py`

### 712a708 — fix(visual): identify the selected organism in terminal inspection

`tests/test_viewer_audit.py`, `visual/terminal_view.py`

### 09929c2 — fix: handle exhausted reproduction, empty diagnostics and invalid benchmark budgets

`benchmarks/run.py`, `neat/diagnostics.py`, `tests/test_boundary_audit.py`, `world/organism.py`

### 92868eb — fix(packaging): ship experiment configs and identify corrected engine as 0.4.0

`.gitignore`, `neat/__init__.py`, `neat/mutation.py`, `pyproject.toml`, `visual/analytics.py`, `world/config.py`

### bec8c03 — docs: present verified results, controls, architecture and compatibility limits

`README.md`, `docs/architecture.md`, `docs/correctness-notes.md`, `docs/img/audit-best-fitness-food-abundance.png`, `docs/img/audit-network.png`, `docs/img/audit-replay-food-high-gen9.png`


## Exact execution ledger

UTC values are completion timestamps. RSS is the Linux maximum child RSS reported by the runner, not a sum of concurrently running processes. Child exit codes, not wrapper exit, are authoritative.

| Phase / label | Exact command | Child exit | Seconds | Max child RSS KiB | Log under audit/ |
| --- | --- | --- | --- | --- | --- |
| baseline / ruff | `.venv/bin/ruff check .` | 0 | 0.017 | 24356 | logs/baseline-ruff.log |
| baseline / mypy | `.venv/bin/mypy neat world diversity experiments benchmarks visual` | 0 | 3.383 | 265404 | logs/baseline-mypy.log |
| baseline / pytest-coverage | `.venv/bin/python -m pytest --cov=neat --cov=world --cov=diversity --cov=experiments --cov=benchmarks --cov=visual --cov-report=term-missing --cov-report=json:../audit/baseline-coverage.json` | 0 | 5.839 | 104696 | logs/baseline-pytest-coverage.log |
| baseline / benchmarks | `.venv/bin/python -m benchmarks.run --problems or,and,xor,sin --trials 5 --generations 300 --report ../audit/baseline/validation_report.md` | 0 | 59.388 | 76984 | logs/baseline-benchmarks.log |
| baseline / food-abundance | `.venv/bin/python -m experiments.run --config experiments/configs/food_abundance.json --out ../audit/baseline/results --record-generation 9` | 0 | 62.296 | 69420 | logs/baseline-food-abundance.log |
| baseline / defect-probes | `.venv/bin/python ../audit/probes.py` | 0 | 0.418 | 62892 | logs/baseline-defect-probes.log |
| baseline / base | `.venv/bin/python -m experiments.run --config experiments/configs/base.json --out ../audit/baseline/all-configs` | 0 | 16.262 | 17400 | logs/baseline-base.log |
| baseline / tui-export | `.venv/bin/python -m visual tui --recording ../audit/baseline/results/food_abundance/recordings/food_high/0.json --export-tick 40 --export-out ../audit/baseline/frame.txt` | 0 | 0.221 | 36284 | logs/baseline-tui-export.log |
| baseline / diagnostics | `.venv/bin/python -m neat.diagnostics` | 0 | 0.325 | 11776 | logs/baseline-diagnostics.log |
| baseline / replay | `.venv/bin/python -m visual replay --recording ../audit/baseline/results/food_abundance/recordings/food_high/0.json --export-tick 40 --export-out ../audit/baseline/tick.png` | 0 | 0.773 | 89284 | logs/baseline-replay.log |
| baseline / network | `.venv/bin/python -m visual network --recording ../audit/baseline/results/food_abundance/recordings/food_high/0.json --organism 3 --export ../audit/baseline/network.png` | 0 | 0.821 | 88124 | logs/baseline-network.log |
| baseline / analyze-food | `.venv/bin/python -m experiments.analyze --results ../audit/baseline/results/food_abundance --report ../audit/baseline/food-report.md --plots ../audit/baseline/analysis-plots` | 0 | 1.633 | 87408 | logs/baseline-analyze-food.log |
| baseline / analytics | `.venv/bin/python -m visual analytics --results ../audit/baseline/results/food_abundance --export ../audit/baseline/analytics` | 0 | 2.942 | 99080 | logs/baseline-analytics.log |
| baseline / available_space | `.venv/bin/python -m experiments.run --config experiments/configs/available_space.json --out ../audit/baseline/all-configs` | 0 | 53.477 | 18812 | logs/baseline-available_space.log |
| baseline / reproduction_cost | `.venv/bin/python -m experiments.run --config experiments/configs/reproduction_cost.json --out ../audit/baseline/all-configs` | 0 | 56.187 | 18940 | logs/baseline-reproduction_cost.log |
| baseline / population_density | `.venv/bin/python -m experiments.run --config experiments/configs/population_density.json --out ../audit/baseline/all-configs` | 0 | 69.986 | 31312 | logs/baseline-population_density.log |
| baseline / food_regeneration | `.venv/bin/python -m experiments.run --config experiments/configs/food_regeneration.json --out ../audit/baseline/all-configs` | 0 | 71.132 | 56104 | logs/baseline-food_regeneration.log |
| baseline / profile-world | `.venv/bin/python ../audit/profile_and_controls.py profile --out ../audit/baseline/world-profile.json` | 0 | 0.416 | 16780 | logs/baseline-profile-world.log |
| baseline / frozen-control | `.venv/bin/python ../audit/profile_and_controls.py controls --out ../audit/baseline/frozen-control.json` | 0 | 55.333 | 21612 | logs/baseline-frozen-control.log |
| baseline / inference | `.venv/bin/python ../audit/profile_and_controls.py inference --out ../audit/baseline/inference.json` | 0 | 2.272 | 14988 | logs/baseline-inference.log |
| baseline / neat-regressions | `.venv/bin/python -m pytest tests/test_neat_audit.py` | 1 | 0.265 | 29084 | logs/baseline-neat-regressions.log |
| fixes / core-tests | `.venv/bin/python -m pytest` | 1 | 1.222 | 85940 | logs/fixes-core-tests.log |
| fixes / core-ruff | `.venv/bin/ruff check .` | 1 | 0.016 | 22748 | logs/fixes-core-ruff.log |
| fixes / core-tests-verified | `.venv/bin/python -m pytest` | 1 | 1.17 | 85020 | logs/fixes-core-tests-verified.log |
| fixes / core-mypy | `.venv/bin/mypy neat world diversity experiments benchmarks visual` | 0 | 2.273 | 215368 | logs/fixes-core-mypy.log |
| baseline / data-regressions | `.venv/bin/python -m pytest tests/test_data_audit.py` | 1 | 0.315 | 31392 | logs/baseline-data-regressions.log |
| fixes / data-ruff | `.venv/bin/ruff check .` | 0 | 0.016 | 22320 | logs/fixes-data-ruff.log |
| fixes / data-tests | `.venv/bin/python -m pytest tests/test_data_audit.py tests/test_experiments.py tests/test_visual.py tests/test_world.py` | 0 | 0.869 | 83980 | logs/fixes-data-tests.log |
| fixes / data-mypy | `.venv/bin/mypy neat world diversity experiments benchmarks visual` | 0 | 2.121 | 216820 | logs/fixes-data-mypy.log |
| baseline / viewer-regressions | `.venv/bin/python -m pytest tests/test_viewer_audit.py` | 1 | 0.818 | 81140 | logs/baseline-viewer-regressions.log |
| fixes / viewer-ruff | `.venv/bin/ruff check .` | 0 | 0.016 | 21664 | logs/fixes-viewer-ruff.log |
| fixes / viewer-tests | `.venv/bin/python -m pytest tests/test_viewer_audit.py tests/test_visual.py` | 0 | 1.071 | 87476 | logs/fixes-viewer-tests.log |
| fixes / viewer-mypy | `.venv/bin/mypy neat world diversity experiments benchmarks visual` | 0 | 2.222 | 204340 | logs/fixes-viewer-mypy.log |
| fixes / hot-path-tests | `.venv/bin/python -m pytest tests/test_performance_audit.py tests/test_phenotype.py tests/test_world.py tests/test_viewer_audit.py` | 0 | 0.919 | 83180 | logs/fixes-hot-path-tests.log |
| research / interpretation-probes | `.venv/bin/python ../audit/science_probes.py` | 0 | 0.064 | 12544 | logs/research-interpretation-probes.log |
| final / base | `.venv/bin/python -m experiments.run --config experiments/configs/base.json --out ../audit/final/all-configs` | 0 | 15.362 | 22996 | logs/final-base.log |
| final / benchmarks | `.venv/bin/python -m benchmarks.run --problems or,and,xor,sin --trials 5 --generations 300 --report ../audit/final/validation_report.md` | 0 | 51.935 | 83776 | logs/final-benchmarks.log |
| final / food-abundance | `.venv/bin/python -m experiments.run --config experiments/configs/food_abundance.json --out ../audit/final/results --record-generation 9` | 0 | 67.417 | 87332 | logs/final-food-abundance.log |
| final / reproduction_cost | `.venv/bin/python -m experiments.run --config experiments/configs/reproduction_cost.json --out ../audit/final/all-configs` | 0 | 43.127 | 32880 | logs/final-reproduction_cost.log |
| final / available_space | `.venv/bin/python -m experiments.run --config experiments/configs/available_space.json --out ../audit/final/all-configs` | 0 | 47.106 | 31600 | logs/final-available_space.log |
| final / population_density | `.venv/bin/python -m experiments.run --config experiments/configs/population_density.json --out ../audit/final/all-configs` | 0 | 56.276 | 35512 | logs/final-population_density.log |
| final / food_regeneration | `.venv/bin/python -m experiments.run --config experiments/configs/food_regeneration.json --out ../audit/final/all-configs` | 0 | 56.63 | 60132 | logs/final-food_regeneration.log |
| final / ruff | `.venv/bin/ruff check .` | 0 | 0.016 | 20820 | logs/final-ruff.log |
| final / mypy | `.venv/bin/mypy neat world diversity experiments benchmarks visual` | 0 | 2.072 | 205336 | logs/final-mypy.log |
| final / pytest | `.venv/bin/python -m pytest --cov=neat --cov=world --cov=diversity --cov=experiments --cov=benchmarks --cov=visual --cov-report=term-missing --cov-report=json:../audit/final-coverage.json` | 0 | 3.278 | 104112 | logs/final-pytest.log |
| final / output-integrity | `.venv/bin/python ../audit/verify_outputs.py data --out ../audit/result-summary.json` | 0 | 0.471 | 16916 | logs/final-output-integrity.log |
| final / widget-interaction | `.venv/bin/python ../audit/verify_outputs.py ui --out ../audit/final/widget-events.json` | 0 | 1.37 | 102984 | logs/final-widget-interaction.log |
| final / analysis | `.venv/bin/python -m experiments.analyze --results ../audit/final/results/food_abundance --report ../audit/final/food-report.md --plots ../audit/final/analysis-plots` | 0 | 1.427 | 84364 | logs/final-analysis.log |
| final / world-profile | `.venv/bin/python ../audit/profile_and_controls.py profile --out ../audit/final/world-profile.json` | 0 | 0.269 | 18572 | logs/final-world-profile.log |
| final / analytics | `.venv/bin/python -m visual analytics --results ../audit/final/results/food_abundance --export ../audit/final/analytics` | 0 | 2.673 | 103504 | logs/final-analytics.log |
| final / terminal-interaction | `.venv/bin/python ../audit/tui_interaction.py` | 1 | 5.077 | 36436 | logs/final-terminal-interaction.log |
| final / terminal-interaction-retry | `.venv/bin/python ../audit/tui_interaction.py` | 1 | 0.666 | 36436 | logs/final-terminal-interaction-retry.log |
| final / terminal-interaction-verified | `.venv/bin/python ../audit/tui_interaction.py` | 0 | 1.719 | 36536 | logs/final-terminal-interaction-verified.log |
| final / inference | `.venv/bin/python ../audit/profile_and_controls.py inference --out ../audit/final/inference.json` | 0 | 1.618 | 16792 | logs/final-inference.log |
| baseline / isolated-world-runtime | `env PYTHONPATH=/workspace/scratch/bac8048257f0/Clage-baseline /workspace/scratch/bac8048257f0/Clage/.venv/bin/python ../audit/verify_outputs.py runtime --out ../audit/baseline/runtime.json` | 0 | 8.44 | 16384 | logs/baseline-isolated-world-runtime.log |
| final / isolated-world-runtime | `.venv/bin/python ../audit/verify_outputs.py runtime --out ../audit/final/runtime.json` | 0 | 6.886 | 16128 | logs/final-isolated-world-runtime.log |
| baseline / neutral-replay | `env PYTHONPATH=/workspace/scratch/bac8048257f0/Clage-baseline /workspace/scratch/bac8048257f0/Clage/.venv/bin/python ../audit/verify_outputs.py neutral --out ../audit/baseline/neutral.json` | 0 | 0.967 | 41048 | logs/baseline-neutral-replay.log |
| final / neutral-replay | `.venv/bin/python ../audit/verify_outputs.py neutral --out ../audit/final/neutral.json` | 0 | 0.866 | 41100 | logs/final-neutral-replay.log |
| final / terminal-export | `.venv/bin/python -m visual tui --recording ../audit/final/results/food_abundance/recordings/food_high/0.json --export-tick 150 --export-out ../audit/final/frame.txt` | 0 | 0.169 | 36444 | logs/final-terminal-export.log |
| final / replay-export | `.venv/bin/python -m visual replay --recording ../audit/final/results/food_abundance/recordings/food_high/0.json --export-tick 40 --export-out ../audit/final/tick.png` | 0 | 0.574 | 89264 | logs/final-replay-export.log |
| final / network-export | `.venv/bin/python -m visual network --recording ../audit/final/results/food_abundance/recordings/food_high/0.json --genome 8 --export ../audit/final/network.png` | 0 | 0.67 | 87904 | logs/final-network-export.log |
| fixes / boundary-before | `.venv/bin/python -m pytest tests/test_boundary_audit.py` | 1 | 0.618 | 73880 | logs/fixes-boundary-before.log |
| fixes / boundary-after | `.venv/bin/python -m pytest tests/test_boundary_audit.py tests/test_benchmarks.py tests/test_world.py tests/test_speciation.py` | 1 | 0.265 | 29212 | logs/fixes-boundary-after.log |
| fixes / boundary-verified | `.venv/bin/python -m pytest tests/test_boundary_audit.py tests/test_benchmarks.py tests/test_world.py tests/test_speciation.py` | 0 | 0.316 | 29212 | logs/fixes-boundary-verified.log |
| final / diagnostics | `.venv/bin/python -m neat.diagnostics` | 0 | 0.165 | 11776 | logs/final-diagnostics.log |
| final / wheel | `uv build --wheel --out-dir ../audit/package` | 0 | 8.539 | 44168 | logs/final-wheel.log |
| final / source-diff-check | `git diff --check 2105fee..HEAD` | 0 | 0.009 | 10368 | logs/final-source-diff-check.log |
| final / verified-ruff | `.venv/bin/ruff check .` | 0 | 0.017 | 24732 | logs/final-verified-ruff.log |
| final / verified-mypy | `.venv/bin/mypy neat world diversity experiments benchmarks visual` | 0 | 0.372 | 91352 | logs/final-verified-mypy.log |
| final / verified-wheel | `uv build --wheel --out-dir ../audit/package-final` | 0 | 0.569 | 43260 | logs/final-verified-wheel.log |
| final / verified-pytest | `.venv/bin/python -m pytest --cov=neat --cov=world --cov=diversity --cov=experiments --cov=benchmarks --cov=visual --cov-report=term-missing --cov-report=json:../audit/final-coverage.json` | 0 | 3.782 | 107896 | logs/final-verified-pytest.log |
| final / wheel-install | `uv pip install --python ../audit/wheel-env/bin/python ../audit/package-final/clage-0.4.0-py3-none-any.whl` | 2 | 0.016 | 12568 | logs/final-wheel-install.log |
| final / current-source-recheck | `.venv/bin/python ../audit/verify_outputs.py current --out ../audit/final/current-source-recheck.json` | 0 | 14.056 | 36428 | logs/final-current-source-recheck.log |
| final / wheel-install-verified | `uv pip install --python ../audit/wheel-env/bin/python ../audit/package-final/clage-0.4.0-py3-none-any.whl` | 0 | 4.426 | 37760 | logs/final-wheel-install-verified.log |
| final / wheel-smoke | `wheel-env/bin/python wheel_smoke.py` | 0 | 0.265 | 19964 | logs/final-wheel-smoke.log |
| final / verified-world-runtime | `.venv/bin/python ../audit/verify_outputs.py runtime --out ../audit/final/runtime.json` | 0 | 7.34 | 16384 | logs/final-verified-world-runtime.log |
| final / verified-neutral-replay | `.venv/bin/python ../audit/verify_outputs.py neutral --out ../audit/final/neutral.json` | 0 | 0.917 | 41108 | logs/final-verified-neutral-replay.log |
| final / verified-world-profile | `.venv/bin/python ../audit/profile_and_controls.py profile --out ../audit/final/world-profile.json` | 0 | 0.315 | 18584 | logs/final-verified-world-profile.log |
| final / packaged-configs-wheel | `uv build --wheel --out-dir ../audit/package-final` | 0 | 0.616 | 44160 | logs/final-packaged-configs-wheel.log |
| final / readme-python | `.venv/bin/python ../audit/readme_example.py` | 0 | 1.468 | 15104 | logs/final-readme-python.log |
| research / hardening-probes | `.venv/bin/python ../audit/hardening_probes.py` | 0 | 0.064 | 11392 | logs/research-hardening-probes.log |
| final / interpretation-after-boundaries | `.venv/bin/python ../audit/science_probes.py --out ../audit/final/science-probes.json` | 0 | 0.064 | 13180 | logs/final-interpretation-after-boundaries.log |

| final / final-diff-check | `git diff --check 2105fee..HEAD` | 0 | 0.016 | 10368 | logs/final-final-diff-check.log |

Every command ran from `/workspace/scratch/bac8048257f0/Clage` unless its ledger row in `commands.jsonl` specifies the baseline worktree or the isolated `audit/` wheel environment. The ledger also retains the exact cwd and UTC for each attempt.

## Handoff

The branch contains ten focused commits including the audit documents. HTTPS push returned an unavailable-credentials error, so no remote pull request was created. The accompanying Git bundle preserves the complete reviewable branch and commit history. Main was not modified or merged. The final local Python 3.12 checks passed; the revised branch has not run the remote Python 3.10/3.12 CI matrix.
