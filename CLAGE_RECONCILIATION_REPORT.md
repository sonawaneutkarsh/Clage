# Clage Studio semantic reconciliation

## 1. Scope, preservation and provenance

This is reconciliation of the existing release candidate, not another Studio
implementation or research roadmap. Both recovered reports were read completely:
`CLAGE_COMPREHENSIVE_AUDIT.md` (1,047 lines) and
`CLAGE_IMPLEMENTATION_REPORT.md` (356 lines). Their 21 **Fixed** findings, seven
**Partial** findings with implemented portions, implementation sections and
stated limitations determine the scope below.

Before edits, `studio/clage-1.0` was clean at:

`9283da52f6beb787b62d90b4f742012537a0dced`

Safe local backup: `backup/studio-pre-semantic-reconciliation-2026-10-07` at that
exact HEAD. The earlier backup remains at
`2c80d0ef0092cf4a8b9bdd771b8a3fbc8f28d34e`. Main remains untouched at
`9bc763a855249ea3a07b7d67054f50dfa8634afd`. No push, merge, force-reset, branch
deletion, remote import or new platform phase was performed. The old blocked
reconciliation report and its verification artifacts are preserved separately.

### Recovered and unavailable evidence

Reports were recovered from `~/Downloads` and archived byte-for-byte under
`docs/reconciliation/semantic-2026-10-07/source-reports/`:

| Report | SHA-256 |
|---|---|
| Comprehensive audit | `db6a4a64e15b285424a873c68d7e7ec803fba5553382593911df0293ed0edda7` |
| Implementation report | `57d5f2f9b44191433f8bfa69145540664774866e15e727c1568788d968ecbf82` |

`CLAGE_AUDIT_CHANGES.bundle` and `CLAGE_AUDIT_EVIDENCE.zip` remain unavailable.
Searches of Downloads/Desktop/Documents found neither. Consequently:

- **Work bundle refs/commits recovered: none.** No bundle verification or import
  could be performed. No Work commits were merged or cherry-picked.
- `git cat-file -t bec8c03a02242d2effdeb6e553c3fdf05c1931ce` cannot resolve the
  reported production commit locally. Its code/tree/lineage is not reconstructed.
- The reports attribute work to `audit/comprehensive-2026-10-07`, baseline
  `2105fee258f2a329c351a4ab255fd585b3437a8a`, and production `bec8c03a...`.
  These are **report-attributed provenance**, not recovered Git objects.
- The reported commit inventory names nine IDs: `251d1fb`, `ad5c9f7`, `a2772c7`,
  `7a075ca`, `39445ad`, `712a708`, `09929c2`, `92868eb`, `bec8c03`.
  The earlier handoff said ten focused commits; an additional commit cannot be
  identified from the available inventory. No missing tenth ID is invented.
- Original audit test files, profiles, raw 160-trial results, frozen-control
  studies and original screenshots are unavailable. New semantic regression
  tests are independently authored, not copies of unavailable patches.

## 2. Reconciliation method and classification

Classification describes the **protected starting Studio source**, not a claim
that the final source still has the gap:

- **A:** implemented equivalently or better already.
- **B:** absent, valuable, implemented now.
- **C:** partially implemented, valuable remaining gap reconciled now.
- **D:** obsolete because its architecture was replaced.
- **E:** incorrect or undesirable in the current architecture.

The 21 Fixed findings contain **8 B and 13 C** classifications. There are no
whole-finding A/D/E entries in this group: retained legacy pipelines still have
users, so a superior Studio screen does **not** make their correctness fixes
obsolete. Several subfeatures are A/superseded, as recorded below. All valuable
B/C portions are integrated. Original scientific parameters/fitness coefficients
and historical results are preserved.

### All 21 implemented Work fixes

Code references describe the final implementation. Test references identify
concrete current regressions; their functions can be selected with pytest `-k`.

| ID / initial class | Starting gap and reconciliation | Current code / current test evidence |
|---|---|---|
| **C01 / C** Champion identity and first-generation preservation | Studio had defensive evaluated snapshots and ancestry logging, but champion presence compared only sizes/innovation numbers, and archival happened after reproduction. Identity now covers sorted node ID/type/bias and connection innovation/endpoints/weight/enabled; evaluation is archived before reproduction, including generation one. Original champion-parent logging remains intact. | `neat/population.py:62`, `_next_generation`, `_archive_best`; `tests/test_semantic_reconciliation.py:35` genotype-identity variants, `:51` first-generation retention; existing `tests/test_reproduction_recording.py:33` actual champion origin. |
| **C02 / B** Species elitism/population budgets | Elite count omitted available members, so the offspring loop could underfill a species allocation. Cap is now min(elitism, budget, members); allocation semantics are unchanged. | `neat/population.py:299`; `tests/test_semantic_reconciliation.py:64` one-member species with budget/elitism four; existing population-size/allocation suites retained. |
| **C03 / B** Crossover bias inheritance | Shared node bias always came from parent A. Unequal values now choose A/B equiprobably; equal values consume no new RNG draw. Neither parent is modified; conflicting shared roles reject. | `neat/crossover.py:33`; `tests/test_semantic_reconciliation.py:87` 20 seeds, both biases, parent immutability and equal-bias RNG state; `:101` conflict cases. |
| **C04 / B** Innovation/pair registration and structural atomicity | Imported genomes did not reserve/register history; custom interface IDs could collide with split IDs; old edges were disabled before ledger success. Registration now validates a one-to-one pair/innovation relationship transactionally and reserves both maxima before structural mutation. Population startup also registers founders. Split disable follows successful registration/minting. Serialized ledger counters/splits receive consistency checks. | `neat/innovation.py:54`, `from_dict`; `neat/mutation.py:166`, `:206`; Population initialization; `tests/test_semantic_reconciliation.py:115` imported 99 and IDs 14/15, conflict leaves genome/database unchanged, round-trip; `:132` rejection without mint/disable; `:245` invalid serialized counters. |
| **C05 / C** Invalid genomes and parent history | Duplicate pairs/cycles were rejected already, but duplicate/overlapping interface IDs, duplicate innovations, hidden-to-input edges and cross-parent history/role conflicts were not. These now reject explicitly while legitimate hidden cycles remain covered as invalid feed-forward topology. | `neat/genome.py:105`, `validate_connection`, `validate`; crossover preflight; `tests/test_semantic_reconciliation.py:144`, `:149`, `:101`; corrected hidden-cycle fixture in `tests/test_crossover.py:153`. |
| **C06 / B** Empty compatibility normalization | With small-genome threshold zero, empty genomes divided by zero. N is now at least one. | `neat/speciation.py:133`; `tests/test_semantic_reconciliation.py:159` empty-genome distance at threshold zero equals zero. |
| **C07 / C** Generation/founder count validation | Constructor counts already had useful validation, but `run()` accepted invalid generation budgets and explicit founder/population-size disagreement. Generation count must be integer/nonbool and nonnegative; invalid calls do not advance; explicit founder size must agree. | `neat/population.py:158`, initialization; `tests/test_semantic_reconciliation.py:74`, `:81`; existing constructor-boundary tests preserved. |
| **C08 / B** Diagnostic callback/extinction recovery | Diagnostics passed a Random object as generation and could empty reproduction after all species were pruned. Callback now receives integer generation; evaluated members are re-speciated when pruning leaves no species. | `neat/diagnostics.py:64`; `tests/test_semantic_reconciliation.py:159` three members, five flat-fitness generations, stagnation one; `diagnostics.txt` captures actual CLI completion. |
| **D01 / C** Experiment seeds/config/path validation | Existing condition/control checks did not protect direct configurations from repeated/noninteger seeds, unsafe path components or invalid generation budgets. These now fail early; direct trial/condition entry points validate before work and reject foreign conditions. | `experiments/config.py:107`; `experiments/run.py:101`, `:170`; `tests/test_semantic_reconciliation.py:174`, `:180`, `:212`, `:238`. |
| **D02 / C** Overwrite protection/provenance/semantic controls | Studio already froze configs and had bounded artifact storage; the legacy runner still overwrote raw results and lacked manifests. Whole runs now require fresh output; direct conditions preflight all seed/config/replay targets. Manifests store UTC, actual source SHA, available commit, resolved configs, Python/platform/matplotlib, seeds and semantic control. Studio additionally fingerprints actual code/assets and records runtime, rather than relying on HEAD/dirty alone. | `experiments/run.py:170`, `:225`; `experiments/report.py:84`; `studio/core.py:82`, `studio/reproduce.py:11`; `tests/test_semantic_reconciliation.py:196`, `:212`, `:359`; existing source-root and installed-provenance tests in `tests/test_studio.py:40`, `:47`; installed smoke verifies unknown Git attribution. |
| **D03 / B** Aggregation validation | Both aggregators accepted empty/ragged/corrupt trials or truncated them. Both now require nonempty equal lengths, sequential one-based generations, all required finite numeric metrics; valid sample-SD behavior remains. A shared standard-library-only validator keeps `visual.data` free of simulation/frontend runtime dependencies. | `visual/data.py:91`, `:108`; `experiments/analysis.py:64`; `tests/test_semantic_reconciliation.py:256` 12 invalid-contract combinations, plus original valid aggregates. |
| **D04 / C** Evaluated replay fitness | Studio's own v2 recordings already distinguished live/provisional and evaluated state, but supported generation recordings still captured inherited fitness. Legacy recording v2 now finalizes scores after evaluation and labels phase. V1 remains viewable with stored/unverified labels; unknown versions/empty recordings reject. | `world/recorder.py:87`, `finalize`, `to_dict`; `world/simulation.py:148`; `visual/data.py:49`; `tests/test_semantic_reconciliation.py:274`; existing `tests/test_studio.py:276`, `:306` preserve newer fidelity. |
| **W01 / B** Odd-grid boundary sensing | Interior span incorrectly produced negative center values on odd grids. Denominator is max(1,(dimension−1)//2); even-grid behavior is unchanged. Observation regime is labeled in Studio provenance. | `world/organism.py:59`; `tests/test_semantic_reconciliation.py:295` 5×7 center; original even-grid world tests remain. |
| **W02 / B** Exhausted reproduction endpoints | Fractions zero/one could leave zero-energy child/parent alive. Such bodies are immediately killed/removed after transfer, while returning the actual child and incrementing the birth count. Shared networks, parent identity and inference instrumentation remain. | `world/organism.py:147`; `tests/test_semantic_reconciliation.py:295` both endpoints, occupancy removal and birth count; existing reproduction/world-index tests. |
| **B01 / C** Benchmark budget/CLI errors | Population size checks existed indirectly, but zero/invalid trial/generation budgets and unknown problem names failed poorly. Positive nonbool integer budgets validate before loops; invalid CLI names/budgets give argparse errors. Benchmark defaults/success criteria were not tuned. | `benchmarks/run.py:63`, `:124`, `:145`; `tests/test_semantic_reconciliation.py:352`; full default-budget OR/AND/XOR/sine verification below. |
| **V01 / C** Faithful legacy world playback | Studio already used correct row orientation/control state. Legacy north disagreed with world coordinates; tick zero/control labels/redraws were wrong. Y is inverted consistently with north=(0,−1), tick zero renders immediately, animation delegates slider redraw, end/pause/step labels reflect state. Existing export directory handling remains. | `visual/world_view.py:22`, `:30`, `WorldViewer`; `tests/test_reconciled_viewers.py:27`; installed legacy PNG and headless canvas-control tests. |
| **V02 / C** Terminal selection/input/rendering | Export directories already worked, but double-width food, invisible cursor, absent body ID/% labels, buffered fragmented input and invalid FPS remained. One-column food/cursor, rendered-grid bounds, explicit body/%/final-score phase, byte-based queued/fragmented keys and redraw suppression are integrated. Tick zero displays before advancement; end stops instead of looping. | `visual/terminal_view.py:74`, `:119`, `:241`, `:306`; `tests/test_reconciled_viewers.py:46`, `:56`, `:90` actual PTY selection, fragmented arrow and quit. |
| **V03 / C** Network/table/CLI fidelity | Studio already had richer live semantic SVG inspection. Legacy topology had cramped labels, no directional arrow/legend, wide tables and ambiguous genome flag. Adaptive physical spacing/figure height, separate bias labels, enabled arrows/sign/disabled legend and persistent unwired explanation were added. Tables are transposed; `--genome` is primary with alias preserved; export parents/unknown metrics handled. | `visual/network_view.py:27`, `:63`; `visual/analytics.py:39`, `:82`; `visual/app.py:73`, `:96`; `tests/test_reconciled_viewers.py:61`, `:76`; inspected `screenshots/legacy-network-final.png`. |
| **P01 / C** Inference/world hot paths | Studio had valuable empty-cell/food indexes, shared networks and cached input membership already. Remaining compiled noninput plans and clipped row-density scans are integrated, preserving accumulation order, full-square denominator, self/exclusion and off-grid behavior. Existing indexes and RNG/tie ordering are retained. | `neat/phenotype.py:37`, `activate_with_trace`; `world/grid.py:158`, `:170`; `tests/test_semantic_reconciliation.py:310` recursive oracle over 200 cases, `:333` density oracle; existing instrumentation and world-index parity tests. |
| **E01 / C** Packaging | Studio assets/build ignore were already present; six JSON configs were absent from wheel and setuptools minimum predates PEP621 support. Config package data and minimum ≥61 are added. Local reconciled release is **0.4.1**, deliberately not presented as recovered Work 0.4.0. | `pyproject.toml:1`, build/package-data sections; `.gitignore:2`; wheel build/install logs and `installed_smoke.py` verify six configs, four assets, imported installed packages, HTTP/replay/legacy exports. |
| **DOC01 / C** Scientific/documentation discipline | Studio already labeled historical results and limitations, but unsupported 100-generation anecdote/strong cross-seed language survived. Those claims are withdrawn, numeric tables/images retained. Earlier correctness history is retained with appended reconciliation contracts; RNG/hash, entropy/coupling/SD/seed-layout limits are explicit; all four tutorial Python blocks execute in order. | `README.md:80`, Programmatic use, Design notes; `docs/correctness-notes.md:1`; `readme-smoke.txt` PASS; original historical benchmark/performance artifacts remain unchanged. |

### Other implemented portions described by the reports

Additional D02 provenance regression:
`tests/test_semantic_reconciliation.py:223` verifies that an installed legacy
runner does not attribute its source to an unrelated parent Git checkout.

These seven **Partial** findings are not additional completed research claims.
Classification applies only to the already implemented capability/clarification,
not to the proposed follow-on experiments.

| ID / class | Semantic reconciliation and current evidence |
|---|---|
| **R01 / A (capability)** Frozen-policy reference | Studio already supplies fixed MOVE, random and privileged hand-forager references plus saved champions on held-out seeds, without training during evaluation. `studio/core.py:497` (`evaluate_policies`), `tests/test_studio.py:232`, `:258` verify deterministic, nonmutating evaluation and seed separation. This supersedes the useful control capability, not the unavailable original 15-trial study. No claim that evolution beats these controls is added. |
| **R06 / C (clarification)** Entropy/scaling | Existing entropy-boundary/food-alignment corrections and Studio histogram caveats were retained. Appended correctness notes explicitly warn that population z-scores erase absolute scale and stochastic actions/placement can mimic diversity. `diversity/metrics.py`, `tests/test_diversity.py`, `docs/correctness-notes.md` are the current evidence. Proposed metric-v2 normalization/null-distribution experiments remain unimplemented. |
| **R07 / C (clarification)** Coupled OFAT factors | Retain shipped configs and their historical identity. Notes clarify that `reproduction_cost.json` varies threshold, not an energy charge, and space/founder/food sweeps alter coupled densities/distances. `experiments/config.py:33`, six config JSONs, original config tests and `docs/correctness-notes.md`. No density-matched sweep is newly claimed. |
| **R08 / C (clarification)** Replicate/uncertainty/layout limits | Studio already exports actual per-seed held-out outcomes and labels sample SD with n=5; original aggregators use sample stdev. Notes extend the caution to dependent generations, small training-seed samples and diverging layouts under equal seeds. `studio/core.py:evaluate_policies`, `tests/test_studio.py:232`, `experiments/analysis.py:55`, research/correctness notes. No inferential significance/generalization claim is added. |
| **R10 / C (wording)** Sine diagnosis | The former "layers work; solutions exist" assertion is weakened to a hypothesis; smoke checks cannot prove reachable bounded solutions or exclude defects. `benchmarks/diagnose.py:197`, current sine result and diagnostic CLI logs. Bounded hand solutions/held-out sine ablations remain open. |
| **E07 / C (verification)** Integration/visual regression coverage | Existing seven browser workflows and Python API/instrumentation/lineage parity tests are preserved. Added legacy canvas-control, export, contract and actual PTY tests, wheel HTTP/replay smoke and independent inference/density oracles. 73 new collected cases, not recovered Work test files. Native GUI/display and multi-OS execution are still unverified. |
| **P02 / C (documentation)** Seed/hash honesty | Seed mapping is not changed. `world/config.py:188` no longer promises impossible collision-free finite hashes. Notes accurately distinguish unused stride in `world_rng_seed` from its use by legacy experiment seed resolution (`experiments/config.py:255`). Configuration debt is acknowledged, not silently removed. |

### The remaining 15 findings are not implemented Work fixes

R02/R03/R04/R05/R09, C09, V04/V05, E02/E03/E04/E05/E06/E08 and DOC02 are
Proposed/Open in the recovered audit. They are accounted for, not silently
misrepresented as part of its 21 fixes. Studio already supplies some stronger
inspection/performance/persistence capabilities (V04/V05/E08/DOC02), but this
session does not begin their broader roadmap. See remaining concerns below.

## 3. What changed, conflicts and compatibility

- From-scratch NEAT/evaluation/world code remains independent of browser rendering.
  No third-party NEAT replacement, broad rewrite, new cloud service or optimizer.
- Existing Studio WorldSession orchestration, background manager/WebSocket stream,
  bounded v2 replays, camera/graphs/inspection, actual activation trace, genome
  comparison, separate organism/evolution ancestry and held-out interface remain.
- Four historical test fixtures encoded inconsistent history or an input-target
  edge. They were corrected rather than relaxing validation: the repeated 1→10
  pair reuses innovation 2; a later manually imported edge does not reuse an
  already minted split innovation; imported history is expected in the ledger;
  the cycle test uses two hidden nodes. Original regression intent is preserved.
- No Git merge conflicts occurred: unavailable patches were never imported.
  Architecture conflicts were resolved semantically—extend the incremental world
  finish boundary and shared immutable networks, do not recreate the old engine.
- Scientific regime is `neat-reconciled-v1`, observations
  `world-observations-v2-odd-boundaries`, original `world-fitness-v1` unchanged.
  Bias inheritance/champion correction legitimately alter evolutionary RNG
  trajectories. Odd-grid/fraction-endpoint runs are also scientifically changed.
- Legacy replay schema becomes v2 with evaluated/unevaluated fitness phase. Studio
  keeps its own v2 schema/lineage extension. Older Studio recordings remain viewable;
  rerun verification rejects missing/different scientific definitions rather than
  pretending the corrected engine regenerates historical observations. Neither
  replay format is a complete RNG/innovation/species checkpoint.
- Actual source hashes now distinguish dirty source from a nominal Git HEAD.
  Installed wheels correctly report no checkout commit. Source hashes are
  provenance, not signatures/authentication of a recorded history.
- Initial legacy network visual QA found crowding and an overwritten unwired
  explanation. Physical spacing/adaptive figure sizing and a separate metadata
  title correct that; final exported image was inspected. The earlier correctness
  notes are preserved, not replaced by the reconciliation narrative.

## 4. Verification

Final production-source verification target:
`d05255547ff9328ff04a9fa6b3f2527f0e199076` (engine 0.4.1).
The delivery report/evidence commit contains no subsequent production changes.

All 350 existing Python cases remain. The new two modules collect **73 cases**;
on the protected starting HEAD, **66 fail / 7 pass**. On corrected source all pass.
This is measured counterexample evidence, not an assumption that a higher count
proves reconciliation. `pre-fix-regressions.txt` retains failure detail.

| Verification | Current outcome / evidence |
|---|---|
| Full Python | **423 passed**, `python-tests.txt` |
| JavaScript | **4 passed**, `javascript-tests.txt` |
| Playwright | **7 passed** on final unchanged workflows, `playwright.txt`, `browser-results.json` |
| Ruff | All checks passed, `ruff.txt` |
| mypy | No issues, 49 production source files, `mypy.txt` |
| Wheel/install | Built/installed `clage-0.4.1-py3-none-any.whl`; six configs, four assets, outside-checkout HTTP and replay smoke, `wheel-build.txt`, `installed-smoke.txt` |
| Deterministic full default Studio run | 8 generations / 2,560 world ticks; retained actual frames regenerate exactly, `determinism.json` |
| Installed portable replay | **18/18** frames verified through sequence 17; JSON/gzip identical, unknown Git provenance correct, actual source digests agree |
| Instrumentation/ancestry/index parity | Existing tests plus recursive 200-case inference and clipped-density oracles pass |
| Default OR/AND/XOR/sine | **5/5, 5/5, 5/5, 0/5**; unchanged 100 founders, 300 generations, seeds 0–4, `neat-validation.md`/`.log` |
| Diagnostic CLI | Completes, `diagnostics.txt` |
| README examples | Four Python blocks execute verbatim in document order/shared namespace, `readme-smoke.txt` |
| Visual QA | Headless browser interaction plus viewed ecosystem/neural/mobile and legacy world/network images; archived `screenshots/`. No native-desktop claim. |

One concurrent browser run timed out waiting for the initial render-FPS indicator;
six workflows passed and one failed in setup. It is preserved in
`playwright-concurrent-failure.txt` and `browser-failure/` (trace, screenshot,
context). Unchanged sequential reruns pass all seven. No assertions/budgets/retry
settings were weakened. Scheduling contention is plausible, **not a proven cause**;
a sporadic readiness flake remains a QA concern. Static verification also caught
an adaptive-layout type annotation omission during iteration, which was fixed
before final verification. No failures are represented as successful checks.

### Exact commands / execution boundaries

From repository root, with existing local development dependencies:

```sh
python3 -m pytest -o addopts='' -q
python3 -m ruff check .
python3 -m mypy neat world diversity experiments benchmarks visual studio
npm test
python3 -m neat.diagnostics
```

Browser/performance/replay verification uses isolated detached worktree
`/private/tmp/clage-semantic-verify-20261007` at the source target above, with a
symlink to the repository's locked node_modules. Browser-generated changes in
that checkout are outputs, not source edits; metadata dirty=true reflects these
outputs. Original Studio benchmark/image files in the main checkout are untouched.
Let `E=/Users/utkarsh/clage/docs/reconciliation/semantic-2026-10-07`:

```sh
E=/Users/utkarsh/clage/docs/reconciliation/semantic-2026-10-07
cd /private/tmp/clage-semantic-verify-20261007
npm run test:e2e
python3 -m studio.validate --out "$E/determinism.json"
python3 -m studio.benchmark --out "$E/backend-benchmark.json" --repeats 3
python3 -m benchmarks.run --problems or,and,xor,sin --trials 5 --generations 300 --seed-base 0 --no-plot --report "$E/neat-validation.md"
```

Wheel/install smoke uses fresh target directories, existing environment
dependencies (not a claim of a fresh-machine dependency installation):

```sh
cd /Users/utkarsh/clage
python3 -m pip wheel --no-deps --no-build-isolation . -w /private/tmp/clage-semantic-delivery-d052-wheel
python3 -m pip install --no-deps --target /private/tmp/clage-semantic-delivery-d052-install /private/tmp/clage-semantic-delivery-d052-wheel/clage-0.4.1-py3-none-any.whl
cd /private/tmp
PYTHONPATH=/private/tmp/clage-semantic-delivery-d052-install python3 "$E/installed_smoke.py" /private/tmp/clage-semantic-delivery-d052-install
```

Smoke launches its own loopback server on 8879 and terminates only that process.
Browser workflows use 8876. The existing user Studio session on 8765 is preserved.
There is no build step for the buildless ES-module frontend; wheel asset checks
and real HTTP/browser execution verify its shipping/build contract.

## 5. Benchmarks, reproducibility and claim limits

### Unchanged-budget NEAT checks

| Seed | OR solve generation | AND solve generation | XOR solve generation | sine |
|---:|---:|---:|---:|---|
| 0 | 9 | 6 | 212 | Not solved by 300 |
| 1 | 6 | 8 | 125 | Not solved by 300 |
| 2 | 5 | 7 | 68 | Not solved by 300 |
| 3 | 6 | 8 | 143 | Not solved by 300 |
| 4 | 8 | 17 | 102 | Not solved by 300 |

These current-source results independently match the recovered reports' aggregate
OR/AND/XOR/sine summaries, including XOR min/median/max 68/125/212 and sine mean
best fitness approximately .9195. They do **not** authenticate unavailable original
test logs or prove identical Work source. Protected Studio previously yielded
XOR 4/5 (243, unsolved, 204, 147, 165); its evidence remains under the older dated
reconciliation folder. Correcting biased bias inheritance and true champion
retention changes the search trajectory without tuning rates, budgets, seeds,
initialization or success criteria. Their separate causal contributions were
not isolated; no causal ablation or general success-rate claim is fabricated.

OR/AND/XOR are training truth-table checks. Sine is 21 training samples, MAE<.1;
XOR requires all four errors strictly<.5. None proves artificial-life foraging,
generalization or learned ecological behavior. Sine remains unresolved.

### Preserved performance methodology

Original evidence is byte-for-byte preserved (SHA-256):

| Original file | SHA-256 |
|---|---|
| `docs/studio/backend-before.json` | `3aba9cef811047547152826cdb9f23f11d8624873d49dbca62ce453eb6330053` |
| `docs/studio/backend-after.json` | `b6ec343c4899ba70bcbf947e7e09cd95601ffaa3aff34ba78ce464cf9740dcce` |
| `docs/studio/browser-benchmark.json` | `225f46867cceb091f153e8fc6a640da5ed52ae1395499e29cddae763db62f40f` |
| `docs/studio/long-run-validation.json` | `b8630a407b319c50d27fb80c5128c6bd7e641c85bfe5f0ae31ad5448a00c5613` |
| `CLAGE_STUDIO_PERFORMANCE_REPORT.md` | `adeb85280ce0b835c72ed5b3476f6faf13399ce00d46e431872126c47e47d113` |

Fresh measurements are separately named `backend-benchmark.json`,
`browser-benchmark.json`, `determinism.json` in the semantic evidence directory,
with raw repetitions and machine/source provenance. Existing workloads remain:

- Backend: N=72/512/1000; seed 42, 30 ticks, reproduction disabled by threshold 2,
  three repetitions per recording mode, same world/food sizes; exact medians are
  in `backend-benchmark.txt`/`.json`. Measures engine/instrumentation/capture and
  separately initialization/serialization—not GPU rendering or long-run evolution.
- Browser: N=72/512/1000/2000, paused 96×96 world, food 900, genome layer/grid off,
  fit camera, viewport 1440×1100, 120 RAF intervals per workload in headless
  Chromium. Results report actual RAF FPS, p95 interval and drawing CPU means.
- Default complete run: eight evaluated generations, 2,560 ticks and deterministic
  rerun of retained frames. Snapshot eviction is explicit, not simulated history.
  Peak RSS includes interpreter and coexisting original/rerun, not a leak test.

The final measured values are summarized in `MEASURED_RESULTS.md` alongside raw
JSON/logs. No Work timing (Linux 28.2% inference/11.5% world elapsed-time reduction)
is imported as a current Studio measurement: those raw profiles are unavailable,
and this macOS ARM workload/runtime is different. No headline speedup is claimed
from comparing old and new scientific regimes or concurrently loaded runs.

### Historical results that remain unverified

The reports' original 349 tests, 43-finding audit execution, ten-commit claim,
160 ecological trials, frozen MOVE study, before/after profile percentages and
original environment freeze are **historical reported evidence only**. Matching
current benchmark summaries and passing independently authored regressions do not
recover those artifacts. Historical README numeric tables/images and Studio
performance reports are not silently replaced by corrected-engine numbers.

## 6. Resulting commits

These are actual local Studio commits, **not** reconstructed Work commits:

| Commit | Focus |
|---|---|
| `bbbed3e12d8afc3c7164baf6c97fbb068fcbfbbd` | NEAT/world/legacy data contracts, packaging and semantic regressions |
| `447282099802ec5606b4b7ecd14ad4cead970663` | Viewer fidelity and scientific presentation |
| `6656ac0ae51c17b18fdb26ec475b7be90cb86948` | Artifact preflight, actual PTY edge cases, interpretation notes |
| `960ce5009c87d0981883caa87835fb973ac5bbbc` | Preserve old correctness history, executable tutorials, visually checked adaptive exports |
| `6f918e70a908b0da396a831999ae76c24f048e3c` | Static type correction for adaptive layout |
| `3d932536c3a6635cfbda299898776e5a98d7a1f0` | Actual Studio source fingerprints and replay provenance |
| `13bd9ff6a84abc83491a78765b666da4b91e1430` | Studio Python/platform execution provenance |
| `d05255547ff9328ff04a9fa6b3f2527f0e199076` | Installed legacy manifest rejects unrelated parent Git attribution |

The report/evidence delivery commit is discoverable with
`git log -1 --format='%H %s' -- CLAGE_RECONCILIATION_REPORT.md`; it cannot contain
its own final hash. No PR or pushed-branch reference is fabricated.

## 7. Remaining correctness/research concerns

- Missing Work bundle/evidence prevents exact patch lineage and complete audit
  reproduction. This is a completed **semantic** reconciliation, not Git recovery.
- Full arbitrary engine/legacy external-schema hardening remains incomplete
  (audit E03): pathological field types/nonfinite node parameters/untrusted huge
  histories need a separately scoped contract. Current validation is meaningful
  but not a proof against every malformed external object.
- Public `best_genome` and nested statistics remain mutable API debt (E04).
  The evaluated-best accessor and Studio snapshots still have defensive isolation.
- Legacy experiment publication is not atomic/resumable (E02). Interrupted output
  requires a fresh destination. No replay is promoted to a resumable checkpoint.
- Historical raw world winners span different layouts; held-out selection and
  replicated training-seed uncertainty remain needed. Dormant inherited nodes,
  bias-insensitive speciation, sensing/reproduction/fitness variants and sine
  representability remain research questions, not silently altered definitions.
- Generic package-name collisions, broad dependency ranges and the license-table
  deprecation remain packaging debt. Native desktop and cross-OS visual coverage
  are not established. The observed browser readiness flake is retained as a risk.
- Generalizable learning remains explicitly unproven.

Stop point: the Studio branch contains the focused reconciliations and archived
verification, preserves main/backups/old evidence, and begins no additional
platform phases. Use `python3 -m studio` for normal local operation.
