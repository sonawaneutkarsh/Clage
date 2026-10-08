# Clage reconciliation report

Date: October 7, 2026 (America/New_York).

**Status: blocked on unavailable Work artifacts. Studio was protected and fully
reverified, but the Work implementation has NOT been reconciled.** No engine,
simulation, frontend, test definitions or scientific settings were changed.
Passing tests and additional Studio features are not evidence of equivalence to
the missing Work fixes.

## 1. Protected baseline

- Starting branch: `studio/clage-1.0`.
- Starting HEAD: `2c80d0ef0092cf4a8b9bdd771b8a3fbc8f28d34e`.
- Initial `git status --porcelain=v1`: empty; no uncommitted work.
- Created local backup branch `backup/studio-pre-reconciliation-2026-10-07`
  pointing to that exact HEAD before searching or writing files.
- Local `main` remains `9bc763a855249ea3a07b7d67054f50dfa8634afd`.
- No push, default-branch merge, cherry-pick, force operation or branch deletion.

## 2. Artifact search and recovery

Searched Downloads, Desktop, Documents, `.codex`, `/private/tmp`, then readable
files throughout `/Users/utkarsh` (including Library/CloudStorage and app/container
locations), `/private/var/folders`, `/Users/Shared` and `/private/var/tmp`.
Searched case-insensitively for Clage/audit names, implementation report names
and Git `.bundle` files; supplemented with Spotlight filename search. No personal
external data drive was mounted; visible additional volumes were application
installer images. Unreadable/macOS-protected directories were not bypassed, so
this is a report of what was accessible, not proof that no other copy exists.

| Requested artifact | Recovery result |
|---|---|
| `CLAGE_COMPREHENSIVE_AUDIT.md` | Not found; could not read the 43 findings |
| `CLAGE_IMPLEMENTATION_REPORT.md` | Not found; could not read the 21 fixes/10 commits |
| `CLAGE_AUDIT_EVIDENCE.zip` | Not found |
| `CLAGE_AUDIT_CHANGES.bundle` | Not found |

Archive search in Downloads/Desktop/Documents/`.codex` found only an unrelated
`Desktop/nytr-demo-review.zip`; its entry names contained no Clage/audit/report
artifacts. Spotlight `.bundle` matches were macOS application resources, not
the requested Git file. No Git bundle was available to verify/list/import.
`git bundle verify` and `git bundle list-heads` were therefore **not run**;
no isolated Work refs or Work commits were fabricated.

Two existing checkouts were found and inspected read-only:

| Location | HEAD | Relevant evidence |
|---|---|---|
| `~/Downloads/Clage` | `5a00427922f686f9684f00f320373ad3eb3405df` | Main/origin refs, untracked Kiro spec directory |
| `~/Downloads/Clage-Kiro-Backup` | Same | Same refs/untracked directory |

Both contain a Codex checkpoint ref targeting
`be3668b0be49685b5097ecb4ebcefaae741631dd`, a **tree**, not a ten-commit Work
history. Its file listing contains no requested reports/artifacts. Read-only
`git fsck --full --no-reflogs` yielded no dangling recovery objects. The previously
reported `bec8c03a02242d2effdeb6e553c3fdf05c1931ce` object is absent in both.
Their `5a00427` HEAD is already an ancestor of protected Studio; that fact does
not establish recovery of the independent Work implementation. Both checkouts
and their untracked Kiro work were left untouched.

**Recovered Work bundle refs/commits: none.**

## 3. All 21 implemented Work fixes: classification unavailable

The supplied summary states 21 implemented fixes but does not identify their
names, requirements, patches or test cases. Neither source report nor bundle
could be recovered. Consequently **all 21 remain unassessed**; no defensible
individual A/B/C/D/E classification is possible yet.

| Category | Assessments established from recovered Work evidence |
|---|---:|
| A — already implemented equivalently/better | 0 |
| B — missing, integrate | 0 |
| C — partial, reconcile | 0 |
| D — obsolete architecture | 0 |
| E — incorrect/no longer desirable | 0 |
| Unassessed because identities/source are unavailable | 21 |

Unassessed is not category E or a claim that the fixes are unnecessary. Inventing
21 names or assigning A because Studio has 350 tests would defeat reconciliation.
The 43 findings, 21 fixes, ten commits and Work's benchmark/test claims remain
user-reported information, not independently verified recovered evidence.

### Existing Studio evidence ready for comparison (NOT Work classifications)

| Domain | Concrete current code/test locations |
|---|---|
| Innovation IDs/split ledger and persistence | `neat/innovation.py:37`; `tests/test_innovation.py:40`, `tests/test_innovation.py:76`; `tests/test_mutation.py:148` |
| Crossover/valid children | `neat/crossover.py:33`; `tests/test_crossover.py`; `tests/test_population.py:115` |
| Speciation/offspring allocation | `neat/speciation.py:171`, `neat/speciation.py:239`; `tests/test_speciation.py:171`, `tests/test_speciation.py:183` |
| Reproduction/champion correctness | `neat/population.py:215`, `neat/population.py:294`; `tests/test_population.py:185`; `tests/test_reproduction_recording.py:11` |
| Determinism/shared-world order | `tests/test_population.py:152`; `tests/test_world.py:233`, `tests/test_world.py:255`; `tests/test_studio.py:67` |
| Fitness/evaluation semantics | `world/fitness.py:13`; `neat/population.py:171`; `tests/test_population.py:209`; `tests/test_studio.py:104` |
| Frozen experiments/serialization | `studio/core.py:43`, `studio/core.py:251`; `tests/test_experiments.py:187`; `tests/test_studio.py:276` |
| Indexed-world RNG/performance parity | `tests/test_world_indexes.py:12`, `tests/test_world_indexes.py:30` |
| Scientific limitations | `CLAGE_STUDIO_RESEARCH_NOTES.md`; fresh XOR/sine evidence below |

These passing tests characterize Studio, not what Work changed or whether its
additional audit regression cases are covered. That comparison is still required.

## 4. Changes integrated / conflicts

**No implementation fixes integrated.** No blind cherry-picks, architecture
rewrites, new platform phases or speculative NEAT changes. No merge conflicts
because no Work history was imported. Changes in this reconciliation attempt
are limited to this report and separately named verification evidence under
`docs/reconciliation/2026-10-07/`.

## 5. Fresh verification of protected Studio

Used detached verification worktree
`/private/tmp/clage-reconciliation-20261007-verify` at the protected HEAD.
Its `node_modules` symlink reused existing locked dependencies. Browser tests
write screenshots/benchmark artifacts in that **isolated** checkout, avoiding
replacement of Studio's original evidence. Python 3.13.6, local macOS ARM;
headless Chromium, not a claim of remote CI or broader OS coverage.

| Command (in verification worktree) | Actual outcome |
|---|---|
| `python3 -m pytest -o addopts='' -q` | **350 passed**, 2.43 s |
| `python3 -m ruff check .` | All checks passed |
| `python3 -m mypy neat world diversity experiments benchmarks visual studio` | No issues, 49 source files |
| `npm test` | **4 passed** |
| `npm run test:e2e` | **7 passed**, 34.3 s; controls, neural inspection, genealogy, JSON/gzip replay, exports, configuration, live evolution, empty/large/responsive worlds |
| `python3 -m studio.validate --out /private/tmp/clage-reconciliation-default-validation.json` | 8 evaluated generations, 2,560 world ticks; retained frames reproduced exactly |
| `python3 -m benchmarks.run --problems xor,sin --trials 5 --generations 300 --seed-base 0 --no-plot --report /private/tmp/clage-reconciliation-neat-validation.md` | XOR **4/5**, sine **0/5** |
| `python3 -m pip wheel --no-deps --no-build-isolation . -w /private/tmp/clage-reconciliation-wheel` | Built `clage-0.3.0-py3-none-any.whl`, four static assets present |
| `python3 -m pip install --no-deps --target /private/tmp/clage-reconciliation-install /private/tmp/clage-reconciliation-wheel/clage-0.3.0-py3-none-any.whl` | Installed wheel, existing environment dependencies reused |

Installed-wheel smoke started `python3 -m studio --port 8878` from `/private/tmp`
with `PYTHONPATH=/private/tmp/clage-reconciliation-install`. Verified static assets,
OpenAPI, paused config, stepping, two evaluated generations, JSON/gzip equivalence
and unknown Git provenance for installed code. Exported a replay to
`/private/tmp/clage-reconciliation-installed-replay.json.gz`, then ran
`python3 -m studio.reproduce` against it in that installed environment: **18/18
frames matched**, last sequence 17. This is not a fresh-machine dependency test.

Logs and measured results are stored separately in `docs/reconciliation/2026-10-07/`.

## 6. Benchmarks and reproducibility

### NEAT result requiring follow-up

100 founders, engine defaults/minimal initialization, seeds 0–4, 300-generation
budget per seed, early stop on the repository's defined success criterion:

| Seed | XOR solved | Generation | Sine solved |
|---|---|---:|---|
| 0 | Yes | 243 | No |
| 1 | No | >300 | No |
| 2 | Yes | 204 | No |
| 3 | Yes | 147 | No |
| 4 | Yes | 165 | No |

XOR success requires all four truth-table errors strictly <0.5. Sine success is
MAE <0.1 over 21 training samples. The actual fresh report is
`docs/reconciliation/2026-10-07/neat-validation.md`.
Studio's XOR **4/5** differs from Work's reported **5/5**. Work's exact seed list,
budget, definitions and source are unavailable, so this is an unresolved discrepancy,
not proof of a regression or proof that Work is incorrect. No hyperparameter tuning
or definition change was made to force five successes. These training benchmarks
do not establish generalizable artificial-life learning; it remains unproven.

### Deterministic default run

Fresh initial run: 13.744 s for 2,560 ticks / eight generations, with reproduction
enabled. 2,567 advancement calls including generation boundaries. Retained 40
frames, explicitly dropped 2,528; 16,678,508 encoded frame bytes. Deterministic
rerun matched **40/40** retained frames through sequence 2,567. Peak process RSS
177.109375 MiB includes interpreter/validation/coexisting rerun, not a leak test.
Metadata correctly names source `2c80d0e`; verification checkout is marked dirty
because browser-generated screenshots/benchmark data changed, not source edits.

### Performance evidence preservation

No backend performance benchmark was rerun or substituted. Existing methods and
numbers in `CLAGE_STUDIO_PERFORMANCE_REPORT.md` remain unchanged. Fresh browser
workloads used the existing exact 120-frame, paused 96×96 / food 900 / genome-layer
method: approximately 60 FPS at 72/512/1,000/2,000 bodies, draw CPU samples
.267/.513/.882/1.608 ms. These are new descriptive observations saved separately
as `docs/reconciliation/2026-10-07/browser-benchmark.json`, not replacements or
claims of live inference/stream throughput.

Original evidence SHA-256 (unchanged at verification):

| Original file | SHA-256 |
|---|---|
| `docs/studio/backend-before.json` | `3aba9cef811047547152826cdb9f23f11d8624873d49dbca62ce453eb6330053` |
| `docs/studio/backend-after.json` | `b6ec343c4899ba70bcbf947e7e09cd95601ffaa3aff34ba78ce464cf9740dcce` |
| `docs/studio/browser-benchmark.json` | `225f46867cceb091f153e8fc6a640da5ed52ae1395499e29cddae763db62f40f` |
| `docs/studio/long-run-validation.json` | `b8630a407b319c50d27fb80c5128c6bd7e641c85bfe5f0ae31ad5448a00c5613` |
| `CLAGE_STUDIO_PERFORMANCE_REPORT.md` | `adeb85280ce0b835c72ed5b3476f6faf13399ce00d46e431872126c47e47d113` |

## 7. Commits, remaining concerns and exact next step

Protected source commits remain `e5366d2`, `f8fd0fe`, `fe94e8c`, `2c80d0e`.
Work commits recovered: **none**. Reconciliation implementation commits: **none**.
The documentation/evidence commit containing this report can be identified exactly
with `git log -1 --format=%H -- CLAGE_RECONCILIATION_REPORT.md`; no self-referential
commit hash is invented inside the file it hashes.

Still unresolved: all 21 Work fixes and their added tests, the independent audit's
43 findings, its ten commits, and XOR's reported 5/5 discrepancy. The baseline
suite passing does not eliminate unknown audit-discovered correctness concerns.

Provide `CLAGE_AUDIT_CHANGES.bundle` and both Markdown reports (or the evidence
ZIP containing them) in Downloads or attach them. Then verify/list the bundle,
import only into isolated refs, recover each actual fix's code/tests and construct
the requested A/B/C/D/E matrix before selective integration. No roadmap work
begins meanwhile. **Studio remains protected and verified, not declared reconciled.**
