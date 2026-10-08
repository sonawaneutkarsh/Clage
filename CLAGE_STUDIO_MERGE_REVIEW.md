# Clage Studio 1.0 — merge preparation

Date: October 8, 2026. Scope: release review and CI stabilization only; no new
platform features, scientific parameter tuning, history rewrite or main merge.

## Branch and diff review

- Starting release HEAD: `5164cd47b633ba2330df74962c891a2ccfb51c5e`.
  The working tree was clean and the local/remote release refs matched.
- Fetched main: `2105fee258f2a329c351a4ab255fd585b3437a8a`. This is the
  release's merge base and an ancestor, rather than the older local main ref.
- Reviewed the complete changed-file inventory against that main, production
  NEAT/world/experiment/legacy-viewer changes, Studio API/contracts/orchestration,
  frontend and browser tests, packaging/CI, README and release documentation.
  Particular attention: evaluated versus provisional fitness, champion identity,
  budgets/innovation integrity, RNG ordering, replay validation, bounded exports,
  artifact paths, origin checks and scientific claims.
- Scanned all 236 changed paths and 299 payloads, including entries in the one
  retained evidence ZIP, for common credential/private-key patterns: no matches.
  This is a bounded automated scan plus review, not a guarantee of secret absence.
- The retained ZIP is an intentional historical browser-failure trace, not the
  unavailable original Work evidence ZIP. Screenshots, initial/failure logs and
  benchmark artifacts are intentional evidence; their historical values are not
  overwritten. Existing evidence has some whitespace diagnostics, not executable
  release defects. Build products, virtual environments, node_modules and scratch
  verification outputs are not included in these merge-preparation commits.

## Issues found and corrections

1. **Clean-machine WebSocket transport missing.** CI run `37727686607` on
   `bc87217048e8b61e492a8cd287ffba7e2db63cce` passed both Python jobs but failed
   five browser workflows (16 passed). Uvicorn logged:
   `No supported WebSocket library detected`. The developer environment had
   websockets installed independently, masking the omission from Studio extras.
   The browser could fetch the initial HTTP state but could not receive later
   API changes; this was a real installation blocker, not a timeout to relax.
   Added `websockets>=12,<16` to the optional Studio runtime dependencies, without
   requiring unrelated Uvicorn standard extras or changing simulation behavior.
2. **Transport regression coverage.** Added
   `test_real_uvicorn_websocket_stream_observes_http_changes`: launch actual
   Uvicorn on an ephemeral loopback socket, connect over a real WebSocket, receive
   initial tick 0, issue HTTP single-step, observe streamed tick 1, and verify
   shutdown. An in-process TestClient alone does not test Uvicorn's optional
   transport installation. This increases Python cases from 428 to **429**.
3. **Documentation drift.** README still reported 350 Python/seven browser
   workflows, and two consecutive experiment examples reused an output path
   despite overwrite protection. Corrected the counts and fresh-output guidance,
   linked final QA and updated the lead screenshot. Distinguished original tables
   from current-source validation. Clarified that checkpointing/full archival
   recording/multiple simultaneous runs are beyond the scoped release, rather
   than hidden current blockers. Marked original implementation/test reports as
   historical and corrected scientific notes to name the reconciled engine and
   odd-boundary observation definitions.

## Current-source verification

Code/package correction commit: `a4ba24b451c45344a66008f941942d2b1daedde0`.
Later merge-report edits are documentation only. Runtime Python/frontend digest
remains `8eeec68025bae4e957e53f2bafe024010adb91b50676a306834e91be5731d060`;
this digest does not include packaging metadata or tests.

| Check | Result |
|---|---|
| Fresh macOS Python 3.13 environment, install `.[dev,studio]`, pytest | **429 passed**, one dependency deprecation warning, 8.04 s |
| Fresh-environment Ruff | All checks passed |
| Fresh-environment mypy | No issues in 49 production source files |
| JavaScript unit tests | **4 passed**, no skipped/failed cases |
| Playwright, fresh-environment backend, isolated worktree | **21 passed**, 43.1 s |
| Isolated wheel build | `clage-0.4.1-py3-none-any.whl` built |
| Wheel installation in a separate fresh virtual environment | Six bundled configs, four static assets, actual HTTP API/export smoke and real Uvicorn WebSocket regression passed |
| Installed-wheel deterministic rerun | **18/18** recorded frames verified |
| Clean-source default complete-run/rerun | 8 generations, **2,560 ticks**, **42/42** retained frames verified through sequence 2,567 |
| GitHub Actions, corrected source | Both Python 3.10/3.12 jobs and Chromium browser job **passed** |

Corrected-source CI evidence:
https://github.com/sonawaneutkarsh/Clage/actions/runs/37728107702
The Linux logs confirm **429 Python cases on each version**, Ruff, mypy,
**4 JS tests and 21 Playwright workflows**. No assertions/timeouts were weakened,
tests skipped, workflow checks disabled, or new retries added to obtain green CI.
The intermediate documentation-only run was superseded by the normal workflow
concurrency policy, not used as successful verification.

The initial manual `--no-build-isolation` wheel command in the empty virtual
environment lacked setuptools and failed. The normal isolated build succeeded
using the declared build requirements; no repository fix was needed for that
invocation error. Fresh dependencies emit a Starlette TestClient/httpx
deprecation warning; tests still pass. This is a non-blocking maintenance concern.

## Unchanged-budget engine validation

Reran with 100 founders, engine defaults, minimal initialization, seeds 0–4 and
300 generations; no budget or scientific settings were tuned:

| Problem | Solved | Generations to solve, seed order |
|---|---|---|
| OR | 5/5 | 9, 6, 5, 6, 8 |
| AND | 5/5 | 6, 8, 7, 8, 17 |
| XOR | 5/5 | 212, 125, 68, 143, 102 |
| sine | 0/5 | all exceed the 300-generation budget; mean best fitness 0.9195 |

These match the final review's current-source outcomes, not proof of learned
foraging or generalization. Historical environmental tables and performance
measurements remain separate and unchanged. Newly generated paused-render timing
data stays in the disposable verification worktree rather than replacing the
published benchmark artifacts. Replay reruns from config/seed, not checkpoints.

## Reproduction commands and evidence

```bash
python3 -m venv /tmp/clage-check
/tmp/clage-check/bin/pip install -e ".[dev,studio]"
/tmp/clage-check/bin/python -m pytest
/tmp/clage-check/bin/python -m ruff check .
/tmp/clage-check/bin/python -m mypy neat world diversity experiments benchmarks visual studio
npm ci
npx playwright install chromium
npm test
PATH=/tmp/clage-check/bin:$PATH npm run test:e2e
/tmp/clage-check/bin/python -m pip wheel --no-deps . -w /tmp/clage-wheel
python -m studio.validate --out /tmp/clage-replay-validation.json
python -m benchmarks.run --problems or,and,xor,sin --trials 5 --generations 300 \
  --seed-base 0 --no-plot --report /tmp/clage-neat-validation.md
```

Run Playwright in a disposable checkout: it regenerates documented screenshots
and renderer measurements. Historical artifacts were preserved during these
reruns. Local scratch logs are in `/private/tmp/clage-pr-evidence-20261008`;
remote CI logs provide durable independently accessible test evidence. Existing
final interface screenshots remain under `docs/final-review/2026-10-08/`.

## PR and readiness

PR: https://github.com/sonawaneutkarsh/Clage/pull/1

The corrected code commit is mergeable and all three available checks are green.
Final documentation commits must also receive green PR checks before handoff;
the PR check summary is the authoritative final-head status. No main merge,
force-push or branch deletion was performed. Recommend maintainer review and
merging the **documented single-run Studio scope**, not declaring the larger
research-platform roadmap complete. No known remaining release blocker was
identified; bounded replay, no checkpoint resume, limited browser/accessibility
coverage, sine failure and unproven generalizable learning remain explicit.
