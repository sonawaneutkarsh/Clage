# Implementation Plan

Base verify command for every task (the default `addopts` reference `pytest-cov`, which is not
installed; `pyproject.toml` must stay untouched — bugfix.md 3.14):

```
python3 -m pytest -q -o addopts=''
```

Ordering rule for the whole plan: **every regression test is written and run against the UNFIXED
code first, and its documented pre-fix failure is observed and recorded, before the matching fix is
applied.** design.md records the exact measured pre-fix value for each test, so each test task
below states the value that must be observed. Bugs land smallest-blast-radius first: Bug 3
(one loop in `diversity/metrics.py`), then Bug 2 (`diversity/metrics.py` surface + helper
extraction), then Bug 1 (`neat/population.py` accessor + `benchmarks/run.py` sourcing).

---

- [ ] 1. Record the pre-fix preservation baseline
  - **Property 2: Preservation** - Pre-fix baseline for non-triggering inputs
  - **IMPORTANT**: observation-first — this task only observes and records, it changes no file
  - Run the full suite on UNFIXED code and record the total: **188 passed** is the expected baseline
  - Record the pinned pre-fix values that later tasks must reproduce bitwise:
    `test_transition_entropy_rate_deterministic` (`0.9649839288802097`),
    `test_transition_entropy_rate_alternating_pairs`,
    `test_per_genome_metrics_pooling_and_window` (`action_entropy == 1.0`, `encounter_rate == 0.25`),
    `test_food_alignment_cosine_directions`, `test_food_alignment_skips_zero_deltas`,
    `test_world_records_trace_when_enabled` (`len(org.trace[0]) == 6`),
    `test_statistics_use_evaluated_fitness_not_carried`, `test_seeded_run_is_deterministic`,
    `test_engine_has_no_environment_dependencies`,
    `test_run_trial_is_deterministic`, `test_run_trial_history_is_well_formed`,
    `test_experiment_records_behavioral_metrics`,
    `test_empty_genome_baseline_zero_entropy_and_diversity`,
    `test_population_diversity_zero_for_identical_genomes` (`== 0.0`)
  - **EXPECTED OUTCOME**: 188 passed — this is the set that must stay green with **zero** assertion
    edits, except where a changed value is a documented intended consequence (bugfix.md 2.14)
  - Files touched: none
  - Verify: `python3 -m pytest -q -o addopts=''`
  - _Requirements: 2.14, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 3.10, 3.11, 3.12, 3.13, 3.14_

## Bug 3 — food-alignment temporal misalignment (smallest blast radius)

- [ ] 2. Write the Bug 3 exploration test and prove it fails on unfixed code
  - **Property 1: Bug Condition** - 1c Food alignment pairs each displacement with the observation that preceded its action
  - **CRITICAL**: this test MUST FAIL on unfixed code — the failure is what confirms the bug
  - **DO NOT fix the test or the code in this task**
  - **Scoped PBT approach**: Bug 3 is deterministic, so the property is scoped to the two concrete
    failing cases measured in design.md rather than randomized
  - File: `tests/test_diversity.py`, new test
    `test_food_alignment_pairs_displacement_with_preceding_observation`
  - Case 4a (off-by-one inside one trace): trace
    `[(0, 0, 0, 1.0, 0.0, 0.0), (0, 1, 0, -1.0, 0.0, 0.0)]`,
    assert `per_genome_metrics([t], window=0, area=100)["food_alignment"] == pytest.approx(-1.0)`
  - Case 4b (no cross-organism pairing): traces
    `[(0, 0, 0, 1.0, 0.0, 0.0), (0, 1, 0, 1.0, 0.0, 0.0)]` and
    `[(0, 5, 5, 0.0, 1.0, 0.0), (0, 5, 6, 0.0, 1.0, 0.0)]`,
    assert `per_genome_metrics([t1, t2], window=0, area=100)["food_alignment"] == pytest.approx(1.0)`
  - **EXPECTED PRE-FIX OUTCOME (measured, must be observed and recorded in the test docstring)**:
    4a returns `+1.0` instead of `-1.0` (full sign inversion — movement directly away from food
    scored as perfect food-seeking); 4b returns `0.5` instead of `1.0` (organism B's southward step
    paired with organism A's eastward food direction, scoring a `+1.0` sample as `0.0`)
  - Confirm the root cause while reading the failure: `food_dirs` appends on every index while
    `deltas` appends only for `index > 0`, and `zip` accepts the mismatch, so the offset accumulates
    across traces. If the observed values differ from the two above, stop and re-hypothesize
  - Mark complete when the test is written, run, and both documented failures are recorded
  - Verify: `python3 -m pytest -q -o addopts='' tests/test_diversity.py::test_food_alignment_pairs_displacement_with_preceding_observation`
  - _Requirements: 1.7, 1.8, 2.9, 2.13_

- [ ] 3. Fix Bug 3 — aligned food-alignment pairs

  - [ ] 3.1 Build the alignment pairs per trace with matching indices
    - File: `diversity/metrics.py`, `per_genome_metrics` per-trace loop only
    - Replace the `deltas` / `food_dirs` accumulation with, for each clipped trace and each
      `i >= 1`, appending `(x_i - x_{i-1}, y_i - y_{i-1})` to `aligned_deltas` **and**
      `(fx_i, fy_i)` to `aligned_food_dirs` in the same iteration, so the two lists cannot
      desynchronize; never pair across traces
    - Call `food_alignment_cosine(aligned_deltas, aligned_food_dirs)` for `food_alignment`
    - Keep `actions`, `points` and the density counters exactly as they are, in the same iteration
      order, so `action_entropy`, `spatial_coverage` and `encounter_rate` stay bitwise identical
    - Do not change `food_alignment_cosine`'s signature or its zero-magnitude skipping
    - Do not change the returned dict keys; do not change the trace tuple shape
    - _Bug_Condition: isBugCondition_3(X) — some clipped trace has length >= 2_
    - _Expected_Behavior: Property 1c — mean cosine over `(pos_i - pos_{i-1}, (fx_i, fy_i))` for `i >= 1`, zero-magnitude samples skipped_
    - _Preservation: Preservation Requirements 3.3, 3.4, 3.5, 3.6_
    - _Requirements: 2.9, 2.10_

  - [ ] 3.2 Verify the Bug 3 exploration test now passes
    - **Property 1: Bug Condition** - 1c Food alignment pairs each displacement with the observation that preceded its action
    - **IMPORTANT**: re-run the SAME test from task 2 — do NOT write a new test
    - **EXPECTED OUTCOME**: PASSES — 4a is now `-1.0` and 4b is now `1.0`
    - Verify: `python3 -m pytest -q -o addopts='' tests/test_diversity.py::test_food_alignment_pairs_displacement_with_preceding_observation`
    - _Requirements: 2.9, 2.13_

  - [ ] 3.3 Verify no regressions from the Bug 3 fix
    - **Property 2: Preservation** - Pooled metrics and the `food_alignment_cosine` contract
    - **IMPORTANT**: re-run the baseline from task 1 — do NOT edit any existing assertion
    - **EXPECTED OUTCOME**: `test_food_alignment_cosine_directions`,
      `test_food_alignment_skips_zero_deltas`, `test_per_genome_metrics_pooling_and_window`,
      `test_world_records_trace_when_enabled` and the `experiments` tests all still pass untouched
    - Verify: `python3 -m pytest -q -o addopts=''`
    - _Requirements: 3.3, 3.4, 3.5, 3.6, 3.11, 2.14_

## Bug 2 — transition entropy across organism boundaries

- [ ] 4. Write the Bug 2 exploration test and prove it fails on unfixed code
  - **Property 1: Bug Condition** - 1b Transition entropy counts only within-organism action pairs
  - **CRITICAL**: this test MUST FAIL on unfixed code — the failure is what confirms the bug
  - **DO NOT fix the test or the code in this task**
  - **Scoped PBT approach**: Bug 2 is deterministic, so the property is scoped to the concrete
    counterexample measured in design.md
  - File: `tests/test_diversity.py`, new test
    `test_transition_entropy_respects_organism_boundaries`
  - Traces `t1 = [(0, 0, 0, 0.0, 0.0, 0.0), (0, 0, 0, 0.0, 0.0, 0.0)]` (actions `[0, 0]`) and
    `t2 = [(1, 5, 5, 0.0, 0.0, 0.0), (1, 5, 5, 0.0, 0.0, 0.0)]` (actions `[1, 1]`); each organism is
    perfectly predictable, so the only honest value is exactly `0.0`
  - Assert `per_genome_metrics([t1, t2], window=0, area=100)["transition_entropy"] == 0.0`
  - Companion assertion for 2.8, in the same test: for one trace with `>= 2` entries,
    `per_genome_metrics([t], ...)["transition_entropy"] == transition_entropy_rate([a for a, *_ in t])`
    using `==`, not `approx`
  - **EXPECTED PRE-FIX OUTCOME (measured, must be observed and recorded in the test docstring)**:
    the pooled assertion fails with `0.6666666666666666` instead of `0.0` — the flattened stream
    `[0, 0, 1, 1]` contributes a fabricated `0 -> 1`, and row 0 splits 1/1 (entropy 1.0, weight
    2/3). The companion 2.8 assertion **passes** pre-fix and must keep passing post-fix
  - Confirm the root cause: `per_genome_metrics` flattens all organisms' actions before measuring,
    and `transition_entropy_rate` counts `zip(actions, actions[1:])`. If the observed value differs
    from `0.6666666666666666`, stop and re-hypothesize
  - Mark complete when the test is written, run, and the documented failure is recorded
  - Verify: `python3 -m pytest -q -o addopts='' tests/test_diversity.py::test_transition_entropy_respects_organism_boundaries`
  - _Requirements: 1.5, 1.6, 2.6, 2.7, 2.8, 2.13_

- [ ] 5. Fix Bug 2 — pooled, boundary-respecting transition counts

  - [ ] 5.1 Extract the shared core and make `transition_entropy_rate` delegate to it
    - File: `diversity/metrics.py`
    - Add private `_transition_counts(sequences, n_actions) -> Tuple[List[List[int]], int]`
      (matrix plus total within-trace transitions) and
      `_conditional_entropy_rate(counts, total_transitions) -> float`
    - **Bitwise-preservation constraint (Property 2a)**: `_conditional_entropy_rate` must keep the
      current loop structure exactly — iterate rows in index order `0..n_actions-1`, skip
      `total == 0` rows, accumulate `row_entropy` iterating the row in index order, then
      `rate += (row_total / total_transitions) * row_entropy`. Denominator is
      `sum(len(s) - 1 for s in sequences if len(s) >= 2)`, which reduces to `len(actions) - 1` for a
      single sequence. `len(actions) < 2` must still short-circuit to `0.0` (`total_transitions == 0`)
    - Rewrite `transition_entropy_rate` as a one-line delegation so the single-sequence and pooled
      paths cannot drift; its signature is unchanged
    - Use fixed-size lists indexed by action id, not dicts, so accumulation order is index-fixed
    - **This sub-task must not change any reported value**: the existing single-sequence tests,
      including the exact `0.9649839288802097`, must pass untouched
    - _Preservation: Property 2a; Preservation Requirement 3.1_
    - Verify: `python3 -m pytest -q -o addopts='' tests/test_diversity.py`
    - _Requirements: 2.7, 2.8, 3.1_

  - [ ] 5.2 Add the public `pooled_transition_entropy_rate`
    - File: `diversity/metrics.py`
    - `pooled_transition_entropy_rate(sequences, n_actions=ACTION_SIZE) -> float` built on the two
      helpers; add it to `__all__`. Purely additive — nothing removed or renamed
    - Verify: `python3 -m pytest -q -o addopts='' tests/test_diversity.py`
    - _Requirements: 2.7_

  - [ ] 5.3 Feed `per_genome_metrics` per-trace action sequences
    - File: `diversity/metrics.py`, `per_genome_metrics` per-trace loop only
    - Collect `action_sequences: List[List[int]]`, one list per clipped trace, and compute
      `transition_entropy` via `pooled_transition_entropy_rate(action_sequences)`
    - Keep the flat `actions` accumulation for `action_entropy`, plus `points` and the density
      counters, exactly as they are and in the same order; returned dict keys unchanged
    - _Bug_Condition: isBugCondition_2(X) — two or more organisms contribute at least one action after clipping_
    - _Expected_Behavior: Property 1b — conditional entropy of counts pooled over within-trace consecutive pairs only, normalized by total within-trace transitions_
    - _Preservation: Preservation Requirements 3.2, 3.3, 3.5_
    - _Requirements: 2.6, 2.7_

  - [ ] 5.4 Verify the Bug 2 exploration test now passes
    - **Property 1: Bug Condition** - 1b Transition entropy counts only within-organism action pairs
    - **IMPORTANT**: re-run the SAME test from task 4 — do NOT write a new test
    - **EXPECTED OUTCOME**: PASSES — pooled value is now exactly `0.0`, and the 2.8 companion
      exact-equality assertion still holds
    - Verify: `python3 -m pytest -q -o addopts='' tests/test_diversity.py::test_transition_entropy_respects_organism_boundaries`
    - _Requirements: 2.6, 2.7, 2.8, 2.13_

  - [ ] 5.5 Verify no regressions from the Bug 2 fix
    - **Property 2: Preservation** - Single-sequence entropy is bitwise unchanged
    - **IMPORTANT**: re-run the baseline from task 1 — do NOT edit any existing assertion
    - **EXPECTED OUTCOME**: `test_transition_entropy_rate_deterministic` (including
      `0.9649839288802097`), `test_transition_entropy_rate_alternating_pairs`,
      `test_action_entropy_*`, `test_spatial_coverage_fraction`, `test_encounter_rate_mean_density`,
      `test_population_diversity_zero_for_identical_genomes`,
      `test_empty_genome_baseline_zero_entropy_and_diversity` all pass untouched
    - Verify: `python3 -m pytest -q -o addopts=''`
    - _Requirements: 3.1, 3.2, 3.3, 3.5, 3.11, 2.14_

## Bug 1 — benchmark evaluation state

- [ ] 6. Write the Bug 1 authoritative exploration test and prove it fails on unfixed code
  - **Property 1: Bug Condition** - 1a An unevaluated offspring cannot declare success
  - **AUTHORITATIVE regression for Bug 1** (bugfix.md 2.13, design.md Test 1): this is the
    value-observable discriminator that carries the burden of proving Bug 1, which is why it comes
    first in this section
  - **CRITICAL**: this test MUST FAIL on unfixed code — the failure is what confirms the bug
  - **DO NOT fix the test or the code in this task**
  - File: `tests/test_benchmarks.py`, new test `test_unevaluated_offspring_cannot_declare_success`
  - Build a synthetic `Problem` with the same `input_ids` / `output_ids` / `cases` as `AND`,
    `fitness_fn = lambda g, gen: -1.0` and `success_fn = lambda g: g.fitness >= 0.0`. Negative
    evaluated fitness makes every crossover child (constructed fresh, `fitness == 0.0`) outrank the
    real champion, so the post-reproduction argmax is guaranteed to be an unevaluated offspring
  - Run `run_trial(probe, seed=0, population_size=6, max_generations=3)` and assert
    `result.solved is False`, `result.generations_to_solve is None`, `len(result.history) == 3`,
    `result.total_generations == 3`, and every `row["solved"] is False`
  - **EXPECTED PRE-FIX OUTCOME (measured; seeds 0/1/2/5 all identical, must be recorded in the test
    docstring)**: `solved=True`, `generations_to_solve=1`, `len(history) == 1`, and the single row
    reads `{"best_fitness": -1.0, "solved": True}` — success declared against `fitness >= 0.0` on a
    row whose recorded fitness is `-1.0`, with the remaining generations discarded. The first
    assertion (`solved is False`) fails
  - **Depends on no reference identity**: the contradiction between `solved=True` and
    `best_fitness=-1.0` under the criterion `fitness >= 0.0` is fully observable in values, which is
    why this test survives the copying accessor of Decision 1a unchanged and needs no identity
    assertion anywhere
  - The pre-fix failure needs at least one crossover child in generation 1, which is deterministic
    per seed (population 6, one species, `crossover_rate=0.75`); seed 0 is confirmed. If the seed is
    changed, re-confirm the pre-fix failure and record the observation in the docstring
  - **Scope guard**: this test uses negative fitness only to expose Bug 1; it must not turn into a
    fix for speciation's negative-fitness offspring allocation (explicitly out of scope)
  - Mark complete when the test is written, run, and the documented failure is recorded
  - Verify: `python3 -m pytest -q -o addopts='' tests/test_benchmarks.py::test_unevaluated_offspring_cannot_declare_success`
  - _Requirements: 1.2, 1.4, 2.2, 2.13_

- [ ] 7. Write the Bug 1 semantic consistency guard (non-discriminating)
  - **Property 1: Bug Condition** - 1a A history row describes the genome evaluated in that generation
  - **NOT A DISCRIMINATOR — this test PASSES BOTH PRE- AND POST-FIX on `AND`, and that is
    expected.** The defect is latent there: `_reproduce_species` prepends a structural `elite.copy()`
    that preserves fitness, and `max` returns the first maximal element, so the wrong object carries
    the right values — measured `nodes 3`, `conns 0`, `fitness 0.8` coincide in all four generations.
    A green pre-fix run MUST NOT be read as evidence the bug is absent; task 6 is the discriminator
  - Its job is to lock the semantic contract going forward, so a future change that re-splits the
    field sourcing gets caught
  - File: `tests/test_benchmarks.py`, new test `test_history_row_describes_the_evaluated_genome`
  - Wrap `AND` in a `benchmarks.problems.Problem(...)` probe: a `fitness_fn` that delegates to
    `AND.fitness_fn` and records `(genome, fitness)` per generation into an oracle dict keyed by the
    `generation` argument, and a `success_fn` that delegates to `AND.success_fn`. The `success_fn`
    wrapper MAY record the genome it was handed, but **only** to assert value agreement with the
    oracle champion — never identity (post-fix it receives a defensive snapshot per Decision 1a)
  - Run `run_trial(probe, seed=0, population_size=6, max_generations=4)`. The oracle champion for
    row `k` is `max(evaluated[k], key=fitness)`; the tracking `fitness_fn` is called in population
    order and `_evaluate` takes `max(self.population, ...)` in the same order, so `max` breaks ties
    on the same element and the oracle champion is the genome the engine evaluated
  - Oracle keys are `0..3` (`_evaluate` receives `self.generation`) while history `generation`
    values are `1..4` — align positionally via `sorted(evaluated)`
  - Assert per generation, **values only — no identity assertion appears anywhere in this test**:
    `row["best_fitness"] == champion.fitness`;
    `row["best_node_count"] == len(champion.nodes)`;
    `row["best_connection_count"] == len(champion.connections)`;
    `row["solved"] == AND.success_fn(champion)`
  - **EXPECTED OUTCOME**: PASSES on the UNFIXED code and still passes after the fix. Record in the
    test docstring that it is a semantic consistency guard, not a bug discriminator, and that the
    pre-fix pass is expected rather than reassuring
  - Mark complete when the test is written, run, and the expected pre-fix pass is recorded
  - Verify: `python3 -m pytest -q -o addopts='' tests/test_benchmarks.py::test_history_row_describes_the_evaluated_genome`
  - _Requirements: 1.1, 1.3, 1.4, 2.1, 2.13_

- [ ] 8. Fix Bug 1 — source every champion field from the evaluated genome

  - [ ] 8.1 Add the `evaluated_best_genome` read accessor
    - File: `neat/population.py`
    - Add a public read-only property `evaluated_best_genome -> Optional[Genome]` next to the
      existing `statistics` property, returning `self._evaluated_best_genome.copy()` — a **defensive
      snapshot, not the cached object** (Decision 1a) — with an explicit `None` short-circuit when
      the cache is empty, so the return type is genuinely `Optional[Genome]`
    - Rationale to keep in mind while writing it: `Genome` is mutable (`NodeGene` / `ConnectionGene`
      are mutable dataclasses, `nodes` a dict, `connections` a list), so exposing the cached instance
      would let any caller corrupt engine state that later generations, `best_genome` and
      `statistics` depend on; `Genome.copy()` preserves everything `run_trial` needs (`fitness`, the
      node and connection genes, innovation numbers, biases, weights, enabled flags); and it matches
      the `best_genome` precedent, so the two adjacent champion accessors do not have opposite
      aliasing semantics
    - Consequence to respect everywhere downstream: every read returns a fresh independent snapshot,
      so identity comparisons — between two reads, or between a read and any engine-internal object —
      are meaningless **by design** and must not appear in tests or consumer code; assert on values
    - `None` before the first `_evaluate()` has run, and when the population is empty (Decision 1b)
    - Add no import: `Genome` and `Optional` are already imported and `copy()` is an existing
      `Genome` method, so `neat/` stays free of any dependency on `world/`, `benchmarks/` or
      `experiments/`
    - Do not touch `_next_generation`, `_evaluate`, `_track_best_and_stats` or any private attribute
    - **Text constraint**: `tests/test_population.py::test_engine_has_no_environment_dependencies`
      asserts the substring `"main"` is absent from this module's source — that forbids `main`,
      `remain`, `remaining`, `domain` and `maintain` in anything added here
    - _Bug_Condition: isBugCondition_1(X) — a generation whose post-reproduction argmax differs from the evaluated champion_
    - _Expected_Behavior: Property 1a — the row's champion fields all derive from one evaluated genome_
    - _Preservation: Preservation Requirements 3.7, 3.12, 3.13, 3.15_
    - Verify: `python3 -m pytest -q -o addopts='' tests/test_population.py`
    - _Requirements: 2.4, 3.15_

  - [ ] 8.2 Source `run_trial`'s champion fields from the accessor
    - File: `benchmarks/run.py`, `run_trial` loop body only
    - Replace `gen_best = max(population.population, key=lambda g: g.fitness)` with a read of
      `population.evaluated_best_genome`; raise `RuntimeError` with an explicit message if it is
      `None` — no silent fallback to `population.population`, which would reintroduce the defect on
      the rarest path
    - Source `best_fitness` (from `evaluated.fitness`), `best_node_count`,
      `best_connection_count` and `problem.success_fn(...)` from that one genome; keep
      `mean_fitness` and `species_count` from `population.statistics[-1]`
    - Keep the row key set, key order and `generation` numbering from 1 exactly as they are
    - Do not touch `TrialResult`, `run_benchmark`, `main` or the CLI; `final_nodes` /
      `final_connections` become correct by construction
    - Leave the pre-existing `max_generations=0` `UnboundLocalError` path alone — the `None` guard
      sits inside the loop body, so the two are not entangled
    - _Bug_Condition: isBugCondition_1(X)_
    - _Expected_Behavior: Property 1a_
    - _Preservation: Preservation Requirements 3.8, 3.9, 3.10_
    - _Requirements: 2.1, 2.2, 2.3_

  - [ ] 8.3 Verify both Bug 1 tests now pass
    - **Property 1: Bug Condition** - 1a A history row describes the genome evaluated in that generation
    - **IMPORTANT**: re-run the SAME tests — the authoritative discriminator from task 6
      (`test_unevaluated_offspring_cannot_declare_success`) and the semantic guard from task 7
      (`test_history_row_describes_the_evaluated_genome`) — do NOT write new tests
    - **EXPECTED OUTCOME**: both PASS. Task 6 flips from failing to passing: the synthetic
      negative-fitness trial now reports `solved=False`, `generations_to_solve=None`, 3 rows — that
      flip is the proof Bug 1 is fixed. Task 7 was already passing and simply keeps passing, with
      every row's values still agreeing with the evaluated champion; it proves nothing about the fix
      on its own
    - Verify: `python3 -m pytest -q -o addopts='' tests/test_benchmarks.py::test_unevaluated_offspring_cannot_declare_success tests/test_benchmarks.py::test_history_row_describes_the_evaluated_genome`
    - _Requirements: 2.1, 2.2, 2.3, 2.13_

  - [ ] 8.4 Add the accessor consistency **and isolation** preservation test
    - **Property 2: Preservation** - The accessor agrees with the recorded statistics and stays isolated
    - File: `tests/test_population.py`, new test
      `test_evaluated_best_genome_matches_recorded_statistics`
    - *Consistency (Decision 1, 1b)*: assert `pop.evaluated_best_genome is None` before the first
      `run`; then after `run(1)` and again after a second `run(1)`, assert
      `pop.evaluated_best_genome.fitness == pop.statistics[-1]["best_fitness"]`
    - *Isolation (Decision 1a, bugfix.md 3.15)*: assert two consecutive reads return **distinct
      objects** (`pop.evaluated_best_genome is not pop.evaluated_best_genome`), so no caller can be
      holding the engine's instance; then mutate a returned snapshot — set its `fitness` to a
      sentinel and mutate one gene field (e.g. a `NodeGene.bias`, or a `ConnectionGene.weight` /
      `enabled`) — and assert the mutation is invisible everywhere: a subsequent read still reports
      the original `fitness` and the original gene values, `pop.statistics[-1]["best_fitness"]` is
      unchanged, and `pop.best_genome` is unchanged. This mirrors
      `test_best_genome_tracked_and_isolated`
    - The old "returned object is **not** a member of `pop.population`" assertion is now trivially
      true (a fresh copy can never be a member), so isolation-by-mutation is the assertion that
      actually constrains the implementation; the membership check MAY be kept as a cheap lifecycle
      assertion documenting 2.5
    - This pins both halves of Decision 1 instead of trusting them
    - **EXPECTED OUTCOME**: PASSES on the fixed code
    - Verify: `python3 -m pytest -q -o addopts='' tests/test_population.py::test_evaluated_best_genome_matches_recorded_statistics`
    - _Requirements: 2.4, 2.5, 3.7, 3.15_

  - [ ] 8.5 Verify no regressions from the Bug 1 fix
    - **Property 2: Preservation** - Engine lifecycle, benchmark determinism and row shape
    - **IMPORTANT**: re-run the baseline from task 1 — do NOT edit any existing assertion
    - **EXPECTED OUTCOME**: all of `tests/test_population.py` passes untouched (notably
      `test_statistics_use_evaluated_fitness_not_carried`, `test_best_genome_tracked_and_isolated`,
      `test_seeded_run_is_deterministic`, `test_engine_has_no_environment_dependencies`), and
      `test_run_trial_is_deterministic` and `test_run_trial_history_is_well_formed` pass untouched
      (measured post-fix for `AND` at population 4, seeds 0 and 5: no early stop, `nodes 3`,
      `conns 0`, `fitness 0.8` in each of generations 1-3)
    - Verify: `python3 -m pytest -q -o addopts=''`
    - _Requirements: 3.7, 3.8, 3.9, 3.10, 3.12, 3.13, 2.14_

## Unit-test additions

- [ ] 9. Unit-test `pooled_transition_entropy_rate`
  - File: `tests/test_diversity.py`
  - Cases: empty input (`[]`); one sequence shorter than 2; several sequences all shorter than 2
    (denominator 0 -> `0.0`); one long sequence equals `transition_entropy_rate` over the same
    actions using `==`, not `approx` (this is the 2.8 equivalence at the function level); mixed
    lengths, weighted by the number of transitions each sequence actually contributed
  - Verify: `python3 -m pytest -q -o addopts='' tests/test_diversity.py -k pooled_transition_entropy`
  - _Requirements: 2.7, 2.8, 3.1_

- [ ] 10. Unit-test the aligned-sample count in `per_genome_metrics`
  - File: `tests/test_diversity.py`
  - Assert the number of alignment samples equals `sum(max(0, len(clip(t)) - 1))` across traces —
    exactly one sample dropped per trace (each trace's first recorded action, whose displacement is
    unrecoverable), and no sample mixing indices from two traces
  - Also assert the pooled transition total equals
    `sum(len(s) - 1 for s in sequences if len(s) >= 2)`, which is exactly the count of legitimate
    transitions and therefore excludes fabricated cross-boundary pairs
  - Verify: `python3 -m pytest -q -o addopts='' tests/test_diversity.py`
  - _Requirements: 2.6, 2.9, 2.10_

## Documentation and surface

- [ ] 11. Re-export `pooled_transition_entropy_rate` from the package
  - File: `diversity/__init__.py`
  - Add the name alongside the existing exports; additive only, nothing removed or renamed
  - Verify: `python3 -m pytest -q -o addopts='' tests/test_diversity.py`
  - _Requirements: 2.7_

- [ ] 12. Document the evaluation lifecycle in `neat/population.py`
  - File: `neat/population.py` — documentation only, no behavior change
  - Property docstring: it is the genome behind the most recently recorded `best_fitness`; `None`
    before the first generation; **every read returns a fresh defensive copy**, so callers may retain
    and mutate it freely and those mutations cannot reach engine state (mirroring `best_genome`);
    identity comparisons — against engine internals or between two reads — are therefore meaningless;
    and it is **not** a member of `self.population` after `run()` returns
  - Module docstring: extend the "One generation" block — after `run(1)` returns, `self.population`
    holds unevaluated offspring carrying copied/stale fitness, so evaluated per-generation values
    come from `statistics[-1]` or `evaluated_best_genome`
  - `run()` docstring: one sentence pointing at the same thing
  - **Text constraint**: `test_engine_has_no_environment_dependencies` asserts the substring
    `"main"` is absent from this module — avoid `main`, `remain`, `remaining`, `domain`, `maintain`
  - Verify: `python3 -m pytest -q -o addopts='' tests/test_population.py::test_engine_has_no_environment_dependencies`
  - _Requirements: 2.4, 2.5_

- [ ] 13. Document row provenance in `benchmarks/run.py`
  - File: `benchmarks/run.py` module docstring — documentation only
  - State that per-generation rows describe the genome evaluated in that generation, read via
    `Population.evaluated_best_genome`
  - Verify: `python3 -m pytest -q -o addopts='' tests/test_benchmarks.py`
  - _Requirements: 2.5_

- [ ] 14. Document the metric definitions in `diversity/metrics.py`
  - File: `diversity/metrics.py` docstrings only — no behavior change
  - Module docstring: the trace tuple's per-field temporal reference (`action`, `food_dx`,
    `food_dy`, `density` are pre-action; `x`, `y` are post-action)
  - `transition_entropy_rate` / `pooled_transition_entropy_rate`: the pooled-counts combination
    rule and its denominator `sum(len(s) - 1 for s in sequences if len(s) >= 2)`, plus the rejected
    alternative (averaging per-organism rates) and why
  - `per_genome_metrics`: which metrics pool across organisms (`action_entropy`,
    `spatial_coverage`, `encounter_rate`) and which respect trace boundaries
    (`transition_entropy`, `food_alignment`)
  - `food_alignment_cosine` and `per_genome_metrics`: the alignment definition
    (`(pos_i - pos_{i-1}, (fx_i, fy_i))` for `i >= 1`, within one trace) and the accepted tradeoff —
    each trace's first recorded action is dropped as unrecoverable, `food_dirs[0]` is now unused,
    and the final observed food direction is now used
  - Verify: `python3 -m pytest -q -o addopts='' tests/test_diversity.py`
  - _Requirements: 2.7, 2.10, 2.12_

- [ ] 15. Document the trace tuple's temporal reference in `world/organism.py`
  - File: `world/organism.py` — **documentation only, no code change**; the tuple shape stays
    `(action, x, y, food_dx, food_dy, density)`
  - Update the `self.trace` comment and add a line to `act`'s docstring: an entry combines the
    pre-action observation (`action` chosen from it, `food_dx`, `food_dy`, `density`) with the
    post-action position (`x`, `y`), so a consumer pairing a displacement with an observation must
    use the observation stored at the *later* index
  - Verify: `python3 -m pytest -q -o addopts='' tests/test_world.py tests/test_diversity.py`
  - _Requirements: 2.11, 2.12_

- [ ] 16. Update the README
  - File: `README.md`
  - Add to "Design notes you should know": the evaluation lifecycle — after `run(1)`,
    `pop.population` holds unevaluated offspring; read evaluated per-generation values from
    `pop.statistics[-1]` or `pop.evaluated_best_genome`
  - Add to the same section: the measurement definitions — `transition_entropy` pools
    boundary-respecting transition counts across the organisms of a genome; `food_alignment` pairs
    each displacement with the food direction observed immediately before the action that caused it,
    excluding each trace's first recorded action; pre-fix result JSON is not comparable to post-fix
    for `transition_entropy`, `food_alignment` and `behavioral_diversity`
  - Update the stated test count in all three places where `188` appears (the intro line, the
    install snippet comment, and the `tests/` bullet under Layout) to the new total from the final
    full-suite run in task 17
  - Leave the existing `food_alignment`-is-confounded note as it is
  - Verify: `python3 -m pytest -q -o addopts=''` (confirm the count written matches the run)
  - _Requirements: 2.5, 2.10_

## Final verification

- [ ] 17. Checkpoint — full verification and reporting
  - Run the new tests first:
    `python3 -m pytest -q -o addopts='' tests/test_diversity.py::test_food_alignment_pairs_displacement_with_preceding_observation tests/test_diversity.py::test_transition_entropy_respects_organism_boundaries tests/test_benchmarks.py::test_unevaluated_offspring_cannot_declare_success tests/test_benchmarks.py::test_history_row_describes_the_evaluated_genome tests/test_population.py::test_evaluated_best_genome_matches_recorded_statistics`
  - Then the full suite: `python3 -m pytest -q -o addopts=''` — expect the pre-fix 188 plus the new
    tests, zero failures
  - Confirm every pre-existing test passes with **zero assertion edits** (2.14); if any existing
    assertion had to change, stop and ask before proceeding
  - Confirm `pyproject.toml` is untouched (3.14)
  - Confirm `Population.evaluated_best_genome` returns a **snapshot**: two consecutive reads are
    distinct objects, and mutating a returned genome leaves `statistics`, `best_genome` and any
    later read unchanged (2.4, 3.15)
  - Confirm **no test anywhere asserts reference identity** between a value obtained from the public
    accessor (or handed to `success_fn`) and any engine-internal object — such assertions are
    meaningless under Decision 1a and must not have crept in
  - Review `git diff` and confirm no unrelated change slipped in: the only files touched should be
    `neat/population.py`, `benchmarks/run.py`, `diversity/metrics.py`, `diversity/__init__.py`,
    `world/organism.py` (comments/docstrings only), `README.md`, `tests/test_diversity.py`,
    `tests/test_benchmarks.py`, `tests/test_population.py`. Confirm nothing from the out-of-scope
    list was changed
  - **Do NOT commit and do NOT push**
  - Report: which files changed and why (one line each); and per bug, the previous failure mode, the
    measured pre-fix value, and why the new test proves it fixed —
    Bug 1 (authoritative proof is the synthetic negative-fitness trial): pre-fix it reported
    `solved=True`, `generations_to_solve=1`, one history row with `best_fitness=-1.0` — success
    declared against `fitness >= 0.0` on a row recording `-1.0`; post-fix it runs all 3 generations
    unsolved with `generations_to_solve=None`. The `AND` test
    (`test_history_row_describes_the_evaluated_genome`) is a **semantic consistency guard** that
    passes both before and after the fix and asserts values only — it is NOT an identity check and
    is not evidence of the fix;
    Bug 2: `0.6666666666666666` from a fabricated `0 -> 1` across the organism boundary, now exactly
    `0.0` with single-sequence values bitwise unchanged;
    Bug 3: `+1.0` for a step directly away from food (4a) and `0.5` from cross-organism leakage
    (4b), now `-1.0` and `1.0`
  - Ask the user if any question arises rather than guessing
  - _Requirements: 2.13, 2.14, 3.14_
