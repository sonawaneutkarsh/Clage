# Evaluation & Measurement Correctness Bugfix Design

## Overview

Three independent defects make reported numbers describe something other than what they claim:

1. `benchmarks/run.py::run_trial` builds each history row from two different generations — fitness from the evaluated generation, topology and `solved` from the post-reproduction population.
2. `diversity/metrics.py::per_genome_metrics` flattens the action streams of all organisms sharing a genome before computing `transition_entropy`, fabricating transitions across organism boundaries.
3. The same function pairs movement displacements with food directions one index apart, and lets that offset accumulate across organisms.

The fix strategy is minimal and local:

- **Bug 1** — add one additive, documented read accessor to `neat.Population` returning a defensive snapshot (`Genome.copy()`) of the evaluated champion the engine already caches, and have `run_trial` source all four per-generation champion fields from it. No engine behavior changes; no new dependency enters `neat/`.
- **Bug 2** — factor the transition-count matrix out of `transition_entropy_rate` so a single code path serves both one sequence and many, and pool boundary-respecting counts across the organisms of a genome.
- **Bug 3** — build the `(displacement, food direction)` pairs per trace with the correct index alignment. The world trace tuple shape is **not** changed.

All three fixes are pure data-flow corrections. No RNG is added, no schema changes, and no public signature is removed.

Three decisions the requirements demanded be settled explicitly (each with its rejected alternative) are resolved in [Decisions](#decisions). Every pre-fix value quoted in this document was measured against the current working tree, not inferred.

## Glossary

- **Bug_Condition (C)** — the predicate identifying inputs that trigger a defect. There are three, `isBugCondition_1..3`, one per bug (see bugfix.md).
- **Property (P)** — the required behavior of the fixed code on inputs satisfying C.
- **Preservation** — behavior on `¬C` inputs that must be bit-identical after the fix.
- **F / F'** — the pipeline before / after the fix.
- **Evaluated champion** — the genome with the highest fitness *at the moment `_evaluate()` finished* for a generation. Cached by the engine as `Population._evaluated_best_genome`; it is the genome behind `statistics[-1]["best_fitness"]`. Exposed publicly as a **defensive snapshot** (`Genome.copy()`) via `Population.evaluated_best_genome`, so consumers hold a value-equivalent independent object, never the engine's instance.
- **Post-reproduction population** — `Population.population` after `run(1)` returns: freshly built offspring plus *copies* of the per-species elites. Copies preserve `fitness` (`Genome.copy()` copies `fitness` and `adjusted_fitness`), so this list carries fitness values that no longer correspond to its members' structure.
- **Trace** — `List[Tuple[action, x, y, food_dx, food_dy, density]]`, one entry per tick, appended in `Organism.act`. **Mixed temporal reference:** `action`, `food_dx`, `food_dy`, `density` are all derived from the observation taken *before* the action; `x`, `y` are the position *after* the action was applied.
- **Boundary-respecting transition** — an ordered action pair `(a_i, a_{i+1})` drawn from within one organism's clipped trace.
- **Aligned alignment sample** — a pair `(pos_i - pos_{i-1}, (fx_i, fy_i))` from one trace with `i >= 1`.

## Bug Details

### Bug Condition 1 — benchmark evaluation state

`run_trial` reads `gen_best = max(population.population, key=fitness)` *after* `population.run(1)`, then uses it for `best_node_count`, `best_connection_count` and `success_fn`, while `best_fitness` / `mean_fitness` / `species_count` come from `population.statistics[-1]`. The two sources describe different generations.

```
FUNCTION isBugCondition_1(X)
  INPUT: X = (problem, seed, population_size, max_generations)
  OUTPUT: boolean

  RETURN EXISTS generation g IN trial(X) SUCH THAT
           argmax_fitness(population_after_reproduction(g)) <> evaluated_best(g)
END FUNCTION
```

**This bug is latent, not loud — and that shapes the tests.** Measured on `AND`, seed 0, population 6, 4 generations: the object passed to `success_fn` is *never* the evaluated champion (identity check `False` in all 4 generations), yet `best_node_count` (3), `best_connection_count` (0) and `best_fitness` (0.8) coincidentally match the evaluated champion in all 4. (That identity check is a pre-fix *measurement*, not a post-fix expectation: post-fix `success_fn` receives a defensive snapshot per Decision 1a, so it is a distinct object by design and only its values are asserted.) The reason is an accidental invariant: with `elitism >= 1`, `_reproduce_species` prepends `elite.copy()` — a structural clone carrying the same fitness — ahead of that species' offspring, and `max` returns the *first* maximal element, so the tie resolves to the structural clone. Correctness therefore currently rests on `elitism >= 1`, on the champion's species receiving a non-zero offspring budget, on `max`'s tie order, and on fitness never being negative. Break any one and the reported topology silently describes a different genome.

Two of those are easy to break, and both are legitimate uses of the public API:

- **Negative fitness.** Crossover children are constructed fresh, so they carry `fitness == 0.0`. If evaluated fitness is negative, every crossover child outranks the real champion.
- **Zero offspring budget** for the champion's species (`_reproduce_species` is skipped entirely when `budget <= 0`), which removes the elite clone that was masking the defect.

**Measured counterexample (negative fitness).** A `Problem` with `fitness_fn -> -1.0` and `success_fn -> g.fitness >= 0.0`, population 6, `max_generations=3`, seeds 0/1/2/5 — all four seeds produce:

```
solved=True  generations_to_solve=1  history=[{best_fitness: -1.0, solved: True}]
```

A trial declared solved on a genome whose recorded fitness in that same row is `-1.0`, evaluated against the success criterion `fitness >= 0.0`. The trial then early-stops, discarding the remaining generations.

### Bug Condition 2 — transition entropy across organism boundaries

```
FUNCTION isBugCondition_2(X)
  INPUT: X = list of traces sharing one genome
  OUTPUT: boolean

  RETURN count(t IN X WHERE length(clip(t)) >= 1) >= 2
END FUNCTION
```

`per_genome_metrics` appends every organism's actions into one `actions` list and calls `transition_entropy_rate(actions)` once. That function counts `zip(actions, actions[1:])`, so the boundary pair `(last_action(t_i), first_action(t_{i+1}))` is counted as if observed.

**Measured counterexample.** Two traces, actions `[0, 0]` and `[1, 1]` (each organism perfectly predictable, so the honest answer is exactly `0.0`):

| | value |
|---|---|
| pre-fix `transition_entropy` | `0.6666666666666666` |
| correct (pooled, boundary-respecting) | `0.0` |

The flattened stream `[0, 0, 1, 1]` contains a fabricated `0 -> 1`; row 0 splits 1/1 (entropy 1.0, weight 2/3) giving 2/3 of a bit out of nothing.

### Bug Condition 3 — food-alignment temporal misalignment

```
FUNCTION isBugCondition_3(X)
  INPUT: X = list of traces sharing one genome
  OUTPUT: boolean

  RETURN EXISTS t IN X SUCH THAT length(clip(t)) >= 2
END FUNCTION
```

`food_dirs` receives an entry for every index; `deltas` only for `index > 0`. `food_alignment_cosine` then zips them positionally, so `deltas[k]` (the displacement caused by the action at index `k+1`) is paired with `food_dirs[k]` (observed before the action at index `k`). Because `food_dirs` grows one faster than `deltas` per trace, the offset **accumulates**: with two traces the second trace's displacements are paired with the *first* trace's food directions.

**Measured counterexamples.**

| input | pre-fix | correct |
|---|---|---|
| one trace: `(0,0,0,+1,0,0)`, `(0,1,0,-1,0,0)` | `+1.0` | `-1.0` |
| two traces: `[(0,0,0,1,0,0),(0,1,0,1,0,0)]`, `[(0,5,5,0,1,0),(0,5,6,0,1,0)]` | `0.5` | `1.0` |

The first is a full sign inversion: the organism moved east while food was reported west, and the metric scored it as perfect food-seeking. The second shows the cross-organism leak — organism B's northward step is scored against organism A's eastward food direction, yielding `0.0` for a sample that is `+1.0`.

## Expected Behavior

### Preservation Requirements

**Unchanged behaviors** (bugfix.md 3.1-3.15):

- `transition_entropy_rate(actions)` keeps its signature and returns bit-identical values for any single sequence, including `< 2` actions -> `0.0` (3.1).
- `action_entropy` keeps pooling the marginal action distribution across organisms (3.2).
- `spatial_coverage` keeps pooling positions across organisms (3.3).
- `food_alignment_cosine` keeps its signature, its zero-magnitude skipping, and its `0.0`-when-no-samples result (3.4).
- `encounter_rate` and the `per_genome_metrics` density path are untouched (3.5).
- `record_trace` / `behavior_window` clipping semantics are untouched (3.6).
- `Population._next_generation` keeps its exact phase order and statistics source (3.7).
- Benchmark trials stay deterministic per seed (3.8) and history rows keep exactly the key set `{generation, best_fitness, mean_fitness, species_count, best_node_count, best_connection_count, solved}` with `generation` counting from 1 (3.9).
- `OR`/`AND`/`XOR`/`SIN` definitions and the `benchmarks.run` CLI arguments are untouched (3.10).
- `experiments/run.py` keeps its closure-captured evaluated best and every `RECORD_FIELDS` entry (3.11).
- Seeded determinism holds end to end (3.12); package separation holds (3.13); `pyproject.toml` is untouched (3.14).
- The engine's cached evaluated champion stays isolated from callers: mutating the value returned by `evaluated_best_genome` changes neither `Population`'s internal state, nor any subsequent read of the accessor, nor `statistics`, nor `best_genome` — the same isolation guarantee `tests/test_population.py::test_best_genome_tracked_and_isolated` already asserts for `best_genome` (3.15).

**Scope.** Inputs satisfying none of `isBugCondition_1..3` must produce byte-identical output: single-organism genomes, single-entry traces, and generations whose post-reproduction argmax already coincides with the evaluated champion. Verified as unaffected:

- `Population` behavior — the fix to `neat/` is a read accessor plus docstrings.
- `experiments/run.py` — already reads the evaluated best from inside the evaluator closure; not touched.
- `action_entropy`, `spatial_coverage`, `encounter_rate` code paths in `per_genome_metrics` — the flat `actions`/`points` accumulation and the density counters are kept exactly as they are, in the same iteration order, so their floating-point results are bitwise unchanged.
- Existing benchmark tests — measured post-fix behavior for `AND`, population 4, seeds 0 and 5: the evaluated champion is `solved=False`, `nodes=3`, `conns=0`, `fitness=0.8` in each of generations 1-3, so `test_run_trial_history_is_well_formed` (which requires 3 rows, no early stop, `best_node_count >= 2`) and `test_run_trial_is_deterministic` stay green with no assertion edits.

## Hypothesized Root Cause

Confirmed by reading the implementation; the "hypothesis" is which design pressure produced each defect.

1. **Missing public accessor, not missing knowledge (Bug 1).** `neat/population.py` is already correct — `_evaluate()` deliberately caches best/mean *before* reproduction, with a comment saying exactly why. The defect is an encapsulation gap: the cache is private and `statistics` rows carry no genome reference, so the only genome a consumer can reach after `run(1)` is the wrong one. `run_trial` then reached for what was reachable.
2. **Aggregation before measurement (Bug 2).** `per_genome_metrics` was written around one accumulator per metric. That is correct for the order-insensitive metrics (`action_entropy`, `spatial_coverage`, `encounter_rate`) and wrong for the only order-sensitive one, because flattening destroys the boundaries that `transition_entropy_rate` needs.
3. **Two accumulators with different cardinality (Bug 3).** In the same loop, `food_dirs` appends on every index and `deltas` only on `index > 0`. Nothing enforces a shared index, and `zip` silently accepts the mismatch. The undocumented mixed temporal reference of the trace tuple (pre-action observation, post-action position) is what made the correct alignment non-obvious in the first place.

A single theme: values that must be derived from one source are derived from two, with nothing checking that the two agree.

## Correctness Properties

Property 1: Bug Condition - Every reported number describes what it claims

_For any_ input where a bug condition holds (`isBugCondition_1`, `_2` or `_3` returns true), the fixed pipeline SHALL report values derived from the correct source:

- **1a (Bug 1)** — _for any_ trial generation, `best_fitness`, `best_node_count`, `best_connection_count` and `solved` in that generation's history row SHALL all be derived from one genome value: a snapshot of the genome the engine evaluated in that generation, with `success_fn` invoked on that same snapshot. The requirement is **value agreement**, not reference identity — the accessor deliberately hands out a defensive copy (Decision 1a), so the snapshot is a distinct object that is equal in fitness, node set, connection set and gene contents to the engine's evaluated champion.
- **1b (Bug 2)** — _for any_ set of traces sharing a genome, `transition_entropy` SHALL equal the conditional entropy of the transition counts pooled over within-trace consecutive action pairs only, normalized by the total number of within-trace transitions; no pair `(last_action(t_i), first_action(t_j))` with `i != j` SHALL be counted.
- **1c (Bug 3)** — _for any_ set of traces sharing a genome, `food_alignment` SHALL equal the mean cosine over samples `(pos_i - pos_{i-1}, (fx_i, fy_i))` built within a single trace for `i >= 1`, with zero-magnitude samples skipped.

**Validates: Requirements 2.1, 2.2, 2.3, 2.6, 2.7, 2.8, 2.9, 2.10**

Property 2: Preservation - Non-triggering inputs are bit-identical

_For any_ input where no bug condition holds (`isBugCondition_1..3` all false), the fixed pipeline SHALL produce results identical to the original:

- **2a** — `transition_entropy_rate(actions)` for a single sequence returns its current value bitwise, and a genome with exactly one organism whose clipped trace has `>= 2` entries reports `transition_entropy` exactly equal to `transition_entropy_rate` over that one sequence.
- **2b** — `action_entropy`, `spatial_coverage`, `encounter_rate` and `food_alignment_cosine`'s zero-magnitude skipping return their current values.
- **2c** — `Population`'s generation lifecycle, statistics contents, and seeded determinism are unchanged; benchmark history key set and `generation` numbering are unchanged.
- **2d** — the public evaluated-champion accessor leaks no mutable engine state: mutating a returned snapshot leaves `Population`'s cached evaluated champion, every subsequent read of the accessor, `statistics` and `best_genome` unchanged.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 3.10, 3.11, 3.12, 3.13, 3.14, 3.15**

## Decisions

### Decision 1 — how benchmarks reach the evaluated genome

**Adopted.** Add a documented public read-only property to `neat.Population`:

```python
@property
def evaluated_best_genome(self) -> Optional[Genome]:
    ...
    if self._evaluated_best_genome is None:
        return None
    return self._evaluated_best_genome.copy()
```

`benchmarks/run.py::run_trial` uses that one genome for `best_fitness`, `best_node_count`, `best_connection_count` and `problem.success_fn(...)`, and keeps taking `mean_fitness` / `species_count` from `statistics[-1]`.

*Rationale.* Purely additive; `neat/` gains no import (2.4) — `Genome` and `Optional` are already imported, and `Genome.copy()` is an existing method. The returned genome is guaranteed consistent with `statistics[-1]["best_fitness"]` because both derive from the same `_evaluate()` cache — `_evaluated_best_fitness` is literally assigned `self._evaluated_best_genome.fitness`, nothing writes `.fitness` between then and the end of `run(1)`, and `Genome.copy()` carries `fitness` over to the snapshot. Genome objects stay out of `statistics` rows, which must remain JSON-friendly.

**1a — defensive snapshot, not a live reference.** The accessor returns `self._evaluated_best_genome.copy()`, typed `Optional[Genome]` and `None` when the cache is empty. Each read produces a fresh, independent genome (2.4, 3.15).

*Rationale.*

- **`Genome` is mutable.** `NodeGene` and `ConnectionGene` are mutable dataclasses, `nodes` is a mutable dict and `connections` a mutable list. Handing out the cached object lets any caller — accidentally or otherwise — corrupt engine state that later generations, `best_genome` and `statistics` depend on. A public accessor that exposes mutable internal state is a correctness hazard regardless of how well-behaved today's single consumer is.
- **`Genome.copy()` already provides exactly the guarantees `run_trial` needs.** It deep-copies genes while preserving node ids, node types, biases, weights, enabled flags, innovation numbers, `fitness` and `adjusted_fitness` — pinned by `tests/test_genome.py::test_copy_preserves_ids_bias_innovation_and_fitness` and `::test_copy_is_deep_no_alias`. `run_trial` reads only `fitness`, `len(nodes)` and `len(connections)`, and decodes a read-only `Network` via `success_fn`; a copy satisfies all four identically.
- **It matches the precedent already set in this class.** `best_genome` stores a copy, and `tests/test_population.py::test_best_genome_tracked_and_isolated` asserts that isolation guarantee. Two adjacent champion accessors with opposite aliasing semantics would be a trap.
- **Cost is negligible.** One genome deep copy per accessor read, and `run_trial` reads once per generation. At benchmark scale this is not measurable against a full population evaluation.

*Accessor contract to document:* every read returns a fresh independent snapshot. Callers may retain and mutate it freely. Consequently, identity comparisons — between two reads, or between a read and any engine-internal object — are **meaningless by design** and must not appear in tests or consumer code; assert on values instead.

*Rejected alternative — returning the live cached reference.* It is cheaper (no copy), and it would let a test assert reference identity between the accessor's result and the engine's evaluated object, which on `AND` — where every value coincides pre-fix — would give that particular test a way to discriminate. Rejected: exposing mutable internal state through a public API is a correctness hazard, and that outweighs a test-expressiveness convenience. The convenience costs nothing real here, because Bug 1's authoritative regression test is the value-observable negative-fitness test (bugfix.md 2.13), which discriminates pre-fix without referring to identity at all; the real-problem `AND` test is retained as a value-only semantic guard. See [Exploratory Bug Condition Checking](#exploratory-bug-condition-checking).

**1b — `None` before the first generation; `run_trial` refuses to guess.** The property returns `Optional[Genome]`, `None` until the first `_evaluate()` has run (and again only if the population is empty). In `run_trial` the accessor is read strictly after `population.run(1)`, so it is non-`None` for any non-empty population. If it is `None`, `run_trial` raises `RuntimeError` with an explicit message rather than falling back to `population.population`. A silent fallback would reintroduce exactly the defect being fixed, on the rarest path, where nobody would notice.

*Rejected alternative.* A benchmarks-side tracking `fitness_fn` wrapper that recomputes the per-generation best. It needs no change to `neat/`, but it forces every consumer to re-implement engine bookkeeping (`experiments/run.py` already does this independently — the duplication the requirements note but leave out of scope), and a private cache with no public accessor is free to diverge from a consumer's reimplementation without any test noticing. Also rejected: putting the genome into `statistics` rows (breaks their JSON-friendliness).

### Decision 2 — transition-entropy combination rule

**Adopted.** Pool boundary-respecting transition **counts** across the organisms sharing a genome, then take the conditional entropy of the pooled count matrix. The normalization denominator is the total number of within-trace transitions, `sum(len(s) - 1 for s in sequences if len(s) >= 2)` — not `len(flat_actions) - 1`.

*Rationale.* Each organism is weighted by the number of transitions it actually contributed, which mirrors how `action_entropy` already pools marginal counts across organisms (3.2) and keeps the two entropy metrics conceptually consistent. It satisfies 2.8 exactly rather than approximately: for one sequence the denominator reduces to `len(actions) - 1`, which is the current formula.

*Rejected alternative.* Averaging per-organism entropy rates. It gives a 2-tick organism the same weight as a 300-tick one, is dominated by noise on short traces (an organism that dies after 2 ticks always reports `0.0`), and would break 2.8's exact-equality requirement for the single-organism case unless special-cased.

**Surface, and how drift is prevented.** The count matrix and the entropy computation move into one shared private core, and *both* public entry points call it:

```python
def _transition_counts(sequences, n_actions) -> Tuple[List[List[int]], int]   # matrix, total transitions
def _conditional_entropy_rate(counts, total_transitions) -> float

def pooled_transition_entropy_rate(sequences, n_actions=ACTION_SIZE) -> float  # new, public
def transition_entropy_rate(actions, n_actions=ACTION_SIZE) -> float:
    return pooled_transition_entropy_rate([actions], n_actions)               # delegation
```

`transition_entropy_rate` becomes a one-line delegation, so the single-sequence and pooled paths **cannot** drift — there is only one implementation. `pooled_transition_entropy_rate` is public (added to `diversity/metrics.__all__` and re-exported from `diversity/__init__.py`) so the rule is directly testable and reusable; the addition is purely additive.

*Bitwise-preservation constraint on the refactor (2a).* `_conditional_entropy_rate` must keep the current loop structure exactly: iterate rows in index order `0..n_actions-1`, skip `total == 0` rows, accumulate `row_entropy` by iterating the row in index order, then `rate += (row_total / total_transitions) * row_entropy`. Same operations in the same order over the same matrix with the same denominator gives bit-identical floats. `len(actions) < 2` must still short-circuit to `0.0` (it does: `total_transitions == 0`).

### Decision 3 — food-alignment temporal definition

**Adopted.** Fix the pairing inside `diversity/` only. The world trace tuple shape is unchanged.

**The definition.** Within one organism's clipped trace, for each index `i >= 1`:

- `pos_i - pos_{i-1}` is the displacement caused by the action recorded at index `i`, because `pos_{i-1}` is the position *after* action `i-1`, i.e. the position immediately *before* action `i`;
- `(fx_i, fy_i)` is the food direction observed immediately *before* action `i`, because a trace entry holds a pre-action observation together with a post-action position.

So the correct sample is `(pos_i - pos_{i-1}, (fx_i, fy_i))`. Samples are built **per trace and never across traces**.

*Rationale.* It is the only pairing consistent with the tuple's actual mixed temporal reference, it is a two-line change confined to one loop, and it fixes both the off-by-one and the cross-organism accumulation at once.

*Accepted tradeoff (2.10).* The displacement caused by each trace's **first** recorded action is excluded, because the pre-first-action position is never recorded. Alignment is therefore defined from the second recorded action onward: at most one lost sample per organism, out of up to `behavior_window` (default 100). Symmetrically, `food_dirs[0]` is now never used (it precedes the unrecoverable first displacement) while the final observed food direction — silently dropped pre-fix — is now used.

*Rejected alternative.* Extending the trace tuple with the pre-action position to recover that first sample. It changes a cross-package schema asserted by `tests/test_diversity.py::test_world_records_trace_when_enabled` (`len(org.trace[0]) == 6`) and world tests, adds per-tick memory for every organism, and forces a coordinated `world/` + `diversity/` + tests change (2.11) — a far larger blast radius than one lost sample per organism warrants.

`food_alignment_cosine` keeps its signature, semantics and zero-magnitude skipping (3.4); only the two sequences handed to it change.

## Fix Implementation

### `neat/population.py`

1. **New property `evaluated_best_genome`**, placed next to the existing `statistics` property. Returns `self._evaluated_best_genome.copy()`, or `None` when the cache is empty. Docstring states: it is the genome behind the most recently recorded `best_fitness`; it is `None` before the first generation; **every read returns a fresh defensive copy**, so callers may retain and mutate it and mutations cannot reach engine state (mirroring `best_genome`); and identity comparisons against engine internals or between two reads are meaningless. Type stays `Optional[Genome]`.
2. **Module docstring** — extend the "One generation" block with the evaluation lifecycle note (2.5): after `run(1)` returns, `self.population` holds unevaluated offspring carrying copied/stale fitness; evaluated per-generation values come from `statistics[-1]` or `evaluated_best_genome`.
3. **`run()` docstring** — one sentence pointing at the same thing.

No behavioral change; `_next_generation`, `_evaluate`, `_track_best_and_stats` and the private attributes are untouched (3.7).

> **Constraint on any text added to this file:** `tests/test_population.py::test_engine_has_no_environment_dependencies` asserts the substring `"main"` does not appear in the module source. That forbids not only `main` but `remain`, `domain`, `maintain`, `remaining`. Currently the substring is absent — keep it absent. (`grid` appears already and is fine: the test only rejects `import grid` / `from grid`.)

### `benchmarks/run.py`

1. **`run_trial` loop body** — replace `gen_best = max(population.population, key=lambda g: g.fitness)` with a read of `population.evaluated_best_genome`; raise `RuntimeError` if it is `None`. Source `best_fitness`, `best_node_count`, `best_connection_count` and `problem.success_fn(...)` from that one genome; keep `mean_fitness` and `species_count` from `population.statistics[-1]`. Row key set, order and `generation` numbering unchanged (3.9).
   - `best_fitness` moves from `stat["best_fitness"]` to `evaluated.fitness`. These are provably the same value (`_evaluated_best_fitness = _evaluated_best_genome.fitness`, never rewritten), and sourcing it from the genome is what makes 2.1's "same evaluated genome" literal rather than incidental.
2. **Module docstring** — state that per-generation rows describe the genome evaluated in that generation, read via `Population.evaluated_best_genome`.

`TrialResult`, `run_benchmark`, `main` and the CLI are untouched (3.10); `final_nodes` / `final_connections` become correct by construction (2.3).

### `diversity/metrics.py`

1. **New private helpers** `_transition_counts(sequences, n_actions)` and `_conditional_entropy_rate(counts, total_transitions)`, carrying the current loop structure verbatim (see Decision 2).
2. **New public `pooled_transition_entropy_rate(sequences, n_actions=ACTION_SIZE)`**; add to `__all__`.
3. **`transition_entropy_rate`** becomes `return pooled_transition_entropy_rate([actions], n_actions)`. Signature unchanged; docstring gains a pointer to the pooled variant.
4. **`per_genome_metrics`** — inside the per-trace loop:
   - keep `actions`, `points` and the density counters exactly as they are, in the same order (preserves `action_entropy`, `spatial_coverage`, `encounter_rate` bitwise);
   - collect `action_sequences: List[List[int]]`, one list per clipped trace;
   - replace the `deltas` / `food_dirs` accumulation with aligned per-trace pairs: for `i >= 1`, append `(x_i - x_{i-1}, y_i - y_{i-1})` to `aligned_deltas` **and** `(fx_i, fy_i)` to `aligned_food_dirs` in the same iteration, so the two lists cannot desynchronize;
   - call `pooled_transition_entropy_rate(action_sequences)` for `transition_entropy` and `food_alignment_cosine(aligned_deltas, aligned_food_dirs)` for `food_alignment`.
   - Returned dict keys unchanged.
5. **Docstrings** — module docstring documents the trace tuple's per-field temporal reference; `per_genome_metrics` states which metrics pool across organisms and which respect trace boundaries; `transition_entropy_rate` / `pooled_transition_entropy_rate` state the pooled-counts combination rule and its denominator (2.7); `food_alignment_cosine` and `per_genome_metrics` state the alignment definition and the dropped-first-action tradeoff (2.10).

`population_behavior`, `_behavior_keys`, `_std`, `_population_diversity`, `action_entropy`, `spatial_coverage`, `encounter_rate`, `_clip_window` are untouched.

### `diversity/__init__.py`

Re-export `pooled_transition_entropy_rate` alongside the existing names (additive; nothing removed).

### `world/organism.py`

**Documentation only — no code change** (2.12). Update the `self.trace` comment and add a line to `act`'s docstring stating that a trace entry combines the pre-action observation (`action` chosen from it, `food_dx`, `food_dy`, `density`) with the post-action position (`x`, `y`), and that consumers pairing a displacement with an observation must therefore use the observation stored at the *later* index. The tuple shape stays `(action, x, y, food_dx, food_dy, density)` (Decision 3).

### `README.md`

Yes, one line is needed — two, in "Design notes you should know":

- Evaluation lifecycle: after `run(1)`, `pop.population` holds unevaluated offspring; read evaluated per-generation values from `pop.statistics[-1]` or `pop.evaluated_best_genome`. The existing programmatic-use snippet prints `pop.statistics[-1]` and `pop.best_genome`, so a reader currently has no hint that `pop.population` is the wrong place to look.
- Measurement definitions: `transition_entropy` pools boundary-respecting transition counts across the organisms of a genome; `food_alignment` pairs each displacement with the food direction observed immediately before the action that caused it, excluding each trace's first recorded action. Pre-fix result JSON is not comparable to post-fix for these fields (and for `behavioral_diversity`, see blast radius).

No change to the `food_alignment`-is-confounded note or the 188-test count line (the count will change; update it to the new total).

### Files NOT changed

`experiments/run.py`, `experiments/metrics.py`, `world/simulation.py`, `world/config.py`, `world/recorder.py`, `visual/*`, `benchmarks/problems.py`, `benchmarks/report.py`, `benchmarks/plot.py`, `benchmarks/diagnose.py`, `pyproject.toml`, `neat/{genome,crossover,mutation,speciation,phenotype,innovation}.py`.

## Testing Strategy

### Validation Approach

Two phases. First, write the exploration tests and run them against the **unfixed** code to surface the counterexamples and confirm the root-cause analysis. Then apply the fix and re-run the same tests plus the full suite.

Every pre-fix value below was measured against the current working tree, so each test's pre-fix failure is a known quantity, not a prediction.

### Exploratory Bug Condition Checking

**Goal.** Surface counterexamples that demonstrate all three bugs before implementing anything. Confirm or refute the root-cause analysis; if refuted, re-hypothesize before touching code.

**Test plan.** Four deterministic tests, no new files. Bug 1 gets two tests, and they play **different roles** — a split forced by Decision 1a, where the accessor returns a defensive copy and identity assertions are therefore meaningless by design:

- **Test 1 (authoritative discriminator)** — the synthetic negative-fitness test. It is value-observable, fails pre-fix, passes post-fix, and is the regression test bugfix.md 2.13 requires. It carries the burden of proving Bug 1.
- **Test 2 (non-discriminating semantic guard)** — the real-problem `AND` test. Because the defect is latent on `AND` (see [Bug Condition 1](#bug-condition-1--benchmark-evaluation-state): the elite clone carries identical fitness and topology, so every value coincides), this test **passes both pre- and post-fix**. It is not a discriminator and is not expected to fail pre-fix; it is an invariant test that locks the semantic contract — history-row fields agree with the evaluated champion's values — so a future change that re-splits the sourcing gets caught.

---

**Test 1 — `tests/test_benchmarks.py::test_unevaluated_offspring_cannot_declare_success` (Bug 1, AUTHORITATIVE regression — value + early stop)**

Build a synthetic `Problem` (same `input_ids`/`output_ids`/`cases` as `AND`) with `fitness_fn = lambda g, gen: -1.0` and `success_fn = lambda g: g.fitness >= 0.0`. Negative evaluated fitness makes every crossover child (constructed fresh, `fitness == 0.0`) outrank the real champion, so the post-reproduction argmax is guaranteed to be an unevaluated offspring. Run `run_trial(probe, seed=0, population_size=6, max_generations=3)` and assert `result.solved is False`, `result.generations_to_solve is None`, `len(result.history) == 3`, `result.total_generations == 3`, and every `row["solved"] is False`.

*Exact pre-fix failure (measured, seeds 0/1/2/5 all identical).* `solved=True`, `generations_to_solve=1`, `len(history) == 1`, and the single row reads `{"best_fitness": -1.0, "solved": True}` — a success declared against the criterion `fitness >= 0.0` on a row whose recorded fitness is `-1.0`, with the remaining generations discarded. The first assertion (`solved is False`) fails.

This test uses no reference identity and needs none: the contradiction between `solved=True` and `best_fitness=-1.0` under the criterion `fitness >= 0.0` is fully observable in values, which is why it survives the switch to a copying accessor unchanged.

*Implementation note.* The pre-fix failure requires at least one crossover child to exist in generation 1, which is deterministic per seed (population 6, one species, `crossover_rate=0.75`). Seed 0 is confirmed; if the seed is changed, re-confirm the pre-fix failure and record the observation in the test docstring.

---

**Test 2 — `tests/test_benchmarks.py::test_history_row_describes_the_evaluated_genome` (Bug 1, SEMANTIC CONSISTENCY guard — not a discriminator)**

**This test passes pre-fix and post-fix.** That is expected, and it is stated here so nobody treats a green run pre-fix as evidence the bug is absent. On `AND` the defect is masked: `_reproduce_species` prepends `elite.copy()`, which preserves `fitness`, and `max` returns the first maximal element, so the wrong object carries the right values — measured `nodes 3`, `conns 0`, `fitness 0.8` coinciding in all four generations. The test's job is to pin the semantic contract going forward, not to expose the defect.

Wrap `AND` with `benchmarks.problems.Problem(...)`, substituting:
- a `fitness_fn` that calls `AND.fitness_fn` and records `(genome, fitness)` per generation into an oracle dict keyed by the `generation` argument;
- a `success_fn` that delegates to `AND.success_fn`. It may record the genome it was handed, but **only** to assert value agreement with the oracle champion — post-fix it receives a snapshot, so recording it for an identity check is not possible and not wanted.

Run `run_trial(probe, seed=0, population_size=6, max_generations=4)`. For each history row `k` (0-based), the oracle's champion is `champion = max(evaluated[k], key=fitness)`. Because the tracking `fitness_fn` is called in population order and `_evaluate` computes `max(self.population, ...)` in the same order, `max` breaks ties on the same element, so the oracle's champion is the genome the engine evaluated. Assert per generation, **values only**:
- `row["best_fitness"] == champion.fitness`;
- `row["best_node_count"] == len(champion.nodes)`;
- `row["best_connection_count"] == len(champion.connections)`;
- `row["solved"] == AND.success_fn(champion)`.

No identity assertion appears anywhere in this test (Decision 1a; bugfix.md 2.13 forbids the authoritative test from depending on identity and does not require this one to fail pre-fix).

*Oracle-generation indexing.* `_evaluate` is called with `self.generation`, which is `0` for the first generation and increments afterwards, so oracle keys are `0..3` while history `generation` values are `1..4`. Align positionally via `sorted(evaluated)`.

---

**Test 3 — `tests/test_diversity.py::test_transition_entropy_respects_organism_boundaries` (Bug 2)**

```python
t1 = [(0, 0, 0, 0.0, 0.0, 0.0), (0, 0, 0, 0.0, 0.0, 0.0)]   # actions [0, 0]
t2 = [(1, 5, 5, 0.0, 0.0, 0.0), (1, 5, 5, 0.0, 0.0, 0.0)]   # actions [1, 1]
assert per_genome_metrics([t1, t2], window=0, area=100)["transition_entropy"] == 0.0
```

Each organism is perfectly predictable, so the only honest value is exactly `0.0`. Add a companion assertion for 2.8: for a single trace with `>= 2` entries, `per_genome_metrics([t], ...)["transition_entropy"] == transition_entropy_rate([a for a, *_ in t])` exactly (`==`, not `approx`).

*Exact pre-fix failure (measured).* `0.6666666666666666` instead of `0.0` — the flattened stream `[0, 0, 1, 1]` contributes a fabricated `0 -> 1`. The companion 2.8 assertion passes pre-fix (single sequence) and must keep passing post-fix.

---

**Test 4 — `tests/test_diversity.py::test_food_alignment_pairs_displacement_with_preceding_observation` (Bug 3)**

Two cases, both asserting exact values.

*4a, off-by-one within one trace:*
```python
t = [(0, 0, 0, 1.0, 0.0, 0.0), (0, 1, 0, -1.0, 0.0, 0.0)]
assert per_genome_metrics([t], window=0, area=100)["food_alignment"] == pytest.approx(-1.0)
```
The single recoverable sample is a step east (`+1, 0`) taken while the observation preceding it reported food west (`-1, 0`) -> `-1.0`.

*4b, no cross-organism pairing:*
```python
t1 = [(0, 0, 0, 1.0, 0.0, 0.0), (0, 1, 0, 1.0, 0.0, 0.0)]   # east step, food east   -> +1
t2 = [(0, 5, 5, 0.0, 1.0, 0.0), (0, 5, 6, 0.0, 1.0, 0.0)]   # south step, food south -> +1
assert per_genome_metrics([t1, t2], window=0, area=100)["food_alignment"] == pytest.approx(1.0)
```

*Exact pre-fix failures (measured).* 4a returns `+1.0` instead of `-1.0` — a full sign inversion, scoring movement directly away from food as perfect food-seeking. 4b returns `0.5` instead of `1.0`: organism B's southward step is paired with organism A's eastward food direction, scoring a `+1.0` sample as `0.0`.

---

**Expected counterexamples, consolidated.**

| test | pre-fix | post-fix |
|---|---|---|
| 1 (Bug 1, **authoritative**) | `solved=True`, `gts=1`, 1 row, `best_fitness=-1.0` | `solved=False`, `gts=None`, 3 rows |
| 2 (Bug 1, semantic guard) | passes pre-fix and post-fix (guard test, not a discriminator) | passes; values keep agreeing with the evaluated champion |
| 3 (Bug 2) | `0.6666666666666666` | `0.0` |
| 4a (Bug 3) | `+1.0` | `-1.0` |
| 4b (Bug 3) | `0.5` | `1.0` |

Possible causes to confirm or refute while examining the failures: (Bug 1) the accessor gap plus elite-copy tie masking; (Bug 2) flattening before measurement; (Bug 3) two accumulators of different cardinality zipped positionally.

### Fix Checking

**Goal.** For all inputs where a bug condition holds, the fixed code produces the expected behavior.

```
FOR ALL X WHERE isBugCondition_1(X) DO
  row := run_trial'(X).history[g]
  ASSERT row.best_fitness          = fitness(evaluated_best(g))
     AND row.best_node_count       = node_count(evaluated_best(g))
     AND row.best_connection_count = connection_count(evaluated_best(g))
     AND row.solved                = success_fn(evaluated_best(g))
END FOR

FOR ALL X WHERE isBugCondition_2(X) DO
  ASSERT per_genome_metrics'(X).transition_entropy
       = conditional_entropy(pool({within_trace_transition_counts(clip(t)) : t IN X}))
END FOR

FOR ALL X WHERE isBugCondition_3(X) DO
  ASSERT per_genome_metrics'(X).food_alignment
       = mean{ cos(pos_i - pos_{i-1}, food_dir_i) : t IN X, i >= 1, both magnitudes > 0 }
END FOR
```

Tests 1-4 are the scoped instantiations. All three bugs are deterministic, so the properties are scoped to concrete failing cases for reproducibility rather than randomized.

### Preservation Checking

**Goal.** For all inputs where no bug condition holds, the fixed code produces the same result as the original.

```
FOR ALL X WHERE NOT (isBugCondition_1(X) OR isBugCondition_2(X) OR isBugCondition_3(X)) DO
  ASSERT F(X) = F'(X)
END FOR
```

**Testing approach.** Observation-first: run the **unfixed** code on non-triggering inputs, record the actual outputs, then assert those recorded values. The existing suite already pins most of this — 188 tests pass via `python3 -m pytest -q -o addopts=''` — so preservation is checked primarily by keeping every existing test green with no assertion edits, plus targeted additions where no existing test covers the boundary.

**Test cases.**

1. **Single-sequence transition entropy (3.1)** — `test_transition_entropy_rate_deterministic` and `test_transition_entropy_rate_alternating_pairs` must pass **untouched**, including the exact `0.9649839288802097`. This is the bitwise check on the Decision 2 refactor.
2. **Single-organism equivalence (2.8)** — the companion assertion in Test 3, using `==` rather than `approx`.
3. **Pooled marginal metrics (3.2, 3.3, 3.5)** — `test_per_genome_metrics_pooling_and_window` (`action_entropy == 1.0`, `encounter_rate == 0.25`), `test_action_entropy_*`, `test_spatial_coverage_fraction`, `test_encounter_rate_mean_density` pass untouched.
4. **`food_alignment_cosine` contract (3.4)** — `test_food_alignment_cosine_directions` and `test_food_alignment_skips_zero_deltas` pass untouched; the function is not modified.
5. **Trace schema (Decision 3)** — `test_world_records_trace_when_enabled` (`len(org.trace[0]) == 6`) passes untouched, proving the tuple shape did not change.
6. **Engine lifecycle and stats (3.7)** — all of `tests/test_population.py` passes untouched, notably `test_statistics_use_evaluated_fitness_not_carried`, `test_best_genome_tracked_and_isolated`, `test_seeded_run_is_deterministic`, and `test_engine_has_no_environment_dependencies` (the `"main"`-substring constraint).
7. **Benchmark determinism and row shape (3.8, 3.9)** — `test_run_trial_is_deterministic` and `test_run_trial_history_is_well_formed` pass untouched. Measured post-fix behavior for `AND` at population 4, seeds 0 and 5, confirms no early stop and identical counts across generations 1-3.
8. **Experiment integration (3.11)** — `test_experiment_records_behavioral_metrics` and `test_empty_genome_baseline_zero_entropy_and_diversity` pass untouched. Empty genomes emit a single constant action, so `transition_entropy` is `0.0` both before and after pooling.

**New preservation test worth adding** — `tests/test_population.py::test_evaluated_best_genome_matches_recorded_statistics`. It pins both halves of Decision 1 rather than trusting them:

*Consistency (Decision 1, 1b).*
- the accessor is `None` before the first `run`;
- after `run(1)`, and again after a second `run(1)`, `pop.evaluated_best_genome.fitness == pop.statistics[-1]["best_fitness"]`.

*Isolation (Decision 1a, bugfix.md 3.15).*
- two consecutive reads return **distinct objects** (`pop.evaluated_best_genome is not pop.evaluated_best_genome`), so no caller can be holding the engine's instance;
- mutate a returned snapshot — set its `fitness` to a sentinel and mutate one of its genes (e.g. flip a `NodeGene.bias`, or a `ConnectionGene.weight`/`enabled`) — then assert the mutation is invisible everywhere: a subsequent read still reports the original `fitness` and the original gene values, `pop.statistics[-1]["best_fitness"]` is unchanged, and `pop.best_genome` is unchanged. This is the meaningful assertion, and it mirrors `test_best_genome_tracked_and_isolated`.

The old "returned object is not a member of `pop.population`" check is now trivially true — a fresh copy can never be a member — so it is no longer the point. It may be kept as a cheap lifecycle assertion documenting 2.5, but isolation-by-mutation is what actually constrains the implementation.

### Unit Tests

- `pooled_transition_entropy_rate`: empty input, one sequence shorter than 2, several sequences all shorter than 2 (denominator 0 -> `0.0`), one long sequence (equals `transition_entropy_rate`), mixed lengths (weighting by contributed transitions).
- `run_trial` field sourcing: `solved` never sourced from offspring (Test 1, the discriminator); row fields agreeing with the evaluated genome's values (Test 2, the guard).
- `per_genome_metrics`: aligned pair count equals `sum(max(0, len(clip(t)) - 1))`, i.e. exactly one sample dropped per trace.

### Property-Based Tests

Scoped to concrete cases, because all three bugs are deterministic (per the exploration-test guidance):

- Bug 1: over the generations of one seeded trial, every row satisfies Property 1a — a bounded universal quantification over generations.
- Bug 2: over the traces of one genome, the pooled count matrix contains no cross-boundary pair — assert the pooled transition total equals `sum(len(s) - 1 for s in sequences if len(s) >= 2)`, which is exactly the count of legitimate transitions and therefore excludes fabricated ones.
- Bug 3: over the traces of one genome, sample count equals `sum(max(0, len(clip(t)) - 1))` and no sample mixes indices from two traces.

### Integration Tests

- `run_trial` end to end on the synthetic negative-fitness problem and on `AND` (Tests 1 and 2 are already integration-level: real engine, real reproduction, real early-stop logic).
- `experiments.run.run_trial` on the tiny resolved config, asserting all `RECORD_FIELDS` present and non-NaN — existing tests, unmodified, covering the `world -> diversity -> experiments` path with the new metric definitions.
- Full suite: `python3 -m pytest -q -o addopts=''`, expecting the current 188 plus the new tests, zero failures, zero edits to existing assertions.

## Determinism Preservation

- **No new RNG.** No fix path touches `self.rng`, `world.rng`, or `random`. The synthetic problem in Test 1 is a pure function of the genome.
- **No new iteration-order dependence.** `per_genome_metrics` keeps consuming `traces` in the order given and each clipped trace in index order. `population_behavior` builds `traces_by_genome` as a dict keyed by genome objects; insertion order (population order, then organism order) is preserved and unchanged.
- **Sets.** The only `set()` in the changed module is inside `spatial_coverage`, used solely for `len()` — order-independent, untouched.
- **Dicts.** `_transition_counts` uses fixed-size lists indexed by action id, not dicts, so accumulation order is fixed by index, not by hash.
- **Float reproducibility.** The Decision 2 refactor preserves the exact operation order (see the bitwise constraint), so single-sequence results are bit-identical, and pooled results are a deterministic function of the trace order.
- **Engine seeding untouched.** `Population` gains a read accessor only; `_next_generation` phase order, `_select_parent` roulette, mutation and `allocate_offspring`'s largest-remainder tie-break are unchanged (3.7, 3.12).
- **Benchmark seeding untouched.** `run_trial` still constructs `Population(..., seed=seed)` identically; the change is which already-computed genome is read afterwards. It consumes no RNG, so `test_run_trial_is_deterministic` and cross-run reproducibility are unaffected.
- **The accessor's `copy()` consumes no RNG.** `Genome.copy()` is a pure structural deep copy — no sampling, no innovation-number allocation — so returning a defensive snapshot per read cannot shift any seeded stream or affect determinism.

## Blast Radius

**`experiments/run.py` — no code change.** `make_tracking_evaluator` already captures `evaluated_best_genome` inside the evaluator closure (before reproduction), so `node_count` / `connection_count` there were already correct; `run_trial` already reads `best_fitness` / `mean_fitness` from that same closure state. `RECORD_FIELDS` is unchanged: no field added, removed or renamed (3.11).

**Values that change (intended, per bugfix.md).**

| field | changes when | reason |
|---|---|---|
| `transition_entropy` | any genome with `>= 2` organisms contributing actions | boundary-respecting pooled counts |
| `food_alignment` | any trace with `>= 2` entries | corrected pairing; no cross-organism leak |
| **`behavioral_diversity`** | whenever `transition_entropy` changes | see below |
| benchmark `solved`, `generations_to_solve`, `best_node_count`, `best_connection_count` | when the post-reproduction argmax differed from the evaluated champion | now sourced from the evaluated genome |

**Flag on requirement 3.11.** It says only `transition_entropy` and `food_alignment` values change. Strictly, `behavioral_diversity` also changes: `population_behavior` feeds `transition_entropy` into `_population_diversity` as one of three z-scored fingerprint dimensions. `action_entropy_diversity` is unaffected (it is the std of the `action_entropy` column only). Read 3.11 as "only `transition_entropy`, `food_alignment`, and values derived from them". No existing test asserts an exact `behavioral_diversity` value — `test_population_diversity_zero_for_identical_genomes` asserts `== 0.0` for two organisms of the *same* genome with identical traces, which stays `0.0` because the single-genome fingerprint column has fewer than 2 entries and `_population_diversity` returns `0.0`; the others assert `> 0.0` / `>= 0.0`. So this is a values-only consequence with no assertion edits, but it belongs in the not-comparable list and in the README note.

**`RECORD_FIELDS` consumers.** `visual/data.py` (`METRICS` list) and `visual/analytics.py` (`DEFAULT_METRICS`, which already excludes `food_alignment`) read recorded JSON by field name. No schema change, so no `visual/` change and no risk of a `KeyError`. Pre-fix and post-fix result JSON must not be pooled for the changed fields.

**Benchmark history consumers.** `benchmarks/report.py` (`final_nodes`, `final_connections`, `result.solved`, `species_count`) and `benchmarks/plot.py` (`_series` over `best_connection_count`, `best_node_count`, `best_fitness`, `mean_fitness`, `species_count`) read by key. The key set and types are unchanged (3.9), so both keep working; their numbers become correct.

**Package separation (3.13).** `neat/population.py` gains no import — the new property returns an existing attribute typed with the already-imported `Genome` and `Optional`. `neat/` still imports nothing from `world/`, `benchmarks/`, `experiments/` or `visual/`; `visual/` still consumes recorded data only. `diversity/metrics.py` keeps its single cross-package import (`world.config.ACTION_SIZE`). Dependency direction is unchanged: `benchmarks -> neat`, `diversity -> world.config`, `experiments -> {neat, world, diversity}`.

**Public API deltas.** Additive only: `Population.evaluated_best_genome` (property — returns a fresh defensive **snapshot** per read, `None` before the first generation; callers must not rely on reference identity), `diversity.metrics.pooled_transition_entropy_rate` (function, plus `__all__` and `diversity/__init__.py` re-export). Nothing removed, renamed, or signature-changed.

## Out of Scope (restated from bugfix.md — do not drift)

- World input/output interface mismatch (observation/action wiring vs. declared ids).
- Grid capacity crash (`random_empty_cell` on a full grid).
- Negative-fitness offspring allocation in speciation. *(Test 1 uses a negative-fitness problem to expose Bug 1; it must not turn into a fix for allocation behavior.)*
- Champion structural-signature identity in `_guarantee_champion`.
- Existing mypy errors.
- The `pytest` `addopts` reference to the uninstalled `pytest-cov` plugin — `pyproject.toml` stays as-is.
- Bugs in `visual/`.
- Duplicated metric aggregation between `experiments/` and `diversity/`. *(Decision 1's rejected alternative notes this duplication; the fix does not consolidate it.)*
- `run_trial`'s `UnboundLocalError` when `max_generations=0` — unless entangled with the evaluated-genome fix. It is **not** entangled: the `None`-guard in Decision 1b sits inside the loop body, and the pre-existing unbound `generation` on the zero-iteration path is untouched.
- Any refactor, rename, or stylistic cleanup not required by the three defects above. *(The Decision 2 helper extraction is required — it is what makes the single-sequence and pooled paths one implementation.)*
