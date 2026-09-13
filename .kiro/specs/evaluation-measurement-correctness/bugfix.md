# Bugfix Requirements Document

## Introduction

Clage's evaluation and behavioral measurement pipeline reports three classes of numbers that do not describe what they claim to describe. All three are confirmed by reading the implementation, and none of them are caught by the current 188-test suite.

1. **Benchmark evaluation-state bug** (`benchmarks/run.py::run_trial`). After `Population.run(1)` returns, `population.population` holds newly reproduced offspring carrying stale/`0.0` fitness. `run_trial` builds each history row from a mix of sources: `best_fitness` / `mean_fitness` / `species_count` come from `population.statistics[-1]` (the evaluated generation, correct), while `best_node_count`, `best_connection_count` and `solved` come from `max(population.population, ...)` — an unevaluated newborn. A problem can therefore be declared solved, and the trial early-stopped with a recorded `generations_to_solve`, on the basis of a genome that was never evaluated in that generation; topology counts describe a different generation than the fitness printed beside them. `neat/population.py` is itself correct here — it deliberately caches the evaluated best/mean before reproduction — but it exposes the evaluated champion only through the private `_evaluated_best_genome` attribute, and `statistics` rows carry no genome reference, so the correct value is not reachable by a consumer.

2. **Transition entropy crosses organism boundaries** (`diversity/metrics.py::per_genome_metrics`). Actions from every organism sharing a genome are appended into one flat list, and `transition_entropy_rate` is called once over that list. Because it counts pairs with `zip(actions, actions[1:])`, the last action of one organism is counted as the predecessor of the first action of the next — a transition that never happened. In-world asexual reproduction makes multi-organism genomes common, so these fabricated transitions are frequent, not marginal.

3. **Food-alignment temporal misalignment** (`world/organism.py::act` + `diversity/metrics.py`). A trace entry mixes temporal references: `action` and the food direction `(fx, fy)` are pre-action, while `(x, y)` is post-action. In `per_genome_metrics`, `food_dirs` gets one entry per trace index but `deltas` only from `index > 0`, so `food_alignment_cosine` zips displacement caused by the action at index *i* against the food direction observed before the action at index *i-1*. Every pair is off by one, and the final observed food direction is dropped. The pre-first-action position is never recorded, so the displacement produced by the first recorded action is not recoverable from the current trace.

### Intended consequences of the fix

Some reported values legitimately change. This is intended, not a regression:

- `transition_entropy` values change for any genome represented by more than one organism.
- `food_alignment` values change wherever a trace has more than one entry.
- Benchmark `solved`, `generations_to_solve`, `best_node_count` and `best_connection_count` rows may change, since they will now describe the evaluated genome.

Result JSON and benchmark reports produced before the fix are **not comparable** to output produced after it for these fields.

### Out of scope

The following issues are known and explicitly **not** addressed by this bugfix:

- World input/output interface mismatch (observation/action wiring vs. declared ids).
- Grid capacity crash (`random_empty_cell` behavior on a full grid).
- Negative-fitness offspring allocation in speciation.
- Champion structural-signature identity in `_guarantee_champion`.
- Existing mypy errors.
- The `pytest` `addopts` reference to the uninstalled `pytest-cov` plugin — configuration stays as-is.
- Bugs in `visual/`.
- Duplicated metric aggregation between `experiments/` and `diversity/`.
- `run_trial`'s `UnboundLocalError` when `max_generations=0` — out of scope unless it turns out to be entangled with the evaluated-genome fix.
- Any refactor, rename, or stylistic cleanup not required by the three defects above.

## Bug Analysis

### Current Behavior (Defect)

**Bug 1 — benchmark evaluation state**

1.1 WHEN `run_trial` records a history row for a generation THEN the system takes `best_node_count`, `best_connection_count` and `solved` from `population.population` after reproduction, so those fields describe an unevaluated offspring genome while `best_fitness` / `mean_fitness` / `species_count` in the same row describe the evaluated generation.

1.2 WHEN an unevaluated offspring genome happens to satisfy `problem.success_fn` THEN the system marks the generation `solved`, stops the trial early, and records `generations_to_solve` for a genome whose fitness was never measured in that generation.

1.3 WHEN a caller reads `TrialResult.final_nodes` or `TrialResult.final_connections` THEN the system returns topology counts that are not guaranteed to belong to the genome behind `TrialResult.final_best_fitness`.

1.4 WHEN a consumer needs the genome that produced a generation's reported `best_fitness` THEN the system offers no public access to it — the value exists only as `Population._evaluated_best_genome`, and `statistics` rows contain no genome reference — so consumers read the post-reproduction population instead.

**Bug 2 — transition entropy across organism boundaries**

1.5 WHEN a genome is represented by more than one organism THEN the system pools all their actions into one flat sequence and counts a transition from the last action of one organism to the first action of the next, inflating or deflating `transition_entropy` with pairs that never occurred.

1.6 WHEN `transition_entropy` is reported for a multi-organism genome THEN the system does not define or document how per-organism transition statistics are combined into the single reported scalar, so the reported number has no stated meaning.

**Bug 3 — food-alignment temporal misalignment**

1.7 WHEN `per_genome_metrics` computes `food_alignment` over a trace with more than one entry THEN the system zips `deltas` and `food_dirs` positionally with a one-index offset, pairing a movement with a food direction observed before a different action.

1.8 WHEN a trace ends THEN the system drops the last observed food direction from the pairing, and because the pre-first-action position is never recorded, the displacement produced by the first recorded action cannot be recovered at all.

1.9 WHEN a reader inspects a trace entry THEN the system provides no documented temporal reference for its fields, even though `action` and `(food_dx, food_dy)` are pre-action while `(x, y)` is post-action.

### Expected Behavior (Correct)

**Bug 1 — benchmark evaluation state**

2.1 WHEN `run_trial` records a history row for a generation THEN the system SHALL derive `best_fitness`, `best_node_count`, `best_connection_count` and `solved` from the same evaluated genome of that same generation.

2.2 WHEN a generation is marked `solved` and the trial early-stops with `generations_to_solve` THEN the system SHALL base that decision on the genome that was evaluated in that generation, never on a post-reproduction offspring.

2.3 WHEN a caller reads `TrialResult.final_nodes` and `TrialResult.final_connections` THEN the system SHALL return the topology of the genome behind `TrialResult.final_best_fitness`.

2.4 WHEN a consumer needs the evaluated best genome for a generation THEN the system SHALL provide a documented, non-private means of obtaining it that keeps `neat/` free of any dependency on `world/`, `benchmarks/` or `experiments/`, and the value it returns SHALL be a defensive snapshot that does not alias the engine's cached genome — no shared `Genome`, `NodeGene`, `ConnectionGene`, or mutable container instance — so that mutating the returned value cannot affect `Population`'s cached evaluated champion or any later read. The snapshot SHALL preserve everything `benchmarks.run_trial` needs — `fitness`, the node set, the connection set and its genes, innovation numbers, biases, weights and enabled flags — i.e. the same guarantees `Genome.copy()` already provides and that `tests/test_genome.py::test_copy_preserves_ids_bias_innovation_and_fitness` and `::test_copy_is_deep_no_alias` pin.

2.5 WHEN the generation/evaluation lifecycle is documented THEN the system SHALL state explicitly that after `Population.run(1)` returns, `population.population` holds unevaluated offspring, and that evaluated per-generation values must be read from the recorded statistics or the documented evaluated-champion accessor.

**Bug 2 — transition entropy across organism boundaries**

2.6 WHEN `per_genome_metrics` computes `transition_entropy` for a genome with multiple organisms THEN the system SHALL count only action pairs that occurred consecutively within a single organism's trace, never across a trace boundary.

2.7 WHEN per-organism transition statistics are combined into the single reported `transition_entropy` scalar THEN the system SHALL use one explicitly chosen and documented combination rule (for example, pooling boundary-respecting transition counts across organisms, or averaging per-organism entropy rates), with the choice and its rationale recorded, since the alternatives yield different numbers.

2.8 WHEN a genome has exactly one organism with a trace of at least two entries THEN the system SHALL report the same `transition_entropy` value as `transition_entropy_rate` computed over that single action sequence.

**Bug 3 — food-alignment temporal misalignment**

2.9 WHEN `food_alignment` is computed THEN the system SHALL pair each movement displacement with the food direction observed immediately before the action that produced that displacement.

2.10 WHEN the temporal definition of `food_alignment` is chosen THEN the system SHALL adopt exactly one internally consistent definition, document it, and state the tradeoff — either correcting the pairing inside `diversity/` and dropping the first action's displacement as unrecoverable, or recording the pre-action position in the world trace so every action's displacement is recoverable.

2.11 IF the trace tuple shape changes to carry the pre-action position THEN the system SHALL update the documented trace schema and every consumer and test that asserts the tuple shape, keeping `world/` and `diversity/` consistent.

2.12 WHEN a trace entry is documented THEN the system SHALL state the temporal reference of each field (which values are pre-action and which are post-action).

**Verification requirements (all three bugs)**

2.13 WHEN the fix is delivered THEN the system SHALL include at least one focused, deterministic regression test per bug that fails on the pre-fix code and passes on the fixed code. For Bugs 2 and 3 this requirement stands as stated. For Bug 1 the authoritative regression test SHALL demonstrate a **value-observable** pre-fix failure — that success and early-stop are declared from an unevaluated genome — rather than object identity: a synthetic `Problem` whose `fitness_fn` returns `-1.0` and whose `success_fn` is `g.fitness >= 0.0` yields, pre-fix, `solved=True`, `generations_to_solve=1`, one history row, and `best_fitness=-1.0`, i.e. a trial declared solved against `fitness >= 0.0` on a row that records `-1.0`. That test SHALL NOT depend on reference identity between the value returned by the public accessor and any engine-internal object. A real-problem test MAY additionally verify semantic consistency (history-row fields agreeing with the evaluated champion's values), but SHALL NOT require reference identity and is not required to fail pre-fix.

2.14 WHEN the full test suite is run after the fix THEN the system SHALL pass all existing tests, with assertion changes permitted only where a changed value is a documented intended consequence listed in this document.

### Unchanged Behavior (Regression Prevention)

3.1 WHEN `transition_entropy_rate` is called with a single action sequence THEN the system SHALL CONTINUE TO return its current values, keeping `tests/test_diversity.py::test_transition_entropy_rate_deterministic` and `::test_transition_entropy_rate_alternating_pairs` passing unchanged.

3.2 WHEN `action_entropy` is computed for a genome with multiple organisms THEN the system SHALL CONTINUE TO report the pooled marginal distribution value it reports today.

3.3 WHEN `spatial_coverage` is computed for a genome with multiple organisms THEN the system SHALL CONTINUE TO pool positions across organisms, since that pooling is an intentional documented design choice.

3.4 WHEN `food_alignment_cosine` encounters a zero-magnitude movement delta or a zero-magnitude food direction THEN the system SHALL CONTINUE TO skip that sample, keeping `::test_food_alignment_cosine_directions` and `::test_food_alignment_skips_zero_deltas` passing.

3.5 WHEN `encounter_rate` is computed THEN the system SHALL CONTINUE TO return the mean observed organism density over the (window-clipped) trace.

3.6 WHEN the behavior window (`behavior_window`) or the trace toggle (`record_trace`) is set THEN the system SHALL CONTINUE TO clip and enable traces exactly as it does today.

3.7 WHEN `Population.run` advances a generation THEN the system SHALL CONTINUE TO evaluate, speciate, share fitness, allocate offspring, reproduce and guarantee the champion in the current order, and SHALL CONTINUE TO record statistics from the evaluated population, keeping `tests/test_population.py::test_statistics_use_evaluated_fitness_not_carried` passing.

3.8 WHEN two benchmark trials run with the same seed and settings THEN the system SHALL CONTINUE TO produce identical histories, keeping `tests/test_benchmarks.py::test_run_trial_is_deterministic` passing.

3.9 WHEN a benchmark history row is produced THEN the system SHALL CONTINUE TO expose exactly the key set `{generation, best_fitness, mean_fitness, species_count, best_node_count, best_connection_count, solved}` with `generation` counting from 1, keeping `::test_run_trial_history_is_well_formed` passing.

3.10 WHEN benchmark problems are used THEN the system SHALL CONTINUE TO define `OR`, `AND`, `XOR` and `SIN` with their current inputs, cases, fitness functions and success criteria, and the `benchmarks.run` CLI SHALL CONTINUE TO accept its current arguments.

3.11 WHEN an experiment trial runs THEN the system SHALL CONTINUE TO capture the evaluated best genome inside the evaluator closure and SHALL CONTINUE TO emit every field in `RECORD_FIELDS`, with only `transition_entropy` and `food_alignment` values changing.

3.12 WHEN any part of the pipeline is seeded THEN the system SHALL CONTINUE TO be deterministic: the same engine seed and the same `(trial seed, generation)` world seed SHALL produce identical results.

3.13 WHEN packages import one another THEN the system SHALL CONTINUE TO respect the existing separation: `neat/` does not import `world/`, and `visual/` contains no evolutionary logic and consumes recorded data only.

3.14 WHEN the test suite is configured THEN the system SHALL CONTINUE TO use the existing `pyproject.toml` pytest configuration unchanged, including the current `addopts`.

3.15 WHEN a caller mutates the value returned by the public evaluated-champion accessor THEN the system SHALL CONTINUE TO keep the engine's cached evaluated champion isolated from callers: that mutation SHALL NOT change `Population`'s internal state, the result of any subsequent read of the accessor, `statistics`, or `best_genome` — mirroring the isolation guarantee already asserted for `best_genome` by `tests/test_population.py::test_best_genome_tracked_and_isolated`.

### Bug Conditions and Properties

**Bug 1 — benchmark evaluation state**

```pascal
FUNCTION isBugCondition_1(X)
  INPUT: X = (problem, seed, population_size, max_generations)
  OUTPUT: boolean

  // A trial generation where the post-reproduction population's best-by-fitness
  // genome differs from the genome evaluated in that generation.
  RETURN EXISTS generation g IN trial(X) SUCH THAT
           argmax_fitness(population_after_reproduction(g)) <> evaluated_best(g)
END FUNCTION
```

```pascal
// Property: Fix Checking - one generation, one genome
FOR ALL X WHERE isBugCondition_1(X) DO
  row ← run_trial'(X).history[g]
  ASSERT row.best_node_count       = node_count(evaluated_best(g))
     AND row.best_connection_count = connection_count(evaluated_best(g))
     AND row.solved                = success_fn(evaluated_best(g))
     AND row.best_fitness          = fitness(evaluated_best(g))
END FOR
```

**Bug 2 — transition entropy across organism boundaries**

```pascal
FUNCTION isBugCondition_2(X)
  INPUT: X = list of traces sharing one genome
  OUTPUT: boolean

  // More than one organism contributes at least one action after window clipping.
  RETURN count(t IN X WHERE length(clip(t)) >= 1) >= 2
END FUNCTION
```

```pascal
// Property: Fix Checking - no fabricated transitions
FOR ALL X WHERE isBugCondition_2(X) DO
  metrics ← per_genome_metrics'(X)
  ASSERT metrics.transition_entropy = combine(
           { transition_counts(clip(t)) : t IN X }   // boundary-respecting only
         )
     AND no pair (last_action(t_i), first_action(t_j)) with i <> j is counted
END FOR
```

**Bug 3 — food-alignment temporal misalignment**

```pascal
FUNCTION isBugCondition_3(X)
  INPUT: X = list of traces sharing one genome
  OUTPUT: boolean

  // Any trace long enough for the off-by-one pairing to occur.
  RETURN EXISTS t IN X SUCH THAT length(clip(t)) >= 2
END FUNCTION
```

```pascal
// Property: Fix Checking - displacement paired with the food direction that preceded it
FOR ALL X WHERE isBugCondition_3(X) DO
  metrics ← per_genome_metrics'(X)
  ASSERT metrics.food_alignment = mean over samples (d_k, f_k) WHERE
           d_k = displacement caused by action a_k
       AND f_k = food direction observed immediately before a_k
       AND |d_k| > 0 AND |f_k| > 0
END FOR
```

**Preservation goal (all three bugs)**

```pascal
// Property: Preservation Checking
FOR ALL X WHERE NOT (isBugCondition_1(X) OR isBugCondition_2(X) OR isBugCondition_3(X)) DO
  ASSERT F(X) = F'(X)
END FOR
```

Where `F` is the current (unfixed) pipeline and `F'` the fixed pipeline: single-organism genomes, single-entry traces, and generations whose evaluated champion already coincides with the post-reproduction best must produce byte-identical results.
