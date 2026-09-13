# Bugfix Requirements Document

## Introduction

Clage accepts invalid configurations and incompatible subsystem combinations without complaint. The
consequences fall into two families, both bad: either an unactionable exception surfaces deep inside
a running simulation (`TypeError: Value after * must be an iterable, not NoneType`,
`ValueError: max() iterable argument is empty`, `ZeroDivisionError`), or nothing raises at all and
the run silently produces invalid behavior — organisms that gain energy every tick, densities frozen
at 0.0, actions the network can never reach, trial seeds that all collapse onto the same world,
populations that grow past their configured size.

This bugfix makes invalid configurations and incompatible combinations fail early, deterministically,
and with actionable errors. Five distinct boundaries are in scope, each verified against HEAD
`ea8a9c2`:

1. **NEAT ↔ world neural-interface contract** — `world/config.py` declares `OBSERVATION_SIZE = 9` and
   `ACTION_SIZE = 4` as a comment, not a check. `neat/genome.py` defaults to 10 inputs. Nothing
   reconciles them before a run starts.
2. **Population ↔ grid capacity** — founder placement in `world/simulation.py` ignores the documented
   `Optional` return of `World.random_empty_cell()`.
3. **Environment numeric field validity** — `EnvironmentConfig` is a plain dataclass with no
   validation at all; several fields crash downstream and several silently invalidate the model.
4. **Experiment condition and inheritance validity** — duplicate condition names, control-count
   problems, unknown requested conditions, wrongly typed values, nested `extends`, and an explicit
   empty `seeds` list are all mishandled.
5. **Fitness validity and offspring-allocation arithmetic** — non-finite fitness either crashes with a
   misleading message or silently enters selection and reporting, and mixed-sign adjusted fitness
   with a positive total breaks the engine's own constant-population-size invariant.

Every clause below describes a behavior reproduced against HEAD. Every proposed rule is justified by
an actual downstream consequence: a crash, a violated conservation law, a broken stated invariant, or
a silent contradiction of documented semantics. Rules that would only encode taste are listed under
"Investigated and rejected as non-defects" so that implementation does not invent restrictions.

### Requirement ownership boundaries

These five concerns have materially different owners and MUST NOT be collapsed into a single shared
validator or a new unifying abstraction:

- The **neural-interface contract** is a world-side boundary check performed at evaluator/generation
  entry. Only `world` knows the contract; `neat` must stay a generic engine and must not gain a
  dependency on `world`.
- **Grid capacity** is a world-side precondition that combines an `EnvironmentConfig` with a
  population size, so it cannot live on the config alone.
- **Numeric field validity** belongs to `EnvironmentConfig` itself, at construction time.
- **Experiment condition and inheritance validity** belongs to the experiment config loader and
  `ExperimentConfig`.
- **Fitness validity and allocation arithmetic** are internal `neat` engine invariants.

## Bug Analysis

### Current Behavior (Defect)

#### Area 1 — NEAT ↔ world neural-interface contract

1.1 WHEN a population built with the default genome interface (`DEFAULT_INPUT_IDS = tuple(range(10))`,
10 inputs) is evaluated by a world evaluator whose `Organism.observe` returns 9 values THEN the system
raises `ValueError: expected 10 inputs, got 9` from `neat/phenotype.py:109` in `activate()`, mid-run,
once per organism per tick, after the world was already built and organisms already placed.

1.2 WHEN a genome interface declares zero outputs (`output_ids=[]`) THEN the system raises
`ValueError: max() iterable argument is empty` from `world/organism.py:96` in `act()`, mid-simulation,
with a message that names neither the interface nor the configuration field at fault.

1.3 WHEN a genome interface declares more than 4 outputs (e.g. 6) THEN the system raises no error, runs
to completion, and an organism whose largest output is at index 4 or 5 selects a `previous_action`
outside the valid `0..3` range, so `_apply_action` matches no branch and the tick silently does nothing
(position, facing, and `food_eaten` all unchanged).

1.4 WHEN a genome interface declares fewer than 4 outputs (e.g. 2) THEN the system raises no error and
`Action.TURN_RIGHT` and `Action.EAT` become unreachable, silently crippling every organism in the run.

1.5 WHEN `world/organism.py` imports `OBSERVATION_SIZE` and `ACTION_SIZE` THEN the system never uses
either value, so the declared interface is documentation rather than enforcement, and the stale comment
in `neat/genome.py` ("10 sensors, 4 actions") contradicts the actual 9-value observation vector.

#### Area 2 — Population ↔ grid capacity

1.6 WHEN `population_size` exceeds grid capacity `width * height` (e.g. 10 founders on a 3×3 grid)
THEN the system raises `TypeError: Value after * must be an iterable, not NoneType` at
`world/simulation.py:44`, because `random_empty_cell()` returned `None` and was unpacked anyway.

1.7 WHEN `width` or `height` is zero or negative (e.g. `width=0`, `width=-3`) THEN the system raises
that same unactionable `TypeError` at `world/simulation.py:44`, since no cells exist at all, rather
than reporting an invalid grid dimension.

#### Area 3 — Environment numeric field validity

1.8 WHEN `density_radius` is 0 THEN the system raises `ZeroDivisionError: division by zero` at
`world/grid.py:109`, because `max_count = (2 * radius + 1) ** 2 - 1` evaluates to 0.

1.9 WHEN `density_radius` is negative (e.g. -2) THEN the system raises no error, but the density loops
iterate over nothing while `max_count` is 8, so every food-density and organism-density observation is
silently 0.0 for the whole run.

1.10 WHEN `max_energy` is 0.0 THEN the system raises `ZeroDivisionError: float division by zero` at
`world/organism.py:77` in `observe()`, evaluating `self.energy / config.max_energy`.

1.11 WHEN `repro_fraction` is greater than 1.0 (e.g. 1.5) THEN the system raises no error, the child
receives 1.5× the parent's energy, and the parent is driven negative — energy is created from nothing,
violating the conservation implied by `self.energy -= child_energy`.

1.12 WHEN `repro_fraction` is negative (e.g. -0.5) THEN the system raises no error, the child receives
negative energy, and the parent gains energy from reproducing.

1.13 WHEN `metabolism` is negative (e.g. -1.0) THEN the system raises no error, organisms gain energy
every tick, never die, and reproduce explosively (24 organisms from 3 founders within 3 ticks).

1.14 WHEN `ticks` is zero or negative (e.g. -5) THEN the system raises no error and silently runs zero
ticks, because `range(-5)` is empty, reporting a completed generation that never simulated anything.

1.15 WHEN `initial_food`, `food_target`, or `food_regrowth_per_tick` is negative (e.g. -10, -10, -3)
THEN the system raises no error and each silently means "none" via an empty `range` or a
never-satisfied `while` condition.

1.16 WHEN `initial_energy` is zero or negative (e.g. -1.0) THEN the system raises no error and every
organism dies on its first tick, producing an empty-world generation that looks like a legitimate result.

1.17 WHEN `seed_stride` is 0 THEN the system raises no error and `experiments/config.py::resolve_config`
maps every trial seed to `seed_base = 0` (`world = replace(world, seed_base=seed * world.seed_stride)`),
so all trials of a condition run the identical world while still reporting distinct seeds — a silent
reproducibility and independence failure.

#### Area 4 — Experiment condition and inheritance validity

1.18 WHEN an experiment file declares two conditions with the same `name` THEN the system accepts it,
`ExperimentConfig.condition(name)` returns only the first match so the second is unreachable, and
`run_condition` writes both to the same output directory, the later overwriting the earlier.

1.19 WHEN an experiment declares zero control conditions, or more than one THEN the system accepts it,
while `experiments/report.py` and `analysis.compare_conditions` key off a condition literally named
`"control"`, so comparisons silently degrade or silently pick one of several controls.

1.20 WHEN a caller requests a condition name that does not exist (`run_experiment(..., ["typo-name"])`)
THEN the system returns `{}` and the CLI prints "0 condition(s)" and exits 0 — a silent no-op that is
indistinguishable from success.

1.21 WHEN a condition value has a type the target field cannot accept (e.g. `available_space` set to
`"twenty"`) THEN the system accepts it through `resolve_config`, producing `EnvironmentConfig.width ==
'twenty'`, which fails later inside `World` far from the offending file.

1.22 WHEN an experiment file's `extends` target itself only declares `extends` (a nested
child → mid → grandparent chain) THEN the system raises `KeyError: 'base'` at
`experiments/config.py:152`, because `parent["base"]` is read unconditionally and only one level of
inheritance is actually supported.

1.23 WHEN an experiment file declares an explicitly empty `seeds` list THEN the system silently
overrides it with the parent's seeds, because `raw.get("seeds") or (...)` treats `[]` as absent.

1.24 WHEN an experiment file omits the `conditions` key THEN the system raises a bare
`KeyError: 'conditions'` that names neither the file nor the expected schema.

#### Area 5 — Fitness validity and offspring allocation

1.25 WHEN any genome's fitness is NaN THEN the system raises `ValueError: cannot convert float NaN to
integer` at `neat/speciation.py:236`, from `int(count)` applied to a NaN proportion, naming neither the
genome nor the evaluator that produced the value.

1.26 WHEN any genome's fitness is positive infinity THEN the system raises that same
`ValueError: cannot convert float NaN to integer`, because `inf / inf` is NaN — a message that
actively misleads, reporting NaN when the caller supplied infinity.

1.27 WHEN every genome's fitness is negative infinity THEN the system raises no error at all: the
adjusted total is `-inf <= 0`, so the documented uniform-split fallback absorbs it and `-inf` silently
becomes `best_fitness` and `statistics[-1]["best_fitness"]`.

1.28 WHEN a single NaN fitness appears among otherwise finite values THEN the system raises no error and
the NaN genome silently loses every `max()` comparison, so it is invisible to best-genome tracking and
to representative selection.

1.29 WHEN adjusted fitness is mixed-sign with a positive total THEN the system produces allocations that
break the population-size invariant: `allocate_offspring(10)` with species adjusted sums `[3.0, -1.0]`
returns `{1: 15, 2: -5}` (species 1 exceeds the entire target; species 2 gets a negative budget), and
sums `[10.0, -4.0, -4.0]` return `{1: 50, 2: -20, 3: -20}`.

1.30 WHEN such an allocation is executed end to end THEN the resulting population exceeds
`population_size`: 12 genomes in two species (compatibility threshold 0.5), wired genomes scoring +5.0
and unwired -2.0, produced generation-1 `species offspring = [20, -8]` and `len(population) == 20`
against `population_size == 12`, because `_next_generation` skips budgets `<= 0` and
`_guarantee_champion` only tops up (`while len < population_size`) and never trims. This contradicts
`neat/population.py`'s own module docstring ("population size is constant") and the intent of
`tests/test_population.py::test_population_size_constant`, which passes today only because it uses
nonnegative fitness.

### Expected Behavior (Correct)

Each clause below corresponds to the Current Behavior clause with the same index.

#### Area 1 — NEAT ↔ world neural-interface contract

2.1 WHEN a population's genome interface input count does not equal the world's observation size THEN
the system SHALL raise an error before the world is built and before any organism is placed, naming the
expected observation size, the received input count, and the interface configuration responsible.

2.2 WHEN a genome interface declares zero outputs THEN the system SHALL raise that same pre-simulation
interface error naming the expected action size and the received output count, rather than failing
inside `max()` mid-tick.

2.3 WHEN a genome interface declares more outputs than the world's action size THEN the system SHALL
raise the pre-simulation interface error, so no run can select an action index outside the valid range.

2.4 WHEN a genome interface declares fewer outputs than the world's action size THEN the system SHALL
raise the pre-simulation interface error, so no run can silently render actions unreachable.

2.5 WHEN the neural interface contract is expressed THEN the system SHALL enforce it from the declared
constants rather than restating them in comments, and the stale interface comment in `neat/genome.py`
SHALL be corrected to describe the generic default it actually provides.

#### Area 2 — Population ↔ grid capacity

2.6 WHEN `population_size` exceeds grid capacity `width * height` THEN the system SHALL raise an error
before any organism is constructed, naming the requested founder count and the available capacity with
its `width`/`height` derivation.

2.7 WHEN `width` or `height` is less than 1 THEN the system SHALL raise a configuration error naming the
offending field and its received value, at `EnvironmentConfig` construction.

#### Area 3 — Environment numeric field validity

2.8 WHEN `density_radius` is less than 1 THEN the system SHALL raise a configuration error at
`EnvironmentConfig` construction naming the field and value, since `radius >= 1` is required for a
nonzero density normalizer.

2.9 WHEN `density_radius` is negative THEN the system SHALL reject it under the same `>= 1` rule rather
than producing observations that are structurally always 0.0.

2.10 WHEN `max_energy` is not strictly positive THEN the system SHALL raise a configuration error naming
the field and value, since `observe()` divides by it.

2.11 WHEN `repro_fraction` is greater than 1.0 THEN the system SHALL raise a configuration error, since
a child cannot receive more energy than its parent holds without creating energy from nothing.

2.12 WHEN `repro_fraction` is negative THEN the system SHALL raise a configuration error, since negative
child energy and parental energy gain both violate the same conservation invariant. Together with 2.11
the accepted domain SHALL be `0.0 <= repro_fraction <= 1.0`.

2.13 WHEN `metabolism` is negative THEN the system SHALL raise a configuration error, since a negative
drain inverts the energy model: nothing dies and reproduction is unbounded.

2.14 WHEN `ticks` is less than 1 THEN the system SHALL raise a configuration error naming the field and
value, rather than reporting a generation that simulated nothing.

2.15 WHEN `initial_food`, `food_target`, or `food_regrowth_per_tick` is negative THEN the system SHALL
raise a configuration error naming the offending field and value; zero SHALL remain valid, since zero is
a meaningful "no food" setting whereas a negative value is an unexpressible quantity.

2.16 WHEN `initial_energy` is not strictly positive THEN the system SHALL raise a configuration error,
since an organism created at or below the death threshold cannot participate in the simulation it was
created for.

2.17 WHEN `seed_stride` is less than 1 THEN the system SHALL raise a configuration error, since a stride
of 0 collapses every trial seed onto one world while reporting distinct seeds, and a negative stride
serves no documented purpose.

#### Area 4 — Experiment condition and inheritance validity

2.18 WHEN an experiment declares two conditions with the same name THEN the system SHALL raise a
validation error naming the duplicated name, at `ExperimentConfig` construction.

2.19 WHEN an experiment declares zero controls, or more than one control THEN the system SHALL raise a
validation error stating how many controls were found and that exactly one is required, because reporting
and comparison both depend on a single identifiable control.

2.20 WHEN a caller requests a condition name that the experiment does not define THEN the system SHALL
raise an error naming the unknown condition and listing the available names, instead of returning an
empty result with a success exit code.

2.21 WHEN a condition value cannot be applied to the fields its parameter targets THEN the system SHALL
raise a validation error naming the condition, the parameter, the received value, and the expected type,
at load or resolution time rather than inside `World`.

2.22 WHEN an experiment file's `extends` chain is nested more than one level, or its `extends` target
provides no `base` THEN the system SHALL raise an error stating that a single level of `extends` is
supported and naming the files involved.

2.23 WHEN an experiment file declares an explicitly empty `seeds` list THEN the system SHALL either
honour it as an explicit choice or reject it as an empty seed set, and SHALL NOT silently substitute the
parent's seeds; the distinction between "absent" and "explicitly empty" SHALL be preserved.

2.24 WHEN an experiment file omits the `conditions` key THEN the system SHALL raise an error naming the
missing key and the file path.

#### Area 5 — Fitness validity and offspring allocation

2.25 WHEN any genome's fitness is NaN THEN the system SHALL raise an error before that value reaches
speciation or allocation, naming the non-finite value and the genome or population position that carries
it.

2.26 WHEN any genome's fitness is positive infinity THEN the system SHALL raise an error that reports
infinity, not NaN, so the message identifies the value the caller actually supplied.

2.27 WHEN any genome's fitness is negative infinity THEN the system SHALL raise the same non-finite
fitness error, so no non-finite value can reach selection, `best_fitness`, or recorded statistics.

2.28 WHEN a single NaN appears among finite fitness values THEN the system SHALL raise the same
non-finite fitness error rather than letting the value pass silently through `max()` comparisons.

2.29 WHEN adjusted fitness is mixed-sign THEN offspring allocation SHALL produce only nonnegative
per-species budgets that sum exactly to `population_size`, consistent with the engine's existing
established semantic that negative fitness confers no selective advantage — `Population._select_parent`
already floors weights at zero (`weights = [max(g.fitness, 0.0) ...]`) and falls back to uniform choice
when the total is `<= 0`. Allocation SHALL be made consistent with that rule rather than banning
negative fitness.

2.30 WHEN a generation completes with any finite fitness values, of any sign THEN `len(self.population)`
SHALL equal `population_size`, upholding the invariant stated in `neat/population.py`'s module
docstring.

### Unchanged Behavior (Regression Prevention)

3.1 WHEN a genome interface uses custom node ids whose counts match the world contract (9 inputs,
4 actions) THEN the system SHALL CONTINUE TO accept it; only the COUNTS are contractual, and ids need
not be `0..8` and `10..13`.

3.2 WHEN the NEAT engine is used without any world THEN it SHALL CONTINUE TO be free of any dependency
on `world`, as asserted by `tests/test_population.py::test_engine_has_no_environment_dependencies`.

3.3 WHEN benchmark workflows run with valid configurations THEN they SHALL CONTINUE TO behave exactly as
today, as covered by `tests/test_benchmarks.py`.

3.4 WHEN `population_size` exactly equals grid capacity (9 founders on a 3×3 grid) THEN the system SHALL
CONTINUE TO place all founders successfully; the defect boundary is strictly
`population_size > width * height`.

3.5 WHEN `initial_food` exceeds remaining grid capacity THEN the system SHALL CONTINUE TO place as much
food as fits without raising, preserving the existing `if cell is not None` guard at
`world/simulation.py:50-52` and the behavior covered by `tests/test_world.py::test_food_regeneration`.

3.6 WHEN in-world reproduction pressure fills the grid THEN the system SHALL CONTINUE TO treat it as a
non-error: `Organism._try_reproduce` returns `None` when `world.adjacent_empty` is empty, and
`World.regenerate_food` returns when `random_empty_cell()` is `None`. Only founder placement is
defective.

3.7 WHEN a valid world simulation runs THEN observation semantics, movement, eating, metabolism, death,
reproduction, and food regeneration SHALL CONTINUE TO behave identically, as covered by
`tests/test_world.py` (`test_observe_vector`, `test_movement_steps_and_blocked_by_wall`,
`test_eat_action_consumes_food`, `test_metabolism_drains_energy`, `test_energy_capped_at_max`,
`test_death_at_zero_energy`, `test_reproduction_split`, `test_boundary_observation_at_wall`).

3.8 WHEN a seeded run is repeated THEN it SHALL CONTINUE TO produce identical results, as covered by
`tests/test_world.py::test_run_generation_is_deterministic`,
`tests/test_world.py::test_world_seeds_distinct_across_trials_and_generations`,
`tests/test_population.py::test_seeded_run_is_deterministic`, and
`tests/test_experiments.py::test_run_is_deterministic`.

3.9 WHEN all genomes have zero or all-negative fitness THEN offspring allocation SHALL CONTINUE TO use
the documented uniform-split fallback and preserve population size, as covered by
`tests/test_speciation.py::test_all_zero_fitness_falls_back_to_uniform` and
`test_zero_fitness_species_gets_zero_offspring`. Negative fitness SHALL NOT be banned.

3.10 WHEN adjusted fitness sums are nonnegative THEN proportional allocation and largest-remainder
rounding SHALL CONTINUE TO produce today's exact counts, as covered by
`tests/test_speciation.py::test_offspring_allocation_proportional` and
`test_offspring_allocation_largest_remainder`.

3.11 WHEN parents are selected THEN `Population._select_parent` SHALL CONTINUE TO floor weights at zero
and fall back to uniform choice on a non-positive total; this existing semantic is the reference the
allocation fix aligns with, not something to change.

3.12 WHEN experiment results are written THEN the record schema `RECORD_FIELDS` SHALL CONTINUE TO be
exactly as today, as covered by `tests/test_experiments.py::test_run_trial_records_expected_fields` and
`test_run_condition_writes_machine_readable_files`.

3.13 WHEN a generation replay is written THEN the `clage-generation-replay` v1 schema SHALL CONTINUE TO
be unchanged, as covered by `tests/test_visual.py`.

3.14 WHEN configuration is serialized THEN `ResolvedConfig.to_dict` and `asdict(EnvironmentConfig)` SHALL
CONTINUE TO produce the same shape and keys, as covered by
`tests/test_experiments.py::test_config_round_trip`.

3.15 WHEN the shipped experiment configs are loaded THEN they SHALL CONTINUE TO load and validate, and an
unknown `parameter` SHALL CONTINUE TO raise `ValueError`, as covered by
`tests/test_experiments.py::test_all_shipped_configs_load_and_validate` and
`test_unknown_parameter_rejected`. The existing `ExperimentConfig._validate` parameter check is correct
and SHALL be kept.

3.16 WHEN an `extends` target file does not exist THEN the system SHALL CONTINUE TO raise
`FileNotFoundError`; that message is already clear and SHALL NOT be gold-plated.

3.17 WHEN `behavior_window` is zero or negative THEN `diversity/metrics.py::_clip_window` SHALL CONTINUE
TO treat it as "no clipping", as covered by `tests/test_diversity.py::test_per_genome_metrics_pooling_and_window`.

3.18 WHEN the public API is used THEN existing public function signatures and the existing CLI command
structure SHALL CONTINUE TO work unchanged.

3.19 WHEN the packages are inspected THEN the separation between `neat`, `world`, `experiments`,
`benchmarks`, `diversity`, and `visual` SHALL CONTINUE TO hold.

3.20 WHEN the full suite is run (`python3 -m pytest -q -p no:cacheprovider -o addopts=''`) THEN all 198
existing tests SHALL CONTINUE TO pass, unless a specific assertion is explicitly identified as encoding
incorrect behavior and that identification is recorded.

### Investigated and Rejected as Non-Defects

These were probed against HEAD and are NOT defects. Implementation MUST NOT add restrictions for them.

- **Negative `behavior_window`** — `diversity/metrics.py::_clip_window` is
  `trace[:window] if window and window > 0 else trace`. "Less than or equal to zero means no clipping" is
  documented, intentional semantics, so a negative window means "unlimited", exactly like 0. Do not ban
  it. Documenting the semantic is the most that is warranted.
- **`max_energy < initial_energy`** — no crash and no invariant violation; energy is simply capped on the
  first `_consume`. Not obviously invalid. Do not ban.
- **`initial_food` exceeding grid capacity** — already handled correctly by the existing `None` guard at
  `world/simulation.py:50-52`. Not a defect (see 3.5).
- **All-negative fitness** — already safe: the adjusted total is `<= 0`, which triggers the documented
  uniform-split fallback and preserves population size. Do not ban negative fitness (see 3.9).
- **`seed_stride` "unused"** — an earlier audit called this field unused; that is WRONG at HEAD.
  `world_rng_seed` no longer uses it (it hashes `(seed_base, generation)`), but
  `experiments/config.py::resolve_config` does: `world = replace(world, seed_base=seed * world.seed_stride)`.
  The real defect is therefore 1.17/2.17 (`seed_stride = 0` silently collapsing all trials), not
  dead-field removal.
- **`PARAMETERS` duplicate aliases** — `food_abundance` and `resource_scarcity` map to the identical field
  pair (`world.initial_food`, `world.food_target`). Noted for documentation only; changing it is an
  experiment-format decision and is out of scope for this bugfix.

### Intended Consequences

This milestone deliberately converts silent acceptance into failure. Configurations that previously ran
without complaint WILL now raise:

- Interface mismatches between genome input/output counts and the world contract (1.1–1.4).
- Founder populations larger than the grid, and non-positive grid dimensions (1.6, 1.7).
- Out-of-range probabilities and fractions, negative metabolism, negative food quantities, non-positive
  `ticks`, `max_energy`, `initial_energy`, `density_radius`, and `seed_stride` (1.8–1.17).
- Malformed experiment files: duplicate condition names, wrong control counts, unknown requested
  conditions, wrongly typed values, nested `extends`, missing `conditions` (1.18–1.24).
- Non-finite fitness values reaching the engine (1.25–1.28).

This is the point of the milestone, but it is a genuine behavior change for any caller that relied on the
silent path — including any external script that passed a negative or zero value and accepted whatever
came out. That change must be stated in release notes rather than treated as invisible.

### Out of Scope

Explicitly not part of this bugfix, and not to be touched opportunistically:

- Performance optimization of any kind.
- Replay scalability.
- Visualization bugs.
- mypy cleanup.
- pytest-cov / tooling repair. `pyproject.toml`'s default `addopts` references the uninstalled
  `pytest-cov`; leave `pyproject.toml` alone.
- CI setup.
- Refactoring duplicated aggregation logic.
- Checkpoint/resume.
- Recurrent networks.
- Behavioral-metric changes and benchmark evaluation-state changes — both already delivered in the
  previous milestone.
- Redesigning the experiment JSON format.
- Any other opportunistic cleanup.

## Bug Conditions and Properties

`F` is the original (unfixed) behavior, `F'` the fixed behavior. Each area has its own bug condition
because each has its own owner.

### Area 1 — Neural interface contract

```pascal
FUNCTION isBugCondition_interface(X)
  INPUT: X = (input_ids, output_ids, observation_size, action_size)
  OUTPUT: boolean

  RETURN LENGTH(X.input_ids) <> X.observation_size
      OR LENGTH(X.output_ids) <> X.action_size
END FUNCTION
```

```pascal
// Fix checking
FOR ALL X WHERE isBugCondition_interface(X) DO
  ASSERT raises_error(start_generation'(X))
  ASSERT error_names(expected_size) AND error_names(received_size)
  ASSERT no_world_was_built(X) AND no_organism_was_placed(X)
END FOR

// Preservation checking
FOR ALL X WHERE NOT isBugCondition_interface(X) DO
  ASSERT F(X) = F'(X)   // including custom ids with correct counts
END FOR
```

### Area 2 — Grid capacity for founder placement

```pascal
FUNCTION isBugCondition_capacity(X)
  INPUT: X = (config, population_size)
  OUTPUT: boolean

  RETURN X.config.width < 1
      OR X.config.height < 1
      OR X.population_size > X.config.width * X.config.height
END FUNCTION
```

```pascal
// Fix checking
FOR ALL X WHERE isBugCondition_capacity(X) DO
  ASSERT raises_error(run_generation'(X))
  ASSERT error_names(population_size) AND error_names(capacity)
  ASSERT NOT raises(TypeError)
  ASSERT no_organism_was_constructed(X)
END FOR

// Preservation checking
FOR ALL X WHERE NOT isBugCondition_capacity(X) DO
  ASSERT F(X) = F'(X)   // including population_size = width * height exactly,
                        // oversized initial_food, and saturated in-world reproduction
END FOR
```

### Area 3 — Environment numeric field validity

```pascal
FUNCTION isBugCondition_envconfig(X)
  INPUT: X of type EnvironmentConfig
  OUTPUT: boolean

  RETURN X.width < 1 OR X.height < 1
      OR X.ticks < 1
      OR X.density_radius < 1
      OR X.max_energy <= 0.0
      OR X.initial_energy <= 0.0
      OR X.metabolism < 0.0
      OR X.repro_fraction < 0.0 OR X.repro_fraction > 1.0
      OR X.initial_food < 0 OR X.food_target < 0
      OR X.food_regrowth_per_tick < 0
      OR X.seed_stride < 1
END FUNCTION
```

```pascal
// Fix checking
FOR ALL X WHERE isBugCondition_envconfig(X) DO
  ASSERT raises_error(construct_EnvironmentConfig'(X))
  ASSERT error_names(offending_field) AND error_includes(received_value)
END FOR

// Preservation checking
FOR ALL X WHERE NOT isBugCondition_envconfig(X) DO
  ASSERT construct_EnvironmentConfig'(X) succeeds
  ASSERT asdict(F'(X)) = asdict(F(X))          // serialized shape unchanged
  ASSERT simulate'(X) = simulate(X)            // includes behavior_window <= 0,
                                               // max_energy < initial_energy,
                                               // zero-valued food fields
END FOR
```

### Area 4 — Experiment condition and inheritance validity

```pascal
FUNCTION isBugCondition_experiment(X)
  INPUT: X = raw experiment document (plus optional requested condition names)
  OUTPUT: boolean

  RETURN has_duplicate_condition_names(X)
      OR count_controls(X) <> 1
      OR requested_name_not_defined(X)
      OR condition_value_type_invalid_for_parameter(X)
      OR extends_chain_depth(X) > 1
      OR extends_target_has_no_base(X)
      OR missing_key(X, "conditions")
      OR seeds_explicitly_empty(X)
END FUNCTION
```

```pascal
// Fix checking
FOR ALL X WHERE isBugCondition_experiment(X) DO
  ASSERT raises_error(load_or_run'(X))
  ASSERT error_names(offending_condition_or_key) AND error_names(file_or_available_names)
  ASSERT NOT silently_returns_empty_result(X)
END FOR

// Preservation checking
FOR ALL X WHERE NOT isBugCondition_experiment(X) DO
  ASSERT F(X) = F'(X)                          // shipped configs load identically
  ASSERT to_dict(F'(X)) = to_dict(F(X))        // config round trip unchanged
  ASSERT RECORD_FIELDS unchanged
  ASSERT unknown_parameter still raises ValueError
  ASSERT missing extends target still raises FileNotFoundError
END FOR
```

### Area 5 — Fitness validity and allocation arithmetic

```pascal
FUNCTION isBugCondition_fitness(X)
  INPUT: X = the evaluated population (fitness values) for one generation
  OUTPUT: boolean

  RETURN EXISTS g IN X WHERE NOT is_finite(g.fitness)
      OR (total_adjusted(X) > 0.0 AND EXISTS s IN species(X)
            WHERE s.adjusted_fitness_sum < 0.0)
END FUNCTION
```

```pascal
// Fix checking - part A: non-finite fitness
FOR ALL X WHERE EXISTS g IN X WITH NOT is_finite(g.fitness) DO
  ASSERT raises_error(next_generation'(X))
  ASSERT error_reports_actual_value(NaN vs +inf vs -inf)
  ASSERT no_non_finite_value_reached(selection, best_fitness, statistics)
END FOR

// Fix checking - part B: finite mixed-sign fitness
FOR ALL X WHERE all_finite(X) DO
  allocation := allocate_offspring'(X, population_size)
  ASSERT FOR ALL sid IN allocation: allocation[sid] >= 0
  ASSERT SUM(allocation.values()) = population_size
  next := next_generation'(X)
  ASSERT LENGTH(next.population) = population_size
END FOR

// Preservation checking
FOR ALL X WHERE all_finite(X) AND all_adjusted_sums_nonnegative(X) DO
  ASSERT allocate_offspring'(X, n) = allocate_offspring(X, n)
  ASSERT F(X) = F'(X)   // includes all-zero and all-negative uniform fallback,
                        // largest-remainder rounding, seeded determinism
END FOR
```
