# Boundary Configuration Validation — Bugfix Design

## Overview

Clage accepts invalid configurations and incompatible subsystem combinations, then either crashes deep
inside a running simulation with an unactionable message or silently produces an invalid run. This
design makes those inputs fail early, deterministically, and with messages that name the offending
field and value.

The fix is **five separate validators owned by five separate modules**, plus one flagged addition.
`bugfix.md` states the five concerns MUST NOT be collapsed into a shared validator or a new unifying
abstraction, and this design keeps them apart. Each validator is a handful of lines placed at the
boundary that owns the rule; there is no new module, no new abstraction, and no shared "validate"
framework. The reason is not stylistic: each rule needs a different set of facts, available at a
different time, in a different package.

| # | Area | Owner (file / function) | Validation TIME | Why this owner |
|---|------|-------------------------|-----------------|----------------|
| 1 | Neural interface contract | `world/simulation.py::run_generation` (entry) | Execution entry, before `World(...)` and before any `Organism` | Only `world` knows `OBSERVATION_SIZE` / `ACTION_SIZE`; needs the genomes, which only exist at call time |
| 2 | Grid capacity for founders | `world/simulation.py::run_generation` (entry, right after Area 1) | Execution entry, before any `Organism` | Combines an `EnvironmentConfig` with a population size; the config alone cannot know the population |
| 3 | Environment numeric field validity | `world/config.py::EnvironmentConfig.__post_init__` | Construction (and again on every `dataclasses.replace`) | Every rule is a property of a single field of that dataclass |
| 4 | Experiment condition / inheritance validity | `experiments/config.py::ExperimentConfig._validate`, `load_experiment`, `experiments/run.py::run_experiment` | Load time, construction time, and run-entry (respectively) | Only the loader has the file path, the raw document, and the requested-condition list |
| 5 | Fitness validity + allocation arithmetic | `neat/population.py::Population._evaluate`, `neat/speciation.py::Speciation.allocate_offspring` | Immediately after evaluation; and inside allocation | Internal `neat` engine invariants; `neat` must gain no `world` dependency |
| 6 | **ADDITION (flagged)** engine config validity | `neat/population.py::Population.__init__`, `MutationConfig.__post_init__`, `SpeciationConfig.__post_init__` | Construction | Same reason as Area 3, for the engine's own dataclasses |

Area 6 is **not** enumerated in `bugfix.md`. It is marked as an addition throughout so it can be
approved or trimmed without disturbing Areas 1–5. See "Area 6 — ADDITION" below.

### Measured compatibility facts

These were measured, not assumed. They are recorded here because they are the whole basis for the claim
that this milestone can be shipped without editing a single existing test assertion.

Verified by the user before this design (stated as measured; not re-run here):

- `EnvironmentConfig()` defaults satisfy every numeric rule proposed in Area 3.
- All six shipped experiment configs satisfy the proposed rules, have exactly **one** control each, have
  no duplicate condition names, and every condition VALUE still satisfies the rules after resolution.
- All shipped conditions resolve to interface 9 -> 4 and to `population_size <= width * height`. The
  smallest margin is the `population_density` / `density_high` condition: 50 founders on a 20x20 = 400
  cell grid.
- Every `EnvironmentConfig(...)` construction in the test suite uses valid values (`width`/`height` 4–6,
  `ticks` 1–5, `seed_stride=1000`, positive energies). No existing test would be rejected.
- Every world-driving test uses `INPUT_IDS = list(range(9))` and `OUTPUT_IDS = [10, 11, 12, 13]`, so the
  interface rule breaks nothing.

Verified while writing this design:

- `python3 -m pytest -q -p no:cacheprovider -o addopts=''` at HEAD `ea8a9c2`: **198 passed**.
- `EnvironmentConfig(width='twenty')` and `EnvironmentConfig(width=20.5)` both survive construction and
  fail later in `World.__init__` with `TypeError: 'str'/'float' object cannot be interpreted as an
  integer`, from `range(self.width)` in `world/grid.py`. So today's observed failure type for a wrongly
  typed dimension is already `TypeError` — see the exception-type policy.
- `neat/population.py` at HEAD contains **zero** occurrences of the substring `main`. See the
  "Substring hazard" note; this constrains the wording of new code in that file.
- `tests/test_experiments.py::test_unknown_parameter_rejected` builds an `ExperimentConfig` with a
  single non-control condition, i.e. **zero controls**. It only asserts `pytest.raises(ValueError)`, so
  it keeps passing either way, but check ordering in `_validate` matters if it is to keep failing for
  its original reason. See Area 4.
- `tests/test_benchmarks.py::test_unevaluated_offspring_cannot_declare_success` scores every genome
  `-1.0`. All-negative adjusted sums keep taking the documented uniform fallback under the Area 5 fix,
  so that test is unaffected. Its scope guard ("must not turn into a fix for speciation's
  negative-fitness offspring allocation") referred to the *previous* milestone; this milestone owns that
  fix by requirement 2.29, and the test's own assertions do not constrain it.

**Conclusion recorded:** NO existing test assertion needs editing, and requirement 3.20 (all 198 tests
keep passing) is achievable.

## Glossary

- **Bug_Condition (C)** — the predicate identifying inputs that trigger a defect. There are five (six
  with the addition), one per owner: `isBugCondition_interface`, `isBugCondition_capacity`,
  `isBugCondition_envconfig`, `isBugCondition_experiment`, `isBugCondition_fitness`, and
  `isBugCondition_engineconfig`.
- **Property (P)** — the required behavior for inputs satisfying C. Here it is almost always "raise a
  specific error, early, naming the offending field and value, before any side effect".
- **Preservation** — for inputs where C is false, the fixed code must behave exactly as HEAD does,
  including byte-identical seeded output.
- **F / F'** — the original (unfixed) and fixed behavior.
- **Validation time** — the moment a rule runs: *construction* (dataclass `__post_init__`),
  *resolution* (`resolve_config`), *evaluator creation* (`make_evaluator`), or *execution entry*
  (`run_generation`, `Population._evaluate`, `run_experiment`).
- **`OBSERVATION_SIZE` / `ACTION_SIZE`** — `world/config.py` constants (9 and 4). They are the neural
  interface contract and they stay in `world`.
- **Interface counts** — `len(genome.inputs)` and `len(genome.outputs)`. Only the COUNTS are
  contractual. Node ids are free (requirement 3.1).
- **Grid capacity** — `config.width * config.height`, the number of cells, hence the maximum number of
  simultaneously placeable founders.
- **`adjusted_fitness_sum`** — per-species sum of `fitness / species_size`, computed by
  `Speciation.share_fitness`, consumed by `allocate_offspring`.
- **Largest-remainder rounding** — allocation takes `int(raw)` per species then hands the shortfall to
  the species with the largest fractional parts, tie-broken by lowest species id.
- **Clipping (the Area 5 fix)** — `weight = max(adjusted_fitness_sum, 0.0)` before proportional
  allocation; a no-op whenever all sums are already nonnegative.
- **Non-finite** — `math.isfinite(x)` is false: `nan`, `+inf`, `-inf`.

## Bug Details

### Area 1 — Neural interface contract

`world/config.py` declares the interface as constants, `world/organism.py` imports them and never uses
them, and nothing reconciles the genome interface with the world before a run starts. `neat/genome.py`
defaults to 10 inputs, and its comment claims "10 sensors, 4 actions" while `Organism.observe` returns 9
values.

**Formal specification:**

```pascal
FUNCTION isBugCondition_interface(X)
  INPUT: X = (population, observation_size, action_size)
  OUTPUT: boolean

  RETURN EXISTS g IN X.population WHERE
           LENGTH(g.inputs) <> X.observation_size
        OR LENGTH(g.outputs) <> X.action_size
END FUNCTION
```

**Examples (measured at HEAD, from bugfix.md 1.1–1.5):**

- 10 inputs (`Genome.minimal()` default) against a 9-value observation:
  `ValueError: expected 10 inputs, got 9` from `neat/phenotype.py:109` in `activate()` — mid-run, once
  per organism per tick, after the world was built and organisms placed. Expected: one error at
  generation entry, before any of that.
- `output_ids=[]`: `ValueError: max() iterable argument is empty` from `world/organism.py:96` in
  `act()`. Names neither the interface nor the field. Expected: the interface error.
- 6 outputs: no error at all. `action = max(range(len(outputs)), ...)` can return 4 or 5,
  `_apply_action` matches no branch, and the tick silently does nothing. Expected: the interface error.
- 2 outputs: no error. `Action.TURN_RIGHT` (2) and `Action.EAT` (3) are unreachable for the whole run.
  Expected: the interface error.
- Edge case that must keep working: custom ids with correct counts, e.g.
  `input_ids=[100..108]`, `output_ids=[200, 201, 202, 203]` — 9 and 4, so valid (requirement 3.1).

### Area 2 — Grid capacity for founder placement

`run_generation` unpacks `world.random_empty_cell()` directly (`Organism(genome, *cell, config)`) while
that method is documented — and typed — `Optional[Tuple[int, int]]`.

```pascal
FUNCTION isBugCondition_capacity(X)
  INPUT: X = (population, config)
  OUTPUT: boolean

  RETURN LENGTH(X.population) > X.config.width * X.config.height
END FUNCTION
```

`width < 1` / `height < 1` were part of this bug condition in `bugfix.md` (1.7). They are moved to Area 3
(requirement 2.7 explicitly assigns them to `EnvironmentConfig` construction). By the time
`run_generation` runs, `width >= 1` and `height >= 1` are already guaranteed, so Area 2 only needs the
capacity comparison. The two areas together still cover the full `isBugCondition_capacity` from
`bugfix.md`, just at two different times.

**Examples (measured at HEAD):**

- 10 founders on a 3x3 grid: `TypeError: Value after * must be an iterable, not NoneType` at
  `world/simulation.py:44`. Expected: an error naming 10 founders, capacity 9, and `width=3 * height=3`,
  raised before any `Organism` exists.
- `width=0` or `width=-3`: the same `TypeError` at the same line. Expected: an `EnvironmentConfig`
  construction error naming `width` and its value (Area 3), which now happens strictly earlier.
- Edge case that must keep working: 9 founders on a 3x3 grid — exactly capacity — places all founders
  (requirement 3.4). The boundary is strictly `>`.
- Edge case that must keep working: `initial_food` larger than the remaining free cells still places as
  much as fits, because the food loop already guards `if cell is not None` (requirement 3.5).
- Edge case that must keep working: reproduction that saturates the grid is a non-error —
  `Organism._try_reproduce` returns `None` on no adjacent empty cell and `World.regenerate_food` returns
  on `None` (requirement 3.6). Only *founder* placement is defective.

### Area 3 — Environment numeric field validity

`EnvironmentConfig` is a plain dataclass with no validation. Several fields crash downstream; several
silently invalidate the model.

```pascal
FUNCTION isBugCondition_envconfig(X)
  INPUT: X of type EnvironmentConfig
  OUTPUT: boolean

  RETURN NOT is_integer(X.width)  OR X.width  < 1
      OR NOT is_integer(X.height) OR X.height < 1
      OR NOT is_integer(X.ticks)  OR X.ticks  < 1
      OR NOT is_integer(X.density_radius) OR X.density_radius < 1
      OR NOT is_number(X.max_energy)     OR X.max_energy     <= 0.0
      OR NOT is_number(X.initial_energy) OR X.initial_energy <= 0.0
      OR NOT is_number(X.metabolism)     OR X.metabolism     <  0.0
      OR NOT is_number(X.repro_fraction)
      OR X.repro_fraction < 0.0 OR X.repro_fraction > 1.0
      OR NOT is_integer(X.initial_food)           OR X.initial_food           < 0
      OR NOT is_integer(X.food_target)            OR X.food_target            < 0
      OR NOT is_integer(X.food_regrowth_per_tick) OR X.food_regrowth_per_tick < 0
      OR NOT is_integer(X.seed_stride) OR X.seed_stride < 1
      OR NOT is_number(X.food_energy)     // type only, no range rule
      OR NOT is_number(X.repro_threshold) // type only, no range rule
      OR NOT is_integer(X.seed_base)      // type only, no range rule
      OR NOT is_integer(X.behavior_window)// type only, no range rule (see 3.17)
END FUNCTION
```

**Examples (measured at HEAD, from bugfix.md 1.8–1.17):**

- `density_radius=0`: `ZeroDivisionError: division by zero` at `world/grid.py:109`
  (`max_count = (2*0+1)**2 - 1 == 0`).
- `density_radius=-2`: no error; the density loops iterate over nothing while `max_count` is 8, so every
  food-density and organism-density observation is 0.0 for the whole run.
- `max_energy=0.0`: `ZeroDivisionError: float division by zero` at `world/organism.py:77`
  (`self.energy / config.max_energy`).
- `repro_fraction=1.5`: no error; the child gets 1.5x the parent's energy and the parent goes negative —
  energy created from nothing.
- `repro_fraction=-0.5`: no error; negative child energy, and the parent *gains* energy by reproducing.
- `metabolism=-1.0`: no error; organisms gain energy every tick, never die, reproduce explosively
  (24 organisms from 3 founders within 3 ticks).
- `ticks=-5`: no error; `range(-5)` is empty, so a "completed generation" simulated nothing.
- `initial_food=-10`, `food_target=-10`, `food_regrowth_per_tick=-3`: no error; each silently means
  "none" through an empty `range` or a never-satisfied `while`.
- `initial_energy<=0`: no error; every organism dies on its first tick.
- `seed_stride=0`: no error; `resolve_config` maps every trial seed to `seed_base = 0`, so all trials of
  a condition run the identical world while reporting distinct seeds.
- `width='twenty'` (reachable from `experiments`, requirement 1.21/2.21): construction succeeds, then
  `TypeError: 'str' object cannot be interpreted as an integer` inside `World.__init__`.
- `width=20.5`: same `TypeError` — a float is not usable in `range()` either.
- Edge cases that must keep working: `initial_food=0`, `food_target=0`,
  `food_regrowth_per_tick=0` (zero is a meaningful "no food"); `metabolism=0.0`;
  `repro_fraction=0.0` and `1.0`; `max_energy < initial_energy`; `behavior_window <= 0`;
  `seed_base=0`; `repro_threshold=2.0` (used all over the test suite to disable reproduction).

### Area 4 — Experiment condition and inheritance validity

```pascal
FUNCTION isBugCondition_experiment(X)
  INPUT: X = (raw experiment document, file path, optional requested condition names)
  OUTPUT: boolean

  RETURN has_duplicate_condition_names(X)
      OR count_controls(X) <> 1
      OR requested_name_not_defined(X)
      OR condition_value_type_invalid_for_parameter(X)
      OR extends_target_provides_no_base(X)   // includes nested extends
      OR missing_key(X, "conditions")
      OR seeds_resolve_to_empty(X)            // explicitly [] or inherited []
END FUNCTION
```

**Examples (measured at HEAD, from bugfix.md 1.18–1.24):**

- Two conditions named `food_low`: accepted. `ExperimentConfig.condition(name)` returns the first, so
  the second is unreachable, and `run_condition` writes both to the same directory — the later
  overwrites the earlier.
- Zero or two conditions named `control`: accepted, while `experiments/report.py` and
  `analysis.compare_conditions` key off a condition literally named `"control"`.
- `run_experiment(..., ["typo-name"])`: returns `{}`; the CLI prints `0 condition(s)` and exits 0 —
  indistinguishable from success.
- `available_space: "twenty"`: accepted through `resolve_config`, producing
  `EnvironmentConfig.width == 'twenty'`, which fails inside `World` far from the offending file.
- A child whose `extends` target itself only declares `extends`: `KeyError: 'base'` at
  `experiments/config.py:152` (`parent["base"]` is read unconditionally).
- `"seeds": []`: silently replaced by the parent's seeds, because `raw.get("seeds") or (...)` treats
  `[]` as absent.
- No `conditions` key: bare `KeyError: 'conditions'`, naming neither the file nor the schema.
- Edge cases that must keep working: all six shipped configs load and validate unchanged; an unknown
  `parameter` still raises `ValueError`; a missing `extends` target still raises `FileNotFoundError`
  (requirements 3.15, 3.16); `ResolvedConfig.to_dict` and `RECORD_FIELDS` unchanged (3.12, 3.14);
  the `food_abundance` / `resource_scarcity` alias is left exactly as it is.

### Area 5 — Fitness validity and allocation arithmetic

Two defects with one owner.

```pascal
FUNCTION isBugCondition_fitness(X)
  INPUT: X = the evaluated population for one generation
  OUTPUT: boolean

  RETURN EXISTS g IN X WHERE NOT is_finite(g.fitness)          // part A
      OR (total_adjusted(X) > 0.0                              // part B
            AND EXISTS s IN species(X) WHERE s.adjusted_fitness_sum < 0.0)
END FUNCTION
```

**Part A — non-finite fitness (measured at HEAD, 1.25–1.28):**

- Any NaN fitness: `ValueError: cannot convert float NaN to integer` at `neat/speciation.py:236`
  (`int(count)`), naming neither the genome nor the evaluator.
- `+inf` fitness: the *same* `ValueError` about NaN, because `inf / inf` is NaN. The message actively
  misleads about which value the caller supplied.
- All `-inf`: no error at all. The adjusted total is `-inf <= 0.0`, so the documented uniform fallback
  absorbs it and `-inf` silently becomes `best_fitness` and `statistics[-1]["best_fitness"]`.
- One NaN among finite values: no error. NaN loses every `max()` comparison, so it is invisible to
  best-genome tracking and to representative selection.

**Part B — the allocation arithmetic. Written out, because the fix depends on it.**

`Speciation.allocate_offspring(n)` at HEAD:

```
total = SUM(s.adjusted_fitness_sum for s in species)
IF total <= 0.0 THEN uniform split (documented fallback); RETURN
raw[sid]        = adj[sid] / total * n
allocation[sid] = int(raw[sid])
shortfall       = n - SUM(allocation.values())
distribute shortfall by largest fractional part, tie-break lowest species id
```

Every step assumes `adj[sid] >= 0`. When `total > 0` but some `adj[sid] < 0`:

1. `raw[sid] = adj[sid] / total * n` is **negative** for that species.
2. Because the ratios must sum to 1, the positive species absorb more than the whole budget:
   `SUM(raw) = n` holds algebraically, but individual terms are no longer in `[0, n]`.
3. `int()` truncates **toward zero**, not down, so for negative raws it rounds *up*. The
   largest-remainder step then works on fractional parts that can be negative, and the invariant
   "each species is within 1 of its share" no longer holds.
4. The sum is still `n` — but only as an algebraic accident. The individual budgets are nonsense.

Measured values:

| adjusted sums | n | HEAD allocation | sum | valid? |
|---|---|---|---|---|
| `[3.0, -1.0]` | 10 | `{1: 15, 2: -5}` | 10 | no — species 1 exceeds the whole target, species 2 is negative |
| `[10.0, -4.0, -4.0]` | 10 | `{1: 50, 2: -20, 3: -20}` | 10 | no — 5x the target for species 1 |

Arithmetic for the first row: `total = 2.0`; `raw[1] = 3.0/2.0*10 = 15.0`; `raw[2] = -1.0/2.0*10 = -5.0`;
`int()` leaves both exact, shortfall `10 - 10 = 0`, so largest-remainder never gets a chance to correct
anything.

End to end (measured): 12 genomes, two species under compatibility threshold 0.5, wired genomes scoring
`+5.0` and unwired `-2.0`. Adjusted sums are `+5.0` and `-2.0` (fitness sharing divides by species size
then sums, so the sum equals the per-member fitness). `total = 3.0`, `raw = [5/3*12, -2/3*12] = [20.0,
-8.0]`, giving generation-1 `species offspring = [20, -8]` and `len(population) == 20` against
`population_size == 12`. It escapes because:

- `Population._next_generation` does `if budget <= 0: continue`, so the negative budget is skipped
  rather than subtracted, and
- `Population._guarantee_champion` only tops *up* (`while len(next_population) < self.population_size`)
  and never trims.

This contradicts `neat/population.py`'s own module docstring ("population size is constant") and the
intent of `tests/test_population.py::test_population_size_constant`, which passes today only because it
uses nonnegative fitness.

**Edge cases that must keep working:** all-zero and all-negative adjusted sums keep the documented
uniform fallback (3.9); nonnegative sums keep today's exact proportional + largest-remainder counts
(3.10); `Population._select_parent` is not touched (3.11).

### Area 6 — ADDITION (flagged for approval)

Not enumerated in `bugfix.md`. Reproduced by the user, reachable from `experiments` through
`neat.population_size`, `neat.elitism`, `neat.crossover_rate`, `neat.mutation` and `neat.speciation`,
and therefore inside the milestone's "invalid population sizes / probability fields outside [0,1]"
language.

```pascal
FUNCTION isBugCondition_engineconfig(X)
  INPUT: X = (population_size, elitism, crossover_rate, initial_population,
              MutationConfig fields, SpeciationConfig fields)
  OUTPUT: boolean

  RETURN X.population_size = 0            // silently becomes 100
      OR X.population_size < 0            // silently empty population
      OR X.elitism < 0                    // breaks the population-size invariant
      OR X.crossover_rate < 0.0 OR X.crossover_rate > 1.0
      OR any mutation probability NOT IN [0.0, 1.0]
      OR X.stagnation_threshold < 1
      OR (X.initial_population IS NOT NULL AND LENGTH(X.initial_population) = 0)
END FUNCTION
```

**Examples (measured):**

- `population_size=0` silently becomes 100, because `Population.__init__` uses
  `self.population_size = population_size or 100`; `0` is falsy.
- `population_size=-5` yields an empty population with no error (`range(-5)` is empty).
- `elitism=-3` with `population_size=6` produced a population of **12**. Arithmetic:
  `elites = min(-3, 6) = -3`; `ranked[:-3]` is the first 3 of 6 members, so 3 elites are appended; then
  `range(budget - elites) = range(6 - (-3)) = range(9)` adds 9 children. 3 + 9 = 12. This breaks the
  **same** population-size invariant as Area 5, by a different route.
- `crossover_rate=5.0` / `-1.0` silently saturate (`rng.random() < 5.0` is always true, `< -1.0` never).
  `MutationConfig` probabilities out of `[0, 1]` saturate the same way.
- `SpeciationConfig(stagnation_threshold=0)`: `prune_stagnant` extincts every species every generation
  (`stagnation >= 0` is always true), allocation returns `{}`, and on generation 1 `best_genome` is still
  `None` so `_guarantee_champion` returns the empty list. The population collapses to 0 and never
  recovers.
- `initial_population=[]` yields size 0 with no error and no champion (`population_size or len([])` = 0).

**Edge cases that must keep working:** `elitism=0` (used by
`tests/test_population.py::test_crossover_produces_valid_children_from_parent_genes`),
`crossover_rate=1.0`, mutation probabilities of exactly `0.0` and `1.0` (used in several tests), and
`population_size=2` with a 2-genome `initial_population`. All rules must therefore be inclusive at the
documented endpoints.

## Expected Behavior

### Preservation Requirements

**Unchanged behaviors, by area:**

- Area 1: genomes with custom node ids and correct counts keep working (3.1); the NEAT engine keeps
  running with no world at all and gains no `world` import (3.2); benchmark problems with 1–3 inputs and
  1 output keep working, because the interface rule lives in `world` and benchmarks never touch it (3.3).
  `DEFAULT_INPUT_IDS` stays `tuple(range(10))`.
- Area 2: `population_size == width * height` exactly still places every founder (3.4); oversized
  `initial_food` still degrades gracefully (3.5); grid-saturating reproduction stays a non-error (3.6).
- Area 3: `asdict(EnvironmentConfig)` keeps exactly today's keys and shape — **no field is added** (3.14);
  all world mechanics are untouched (3.7); seeded runs stay byte-identical (3.8);
  `behavior_window <= 0` keeps meaning "no clipping" (3.17); `max_energy < initial_energy` stays legal.
- Area 4: the six shipped configs load and validate identically; unknown `parameter` still raises
  `ValueError` and the existing `_validate` check is kept verbatim (3.15); a missing `extends` target
  still raises `FileNotFoundError` and its message is not gold-plated (3.16); `RECORD_FIELDS` (3.12),
  `ResolvedConfig.to_dict` (3.14) and the replay schema (3.13) are untouched; the
  `food_abundance` / `resource_scarcity` alias is left in place; the JSON format is not redesigned.
- Area 5: the uniform fallback for all-zero and all-negative adjusted fitness (3.9); exact proportional
  and largest-remainder counts for nonnegative sums (3.10); `_select_parent`'s existing
  floor-at-zero-then-uniform behavior (3.11); negative fitness stays legal.
- Area 6: `elitism=0`, `crossover_rate` of `0.0`/`1.0`, mutation probabilities of `0.0`/`1.0`, and the
  documented default `population_size=100` when the argument is omitted.
- All areas: public function signatures and the CLI command structure are unchanged (3.18); package
  separation holds (3.19); all 198 tests keep passing (3.20).

**Scope.** Every input where the relevant bug condition is false must be completely unaffected. In
particular:

- No validator consumes randomness, so no seeded stream shifts (see "Determinism").
- No validator changes a return value, a serialized key, or a signature.
- Areas 1 and 2 run once per `run_generation` call, before the world's `random.Random` is even
  constructed.

## Hypothesized Root Cause

1. **Declared-but-unenforced contracts.** `OBSERVATION_SIZE` and `ACTION_SIZE` exist as constants and are
   even imported by `world/organism.py`, but never compared against anything. The contract is
   documentation. The same pattern produced the stale `neat/genome.py` comment ("10 sensors, 4 actions")
   that contradicts the 9-value observation vector.

2. **`Optional` return values consumed as if total.** `World.random_empty_cell()` is correctly typed
   `Optional[...]` and two of its three call sites guard the `None` (food placement, regeneration). The
   founder loop does not. This is a missed case, not a design disagreement — which is why the fix is a
   precondition check rather than a `None` guard: a `None` there means the configuration was wrong, and
   silently placing fewer founders than requested would be a second silent defect.

3. **Plain dataclasses as configuration.** `EnvironmentConfig`, `MutationConfig` and `SpeciationConfig`
   are `@dataclass` declarations with defaults and no `__post_init__`. The defaults are all valid, so the
   absence of validation is invisible until a caller (or an experiment JSON file) supplies something
   else. Every Area 3 and Area 6 defect is an instance of this.

4. **Falsy-versus-absent confusion.** `population_size or 100`, `raw.get("seeds") or (...)`, and
   `conditions or [all names]` all treat a legitimate zero/empty value as "not supplied". Two of the
   enumerated defects (1.23, and the `population_size=0 -> 100` addition) are exactly this idiom.

5. **Arithmetic written for one sign.** `allocate_offspring` is correct proportional allocation with
   largest-remainder rounding **for nonnegative weights**. The `total <= 0` fallback shows the author
   thought about the degenerate case, but the mixed-sign case (positive total, some negative members)
   falls between the two branches. `Population._select_parent` in the same package already solves the
   equivalent problem by flooring at zero, so the engine already has an answer — allocation just does not
   use it.

6. **No validity gate between an injected callback and the engine's math.** `fitness_fn` / `evaluator`
   are caller-supplied, and their output flows straight into statistics, tracking, speciation and
   integer arithmetic. `int(NaN)` is the first place that objects, which is four steps too late and in
   the wrong vocabulary.

7. **Error messages assembled from the wrong frame.** Every misleading message in the report
   (`inf` reported as NaN, `max() iterable argument is empty`, `Value after * must be an iterable`) comes
   from a low-level operation that has lost the context needed to name the field. The fix is to check
   where the context exists, not to improve messages where it does not.

## Correctness Properties

Numbering convention for this multi-owner bugfix: **odd-numbered properties are Bug Conditions**
(exploration tests, must FAIL pre-fix) and **even-numbered properties are their Preservation partners**
(must PASS pre-fix and post-fix). Each area owns one pair, in area order.

Property 1: Bug Condition - Neural interface counts are enforced before a run starts

_For any_ population containing a genome whose input count differs from `OBSERVATION_SIZE` or whose
output count differs from `ACTION_SIZE`, `run_generation` SHALL raise `ValueError` before constructing
the `World` and before constructing any `Organism`, and the message SHALL name the expected count, the
received count, and which side of the interface (`input_ids` or `output_ids`) is at fault.

**Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5**

Property 2: Preservation - Correct interface counts, custom ids, and world-free engine use

_For any_ population whose genomes all have exactly `OBSERVATION_SIZE` inputs and `ACTION_SIZE` outputs —
including genomes using arbitrary node ids rather than `0..8` and `10..13` — and _for any_ use of the
NEAT engine without a world, the fixed code SHALL produce the same result as the original code, and
`neat` SHALL contain no import of `world`.

**Validates: Requirements 3.1, 3.2, 3.3**

Property 3: Bug Condition - Founder population never exceeds grid capacity

_For any_ population whose length exceeds `config.width * config.height`, `run_generation` SHALL raise
`ValueError` — not `TypeError` — before constructing any `Organism`, and the message SHALL name the
requested founder count, the available capacity, and the `width`/`height` derivation of that capacity.

**Validates: Requirements 2.6, 2.7**

Property 4: Preservation - Exact-capacity placement, oversized food, saturated reproduction

_For any_ population whose length is at most `config.width * config.height` — including exactly equal —
the fixed code SHALL behave identically to the original, and oversized `initial_food` SHALL continue to
place as much food as fits without raising, and grid-saturating in-world reproduction SHALL continue to
be a non-error.

**Validates: Requirements 3.4, 3.5, 3.6, 3.7**

Property 5: Bug Condition - Invalid environment fields are rejected at construction

_For any_ field assignment satisfying `isBugCondition_envconfig`, constructing an `EnvironmentConfig`
SHALL raise — `ValueError` for an out-of-range numeric value, `TypeError` for a value of the wrong type —
with a message naming the field, the received value, and the rule, and the same SHALL hold for every
`dataclasses.replace` of a config, including the per-trial seeded replace in `resolve_config`.

**Validates: Requirements 2.7, 2.8, 2.9, 2.10, 2.11, 2.12, 2.13, 2.14, 2.15, 2.16, 2.17, 2.21**

Property 6: Preservation - Valid environment configs, serialized shape, and world mechanics

_For any_ `EnvironmentConfig` satisfying every rule — including `EnvironmentConfig()` defaults, zero-valued
food fields, `metabolism=0.0`, `repro_fraction` of `0.0` or `1.0`, `max_energy < initial_energy`, and
`behavior_window <= 0` — construction SHALL succeed, `asdict()` SHALL produce exactly today's keys and
values, and every simulation result including seeded byte-for-byte output SHALL be unchanged.

**Validates: Requirements 3.7, 3.8, 3.14, 3.17**

Property 7: Bug Condition - Malformed experiment documents and unknown conditions are rejected

_For any_ experiment document or request satisfying `isBugCondition_experiment`, loading, constructing or
running it SHALL raise an error that names the offending condition, key or name together with the file
path or the list of available condition names, and SHALL NOT silently return an empty result or a
success exit code.

**Validates: Requirements 2.18, 2.19, 2.20, 2.21, 2.22, 2.23, 2.24**

Property 8: Preservation - Shipped configs, existing rejections, and record schemas

_For any_ well-formed experiment document — the six shipped configs in particular — loading, resolution
and running SHALL produce identical results to today, an unknown `parameter` SHALL still raise
`ValueError`, a missing `extends` target SHALL still raise `FileNotFoundError`, and `RECORD_FIELDS`,
`ResolvedConfig.to_dict` and the replay schema SHALL be unchanged.

**Validates: Requirements 3.12, 3.13, 3.14, 3.15, 3.16, 3.18, 3.19**

Property 9: Bug Condition - Non-finite fitness is rejected, and allocation stays nonnegative

_For any_ generation in which some genome carries a non-finite fitness, `Population._evaluate` SHALL
raise `ValueError` naming the actual value kind — distinguishing `nan`, `inf` and `-inf` — and the
population index carrying it, before that value can reach recorded statistics, `best_fitness`,
speciation or allocation; and _for any_ generation whose fitness values are all finite, of any sign,
`allocate_offspring(n)` SHALL return only nonnegative budgets summing exactly to `n` and the resulting
`len(population)` SHALL equal `population_size`.

**Validates: Requirements 2.25, 2.26, 2.27, 2.28, 2.29, 2.30**

Property 10: Preservation - Uniform fallback, largest-remainder counts, and seeded determinism

_For any_ generation whose adjusted-fitness sums are all nonnegative, `allocate_offspring` SHALL return
exactly today's counts, and _for any_ generation whose adjusted total is non-positive (all-zero or
all-negative) the documented uniform split SHALL still apply; negative fitness SHALL remain legal,
`_select_parent` SHALL be unchanged, and seeded runs SHALL stay byte-identical.

**Validates: Requirements 3.8, 3.9, 3.10, 3.11**

Property 11: Bug Condition - ADDITION - Invalid engine configuration is rejected at construction

_For any_ engine configuration satisfying `isBugCondition_engineconfig`, constructing the owning object
SHALL raise with a message naming the field and value, so that `population_size=0` can no longer become
100, a negative `population_size` or an empty `initial_population` can no longer yield an empty
population, `elitism < 0` can no longer inflate the population past `population_size`, out-of-range
probabilities can no longer silently saturate, and `stagnation_threshold < 1` can no longer collapse the
population permanently.

**Validates: ADDITION beyond bugfix.md; supports the invariant asserted by 2.30**

Property 12: Preservation - ADDITION - Valid engine configuration and documented defaults

_For any_ valid engine configuration — `elitism=0`, `crossover_rate` and mutation probabilities at the
inclusive endpoints `0.0` and `1.0`, `population_size` omitted (default 100), `population_size` omitted
with a non-empty `initial_population` (default `len(initial_population)`) — construction SHALL succeed
and behavior SHALL be identical to today's.

**Validates: Requirements 3.18, 3.20 (and the engine defaults documented in `Population.__init__`)**

## Fix Implementation

### Cross-cutting policy

**Exception types.**

- `ValueError` for a value of the right type that is outside the allowed range or set. This covers
  Areas 1, 2, 4 (except type errors), 5 and 6, and every numeric range rule in Area 3. It matches the
  existing conventions in this codebase: `Genome.validate`, `Network.__init__`,
  `ExperimentConfig._validate` and `Population.__init__` all raise `ValueError` today.
- `TypeError` when the value's *type* cannot support the field at all — `width='twenty'`,
  `width=20.5`, `metabolism=None`. This is more honest than `ValueError` and, importantly, it is
  **the type HEAD already raises** for these inputs (measured: `TypeError: 'str' object cannot be
  interpreted as an integer` from `range()` in `world/grid.py`). So the fix moves the error earlier and
  improves the message without changing what a caller catches.
- `FileNotFoundError` stays as-is for a missing `extends` target (3.16).
- `ExperimentConfig.condition(name)` keeps raising `KeyError`; it is a lookup, its behavior is fine, and
  it is not in scope.
- One deliberate type change: the missing-`conditions` case becomes `ValueError` instead of the bare
  `KeyError: 'conditions'`. A malformed document is a value problem, and 2.24 asks for a message naming
  the key and the file, which a `KeyError` cannot carry cleanly (its `str()` is just the repr of the key).
  No existing test asserts `KeyError` there. **Flagged** so it can be vetoed in favour of a
  `KeyError` subclassing nothing new.

*If the reviewer prefers a single catchable type across the whole milestone, the alternative is
`ValueError` everywhere including wrong types. It is defensible but it silently changes the exception
type callers currently see for `width='twenty'`, which is why it is not the recommendation.*

**Message format.** One shape per owner, all naming the field and the received value, all deterministic
(no dict iteration order, no set ordering, no rng):

| Owner | Format |
|---|---|
| `EnvironmentConfig` | `EnvironmentConfig.{field} must be {rule}, got {value!r}` |
| Interface (Area 1) | `genome interface mismatch at population index {i}: world expects {n} {input_ids\|output_ids} ({OBSERVATION_SIZE\|ACTION_SIZE}), got {m}` |
| Capacity (Area 2) | `population of {n} founders exceeds grid capacity {cap} (width={w} * height={h})` |
| Experiment loader | `{path}: {problem}` — the file path always leads |
| `ExperimentConfig` | `condition {name!r}: {problem}` — matching the existing unknown-parameter message |
| Fitness (Area 5) | `non-finite fitness at population index {i}: {kind}` where `{kind}` is `nan`, `inf` or `-inf` |
| Engine config (Area 6) | `{ClassName}.{field} must be {rule}, got {value!r}` |

Field-checking order inside a dataclass follows **declaration order**, so when several fields are invalid
the message always names the same one. That makes the error text testable.

**Determinism.** No validator draws from any `random.Random`, and none is placed between two rng draws:

- Areas 1 and 2 run at the top of `run_generation`, *before* `random.Random(config.world_rng_seed(...))`
  is constructed.
- Area 3 and Area 6 run in `__post_init__` / `__init__`, before any rng exists.
- Area 5's finiteness check runs in `_evaluate` and consumes nothing; the clipping change in
  `allocate_offspring` is pure arithmetic in a class documented as "Deterministic by design — no RNG
  inside".

Therefore every seeded stream is bit-identical for inputs where the bug condition is false, satisfying
3.8 and keeping `tests/test_experiments.py::test_run_is_deterministic` (a byte comparison of written
JSON) green. Where the bug condition *is* true the trajectory necessarily changes — that is the point of
the milestone, and no seeded test exercises those inputs.

**Package separation (3.19).** `neat` gains no import from `world`. The interface constants stay in
`world/config.py`, and the only thing that reads them is `world/simulation.py`. Area 5 and Area 6 touch
`neat` but only its own state. Verified by inspection against
`tests/test_population.py::test_engine_has_no_environment_dependencies`.

**Substring hazard in `neat/population.py` (measured).**
`test_engine_has_no_environment_dependencies` asserts that `inspect.getsource(neat.population)` does not
contain `"import grid"`, `"from grid"`, `"pygame"`, `"evolve_sim"`, or **`"main"`**. HEAD's
`population.py` contains zero occurrences of `main`. Any new code, docstring or error message added to
that file must avoid the substring, which rules out the natural words **remaining**, **remainder**,
**remains** and **domain**. Use "left to fill", "leftover", "stays", "range" instead. This is a real
tripwire, not a hypothetical: the obvious phrasing "the population size remains constant" would fail the
suite.

### Area 1 — Neural interface contract

**File:** `world/simulation.py`
**Function:** new module-private `_check_interface(population)`, called from `run_generation`
**Time:** execution entry, first statement of `run_generation`

1. Add a private helper that walks the population in list order and, for the first genome whose counts
   disagree with the contract, raises `ValueError` using the interface message format. It compares
   `len(genome.inputs)` to `OBSERVATION_SIZE` and `len(genome.outputs)` to `ACTION_SIZE`; both constants
   are already in `world/config.py` and both `Genome.inputs` and `Genome.outputs` already exist as
   properties.
2. Import `ACTION_SIZE` and `OBSERVATION_SIZE` in `world/simulation.py` (it currently imports only
   `EnvironmentConfig` from `.config`).
3. Call it as the first statement of `run_generation`, before `random.Random(...)`, before `World(...)`,
   before the founder loop. This satisfies "before the world is built and before any organism is placed"
   in 2.1 literally.
4. Only counts are compared. Node ids are never inspected, so 3.1 holds by construction.
5. **Validate every genome, not only distinct interface shapes.** Determining the set of distinct shapes
   requires computing `(len(inputs), len(outputs))` for every genome anyway, so "distinct shapes only"
   saves nothing — it only loses the population index in the message. Cost is `O(population x nodes)`
   list comprehensions once per generation, against `ticks x organisms` network activations in the same
   call (300 x 20 at shipped settings); it is not measurable. Report the **first** offending index so the
   message is deterministic.
6. **`make_evaluator` does NOT validate.** It receives only the `EnvironmentConfig`; the genomes do not
   exist yet and arrive later as an argument to the returned closure. It therefore *cannot* check counts
   at evaluator-creation time. Stating this explicitly because it is the natural place to look: the check
   has to be where the genomes are, and that is `run_generation`. Since `make_evaluator`'s closure calls
   `run_generation`, and `world/recorder.py::record_generation_to_file` calls it too, one check at that
   choke point covers every world-driving path in the repo.
7. **Rejected: validating in `resolve_config`.** `ResolvedConfig` does carry `input_ids` / `output_ids`,
   so `experiments` *could* compare their lengths to the world constants. Rejected because it would
   duplicate the rule in a second package, would import world constants into the experiment loader, and
   would not cover callers that build a `Population` directly (tests, `benchmarks`, ad-hoc scripts).
8. **Rejected: `neat` owning the rule.** Any check inside `neat/genome.py`, `neat/phenotype.py` or
   `neat/population.py` needs `OBSERVATION_SIZE` / `ACTION_SIZE`, i.e. an `import world` in the generic
   engine. That breaks 3.2 and `test_engine_has_no_environment_dependencies` directly.
9. **Rejected: `Network.activate`.** It is per-tick and per-organism, runs after the world exists and
   after organisms are placed, and has no access to the config field names. It is exactly the site of
   today's mid-run `ValueError: expected 10 inputs, got 9`, which is the defect. Its existing length
   check stays as a low-level invariant — it just stops being the first thing a misconfigured run hits.
10. **Documentation-only correction (2.5):** the comment above `DEFAULT_INPUT_IDS` in `neat/genome.py`
    currently reads "The Clage experiment's fixed interface to the world: 10 sensors, 4 actions." Replace
    it with a description of the generic default the module actually provides (10 input ids and 4 output
    ids as an engine convenience default, with no claim about any world's observation vector).
    **`DEFAULT_INPUT_IDS` itself is NOT changed.** Changing `tuple(range(10))` to `tuple(range(9))` would
    silently alter the genome every engine-only caller gets from `Genome.minimal()` — including
    `benchmarks`, which passes explicit ids and would be unaffected, but also any external caller relying
    on the documented default. That is a silent behavior change in the generic engine to satisfy a
    world-specific contract, which is precisely the coupling 3.2 forbids. World callers already pass
    explicit `input_ids=list(range(9))`; after this fix, ones that forget get a clear error instead of a
    mid-run one.

### Area 2 — Grid capacity

**File:** `world/simulation.py`
**Function:** new module-private `_check_capacity(population, config)`, called from `run_generation`
**Time:** execution entry, immediately after the Area 1 check

1. Compare `len(population)` to `config.width * config.height`; raise `ValueError` in the capacity
   message format when strictly greater. Strictly greater, so exactly-equal succeeds (3.4).
2. Place it after the interface check and before `World(...)`, so no `Organism` is constructed (2.6) and
   the `TypeError` at `world/simulation.py:44` becomes unreachable for this cause.
3. `width >= 1` and `height >= 1` are **not** re-checked here. Area 3 guarantees them at construction, so
   by the time any `EnvironmentConfig` reaches `run_generation` the capacity product is a positive
   integer. Re-checking would duplicate a rule across owners for no gain. Requirement 2.7 assigns those
   two rules to `EnvironmentConfig` explicitly.
4. The `None` guard on food placement (`world/simulation.py:50-52`) is **left exactly as is** — that is
   the correct behavior for oversized `initial_food` (3.5), and the rejected-non-defect list forbids
   adding a rule for it.
5. `Organism._try_reproduce` and `World.regenerate_food` are untouched, so a saturated grid stays a
   non-error (3.6).
6. **Why this cannot live on `EnvironmentConfig`:** the rule needs the population size, which the config
   does not know and should not — `population_size` is a `neat` concern (`ResolvedConfig.population_size`
   lives on the resolved config, not on `EnvironmentConfig`). A config-level rule would either need a new
   field (changing `asdict()` and breaking 3.14) or a cross-package lookup. The check belongs where both
   facts meet, which is the generation entry point.

### Area 3 — `EnvironmentConfig.__post_init__`

**File:** `world/config.py`
**Function:** new `EnvironmentConfig.__post_init__`
**Time:** construction — and therefore also on every `dataclasses.replace`

1. Add `__post_init__` to the existing `@dataclass`. **No field is added**, so `asdict()` keeps exactly
   today's keys, values and order (3.14, and `test_config_round_trip` / `test_experiments.py` stay green).
2. Range rules, in declaration order, exactly the set enumerated in 2.7–2.17:
   `width >= 1`, `height >= 1`, `ticks >= 1`, `initial_energy > 0`, `max_energy > 0`, `metabolism >= 0`,
   `initial_food >= 0`, `food_target >= 0`, `food_regrowth_per_tick >= 0`, `density_radius >= 1`,
   `0.0 <= repro_fraction <= 1.0`, `seed_stride >= 1`.
3. **No rules are added for the rejected non-defects:** `behavior_window` keeps accepting zero and
   negative values (3.17 — `_clip_window`'s "<= 0 means no clipping" is intentional documented
   semantics), `max_energy < initial_energy` stays legal, and nothing compares `initial_food` to grid
   capacity. Also no rules for `food_energy`, `repro_threshold`, `record_trace` or `seed_base` beyond
   type, because no defect was reproduced for them and `repro_threshold=2.0` (deliberately unreachable)
   is used throughout the test suite.
4. **Type validation as well as ranges — recommended.** `experiments` can inject any JSON value, and a
   bare `'twenty' < 1` comparison raises `TypeError: '<' not supported between instances of 'str' and
   'int'`, which is no more actionable than today's error. So each field is first checked for an
   acceptable numeric type and only then range-checked:
   - int-typed fields (`width`, `height`, `ticks`, `initial_food`, `food_target`,
     `food_regrowth_per_tick`, `density_radius`, `behavior_window`, `seed_base`, `seed_stride`) require
     an `int`. A `float` is rejected too, because `range(20.0)` raises (measured) — `available_space:
     20.5` reproduces `TypeError: 'float' object cannot be interpreted as an integer` inside
     `World.__init__` today.
   - float-typed fields (`initial_energy`, `max_energy`, `metabolism`, `food_energy`, `repro_threshold`,
     `repro_fraction`) accept `int` or `float`, so `max_energy=1` keeps working.
   - `bool` is a subclass of `int` in Python and is accepted where an `int` is; no defect was reproduced
     for `width=True`, and rejecting it would be a taste rule. `record_trace` is not type-checked.
   This is what makes 2.21 work **without `experiments` duplicating any rule**: the loader does not know
   what `available_space` means numerically, it only knows which fields the parameter targets, and the
   fields validate themselves.
5. **`dataclasses.replace` re-runs `__post_init__` — confirmed and desirable.** `replace()` calls the
   generated `__init__`, which calls `__post_init__`. So the per-trial seeded config produced by
   `resolve_config` (`replace(world, seed_base=seed * world.seed_stride)`) is validated too. `seed_base`
   has no range rule, so `seed=0` -> `seed_base=0` stays legal; and because `seed_stride >= 1` is enforced
   on the original config, the collapse-all-trials-onto-one-world defect (1.17) cannot survive to the
   replace.
6. **`resolve_config` wraps construction to name the condition (2.21).** `resolve_config` builds
   `EnvironmentConfig(**base["world"])`. Wrap that single call so the raised error is re-raised with a
   prefix naming the experiment, the condition, the parameter and the value, chained with
   `raise ... from exc` so the original field-level message is preserved. Re-raise the **same** exception
   type (`ValueError` stays `ValueError`, `TypeError` stays `TypeError`) so the policy above holds end to
   end. This adds context, not rules.

### Area 4 — Experiment loader and `ExperimentConfig`

**File:** `experiments/config.py` (`ExperimentConfig._validate`, `load_experiment`, `resolve_config`)
and `experiments/run.py` (`run_experiment`)
**Time:** construction, load, resolution, and run entry respectively

**`ExperimentConfig._validate` — order matters:**

1. Keep the existing unknown-`parameter` check **first and verbatim** (3.15). Ordering is deliberate:
   `tests/test_experiments.py::test_unknown_parameter_rejected` builds a config with a single non-control
   condition, i.e. zero controls. It asserts only `pytest.raises(ValueError)` so it passes either way, but
   with the parameter check first it keeps failing for its original reason instead of tripping over the
   new control-count rule.
2. Then duplicate names: collect names in order, raise `ValueError` naming the first duplicate (2.18).
3. Then control count: `controls = [c for c in conditions if c.is_control]`; raise `ValueError` stating
   how many were found and that exactly one is required, listing their names when there are several
   (2.19). Note `is_control` is `parameter is None`, which is the existing definition — the reporting
   layer's `"control"`-by-name convention is documented in the message but not enforced as a naming rule,
   since 2.19 speaks about control *count*.

**`load_experiment`:**

4. Missing `conditions`: check `if "conditions" not in raw` before building conditions and raise
   `ValueError(f"{path}: missing required key 'conditions' ...")` listing the expected top-level keys
   (2.24). Replaces the bare `KeyError: 'conditions'`.
5. Nested `extends`: after reading the parent document, `if "base" not in parent:` raise `ValueError`
   stating that only one level of `extends` is supported and naming **both** files (the child path and
   the resolved parent path) (2.22). This replaces `KeyError: 'base'` at `experiments/config.py:152` and
   covers both shapes of the defect — a parent that itself only `extends`, and a parent with no `base` at
   all. The `FileNotFoundError` from `extends_path.read_text()` is left untouched (3.16).
6. Seeds — absent versus explicitly empty (2.23): read with `raw.get("seeds")` replaced by an explicit
   presence test.
   - Key absent: inherit exactly as today — parent seeds when `extends` is set, else `[0, 1, 2, 3, 4]`.
   - Key present and non-empty: use it, as today.
   - Key present and empty (`[]`): **reject** with `ValueError(f"{path}: 'seeds' is empty; a condition
     with no seeds runs zero trials")`.
   - Also reject an empty *resolved* seed list (key absent, `extends` set, parent has no seeds — today
     that silently yields `[]`), with the same message naming the parent.
   **Decision and justification:** honouring `[]` is the other option 2.23 allows, and it is rejected
   because it produces exactly the failure mode 2.20 forbids — `run_condition` would loop zero times,
   write no files, and return a directory path as if it had worked. Rejecting is defensible precisely
   because an empty seed set means zero trials. Either way the absent-versus-empty distinction is now
   preserved: absent inherits, empty raises, and `[]` is never silently replaced by the parent's seeds.

**`experiments/run.py::run_experiment`:**

7. Unknown requested condition (2.20): after `load_experiment`, when `conditions` is truthy, compare it
   against `{c.name for c in experiment.conditions}` and raise `ValueError` naming the unknown name(s)
   and listing the available names sorted. This replaces today's silent `{}` + `0 condition(s)` +
   exit 0.
8. Keep `names = conditions or [c.name for c in experiment.conditions]` as-is. An explicitly empty list
   currently means "all conditions", and changing that to `is None` would create a new zero-run silent
   path. Not enumerated, so left alone.

**Explicitly not done:** the JSON format is not redesigned; the `food_abundance` /
`resource_scarcity` alias is untouched (documentation-only note in the rejected list); no new keys, no
schema version, no validation of `name` or `base` presence beyond what is enumerated.

### Area 5 — Fitness validity and allocation arithmetic

**File A:** `neat/speciation.py::Speciation.allocate_offspring`
**Time:** inside allocation, each generation

1. Compute clipped weights once: `weights = {s.id: max(s.adjusted_fitness_sum, 0.0) for s in
   self.species}`.
2. `if sum(weights.values()) <= 0.0:` keep the **existing** uniform-split fallback code path, unchanged.
3. Otherwise allocate proportionally over `weights` instead of over `adjusted_fitness_sum`, with the
   **same** `int()` truncation and the **same** largest-remainder distribution and the **same**
   `(fraction, -id)` tie-break. The only edit is which number feeds the ratio.
4. `species.offspring = allocation[species.id]` at the end is unchanged.

Why clipping is the right fix, recorded:

- **(a) Consistency with the engine's established semantic.** `Population._select_parent` already does
  `weights = [max(g.fitness, 0.0) for g in species.members]` and falls back to
  `self.rng.choice(species.members)` when the total is non-positive. "Negative fitness confers no
  selective advantage, and a non-positive total means uniform" is therefore already this engine's rule,
  stated in code, in the same package, one layer up. Allocation is simply inconsistent with it today.
  Requirement 2.29 names this reference explicitly and 3.11 forbids changing it.
- **(b) Bitwise preservation of the common case.** When every `adjusted_fitness_sum >= 0`, `max(x, 0.0)`
  is the identity, so `weights == adj`, the same branch is taken, the same ratios are computed and the
  same rounding runs. `test_offspring_allocation_proportional`,
  `test_offspring_allocation_largest_remainder`, `test_zero_fitness_species_gets_zero_offspring` and
  `test_all_zero_fitness_falls_back_to_uniform` are untouched (3.9, 3.10). All-negative sums still land
  in the fallback: clipping makes every weight `0.0`, so the total is `0.0 <= 0.0`, the same branch as
  today's `-inf <= 0` / `-3.0 <= 0`. That also keeps
  `tests/test_benchmarks.py::test_unevaluated_offspring_cannot_declare_success` (every genome scores
  `-1.0`) behaving exactly as it does now.
- **(c) It restores the invariant.** With all weights `>= 0` and a positive total, every
  `raw[sid] = weights[sid]/total*n` lies in `[0, n]`, `int()` is a true floor, each species is within 1
  of its exact share, the shortfall `n - SUM(floors)` lies in `[0, len(species)-1]` so the
  largest-remainder loop always has enough recipients, and the budgets are nonnegative and sum to exactly
  `n` (2.29). Every budget being `>= 0` means `_next_generation`'s `if budget <= 0: continue` only ever
  skips genuinely-zero species, so the produced population is exactly `population_size` and
  `_guarantee_champion` has nothing to top up (2.30). Worked example, the measured end-to-end case:
  sums `[+5.0, -2.0]`, `n = 12` -> weights `[5.0, 0.0]` -> total `5.0` -> raw `[12.0, 0.0]` ->
  allocation `{1: 12, 2: 0}` -> `len(population) == 12`.
- **(d) Negative fitness is not banned.** Nothing rejects a negative fitness anywhere; it just stops
  conferring advantage. The user forbade banning it and 3.9 requires it to keep working.

**Rejected alternative — min-shift normalization** (`weights = adj - min(adj)`, or
`adj - min(adj) + epsilon`):

- What it would buy: it preserves the *ranking* of species, supports arbitrary finite fitness with full
  discrimination even among all-negative populations (a species at `-1.0` would out-allocate one at
  `-9.0` instead of both falling back to uniform), and never needs a fallback branch except for the
  degenerate all-equal case.
- Why it is rejected: when all sums are already positive and `min > 0`, subtracting the minimum
  **changes the proportions**. Sums `[6.0, 4.0, 2.0]` with `n = 12` go from `{6, 4, 2}` today to
  `{8, 4, 0}` under min-shift. That is a different allocation for inputs where the bug condition is
  false, so it violates preservation for the common case and would break
  `test_offspring_allocation_proportional` (which asserts strict ordering across three species) and
  `test_zero_fitness_species_gets_zero_offspring`. Fixing a mixed-sign bug must not rewrite the
  nonnegative path.
- **Honest tradeoff:** clipping loses discrimination among all-negative populations — they collapse to
  the uniform fallback. Min-shift would preserve it. But the uniform fallback for all-negative fitness is
  *already* today's behavior (`total <= 0`), and 3.9 explicitly requires it to continue, so clipping
  loses nothing that exists.

**Also rejected — ban negative fitness.** Raise on `fitness < 0` and the mixed-sign case never arises.
Rejected on the user's explicit instruction and on 3.9 ("Negative fitness SHALL NOT be banned"); it would
break `tests/test_benchmarks.py::test_unevaluated_offspring_cannot_declare_success`, whose probe problem
scores every genome `-1.0` by design.

**File B:** `neat/population.py::Population._evaluate`
**Time:** immediately after the evaluator/`fitness_fn` has stamped fitness, before any statistic is
computed

5. Add `import math` at the top of `neat/population.py` (not currently imported). `math.isfinite` is the
   natural predicate: one call, correct for `nan`, `+inf` and `-inf`, no float trickery.
6. Insert the check **after** the `if self.evaluator is not None: ... else: ...` block and **before**
   `self._evaluated_best_genome = max(...)`. That position is the whole point: nothing non-finite can
   reach `_evaluated_best_genome`, `_evaluated_best_fitness`, `_evaluated_mean_fitness`,
   `_track_best_and_stats`, `best_fitness`, `statistics[-1]`, `speciate`, `share_fitness` or
   `allocate_offspring` (2.25–2.28). It is the single choke point through which every fitness value
   flows, for both the per-genome and the batch hook.
7. Walk `enumerate(self.population)` in order and raise on the first non-finite value, naming the
   **actual kind**: `nan` for NaN, `inf` for `+inf`, `-inf` for `-inf`. This is the specific fix for
   1.26 — today `+inf` is misreported as NaN because the failure surfaces at `int(inf/inf)` in
   speciation, four steps downstream. Include the population index so the caller can find the genome
   (2.25).
8. **`Population.best_fitness` is initialized to `float("-inf")` as an internal sentinel** (so that the
   first evaluated generation always wins the `best.fitness > self.best_fitness` comparison in
   `_track_best_and_stats`). The check must therefore validate **genome** fitness values only — iterate
   `self.population`, never touch `self.best_fitness`, `_evaluated_best_fitness` or any other engine
   field. Otherwise the very first generation of every run would raise on the engine's own sentinel. The
   sentinel is left as it is; it is correct and it is not part of this bugfix.
9. Word the new code carefully: it lives in `neat/population.py`, so it must not contain the substring
   `main` (see the substring hazard). "non-finite fitness at population index 3: nan" is safe;
   "the population size remains constant" is not.

### Area 6 — ADDITION (flagged for approval or trimming)

Marked as an addition beyond the five areas `bugfix.md` enumerates. Recommended because two of these
break stated invariants rather than merely being untidy: `elitism < 0` inflates the population past
`population_size` (the same invariant Area 5 restores, by a different route), and
`population_size=0 -> 100` is a silent surprise that contradicts the caller's explicit argument.

**File:** `neat/population.py::Population.__init__` — recommended:

1. `population_size`: when not `None`, require an `int >= 1`.
2. Replace the two `or` defaults with explicit `is None` tests, so a supplied `0` is rejected rather than
   silently replaced:
   - `self.population_size = len(self.population) if population_size is None else population_size`
     in the `initial_population` branch;
   - `self.population_size = 100 if population_size is None else population_size` in the other branch.
   The documented default of 100 for an omitted argument is unchanged (3.18) — only the `0 -> 100` path
   disappears.
3. `initial_population`: when not `None`, require it to be non-empty.
4. `elitism`: require an `int >= 0`. Zero must stay legal —
   `test_crossover_produces_valid_children_from_parent_genes` passes `elitism=0`.
5. `crossover_rate`: require a number in `[0.0, 1.0]` inclusive; `1.0` is used in the tests.
6. Keep the existing `fitness_fn is None and evaluator is None` check exactly as it is.

**File:** `neat/mutation.py::MutationConfig` — recommended: add `__post_init__` requiring each of
`weight_prob`, `replace_prob`, `bias_prob`, `add_connection_prob`, `add_node_prob`,
`enable_connection_prob`, `disable_connection_prob` to be a number in `[0.0, 1.0]` inclusive. Both
endpoints are exercised by the test suite (`weight_prob=0.0`, `add_connection_prob=1.0`), so the rule
must be inclusive. *Optional sub-items, flagged separately:* `weight_sigma >= 0`, `bias_sigma >= 0`, and
`weight_bounds` being a 2-tuple with `low <= high`. No defect was reproduced for these three; include
only if the reviewer wants them.

**File:** `neat/speciation.py::SpeciationConfig` — recommended: add `__post_init__` requiring
`stagnation_threshold >= 1`, since `0` extincts every species every generation and the population
collapses to zero permanently. *Optional sub-items, flagged separately:* `compatibility_threshold > 0`,
non-negative coefficients, `small_genome_threshold >= 0`. None of these was measured as a defect
(threshold `0` merely puts every genome in its own species, which is degenerate but size-preserving), so
they are listed and not recommended.

**Not included even as options:** `generations`, `record_generation`, and any `ResolvedConfig` field rule
— out of scope, no defect reproduced.

### Blast radius

- **`world`** — `world/config.py` gains `__post_init__`; `world/simulation.py` gains two private checks
  and two constant imports. `world/grid.py`, `world/organism.py`, `world/fitness.py` and
  `world/recorder.py` are **not modified**. `record_generation_to_file` inherits both new checks for free
  because it calls `run_generation`.
- **`neat`** — `neat/population.py` (`import math`, the finiteness check, and the Area 6 constructor
  rules), `neat/speciation.py` (clipped weights, plus the Area 6 `SpeciationConfig` rule),
  `neat/mutation.py` (Area 6 `MutationConfig` rule), and a comment-only edit in `neat/genome.py`.
  `neat/crossover.py`, `neat/innovation.py`, `neat/phenotype.py` and `neat/diagnostics.py` are
  **not modified**. No `world` import is added anywhere in `neat`.
- **`experiments`** — `experiments/config.py` (`_validate`, `load_experiment`, and the `resolve_config`
  wrap) and `experiments/run.py` (`run_experiment`). All six shipped configs already satisfy every rule
  (measured), so `experiments/configs/*.json` are **not modified**. `experiments/metrics.py`,
  `analysis.py`, `analyze.py` and `report.py` are untouched, and `RECORD_FIELDS` and
  `ResolvedConfig.to_dict` are unchanged, so downstream aggregation and CSV output are unaffected
  (3.12, 3.14).
- **`benchmarks`** — **no changes and no behavior change.** `benchmarks/run.py` passes
  `population_size=100`, `elitism`/`crossover_rate` defaults, and explicit `input_ids`/`output_ids` from
  each `Problem` (1–3 inputs, 1 output). The interface rule lives in `world`, which benchmarks never
  touch, so `AND`/`XOR`/`SIN` keep running (3.3). `mse_fitness` returns `1/(1+error)`, always finite and
  positive, so the finiteness check never fires and clipping is a no-op. The probe problem scoring `-1.0`
  keeps taking the uniform fallback.
- **`visual`** — **no changes and no behavior change.** Verified by inspection: `visual` never
  constructs an `EnvironmentConfig`, a `World`, an `Organism` or a `Population`; it reads recording JSON
  and results directories from disk. The `clage-generation-replay` v1 schema is untouched (3.13).
- **`diversity`** — **no changes.** `_clip_window` keeps its "<= 0 means no clipping" semantics because
  Area 3 adds no `behavior_window` range rule (3.17).
- **`pyproject.toml`** — **not modified** (explicit out-of-scope item; its `addopts` still references
  the uninstalled `pytest-cov`, hence the `-o addopts=''` verify command).
- **External callers** — this milestone intentionally converts silent acceptance into failure. Any
  script that passed a zero or negative value and accepted whatever came out will now raise. That is the
  point of the milestone (`bugfix.md`, "Intended Consequences") and belongs in release notes rather than
  being treated as invisible.

## Testing Strategy

### Validation Approach

Two phases, per area. First surface counterexamples on the UNFIXED code — each exploration test must FAIL
at HEAD, and the recorded pre-fix error below is the confirmation that the test reaches the real defect
rather than a typo. Then, still before any fix, write preservation tests and confirm they PASS at HEAD, so
they capture measured behavior rather than assumed behavior.

Verify command for every phase: `python3 -m pytest -q -p no:cacheprovider -o addopts=''`
(198 passing at HEAD `ea8a9c2`; `pyproject.toml` must not be modified).

Six areas means six exploration tests and six preservation tests, kept in separate test functions so a
single area can be reverted or deferred without disturbing the others. Suggested homes, following the
existing layout: Areas 1, 2 in `tests/test_world.py`; Area 3 in `tests/test_world.py`; Area 4 in
`tests/test_experiments.py`; Area 5 across `tests/test_speciation.py` (allocation) and
`tests/test_population.py` (finiteness, population size); Area 6 in `tests/test_population.py`.

### Exploratory Bug Condition Checking

**Goal:** surface counterexamples that demonstrate each defect BEFORE implementing anything, and confirm
or refute the root-cause hypotheses. If a hypothesis is refuted, re-hypothesize before writing code.

**Test plan:** one property-based test per area, scoped to concrete failing inputs where the defect is
deterministic (all of these are — no rng is involved in any bug condition). Run them at HEAD and record
what actually happens.

**Exact expected pre-fix failures, per area** (from `bugfix.md`, to be confirmed when the tests run):

| Area | Exploration input | Expected pre-fix behavior at HEAD |
|---|---|---|
| 1 | `Genome.minimal()` (10 inputs) into `run_generation` | `ValueError: expected 10 inputs, got 9` from `neat/phenotype.py:109`, raised mid-tick after the world was built and organisms placed |
| 1 | `output_ids=[]` | `ValueError: max() iterable argument is empty` from `world/organism.py:96` |
| 1 | 6 outputs | **no error**; run completes; an organism can select action 4 or 5 and the tick silently does nothing (position, facing, `food_eaten` all unchanged) |
| 1 | 2 outputs | **no error**; `TURN_RIGHT` and `EAT` unreachable for the whole run |
| 2 | 10 founders on a 3x3 grid | `TypeError: Value after * must be an iterable, not NoneType` at `world/simulation.py:44` |
| 3 | `density_radius=0` | `ZeroDivisionError: division by zero` at `world/grid.py:109` |
| 3 | `density_radius=-2` | **no error**; every food/organism density observation is exactly `0.0` |
| 3 | `max_energy=0.0` | `ZeroDivisionError: float division by zero` at `world/organism.py:77` |
| 3 | `repro_fraction=1.5` | **no error**; child energy is 1.5x the parent's and the parent goes negative |
| 3 | `repro_fraction=-0.5` | **no error**; negative child energy, parent gains energy |
| 3 | `metabolism=-1.0` | **no error**; nothing dies; 24 organisms from 3 founders within 3 ticks |
| 3 | `ticks=-5` | **no error**; zero ticks simulated, generation reported complete |
| 3 | `initial_food=-10` / `food_target=-10` / `food_regrowth_per_tick=-3` | **no error**; each silently means "none" |
| 3 | `initial_energy=-1.0` | **no error**; every organism dies on tick 1 |
| 3 | `seed_stride=0` | **no error**; `resolve_config` maps every trial seed to `seed_base=0`; all trials share one world while reporting distinct seeds |
| 3 | `width=0` / `width=-3` | `TypeError: Value after * must be an iterable, not NoneType` at `world/simulation.py:44` |
| 3 | `width='twenty'` via `available_space` | construction succeeds; `TypeError: 'str' object cannot be interpreted as an integer` inside `World.__init__` (measured) |
| 4 | duplicate condition name | **no error**; `condition()` returns only the first; both write to the same directory, later overwrites earlier |
| 4 | zero controls / two controls | **no error**; reporting and `compare_conditions` silently degrade or pick one |
| 4 | `run_experiment(..., ["typo-name"])` | returns `{}`; CLI prints `0 condition(s)`; exit code 0 |
| 4 | nested `extends` | `KeyError: 'base'` at `experiments/config.py:152` |
| 4 | `"seeds": []` | **no error**; parent's seeds silently substituted |
| 4 | missing `conditions` | `KeyError: 'conditions'`, naming neither file nor schema |
| 5 | any NaN fitness | `ValueError: cannot convert float NaN to integer` at `neat/speciation.py:236` |
| 5 | `+inf` fitness | the **same** NaN message — misleading, because `inf/inf` is NaN |
| 5 | all `-inf` fitness | **no error**; `-inf` becomes `best_fitness` and `statistics[-1]["best_fitness"]` |
| 5 | one NaN among finite values | **no error**; the NaN genome loses every `max()` and is invisible to tracking |
| 5 | sums `[3.0, -1.0]`, `n=10` | `allocate_offspring` returns `{1: 15, 2: -5}` |
| 5 | sums `[10.0, -4.0, -4.0]`, `n=10` | returns `{1: 50, 2: -20, 3: -20}` |
| 5 | 12 genomes, two species (threshold 0.5), `+5.0` / `-2.0` | generation-1 species offspring `[20, -8]` and `len(population) == 20` for `population_size == 12` |
| 6 | `population_size=0` | **no error**; population of 100 |
| 6 | `population_size=-5` | **no error**; empty population |
| 6 | `elitism=-3`, `population_size=6` | **no error**; population of 12 |
| 6 | `crossover_rate=5.0` / `-1.0` | **no error**; silently saturates |
| 6 | `MutationConfig(weight_prob=5.0)` | **no error**; silently saturates |
| 6 | `SpeciationConfig(stagnation_threshold=0)` | **no error**; population collapses to 0 and never recovers |
| 6 | `initial_population=[]` | **no error**; size 0, no champion |

**Expected counterexamples and what they confirm:** an unactionable `TypeError`/`ZeroDivisionError`/
`KeyError` confirms hypothesis 2, 3 or 7 (a low-level frame that has lost the context to name the field);
a silent no-error run confirms hypothesis 1 or 4 (an unenforced contract, or a falsy value read as
absent); the `{1: 15, 2: -5}` allocation confirms hypothesis 5. If any row above does **not** reproduce,
the root cause for that area must be re-hypothesized before writing the fix.

### Fix Checking

**Goal:** for all inputs where a bug condition holds, the fixed code produces the required error, early,
with the required information.

```
FOR ALL X WHERE isBugCondition_interface(X) DO
  ASSERT raises(ValueError, run_generation'(X))
  ASSERT error_names(expected_count) AND error_names(received_count)
  ASSERT error_names("input_ids" OR "output_ids")
  ASSERT no_world_was_built(X) AND no_organism_was_placed(X)
END FOR

FOR ALL X WHERE isBugCondition_capacity(X) DO
  ASSERT raises(ValueError, run_generation'(X))
  ASSERT NOT raises(TypeError, run_generation'(X))
  ASSERT error_names(founder_count) AND error_names(capacity) AND error_names(width, height)
  ASSERT no_organism_was_constructed(X)
END FOR

FOR ALL X WHERE isBugCondition_envconfig(X) DO
  ASSERT raises(ValueError, EnvironmentConfig'(X))   // out-of-range value
     OR  raises(TypeError,  EnvironmentConfig'(X))   // wrong type
  ASSERT error_names(offending_field) AND error_includes(received_value)
  ASSERT raises(..., replace(valid_config, **X))     // replace re-validates
END FOR

FOR ALL X WHERE isBugCondition_experiment(X) DO
  ASSERT raises(load_or_run'(X))
  ASSERT error_names(offending_condition_or_key)
  ASSERT error_names(file_path) OR error_names(available_condition_names)
  ASSERT NOT silently_returns_empty_result(X)
END FOR

FOR ALL X WHERE EXISTS g IN X WITH NOT is_finite(g.fitness) DO
  ASSERT raises(ValueError, next_generation'(X))
  ASSERT error_reports_actual_kind(X)                // nan vs inf vs -inf, distinctly
  ASSERT error_names(population_index)
  ASSERT no_non_finite_value_reached(statistics, best_fitness, speciation, allocation)
END FOR

FOR ALL X WHERE all_finite(X) DO
  allocation := allocate_offspring'(X, population_size)
  ASSERT FOR ALL sid IN allocation: allocation[sid] >= 0
  ASSERT SUM(allocation.values()) = population_size
  ASSERT LENGTH(next_generation'(X).population) = population_size
END FOR

FOR ALL X WHERE isBugCondition_engineconfig(X) DO            // ADDITION
  ASSERT raises(ValueError, construct'(X))
  ASSERT error_names(offending_field) AND error_includes(received_value)
END FOR
```

The "no world was built / no organism was constructed" assertions are observable without touching
production code: monkeypatch `world.simulation.World` and `world.simulation.Organism` with objects that
raise if called, or count constructions with a counting subclass. That keeps the ordering requirement of
2.1 and 2.6 testable rather than aspirational.

### Preservation Checking

**Goal:** for all inputs where a bug condition is false, the fixed code equals the original.

```
FOR ALL X WHERE NOT isBugCondition_interface(X) DO
  ASSERT F(X) = F'(X)     // includes custom node ids with correct counts (3.1)
  ASSERT no_world_import_in(neat)                                        // 3.2
END FOR

FOR ALL X WHERE NOT isBugCondition_capacity(X) DO
  ASSERT F(X) = F'(X)     // includes population_size = width*height exactly (3.4),
                          // oversized initial_food (3.5), saturated reproduction (3.6)
END FOR

FOR ALL X WHERE NOT isBugCondition_envconfig(X) DO
  ASSERT EnvironmentConfig'(X) constructs
  ASSERT asdict(F'(X)) = asdict(F(X))                                    // 3.14
  ASSERT simulate'(X) = simulate(X)     // includes behavior_window <= 0 (3.17),
                                        // max_energy < initial_energy, zero food fields
END FOR

FOR ALL X WHERE NOT isBugCondition_experiment(X) DO
  ASSERT F(X) = F'(X)                            // six shipped configs load identically
  ASSERT to_dict(F'(X)) = to_dict(F(X))          // 3.14
  ASSERT RECORD_FIELDS unchanged                 // 3.12
  ASSERT unknown_parameter still raises ValueError                       // 3.15
  ASSERT missing extends target still raises FileNotFoundError           // 3.16
END FOR

FOR ALL X WHERE all_finite(X) AND all_adjusted_sums_nonnegative(X) DO
  ASSERT allocate_offspring'(X, n) = allocate_offspring(X, n)            // 3.10
  ASSERT F(X) = F'(X)     // includes all-zero and all-negative uniform fallback (3.9)
                          // and seeded byte-identical output (3.8)
END FOR

FOR ALL X WHERE NOT isBugCondition_engineconfig(X) DO         // ADDITION
  ASSERT F(X) = F'(X)     // elitism=0, rates at 0.0/1.0, omitted population_size -> 100
END FOR
```

**Testing approach.** Property-based testing is the right tool for preservation here because preservation
is literally a universal claim ("for all non-buggy inputs"), it generates far more cases than hand-written
examples, and it catches boundary values a human would skip — `repro_fraction` at exactly `0.0` and
`1.0`, `metabolism` at exactly `0.0`, food fields at exactly `0`, `population_size` at exactly
`width * height`, adjusted sums at exactly `0.0`. Generators should draw from the *valid* domain of each
field and assert equality against values observed on the UNFIXED code.

**Observation-first method:** run the UNFIXED code on non-buggy inputs, record the actual outputs, then
write the property tests to assert those recorded outputs, and confirm the tests pass at HEAD before any
production edit. Concretely, the highest-value observations to capture first:

1. `allocate_offspring` results for the four existing nonnegative-sum test scenarios.
2. `asdict(EnvironmentConfig())` — the exact key list and values.
3. The bytes of `run_condition(tiny_experiment(), ..., seed 0)`'s written JSON (already asserted by
   `test_run_is_deterministic`; reuse it as the determinism oracle).
4. `experiment.to_dict()` for all six shipped configs.
5. `len(population)` per generation for a nonnegative-fitness run at several seeds.

### Unit Tests

- Area 1: correct counts pass; 10-input default rejected; `[]`, 2, 6 outputs rejected; custom ids with
  9/4 counts accepted; the error message names both counts and the offending side; `make_evaluator`'s
  closure raises on the first call for a bad population (proving the check is reached through that path
  too); `record_generation_to_file` inherits the check.
- Area 2: `population_size == capacity` succeeds; `capacity + 1` raises `ValueError` (not `TypeError`);
  oversized `initial_food` still places what fits; reproduction into a full grid still returns `None`.
- Area 3: one rejection test per rule, each asserting the exception type, the field name and the received
  value in the message; one acceptance test per boundary (`0` food fields, `metabolism=0.0`,
  `repro_fraction` `0.0`/`1.0`, `max_energy < initial_energy`, `behavior_window=0` and `-5`);
  `EnvironmentConfig()` defaults construct; `asdict()` key list unchanged;
  `replace(config, density_radius=0)` raises (proving `replace` re-validates); `width='twenty'` and
  `width=20.5` raise `TypeError` at construction.
- Area 4: duplicate names; zero controls; two controls; unknown requested condition (message lists
  available names); nested `extends` (message names both files); missing `conditions` (message names the
  file); `"seeds": []` rejected; absent `seeds` still inherits; unknown `parameter` still `ValueError`
  and still reported as a parameter problem; missing `extends` target still `FileNotFoundError`; all six
  shipped configs still load; `available_space: "twenty"` raises with the condition, parameter, value and
  expected type named.
- Area 5: NaN, `+inf`, `-inf` each raise with the correct kind named and the correct index; a single NaN
  among finite values raises; `best_fitness`'s `-inf` sentinel does not trip the check on generation 1;
  `statistics` never records a non-finite value; mixed-sign allocation is nonnegative and sums to `n`;
  the four existing allocation tests unchanged; `len(population) == population_size` for a mixed-sign
  generation.
- Area 6: `population_size=0`, `-5`; `elitism=-3`; `crossover_rate=5.0`, `-1.0`;
  `MutationConfig(weight_prob=5.0)`; `SpeciationConfig(stagnation_threshold=0)`;
  `initial_population=[]`; plus acceptance tests for `elitism=0`, rates at `0.0`/`1.0`, omitted
  `population_size` defaulting to 100, and omitted `population_size` with a non-empty
  `initial_population` defaulting to its length.

### Property-Based Tests

- Area 1: for any `(n_in, n_out)` other than `(9, 4)`, `run_generation` raises before any world exists;
  for any permutation of arbitrary distinct node ids with counts `(9, 4)`, it runs.
- Area 2: for any grid `w x h` and any founder count `<= w*h`, placement succeeds; for any count
  `> w*h`, `ValueError`.
- Area 3: for any field/value pair drawn from the invalid domain, construction raises naming that field;
  for any config drawn entirely from the valid domain, construction succeeds and `asdict()` round-trips.
- Area 4: for any list of condition names containing a duplicate, construction raises; for any control
  count `!= 1`, construction raises; for any requested-name list containing an undefined name,
  `run_experiment` raises.
- Area 5, fix: for any finite fitness assignment of any sign, `SUM(allocate_offspring(n)) == n` and every
  budget `>= 0`, and `len(population) == population_size` after a generation.
- Area 5, preservation: for any assignment of nonnegative adjusted sums,
  `allocate_offspring_fixed == allocate_offspring_head` (compare against values recorded pre-fix).
- Area 6: for any probability outside `[0, 1]`, construction raises; for any probability inside it,
  construction succeeds.

### Integration Tests

- A full `run_experiment` over a temp copy of a shipped config: unchanged results, byte-identical to a
  second run with the same seeds (reuses the `test_run_is_deterministic` oracle).
- A misconfigured experiment file end to end: `available_space: "twenty"` fails at load/resolve with the
  condition and parameter named, before any trial directory is created.
- A CLI run with `--conditions typo-name`: raises rather than printing `0 condition(s)` and exiting 0.
- A world run through `make_evaluator` with the default 10-input genome: fails at generation entry, and no
  `Organism` was constructed.
- A `Population` driven by a world evaluator across several generations with mixed-sign fitness:
  `len(population) == population_size` in every generation and `statistics` holds only finite values.
- A benchmark run (`AND`, seed 0) before and after the fix: identical history rows, proving `neat` stayed
  world-free and the engine path is untouched for valid input.
- The full suite: 198 passing, unchanged, with `python3 -m pytest -q -p no:cacheprovider -o addopts=''`.

## Out of Scope

Restated from `bugfix.md`, and not to be touched opportunistically:

- Performance optimization of any kind.
- Replay scalability.
- Visualization bugs.
- mypy cleanup.
- pytest-cov / tooling repair — `pyproject.toml`'s default `addopts` references the uninstalled
  `pytest-cov`; **leave `pyproject.toml` alone**.
- CI setup.
- Refactoring duplicated aggregation logic.
- Checkpoint/resume.
- Recurrent networks.
- Behavioral-metric changes and benchmark evaluation-state changes — delivered in the previous milestone.
- Redesigning the experiment JSON format.
- Any other opportunistic cleanup.

Also explicitly not done, from the rejected-non-defect list: no rule for `behavior_window <= 0`, no rule
for `max_energy < initial_energy`, no rule for `initial_food` exceeding grid capacity, no ban on negative
fitness, no removal of `seed_stride`, and no change to the `food_abundance` / `resource_scarcity` alias.

## Open Decisions for the Reviewer

1. **Area 6 in or out?** It is an addition beyond `bugfix.md`'s five areas. Recommended in, at least for
   `population_size >= 1`, `elitism >= 0`, `crossover_rate` in `[0, 1]`, the `MutationConfig`
   probabilities and `SpeciationConfig.stagnation_threshold >= 1`. The optional sub-items (sigmas, weight
   bounds ordering, compatibility threshold, coefficients) are listed separately and can be dropped
   without affecting the rest.
2. **`TypeError` for wrong types, `ValueError` for bad ranges** — recommended, and it matches the type
   HEAD already raises for `width='twenty'`. The alternative is `ValueError` uniformly.
3. **Missing `conditions` becomes `ValueError` instead of `KeyError`** — recommended so the message can
   name the file and the schema; no test depends on the current type.
4. **Explicitly empty `seeds` is rejected rather than honoured** — 2.23 permits either. Rejection is
   recommended because honouring `[]` reproduces the zero-trial silent success that 2.20 forbids.
5. **Int-typed fields reject floats** (`width=20.5`) — recommended, since `range()` rejects them anyway;
   this is slightly wider than the enumerated clauses, so it is flagged.
