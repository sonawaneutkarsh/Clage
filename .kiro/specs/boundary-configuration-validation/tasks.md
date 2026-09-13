# Implementation Plan

**Verify command for EVERY task** (`pyproject.toml` MUST NOT be modified — its default `addopts`
references the uninstalled `pytest-cov`, which is why `-o addopts=''` is required):

```
python3 -m pytest -q -p no:cacheprovider -o addopts=''
```

**HEAD baseline:** 198 passed at `ea8a9c2`.

## Pending reviewer decisions (carried forward from design.md "Open Decisions")

The implementer must know which choices are still awaiting approval. Do not silently resolve them;
if a decision is reversed, only the tasks that name it change.

1. **Area 6 in or out?** It is an ADDITION beyond `bugfix.md`'s five areas. Recommended in for
   `population_size >= 1`, `elitism >= 0`, `crossover_rate` in `[0, 1]`, `MutationConfig`
   probabilities, and `SpeciationConfig.stagnation_threshold >= 1`. Every Area 6 task below is marked
   ADDITION/OPTIONAL and can be skipped without touching Areas 1–5.
2. **`TypeError` for wrong types, `ValueError` for bad ranges** — recommended; it matches the type
   HEAD already raises for `width='twenty'`. Alternative: `ValueError` uniformly.
3. **Missing `conditions` becomes `ValueError` instead of the bare `KeyError`** — recommended so the
   message can name the file and the schema. No existing test depends on the current type.
4. **Explicitly empty `seeds` is rejected rather than honoured** — requirement 2.23 permits either.
   Rejection is recommended because honouring `[]` reproduces the zero-trial silent success that 2.20
   forbids.
5. **Int-typed fields reject floats** (`width=20.5`) — recommended, since `range()` rejects them
   anyway. Slightly wider than the enumerated clauses, so it is flagged.

## Ordering rationale — smallest blast radius first

The six areas are sequenced by blast radius, not by `bugfix.md` clause order, so that each area lands
on top of guarantees the previous one already established and every area can be reverted without
disturbing its predecessors.

1. **Area 3 — `EnvironmentConfig.__post_init__` (`world/config.py`) goes FIRST.** It is entirely
   self-contained in one dataclass: every rule is a property of a single field, it consumes no
   randomness, and it adds no field so `asdict()` is unchanged. It also supplies guarantees the later
   areas depend on: Area 2's capacity product relies on `width/height >= 1` being already enforced
   (which is why Area 2 does not re-check them), and Area 4's condition-value typing (2.21) works only
   because the config validates its own field types rather than the loader duplicating the rules.
2. **Area 1 — interface contract at `run_generation` entry (`world/simulation.py`).** One private
   helper plus two constant imports plus one comment-only edit in `neat/genome.py`. It establishes the
   entry-point check site that Area 2 reuses.
3. **Area 2 — grid capacity, same entry point, immediately after Area 1.** Shares the check site just
   established, so it is a few lines placed directly after the interface check.
4. **Area 5 — fitness finiteness in `neat/population.py::Population._evaluate` plus clipped weights in
   `neat/speciation.py::allocate_offspring`.** Touches engine arithmetic that four existing allocation
   tests pin down, so it comes after the world-side work is green and isolatable.
5. **Area 4 — experiment loader / `ExperimentConfig` / `run_experiment`.** Widest surface of the five:
   three functions across two files, plus the `resolve_config` wrap, and it is the one area whose
   correctness depends on Area 3 already validating field types.
6. **Area 6 — ADDITION, engine config validity — goes LAST** so it can be trimmed or vetoed without
   disturbing anything before it.

---

- [ ] 1. Record the pre-fix preservation baseline (observation only — change no file)
  - **Observation-only task**: this task creates and modifies NOTHING. It records measured facts that
    every later preservation test asserts against, per design.md "Observation-first method".
  - Confirm the working tree is clean and at `ea8a9c2` (`git status`, `git log -1`).
  - Run the full suite and record the exact count: `python3 -m pytest -q -p no:cacheprovider -o addopts=''`
    → expect **198 passed**. If it is not 198, STOP and report before doing anything else.
  - Record oracle 1: `allocate_offspring` results for the four existing nonnegative-sum scenarios in
    `tests/test_speciation.py` — `test_offspring_allocation_proportional`,
    `test_offspring_allocation_largest_remainder`, `test_zero_fitness_species_gets_zero_offspring`,
    `test_all_zero_fitness_falls_back_to_uniform`. Record the exact returned dicts.
  - Record oracle 2: `asdict(EnvironmentConfig())` — the exact key list, in order, with exact values.
  - Record oracle 3: the seeded `run_condition` JSON bytes. Reuse
    `tests/test_experiments.py::test_run_is_deterministic` as the determinism oracle rather than
    inventing a second one.
  - Record oracle 4: `experiment.to_dict()` for all six shipped configs in `experiments/configs/`
    (`base.json`, `available_space.json`, `food_abundance.json`, `food_regeneration.json`,
    `population_density.json`, `reproduction_cost.json`).
  - Record oracle 5: `len(population)` per generation for a nonnegative-fitness run, at several seeds.
  - Record the substring fact: `grep -c main neat/population.py` → expect **0**.
  - Mark complete when all five oracles plus the 198 count and the `main` count are written down.
  - _Requirements: 3.8, 3.9, 3.10, 3.12, 3.14, 3.20_

- [ ] 2. Fix the cross-cutting message format and exception-type policy before writing any validator
  - **Policy task**: no production behavior of its own; it pins the wording and types every later task
    must follow, from design.md "Cross-cutting policy". Write the agreed table down in the PR/notes.
  - Exception types: `ValueError` for a right-typed value outside the allowed range or set (Areas 1, 2,
    4 non-type cases, 5, 6, and every Area 3 range rule). `TypeError` when the value's TYPE cannot
    support the field at all (`width='twenty'`, `width=20.5`, `metabolism=None`) — this is the type HEAD
    already raises from `range()` in `world/grid.py`, so callers catch the same thing. See pending
    decision 2.
  - `FileNotFoundError` stays as-is for a missing `extends` target (3.16). `ExperimentConfig.condition(name)`
    keeps raising `KeyError`. One deliberate type change only: missing `conditions` becomes `ValueError`
    (pending decision 3).
  - Message format, one shape per owner (design.md's per-owner format table):
    - `EnvironmentConfig`: `EnvironmentConfig.{field} must be {rule}, got {value!r}`
    - Interface (Area 1): `genome interface mismatch at population index {i}: world expects {n} {input_ids|output_ids} ({OBSERVATION_SIZE|ACTION_SIZE}), got {m}`
    - Capacity (Area 2): `population of {n} founders exceeds grid capacity {cap} (width={w} * height={h})`
    - Experiment loader: `{path}: {problem}` — the file path always leads
    - `ExperimentConfig`: `condition {name!r}: {problem}` — matching the existing unknown-parameter message
    - Fitness (Area 5): `non-finite fitness at population index {i}: {kind}` where `{kind}` is `nan`, `inf` or `-inf`
    - Engine config (Area 6): `{ClassName}.{field} must be {rule}, got {value!r}`
  - Determinism of messages: field checks inside a dataclass follow **declaration order**, so when
    several fields are invalid the same one is always named; no dict-iteration order, no set ordering,
    no rng in any message.
  - No validator draws from any `random.Random` and none is placed between two rng draws.
  - _Requirements: 2.1, 2.6, 2.7, 2.21, 2.24, 2.25, 3.16, 3.18_

---

## Area 3 — `EnvironmentConfig` numeric field validity (FIRST: smallest blast radius)

- [ ] 3. Write the Area 3 exploration test
  - **Property 5: Bug Condition** - Invalid environment fields are rejected at construction
  - **CRITICAL**: this test MUST FAIL on unfixed code. The failure is the confirmation that the test
    reaches the real defect rather than a typo.
  - **DO NOT attempt to fix the test or the code when it fails.**
  - **NOTE**: this test encodes the expected post-fix behavior; it validates the fix when it passes.
  - Home: `tests/test_world.py`. Scope the property to the concrete measured cases below (the defect is
    deterministic — no rng is involved in any bug condition).
  - Assert that constructing `EnvironmentConfig` raises for each of: `width=0`, `width=-3`, `height=0`,
    `ticks=-5`, `density_radius=0`, `density_radius=-2`, `max_energy=0.0`, `initial_energy=-1.0`,
    `metabolism=-1.0`, `repro_fraction=1.5`, `repro_fraction=-0.5`, `initial_food=-10`,
    `food_target=-10`, `food_regrowth_per_tick=-3`, `seed_stride=0`, `width='twenty'`, `width=20.5`.
  - **EXACT expected pre-fix behavior at HEAD** (design.md's pre-fix failure table — if what you observe
    differs from ANY of these, STOP and re-hypothesize before writing a line of production code):
    - `density_radius=0`: construction succeeds, then `ZeroDivisionError: division by zero` at
      `world/grid.py:109` (`max_count = (2*0+1)**2 - 1 == 0`).
    - `density_radius=-2`: **no error**; every food-density and organism-density observation is exactly
      `0.0` for the whole run while `max_count` is 8.
    - `max_energy=0.0`: `ZeroDivisionError: float division by zero` at `world/organism.py:77`
      (`self.energy / config.max_energy`).
    - `repro_fraction=1.5`: **no error**; child energy is 1.5x the parent's and the parent goes negative.
    - `repro_fraction=-0.5`: **no error**; negative child energy, and the parent GAINS energy.
    - `metabolism=-1.0`: **no error**; nothing dies; 24 organisms from 3 founders within 3 ticks.
    - `ticks=-5`: **no error**; zero ticks simulated, generation reported complete (`range(-5)` is empty).
    - `initial_food=-10` / `food_target=-10` / `food_regrowth_per_tick=-3`: **no error**; each silently
      means "none" via an empty `range` or a never-satisfied `while`.
    - `initial_energy=-1.0`: **no error**; every organism dies on tick 1.
    - `seed_stride=0`: **no error**; `resolve_config` maps every trial seed to `seed_base=0`, so all
      trials share one world while reporting distinct seeds.
    - `width=0` / `width=-3`: `TypeError: Value after * must be an iterable, not NoneType` at
      `world/simulation.py:44`.
    - `width='twenty'`: construction succeeds, then `TypeError: 'str' object cannot be interpreted as an
      integer` inside `World.__init__`. `width=20.5`: the same `TypeError` for `'float'`.
  - Include a case asserting `dataclasses.replace(valid_config, density_radius=0)` raises, to prove
    `replace` re-validates.
  - Run on UNFIXED code. **EXPECTED OUTCOME: test FAILS** (this proves the bug exists).
  - Document the counterexamples observed; a `ZeroDivisionError`/`TypeError` from a low-level frame
    confirms root-cause hypotheses 3 and 7, a silent no-error run confirms hypothesis 3.
  - Verify: `python3 -m pytest -q -p no:cacheprovider -o addopts=''`
  - _Requirements: 1.7, 1.8, 1.9, 1.10, 1.11, 1.12, 1.13, 1.14, 1.15, 1.16, 1.17, 1.21_

- [ ] 4. Write the Area 3 preservation test (BEFORE implementing the fix)
  - **Property 6: Preservation** - Valid environment configs, serialized shape, and world mechanics
  - **IMPORTANT**: follow the observation-first methodology — assert the values recorded in task 1, not
    assumed values.
  - Home: `tests/test_world.py`.
  - Assert `EnvironmentConfig()` defaults construct, and that `asdict(EnvironmentConfig())` produces
    exactly the key list and values recorded as oracle 2 (no field is added, so the shape cannot change).
  - Assert every boundary that must stay legal constructs: `initial_food=0`, `food_target=0`,
    `food_regrowth_per_tick=0`, `metabolism=0.0`, `repro_fraction=0.0`, `repro_fraction=1.0`,
    `max_energy < initial_energy`, `behavior_window=0`, `behavior_window=-5`, `seed_base=0`,
    `repro_threshold=2.0`, `max_energy=1` (an `int` in a float-typed field).
  - Assert world mechanics and seeded output are unchanged, leaning on the existing coverage rather
    than duplicating it: `test_observe_vector`, `test_movement_steps_and_blocked_by_wall`,
    `test_eat_action_consumes_food`, `test_metabolism_drains_energy`, `test_energy_capped_at_max`,
    `test_death_at_zero_energy`, `test_reproduction_split`, `test_boundary_observation_at_wall`,
    `test_run_generation_is_deterministic`.
  - Run on UNFIXED code. **EXPECTED OUTCOME: tests PASS** (this is the baseline to preserve).
  - Verify: `python3 -m pytest -q -p no:cacheprovider -o addopts=''`
  - _Requirements: 3.7, 3.8, 3.14, 3.17_

- [ ] 5. Fix Area 3 — validate environment fields at construction

  - [ ] 5.1 Add `EnvironmentConfig.__post_init__` in `world/config.py`
    - **File**: `world/config.py`. **Function**: new `EnvironmentConfig.__post_init__` on the existing
      `@dataclass`.
    - **NO FIELD IS ADDED** to `EnvironmentConfig`. `asdict()` must keep exactly today's keys, values
      and order (3.14, `test_config_round_trip`).
    - Check fields in **declaration order** so the error message is deterministic when several fields
      are invalid.
    - For each field, **type check before range check** — a bare `'twenty' < 1` raises
      `TypeError: '<' not supported between instances of 'str' and 'int'`, which is no more actionable
      than today's error.
    - Int-typed fields require an `int` and **reject floats and strings** (`range(20.0)` raises):
      `width`, `height`, `ticks`, `initial_food`, `food_target`, `food_regrowth_per_tick`,
      `density_radius`, `behavior_window`, `seed_base`, `seed_stride`. `bool` is an `int` subclass and
      is accepted; `record_trace` is not type-checked.
    - Float-typed fields accept `int` or `float`, so `max_energy=1` keeps working: `initial_energy`,
      `max_energy`, `metabolism`, `food_energy`, `repro_threshold`, `repro_fraction`.
    - Range rules, exactly the enumerated set: `width >= 1`, `height >= 1`, `ticks >= 1`,
      `initial_energy > 0`, `max_energy > 0`, `metabolism >= 0`, `initial_food >= 0`,
      `food_target >= 0`, `food_regrowth_per_tick >= 0`, `density_radius >= 1`,
      `0.0 <= repro_fraction <= 1.0`, `seed_stride >= 1`.
    - **NO rule for `behavior_window`** beyond type (3.17 — `_clip_window`'s "<= 0 means no clipping" is
      intentional documented semantics). **NO rule for `max_energy < initial_energy`.** **NO rule
      comparing `initial_food` to grid capacity.** **NO range rule for `food_energy`,
      `repro_threshold` or `seed_base`.** These are on the rejected-non-defect list; adding them is a
      scope violation.
    - Message format: `EnvironmentConfig.{field} must be {rule}, got {value!r}`. `ValueError` for range,
      `TypeError` for type.
    - Verify: `python3 -m pytest -q -p no:cacheprovider -o addopts=''`
    - _Bug_Condition: `isBugCondition_envconfig(X)` from design.md Area 3_
    - _Expected_Behavior: Property 5 — raise at construction naming the field, the received value and the rule_
    - _Preservation: Property 6 — defaults construct, `asdict()` shape unchanged, all rejected non-defects stay legal_
    - _Requirements: 2.7, 2.8, 2.9, 2.10, 2.11, 2.12, 2.13, 2.14, 2.15, 2.16, 2.17_

  - [ ] 5.2 Add a `replace`-revalidation test and the `resolve_config` context wrap
    - `dataclasses.replace` calls the generated `__init__`, which calls `__post_init__`, so the
      per-trial seeded config from `resolve_config` (`replace(world, seed_base=seed * world.seed_stride)`)
      is validated too. Add an **explicit test** that `replace(valid_config, density_radius=0)` raises,
      so this is pinned rather than assumed. `seed_base` has no range rule, so `seed=0` →
      `seed_base=0` stays legal.
    - **File**: `experiments/config.py::resolve_config`. Wrap the single
      `EnvironmentConfig(**base["world"])` call so the raised error is re-raised with a prefix naming
      the experiment, the condition, the parameter and the value.
    - Re-raise the **SAME exception type** (`ValueError` stays `ValueError`, `TypeError` stays
      `TypeError`) and chain with `raise ... from exc` so the original field-level message is preserved.
    - This adds **context, not rules** — the loader still knows nothing about what `available_space`
      means numerically; the fields validate themselves. That is what makes 2.21 work without
      `experiments` duplicating an Area 3 rule.
    - Verify: `python3 -m pytest -q -p no:cacheprovider -o addopts=''`
    - _Bug_Condition: `condition_value_type_invalid_for_parameter(X)` — `available_space: "twenty"`_
    - _Expected_Behavior: Property 5 — the error names the condition, parameter, value and expected type_
    - _Preservation: Property 8 — the six shipped configs still resolve identically_
    - _Requirements: 2.21_

  - [ ] 5.3 Re-run the SAME Area 3 exploration test
    - **Property 5: Bug Condition** - Invalid environment fields are rejected at construction
    - **IMPORTANT**: re-run the SAME test written in task 3 — do NOT write a new test and do NOT relax
      its assertions.
    - **EXPECTED OUTCOME: test PASSES** (confirms the bug is fixed).
    - Verify: `python3 -m pytest -q -p no:cacheprovider -o addopts=''`
    - _Requirements: 2.7, 2.8, 2.9, 2.10, 2.11, 2.12, 2.13, 2.14, 2.15, 2.16, 2.17, 2.21_

  - [ ] 5.4 Re-run the Area 3 preservation test and the full baseline
    - **Property 6: Preservation** - Valid environment configs, serialized shape, and world mechanics
    - Re-run the SAME tests from task 4. **EXPECTED OUTCOME: PASS** (no regressions).
    - Then run the full suite: expect **198 + new**, zero failures.
    - **No assertion of any pre-existing test may be edited.**
    - Verify: `python3 -m pytest -q -p no:cacheprovider -o addopts=''`
    - _Requirements: 3.7, 3.8, 3.14, 3.17, 3.20_

---

## Area 1 — Neural interface contract at `run_generation` entry

- [ ] 6. Write the Area 1 exploration test
  - **Property 1: Bug Condition** - Neural interface counts are enforced before a run starts
  - **CRITICAL**: this test MUST FAIL on unfixed code. **DO NOT fix the test or the code when it fails.**
  - Home: `tests/test_world.py`. Scoped to the concrete measured cases below.
  - Cases: `Genome.minimal()` default (10 inputs) into `run_generation`; `output_ids=[]`; 6 outputs;
    2 outputs.
  - **EXACT expected pre-fix behavior at HEAD** (if any row differs, STOP and re-hypothesize):
    - 10 inputs: `ValueError: expected 10 inputs, got 9` from `neat/phenotype.py:109` in `activate()` —
      raised **mid-tick**, once per organism per tick, AFTER the world was built and organisms placed.
    - `output_ids=[]`: `ValueError: max() iterable argument is empty` from `world/organism.py:96` in
      `act()`, naming neither the interface nor the field at fault.
    - 6 outputs: **no error**; the run completes; an organism whose largest output is at index 4 or 5
      selects a `previous_action` outside `0..3`, `_apply_action` matches no branch, and the tick
      silently does nothing (position, facing and `food_eaten` all unchanged).
    - 2 outputs: **no error**; `Action.TURN_RIGHT` (2) and `Action.EAT` (3) are unreachable for the
      whole run.
  - Assert the post-fix ordering requirement too: **no `World` and no `Organism` is constructed**.
    Observe this without touching production code — monkeypatch `world.simulation.World` and
    `world.simulation.Organism` with objects that raise if called, or count constructions with a
    counting subclass.
  - Run on UNFIXED code. **EXPECTED OUTCOME: test FAILS.** A mid-run `ValueError` and the two silent
    rows confirm root-cause hypothesis 1 (declared-but-unenforced contract).
  - Verify: `python3 -m pytest -q -p no:cacheprovider -o addopts=''`
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [ ] 7. Write the Area 1 preservation test (BEFORE implementing the fix)
  - **Property 2: Preservation** - Correct interface counts, custom ids, and world-free engine use
  - **IMPORTANT**: observation-first — record the behavior on UNFIXED code, then assert it.
  - Home: `tests/test_world.py`.
  - Assert a genome with **custom node ids and correct counts** works: `input_ids=[100..108]`,
    `output_ids=[200, 201, 202, 203]` — 9 and 4. Only the COUNTS are contractual; ids need not be
    `0..8` and `10..13` (3.1).
  - Assert the NEAT engine still runs with no world at all and `neat` gains no `world` import — lean on
    `tests/test_population.py::test_engine_has_no_environment_dependencies` (3.2).
  - Assert benchmark workflows are unaffected: `benchmarks` passes explicit `input_ids`/`output_ids`
    with 1–3 inputs and 1 output, and the interface rule lives in `world`, which benchmarks never touch
    (3.3, `tests/test_benchmarks.py`).
  - Run on UNFIXED code. **EXPECTED OUTCOME: tests PASS.**
  - Verify: `python3 -m pytest -q -p no:cacheprovider -o addopts=''`
  - _Requirements: 3.1, 3.2, 3.3_

- [ ] 8. Fix Area 1 — enforce the interface contract at generation entry

  - [ ] 8.1 Add the private `_check_interface` helper in `world/simulation.py`
    - **File**: `world/simulation.py`. **Function**: new module-private `_check_interface(population)`.
    - Import `ACTION_SIZE` and `OBSERVATION_SIZE` from `.config` into `world/simulation.py` (it
      currently imports only `EnvironmentConfig`). The constants stay in `world/config.py`.
    - Compare `len(genome.inputs)` to `OBSERVATION_SIZE` and `len(genome.outputs)` to `ACTION_SIZE`.
      Both `Genome.inputs` and `Genome.outputs` already exist as properties.
    - Walk the population in list order and raise `ValueError` for the **FIRST** offending index, so the
      message is deterministic. Validate every genome, not only distinct interface shapes — computing
      the distinct set costs the same and loses the index.
    - **Counts only. Never inspect node ids**, so 3.1 holds by construction.
    - Message: `genome interface mismatch at population index {i}: world expects {n} {input_ids|output_ids} ({OBSERVATION_SIZE|ACTION_SIZE}), got {m}`.
    - `neat` gains **no** `world` import; the rule lives in `world` only.
    - Verify: `python3 -m pytest -q -p no:cacheprovider -o addopts=''`
    - _Bug_Condition: `isBugCondition_interface(X)` from design.md Area 1_
    - _Expected_Behavior: Property 1 — `ValueError` naming expected count, received count and the offending side_
    - _Preservation: Property 2 — custom ids with correct counts keep working; `neat` stays world-free_
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

  - [ ] 8.2 Call `_check_interface` as the FIRST statement of `run_generation`
    - **File**: `world/simulation.py`. **Function**: `run_generation`.
    - It must be the **first statement**: before `random.Random(config.world_rng_seed(...))`, before
      `World(...)`, and before any `Organism` is constructed. This satisfies 2.1's "before the world is
      built and before any organism is placed" literally, and it draws no randomness so no seeded
      stream shifts.
    - Document in a comment why **`make_evaluator` cannot validate at creation time**: it receives only
      the `EnvironmentConfig`; the genomes do not exist yet and arrive later as an argument to the
      returned closure. The check has to be where the genomes are. Because that closure calls
      `run_generation`, and `world/recorder.py::record_generation_to_file` calls it too, one check at
      this choke point covers every world-driving path in the repo.
    - Do NOT validate in `resolve_config` (would duplicate the rule in a second package and would miss
      direct `Population` callers) and do NOT put the rule in `neat` (would require importing `world`
      into the generic engine, breaking 3.2). `Network.activate`'s existing length check stays as a
      low-level invariant.
    - Add the test that no `Organism` and no `World` is constructed for a bad population (monkeypatch or
      counting subclass), plus a test that `make_evaluator`'s closure raises on its first call and that
      `record_generation_to_file` inherits the check.
    - Verify: `python3 -m pytest -q -p no:cacheprovider -o addopts=''`
    - _Bug_Condition: `isBugCondition_interface(X)`_
    - _Expected_Behavior: Property 1 — raised before `World` and before any `Organism`_
    - _Preservation: Property 2, Property 4 — valid populations run exactly as today_
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

  - [ ] 8.3 Correct the stale interface comment in `neat/genome.py` (comment only)
    - **File**: `neat/genome.py`. The comment above `DEFAULT_INPUT_IDS` currently reads "The Clage
      experiment's fixed interface to the world: 10 sensors, 4 actions", which contradicts the 9-value
      observation vector.
    - Replace it with a description of the generic default the module actually provides: 10 input ids
      and 4 output ids as an engine convenience default, making no claim about any world's observation
      vector.
    - **`DEFAULT_INPUT_IDS` itself is NOT changed** — it stays `tuple(range(10))`. Changing it to
      `tuple(range(9))` would silently alter the genome every engine-only caller gets from
      `Genome.minimal()`, which is a silent behavior change in the generic engine made to satisfy a
      world-specific contract — precisely the coupling 3.2 forbids.
    - Verify: `python3 -m pytest -q -p no:cacheprovider -o addopts=''`
    - _Expected_Behavior: Property 1 — the contract is enforced from the constants, not restated in comments_
    - _Preservation: Property 2 — `DEFAULT_INPUT_IDS` unchanged, `neat` unchanged behaviorally_
    - _Requirements: 2.5_

  - [ ] 8.4 Re-run the SAME Area 1 exploration test
    - **Property 1: Bug Condition** - Neural interface counts are enforced before a run starts
    - **IMPORTANT**: re-run the SAME test from task 6 — do NOT write a new test.
    - **EXPECTED OUTCOME: test PASSES**, including the "no `World`, no `Organism` constructed"
      assertions.
    - Verify: `python3 -m pytest -q -p no:cacheprovider -o addopts=''`
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [ ] 8.5 Re-run the Area 1 preservation test and the full baseline
    - **Property 2: Preservation** - Correct interface counts, custom ids, and world-free engine use
    - Re-run the SAME tests from task 7. **EXPECTED OUTCOME: PASS.**
    - Then the full suite: expect 198 + new, zero failures. Confirm
      `test_engine_has_no_environment_dependencies` is still green.
    - **No assertion of any pre-existing test may be edited.**
    - Verify: `python3 -m pytest -q -p no:cacheprovider -o addopts=''`
    - _Requirements: 3.1, 3.2, 3.3, 3.20_

---

## Area 2 — Grid capacity for founder placement (same entry point as Area 1)

- [ ] 9. Write the Area 2 exploration test
  - **Property 3: Bug Condition** - Founder population never exceeds grid capacity
  - **CRITICAL**: this test MUST FAIL on unfixed code. **DO NOT fix the test or the code when it fails.**
  - Home: `tests/test_world.py`.
  - Case: 10 founders on a 3x3 grid through `run_generation`.
  - **EXACT expected pre-fix behavior at HEAD**: `TypeError: Value after * must be an iterable, not
    NoneType` at `world/simulation.py:44` — `random_empty_cell()` returned `None` (it is correctly typed
    `Optional[Tuple[int, int]]`) and was unpacked anyway by `Organism(genome, *cell, config)`. If you
    observe anything else, STOP and re-hypothesize.
  - Assert the post-fix requirement: `ValueError`, **not `TypeError`**, and **no `Organism` constructed**
    (monkeypatch or counting subclass).
  - Note that `width < 1` / `height < 1` (bugfix.md 1.7) is **not** tested here — requirement 2.7
    assigns it to `EnvironmentConfig` construction, already covered by task 3.
  - Run on UNFIXED code. **EXPECTED OUTCOME: test FAILS.** The unactionable `TypeError` confirms
    root-cause hypothesis 2 (an `Optional` return consumed as if total).
  - Verify: `python3 -m pytest -q -p no:cacheprovider -o addopts=''`
  - _Requirements: 1.6_

- [ ] 10. Write the Area 2 preservation test (BEFORE implementing the fix)
  - **Property 4: Preservation** - Exact-capacity placement, oversized food, saturated reproduction
  - **IMPORTANT**: observation-first — record the behavior on UNFIXED code, then assert it.
  - Home: `tests/test_world.py`.
  - Assert **9 founders on a 3x3 grid** — exactly capacity — places all founders. The defect boundary is
    strictly `population_size > width * height` (3.4).
  - Assert oversized `initial_food` still places as much as fits without raising, preserving the existing
    `if cell is not None` guard at `world/simulation.py:50-52` (3.5, `test_food_regeneration`).
  - Assert grid-saturating in-world reproduction stays a non-error: `Organism._try_reproduce` returns
    `None` when `world.adjacent_empty` is empty, and `World.regenerate_food` returns when
    `random_empty_cell()` is `None` (3.6).
  - Run on UNFIXED code. **EXPECTED OUTCOME: tests PASS.**
  - Verify: `python3 -m pytest -q -p no:cacheprovider -o addopts=''`
  - _Requirements: 3.4, 3.5, 3.6, 3.7_

- [ ] 11. Fix Area 2 — check grid capacity at generation entry

  - [ ] 11.1 Add the private `_check_capacity` helper and call it immediately after `_check_interface`
    - **File**: `world/simulation.py`. **Function**: new module-private
      `_check_capacity(population, config)`, called from `run_generation` **immediately after** the
      Area 1 interface check and before `World(...)`.
    - Compare `len(population)` to `config.width * config.height`; raise **strictly greater than**, so
      exactly-equal succeeds (3.4).
    - Raise **`ValueError`, not `TypeError`** — this is the whole point; it makes the `TypeError` at
      `world/simulation.py:44` unreachable for this cause.
    - Message: `population of {n} founders exceeds grid capacity {cap} (width={w} * height={h})`.
    - **Do NOT re-check `width`/`height`** — Area 3 owns them (requirement 2.7 assigns them to
      `EnvironmentConfig` explicitly), so by the time any config reaches `run_generation` the capacity
      product is already a positive integer. Re-checking would duplicate a rule across owners.
    - **Leave the food `None` guard at `world/simulation.py:50-52` exactly as is** (3.5) and leave
      `Organism._try_reproduce` and `World.regenerate_food` untouched (3.6). The rejected-non-defect list
      forbids adding a rule for oversized `initial_food`.
    - Do not move this rule onto `EnvironmentConfig`: it needs the population size, which the config
      does not know and should not — a config-level rule would need a new field, changing `asdict()` and
      breaking 3.14.
    - Verify: `python3 -m pytest -q -p no:cacheprovider -o addopts=''`
    - _Bug_Condition: `isBugCondition_capacity(X)` — `LENGTH(population) > config.width * config.height`_
    - _Expected_Behavior: Property 3 — `ValueError` naming founder count, capacity and the width/height derivation, before any `Organism`_
    - _Preservation: Property 4 — exact capacity places all founders; oversized food and saturated reproduction unchanged_
    - _Requirements: 2.6_

  - [ ] 11.2 Re-run the SAME Area 2 exploration test
    - **Property 3: Bug Condition** - Founder population never exceeds grid capacity
    - **IMPORTANT**: re-run the SAME test from task 9 — do NOT write a new test.
    - **EXPECTED OUTCOME: test PASSES**, raising `ValueError` (not `TypeError`) with no `Organism`
      constructed.
    - Verify: `python3 -m pytest -q -p no:cacheprovider -o addopts=''`
    - _Requirements: 2.6_

  - [ ] 11.3 Re-run the Area 2 preservation test and the full baseline
    - **Property 4: Preservation** - Exact-capacity placement, oversized food, saturated reproduction
    - Re-run the SAME tests from task 10. **EXPECTED OUTCOME: PASS.**
    - Then the full suite: expect 198 + new, zero failures.
    - **No assertion of any pre-existing test may be edited.**
    - Verify: `python3 -m pytest -q -p no:cacheprovider -o addopts=''`
    - _Requirements: 3.4, 3.5, 3.6, 3.7, 3.20_

---

## Area 5 — Fitness finiteness and offspring-allocation arithmetic

- [ ] 12. Write the Area 5 exploration test
  - **Property 9: Bug Condition** - Non-finite fitness is rejected, and allocation stays nonnegative
  - **CRITICAL**: this test MUST FAIL on unfixed code. **DO NOT fix the test or the code when it fails.**
  - Homes: `tests/test_speciation.py` (allocation) and `tests/test_population.py` (finiteness and
    population size), kept in separate functions so either half can be deferred.
  - **EXACT expected pre-fix behavior at HEAD** (if any row differs, STOP and re-hypothesize):
    - Any NaN fitness: `ValueError: cannot convert float NaN to integer` at `neat/speciation.py:236`
      (`int(count)`), naming neither the genome nor the evaluator.
    - `+inf` fitness: the **SAME** NaN message — actively misleading, because `inf / inf` is NaN.
    - All `-inf` fitness: **no error**; the adjusted total is `-inf <= 0`, the documented uniform
      fallback absorbs it, and `-inf` silently becomes `best_fitness` and
      `statistics[-1]["best_fitness"]`.
    - One NaN among finite values: **no error**; the NaN genome loses every `max()` comparison and is
      invisible to best-genome tracking and representative selection.
    - Adjusted sums `[3.0, -1.0]`, `n=10`: `allocate_offspring` returns `{1: 15, 2: -5}`.
    - Adjusted sums `[10.0, -4.0, -4.0]`, `n=10`: returns `{1: 50, 2: -20, 3: -20}`.
    - 12 genomes, two species at compatibility threshold 0.5, wired genomes `+5.0` and unwired `-2.0`:
      generation-1 species offspring `[20, -8]` and `len(population) == 20` against
      `population_size == 12`.
  - Assert the post-fix requirements: the error names the actual kind distinctly (`nan` vs `inf` vs
    `-inf`) and the population index; every allocation budget is `>= 0` and the budgets sum to exactly
    `n`; `len(population) == population_size` for a mixed-sign generation; `statistics` never holds a
    non-finite value.
  - Run on UNFIXED code. **EXPECTED OUTCOME: test FAILS.** The `{1: 15, 2: -5}` allocation confirms
    root-cause hypothesis 5 (arithmetic written for one sign); the misleading NaN message confirms
    hypotheses 6 and 7.
  - Verify: `python3 -m pytest -q -p no:cacheprovider -o addopts=''`
  - _Requirements: 1.25, 1.26, 1.27, 1.28, 1.29, 1.30_

- [ ] 13. Write the Area 5 preservation test (BEFORE implementing the fix)
  - **Property 10: Preservation** - Uniform fallback, largest-remainder counts, and seeded determinism
  - **IMPORTANT**: observation-first — assert the exact dicts recorded as oracle 1 in task 1, not
    assumed values.
  - Homes: `tests/test_speciation.py` and `tests/test_population.py`.
  - Assert nonnegative adjusted sums produce today's exact proportional and largest-remainder counts —
    the four existing scenarios (`test_offspring_allocation_proportional`,
    `test_offspring_allocation_largest_remainder`, `test_zero_fitness_species_gets_zero_offspring`,
    `test_all_zero_fitness_falls_back_to_uniform`) must stay untouched and green (3.10).
  - Assert a non-positive adjusted total (all-zero and all-negative) still takes the documented uniform
    split, and that **negative fitness stays legal** — `tests/test_benchmarks.py::test_unevaluated_offspring_cannot_declare_success`
    scores every genome `-1.0` by design (3.9).
  - Assert `Population._select_parent` is unchanged: it keeps flooring weights at zero
    (`max(g.fitness, 0.0)`) and falling back to `rng.choice` on a non-positive total (3.11).
  - Assert seeded runs stay byte-identical: `test_seeded_run_is_deterministic` and oracle 3
    (`test_run_is_deterministic`).
  - Assert `len(population)` per generation matches oracle 5 for a nonnegative-fitness run.
  - Run on UNFIXED code. **EXPECTED OUTCOME: tests PASS.**
  - Verify: `python3 -m pytest -q -p no:cacheprovider -o addopts=''`
  - _Requirements: 3.8, 3.9, 3.10, 3.11_

- [ ] 14. Fix Area 5 — clip allocation weights and reject non-finite fitness

  - [ ] 14.1 Clip weights in `neat/speciation.py::Speciation.allocate_offspring`
    - **File**: `neat/speciation.py`. **Function**: `Speciation.allocate_offspring`.
    - Compute clipped weights once:
      `weights = {s.id: max(s.adjusted_fitness_sum, 0.0) for s in self.species}`.
    - Keep the **existing** uniform-split fallback code path unchanged for
      `sum(weights.values()) <= 0.0`. All-negative sums still land there: clipping makes every weight
      `0.0`, so the total is `0.0 <= 0.0` — the same branch as today's `-3.0 <= 0` (3.9).
    - Otherwise allocate proportionally over `weights` instead of over `adjusted_fitness_sum`, keeping
      the **SAME** `int()` truncation, the **SAME** largest-remainder distribution and the **SAME**
      `(fraction, -id)` tie-break. **Only the number feeding the ratio changes.**
    - `species.offspring = allocation[species.id]` at the end is unchanged.
    - When every sum is already `>= 0`, `max(x, 0.0)` is the identity, so the same branch, the same
      ratios and the same rounding run — bitwise preservation of the common case (3.10).
    - **Do NOT use min-shift normalization** (`adj - min(adj)`): it changes the proportions when all
      sums are already positive (`[6.0, 4.0, 2.0]`, `n=12` goes from `{6, 4, 2}` to `{8, 4, 0}`),
      violating preservation and breaking `test_offspring_allocation_proportional`.
    - **Do NOT ban negative fitness** (3.9) and do NOT touch `_select_parent` (3.11).
    - Verify: `python3 -m pytest -q -p no:cacheprovider -o addopts=''`
    - _Bug_Condition: `total_adjusted(X) > 0.0 AND EXISTS s WHERE s.adjusted_fitness_sum < 0.0`_
    - _Expected_Behavior: Property 9 — every budget `>= 0`, budgets sum to exactly `n`, `len(population) == population_size`_
    - _Preservation: Property 10 — identity for nonnegative sums; uniform fallback for non-positive totals_
    - _Requirements: 2.29, 2.30_

  - [ ] 14.2 Reject non-finite fitness in `neat/population.py::Population._evaluate`
    - **File**: `neat/population.py`. **Function**: `Population._evaluate`.
    - Add `import math` at the top of `neat/population.py` (not currently imported). `math.isfinite` is
      the predicate: one call, correct for `nan`, `+inf` and `-inf`.
    - Placement is the whole point: **AFTER** the `if self.evaluator is not None: ... else: ...` block
      and **BEFORE** `self._evaluated_best_genome = max(...)` is computed. Nothing non-finite can then
      reach `_evaluated_best_genome`, `_evaluated_best_fitness`, `_evaluated_mean_fitness`,
      `_track_best_and_stats`, `best_fitness`, `statistics[-1]`, `speciate`, `share_fitness` or
      `allocate_offspring`.
    - Iterate `enumerate(self.population)` in order and raise `ValueError` on the **first** non-finite
      value, naming the kind **distinctly** — `nan` for NaN, `inf` for `+inf`, `-inf` for `-inf` — and
      the population index. Message: `non-finite fitness at population index {i}: {kind}`. The distinct
      kind is the specific fix for 1.26, where `+inf` is today misreported as NaN.
    - **MUST NOT inspect `self.best_fitness`**, `_evaluated_best_fitness`, or any other engine field.
      `Population.best_fitness` is initialized to `float("-inf")` as an internal sentinel so the first
      evaluated generation always wins the `best.fitness > self.best_fitness` comparison. Validate
      **genome** fitness values only. Add a test proving generation 1 does not trip on the sentinel.
      The sentinel is left exactly as it is.
    - **CRITICAL text constraint**: `tests/test_population.py::test_engine_has_no_environment_dependencies`
      asserts `inspect.getsource(neat.population)` does not contain the substring **`main`**, and HEAD
      has zero occurrences. Any added code, docstring or message in this file must avoid it, which
      **forbids `remaining`, `remainder`, `remains` and `domain`**. Use "left to fill", "leftover",
      "stays", "range" instead. "non-finite fitness at population index 3: nan" is safe; "the
      population size remains constant" would fail the suite.
    - Verify: `python3 -m pytest -q -p no:cacheprovider -o addopts=''` and `grep -c main neat/population.py` → 0
    - _Bug_Condition: `EXISTS g IN X WHERE NOT is_finite(g.fitness)`_
    - _Expected_Behavior: Property 9 — `ValueError` naming the actual kind and index, before statistics, tracking, speciation or allocation_
    - _Preservation: Property 10 — finite fitness of any sign behaves as today; seeded runs byte-identical_
    - _Requirements: 2.25, 2.26, 2.27, 2.28_

  - [ ] 14.3 Re-run the SAME Area 5 exploration test
    - **Property 9: Bug Condition** - Non-finite fitness is rejected, and allocation stays nonnegative
    - **IMPORTANT**: re-run the SAME tests from task 12 — do NOT write new ones.
    - **EXPECTED OUTCOME: tests PASS.** Confirm the worked example: sums `[+5.0, -2.0]`, `n=12` →
      weights `[5.0, 0.0]` → total `5.0` → raw `[12.0, 0.0]` → allocation `{1: 12, 2: 0}` →
      `len(population) == 12`.
    - Verify: `python3 -m pytest -q -p no:cacheprovider -o addopts=''`
    - _Requirements: 2.25, 2.26, 2.27, 2.28, 2.29, 2.30_

  - [ ] 14.4 Re-run the Area 5 preservation test and the full baseline
    - **Property 10: Preservation** - Uniform fallback, largest-remainder counts, and seeded determinism
    - Re-run the SAME tests from task 13. **EXPECTED OUTCOME: PASS.**
    - Then the full suite: expect 198 + new, zero failures. Confirm the four existing allocation tests
      and `test_unevaluated_offspring_cannot_declare_success` are untouched and green.
    - **No assertion of any pre-existing test may be edited.**
    - Verify: `python3 -m pytest -q -p no:cacheprovider -o addopts=''`
    - _Requirements: 3.8, 3.9, 3.10, 3.11, 3.20_

---

## Area 4 — Experiment condition and inheritance validity

- [ ] 15. Write the Area 4 exploration test
  - **Property 7: Bug Condition** - Malformed experiment documents and unknown conditions are rejected
  - **CRITICAL**: this test MUST FAIL on unfixed code. **DO NOT fix the test or the code when it fails.**
  - Home: `tests/test_experiments.py`, using temp files.
  - **EXACT expected pre-fix behavior at HEAD** (if any row differs, STOP and re-hypothesize):
    - Duplicate condition name (two conditions named `food_low`): **no error**;
      `ExperimentConfig.condition(name)` returns only the first so the second is unreachable, and
      `run_condition` writes both to the same output directory, the later overwriting the earlier.
    - Zero controls, or two controls: **no error**; `experiments/report.py` and
      `analysis.compare_conditions` key off a condition literally named `"control"`, so comparison
      silently degrades or silently picks one.
    - `run_experiment(..., ["typo-name"])`: returns `{}`; the CLI prints `0 condition(s)` and exits 0 —
      indistinguishable from success.
    - Nested `extends` (a child whose `extends` target itself only declares `extends`):
      `KeyError: 'base'` at `experiments/config.py:152`, because `parent["base"]` is read
      unconditionally.
    - `"seeds": []`: **no error**; the parent's seeds are silently substituted, because
      `raw.get("seeds") or (...)` treats `[]` as absent.
    - Missing `conditions` key: bare `KeyError: 'conditions'`, naming neither the file nor the schema.
    - `available_space: "twenty"`: accepted through `resolve_config`, producing
      `EnvironmentConfig.width == 'twenty'`, which fails later inside `World`. (Now covered by Area 3
      + task 5.2; keep the end-to-end assertion here that the error names the condition, parameter,
      value and expected type.)
  - Assert the post-fix requirements: every error names the offending condition/key/name together with
    the file path or the sorted list of available condition names, and **nothing silently returns an
    empty result or a success exit code**.
  - Run on UNFIXED code. **EXPECTED OUTCOME: test FAILS.** The bare `KeyError`s confirm hypothesis 7
    (a frame that has lost the context to name the field); the silent `[]`-as-absent and the silent `{}`
    confirm hypothesis 4 (falsy-versus-absent confusion).
  - Verify: `python3 -m pytest -q -p no:cacheprovider -o addopts=''`
  - _Requirements: 1.18, 1.19, 1.20, 1.21, 1.22, 1.23, 1.24_

- [ ] 16. Write the Area 4 preservation test (BEFORE implementing the fix)
  - **Property 8: Preservation** - Shipped configs, existing rejections, and record schemas
  - **IMPORTANT**: observation-first — assert oracles 3 and 4 from task 1, not assumed values.
  - Home: `tests/test_experiments.py`.
  - Assert all six shipped configs still load and validate, and that `experiment.to_dict()` for each
    matches oracle 4 exactly (`test_all_shipped_configs_load_and_validate`).
  - Assert an unknown `parameter` still raises `ValueError` and is still reported as a **parameter**
    problem (`test_unknown_parameter_rejected`) — this is the ordering constraint in task 17.1.
  - Assert a missing `extends` target still raises `FileNotFoundError`, message not gold-plated (3.16).
  - Assert `RECORD_FIELDS` is exactly as today (3.12, `test_run_trial_records_expected_fields`,
    `test_run_condition_writes_machine_readable_files`), `ResolvedConfig.to_dict` is unchanged (3.14,
    `test_config_round_trip`), and the `clage-generation-replay` v1 schema is unchanged (3.13,
    `tests/test_visual.py`).
  - Assert seeded determinism through oracle 3 (`test_run_is_deterministic`).
  - Assert an **absent** `seeds` key still inherits exactly as today (parent seeds when `extends` is
    set, else `[0, 1, 2, 3, 4]`).
  - Assert the `food_abundance` / `resource_scarcity` alias is untouched and the JSON format is not
    redesigned.
  - Run on UNFIXED code. **EXPECTED OUTCOME: tests PASS.**
  - Verify: `python3 -m pytest -q -p no:cacheprovider -o addopts=''`
  - _Requirements: 3.12, 3.13, 3.14, 3.15, 3.16, 3.18, 3.19_

- [ ] 17. Fix Area 4 — validate experiment conditions, inheritance and requested names

  - [ ] 17.1 Add duplicate-name and control-count rules to `ExperimentConfig._validate`, in that order
    - **File**: `experiments/config.py`. **Function**: `ExperimentConfig._validate`.
    - **Keep the existing unknown-`parameter` check FIRST and VERBATIM** (3.15). Ordering is deliberate:
      `tests/test_experiments.py::test_unknown_parameter_rejected` builds a config with a single
      non-control condition, i.e. **zero controls**. It asserts only `pytest.raises(ValueError)` so it
      passes either way, but with the parameter check first it keeps failing **for its original reason**
      instead of tripping over the new control-count rule.
    - **Then** duplicate names: collect names in order and raise `ValueError` naming the **first**
      duplicate (2.18).
    - **Then** exactly-one-control: `controls = [c for c in conditions if c.is_control]`; raise
      `ValueError` stating how many were found and that exactly one is required, listing their names
      when there are several (2.19). `is_control` is `parameter is None`, the existing definition —
      the reporting layer's `"control"`-by-name convention may be mentioned in the message but is NOT
      enforced as a naming rule, since 2.19 speaks about control **count**.
    - Message shape: `condition {name!r}: {problem}`, matching the existing unknown-parameter message.
    - Verify: `python3 -m pytest -q -p no:cacheprovider -o addopts=''`
    - _Bug_Condition: `has_duplicate_condition_names(X) OR count_controls(X) <> 1`_
    - _Expected_Behavior: Property 7 — `ValueError` naming the duplicated name / the control count found_
    - _Preservation: Property 8 — unknown `parameter` still `ValueError`, for its original reason_
    - _Requirements: 2.18, 2.19_

  - [ ] 17.2 Fix `load_experiment` — missing `conditions`, nested `extends`, and seeds presence
    - **File**: `experiments/config.py`. **Function**: `load_experiment`.
    - Missing `conditions`: test `if "conditions" not in raw` before building conditions and raise
      `ValueError(f"{path}: missing required key 'conditions' ...")`, listing the expected top-level
      keys (2.24). Replaces the bare `KeyError: 'conditions'`. See pending decision 3.
    - Nested `extends` / parent without `base`: after reading the parent document,
      `if "base" not in parent:` raise `ValueError` stating that only ONE level of `extends` is
      supported and naming **both** files — the child path and the resolved parent path (2.22). This
      replaces `KeyError: 'base'` at `experiments/config.py:152` and covers both shapes: a parent that
      itself only `extends`, and a parent with no `base` at all.
    - **Leave the `FileNotFoundError`** from `extends_path.read_text()` untouched and un-gold-plated
      (3.16).
    - Seeds — preserve absent versus explicitly empty (2.23). Replace `raw.get("seeds") or (...)` with
      an explicit presence test:
      - key **absent**: inherit exactly as today — parent seeds when `extends` is set, else
        `[0, 1, 2, 3, 4]`;
      - key present and non-empty: use it, as today;
      - key present and **empty (`[]`): REJECT** with
        `ValueError(f"{path}: 'seeds' is empty; a condition with no seeds runs zero trials")`;
      - also reject an empty **resolved** seed list (key absent, `extends` set, parent has no seeds —
        today that silently yields `[]`), with the same message naming the parent.
      Rejecting rather than honouring `[]` is pending decision 4: honouring it reproduces exactly the
      zero-trial silent success that 2.20 forbids. Either way, `[]` is never silently replaced by the
      parent's seeds.
    - Message shape: `{path}: {problem}` — the file path always leads.
    - Verify: `python3 -m pytest -q -p no:cacheprovider -o addopts=''`
    - _Bug_Condition: `missing_key(X, "conditions") OR extends_target_has_no_base(X) OR seeds_explicitly_empty(X)`_
    - _Expected_Behavior: Property 7 — errors name the offending key and the file(s) involved_
    - _Preservation: Property 8 — six shipped configs load identically; absent `seeds` still inherits; `FileNotFoundError` unchanged_
    - _Requirements: 2.22, 2.23, 2.24_

  - [ ] 17.3 Reject unknown requested condition names in `run_experiment`
    - **File**: `experiments/run.py`. **Function**: `run_experiment`.
    - After `load_experiment`, when `conditions` is truthy, compare it against
      `{c.name for c in experiment.conditions}` and raise `ValueError` naming the unknown name(s) and
      listing the available names **sorted** (deterministic message, no set ordering). This replaces
      today's silent `{}` + `0 condition(s)` + exit 0 (2.20).
    - **Keep `names = conditions or [c.name for c in experiment.conditions]` exactly as is.** An
      explicitly empty list currently means "all conditions"; changing it to `is None` would create a
      new zero-run silent path and is not enumerated.
    - Add the CLI-level assertion: `--conditions typo-name` raises rather than printing
      `0 condition(s)` and exiting 0.
    - Verify: `python3 -m pytest -q -p no:cacheprovider -o addopts=''`
    - _Bug_Condition: `requested_name_not_defined(X)`_
    - _Expected_Behavior: Property 7 — `ValueError` naming the unknown condition and listing available names sorted_
    - _Preservation: Property 8 — `conditions or [all]` semantics unchanged; public signatures and CLI structure unchanged (3.18)_
    - _Requirements: 2.20_

  - [ ] 17.4 Re-run the SAME Area 4 exploration test
    - **Property 7: Bug Condition** - Malformed experiment documents and unknown conditions are rejected
    - **IMPORTANT**: re-run the SAME test from task 15 — do NOT write a new test.
    - **EXPECTED OUTCOME: test PASSES.** Confirm the `available_space: "twenty"` case now fails at
      load/resolve with the condition and parameter named, **before any trial directory is created**
      (that path is Area 3 task 5.2 plus this area's coverage).
    - Verify: `python3 -m pytest -q -p no:cacheprovider -o addopts=''`
    - _Requirements: 2.18, 2.19, 2.20, 2.21, 2.22, 2.23, 2.24_

  - [ ] 17.5 Re-run the Area 4 preservation test and the full baseline
    - **Property 8: Preservation** - Shipped configs, existing rejections, and record schemas
    - Re-run the SAME tests from task 16. **EXPECTED OUTCOME: PASS.**
    - Then the full suite: expect 198 + new, zero failures. Confirm
      `experiments/configs/*.json` were **not modified**, and that `experiments/metrics.py`,
      `analysis.py`, `analyze.py` and `report.py` are untouched.
    - **No assertion of any pre-existing test may be edited.**
    - Verify: `python3 -m pytest -q -p no:cacheprovider -o addopts=''`
    - _Requirements: 3.12, 3.13, 3.14, 3.15, 3.16, 3.18, 3.19, 3.20_

---

## Area 6 — ADDITION / OPTIONAL: engine configuration validity (LAST, trimmable)

> **ADDITION / OPTIONAL — every task in this area may be skipped.** Area 6 is not enumerated in
> `bugfix.md`. It is placed last so it can be vetoed or trimmed without disturbing Areas 1–5. See
> pending decision 1. It is recommended because two of its cases break stated invariants rather than
> merely being untidy: `elitism < 0` inflates the population past `population_size` (the same invariant
> Area 5 restores, by a different route), and `population_size=0 -> 100` silently contradicts the
> caller's explicit argument.

- [ ] 18. **ADDITION / OPTIONAL** — Write the Area 6 exploration test
  - **Property 11: Bug Condition** - ADDITION - Invalid engine configuration is rejected at construction
  - **CRITICAL**: this test MUST FAIL on unfixed code. **DO NOT fix the test or the code when it fails.**
  - Home: `tests/test_population.py`.
  - **EXACT expected pre-fix behavior at HEAD** (if any row differs, STOP and re-hypothesize):
    - `population_size=0`: **no error**; a population of **100**, because `Population.__init__` uses
      `population_size or 100` and `0` is falsy.
    - `population_size=-5`: **no error**; an empty population (`range(-5)` is empty).
    - `elitism=-3` with `population_size=6`: **no error**; a population of **12**. Arithmetic:
      `elites = min(-3, 6) = -3`; `ranked[:-3]` is the first 3 of 6, so 3 elites are appended; then
      `range(budget - elites) = range(6 - (-3)) = range(9)` adds 9 children; 3 + 9 = 12.
    - `crossover_rate=5.0` / `-1.0`: **no error**; silently saturates (`rng.random() < 5.0` always
      true, `< -1.0` never).
    - `MutationConfig(weight_prob=5.0)`: **no error**; silently saturates the same way.
    - `SpeciationConfig(stagnation_threshold=0)`: **no error**; `prune_stagnant` extincts every species
      every generation (`stagnation >= 0` is always true), allocation returns `{}`, and on generation 1
      `best_genome` is still `None` so `_guarantee_champion` returns the empty list — the population
      collapses to 0 and never recovers.
    - `initial_population=[]`: **no error**; size 0 and no champion (`population_size or len([])` = 0).
  - Run on UNFIXED code. **EXPECTED OUTCOME: test FAILS.** All rows confirm root-cause hypotheses 3
    (plain dataclasses as configuration) and 4 (falsy-versus-absent confusion).
  - Verify: `python3 -m pytest -q -p no:cacheprovider -o addopts=''`
  - _Requirements: ADDITION beyond bugfix.md; supports the invariant asserted by 2.30_

- [ ] 19. **ADDITION / OPTIONAL** — Write the Area 6 preservation test (BEFORE implementing the fix)
  - **Property 12: Preservation** - ADDITION - Valid engine configuration and documented defaults
  - **IMPORTANT**: observation-first — record on UNFIXED code, then assert.
  - Home: `tests/test_population.py`.
  - Assert every documented endpoint stays legal: `elitism=0` (used by
    `test_crossover_produces_valid_children_from_parent_genes`), `crossover_rate=0.0` and `1.0`,
    mutation probabilities of exactly `0.0` and `1.0` (`weight_prob=0.0`, `add_connection_prob=1.0` are
    used in existing tests), `population_size=2` with a 2-genome `initial_population`.
  - Assert omitted `population_size` still defaults to **100**, and omitted `population_size` with a
    non-empty `initial_population` still defaults to `len(initial_population)`.
  - All rules must therefore be **inclusive** at the documented endpoints.
  - Run on UNFIXED code. **EXPECTED OUTCOME: tests PASS.**
  - Verify: `python3 -m pytest -q -p no:cacheprovider -o addopts=''`
  - _Requirements: 3.18, 3.20 (and the engine defaults documented in `Population.__init__`)_

- [ ] 20. **ADDITION / OPTIONAL** — Fix Area 6

  - [ ] 20.1 **ADDITION / OPTIONAL** — Validate `Population.__init__` arguments
    - **File**: `neat/population.py`. **Function**: `Population.__init__`.
    - `population_size`: when not `None`, require an `int >= 1`.
    - Replace the two `or` defaults with explicit `is None` tests, so a supplied `0` is rejected rather
      than silently replaced:
      - `self.population_size = len(self.population) if population_size is None else population_size`
        in the `initial_population` branch;
      - `self.population_size = 100 if population_size is None else population_size` in the other
        branch.
      The **documented default of 100** for an omitted argument is unchanged (3.18), and
      `len(initial_population)` is preserved — only the `0 -> 100` path disappears.
    - `initial_population`: when not `None`, require it to be **non-empty**.
    - `elitism`: require an `int >= 0`. **Zero must stay legal.**
    - `crossover_rate`: require a number in `[0.0, 1.0]` **inclusive**; `1.0` is used in the tests.
    - Keep the existing `fitness_fn is None and evaluator is None` check exactly as it is.
    - Message: `{ClassName}.{field} must be {rule}, got {value!r}`, `ValueError`.
    - **CRITICAL text constraint**: this file must not contain the substring **`main`**
      (`test_engine_has_no_environment_dependencies`), so avoid `remaining`, `remainder`, `remains`,
      `domain` in any added code, docstring or message.
    - Verify: `python3 -m pytest -q -p no:cacheprovider -o addopts=''` and `grep -c main neat/population.py` → 0
    - _Bug_Condition: `isBugCondition_engineconfig(X)` — population_size, elitism, crossover_rate, initial_population_
    - _Expected_Behavior: Property 11 — raise at construction naming the field and value_
    - _Preservation: Property 12 — `elitism=0`, rates at `0.0`/`1.0`, omitted `population_size` → 100 or `len(initial_population)`_
    - _Requirements: ADDITION; supports 2.30. Preservation: 3.18, 3.20_

  - [ ] 20.2 **ADDITION / OPTIONAL** — Add `MutationConfig.__post_init__`
    - **File**: `neat/mutation.py`. **Class**: `MutationConfig`.
    - Require each of `weight_prob`, `replace_prob`, `bias_prob`, `add_connection_prob`,
      `add_node_prob`, `enable_connection_prob`, `disable_connection_prob` to be a number in
      `[0.0, 1.0]` **inclusive**. Both endpoints are exercised by the existing suite, so the rule must
      be inclusive.
    - **NOT RECOMMENDED, explicitly optional sub-items — do not add unless the reviewer asks:**
      `weight_sigma >= 0`, `bias_sigma >= 0`, and `weight_bounds` being a 2-tuple with `low <= high`.
      No defect was reproduced for any of the three.
    - Verify: `python3 -m pytest -q -p no:cacheprovider -o addopts=''`
    - _Bug_Condition: `any mutation probability NOT IN [0.0, 1.0]`_
    - _Expected_Behavior: Property 11 — construction raises naming the field and value_
    - _Preservation: Property 12 — probabilities of exactly `0.0` and `1.0` still construct_
    - _Requirements: ADDITION. Preservation: 3.18, 3.20_

  - [ ] 20.3 **ADDITION / OPTIONAL** — Add `SpeciationConfig.__post_init__`
    - **File**: `neat/speciation.py`. **Class**: `SpeciationConfig`.
    - Require `stagnation_threshold >= 1` — `0` extincts every species every generation and the
      population collapses to zero permanently.
    - **NOT RECOMMENDED, explicitly optional sub-items — do not add unless the reviewer asks:**
      `compatibility_threshold > 0`, non-negative coefficients, `small_genome_threshold >= 0`. None was
      measured as a defect (threshold `0` merely puts every genome in its own species, which is
      degenerate but size-preserving).
    - **Not included even as options:** `generations`, `record_generation`, and any `ResolvedConfig`
      field rule — out of scope, no defect reproduced.
    - Verify: `python3 -m pytest -q -p no:cacheprovider -o addopts=''`
    - _Bug_Condition: `X.stagnation_threshold < 1`_
    - _Expected_Behavior: Property 11 — construction raises naming the field and value_
    - _Preservation: Property 12 — valid thresholds behave identically_
    - _Requirements: ADDITION. Preservation: 3.18, 3.20_

  - [ ] 20.4 **ADDITION / OPTIONAL** — Re-run the SAME Area 6 exploration test
    - **Property 11: Bug Condition** - ADDITION - Invalid engine configuration is rejected at construction
    - **IMPORTANT**: re-run the SAME test from task 18 — do NOT write a new test.
    - **EXPECTED OUTCOME: test PASSES.**
    - Verify: `python3 -m pytest -q -p no:cacheprovider -o addopts=''`
    - _Requirements: ADDITION; supports 2.30_

  - [ ] 20.5 **ADDITION / OPTIONAL** — Re-run the Area 6 preservation test and the full baseline
    - **Property 12: Preservation** - ADDITION - Valid engine configuration and documented defaults
    - Re-run the SAME tests from task 19. **EXPECTED OUTCOME: PASS.**
    - Then the full suite: expect 198 + new, zero failures.
    - **No assertion of any pre-existing test may be edited.**
    - Verify: `python3 -m pytest -q -p no:cacheprovider -o addopts=''`
    - _Requirements: 3.18, 3.20_

---

- [ ] 21. Checkpoint — full verification, then STOP
  - Run the **new** tests first, area by area, then the **complete** suite:
    `python3 -m pytest -q -p no:cacheprovider -o addopts=''` → expect **198 + new, zero failures**.
  - Confirm **ZERO assertion edits to pre-existing tests**. If any pre-existing assertion appears to
    require an edit, STOP and report it as an explicit finding rather than editing it (3.20).
  - Confirm `pyproject.toml` is **untouched** (out of scope; its `addopts` still references the
    uninstalled `pytest-cov`).
  - Confirm `grep -c main neat/population.py` is **0**.
  - Confirm `neat` imports **nothing** from `world` — no `import world`, no `from world`, anywhere in
    the package (3.2, 3.19).
  - Confirm `asdict(EnvironmentConfig())` keys and values match oracle 2 from task 1 exactly (3.14).
  - Confirm all **six** shipped configs in `experiments/configs/` still load and validate, and that
    none of those JSON files was modified (3.15).
  - Confirm seeded determinism via the recorded oracle 3 (`test_run_is_deterministic`) and
    `test_seeded_run_is_deterministic` (3.8).
  - Inspect `git diff` and `git status` against this explicit **allowed-file list**; anything outside it
    is a scope violation to be reverted or reported:
    - `world/config.py`, `world/simulation.py`
    - `neat/population.py`, `neat/speciation.py`, `neat/mutation.py` (Area 6), `neat/genome.py`
      (comment only)
    - `experiments/config.py`, `experiments/run.py`
    - `tests/test_world.py`, `tests/test_experiments.py`, `tests/test_speciation.py`,
      `tests/test_population.py`
    - Expected **NOT** modified: `pyproject.toml`, `world/grid.py`, `world/organism.py`,
      `world/fitness.py`, `world/recorder.py`, `neat/crossover.py`, `neat/innovation.py`,
      `neat/phenotype.py`, `neat/diagnostics.py`, `experiments/configs/*.json`,
      `experiments/metrics.py`, `experiments/analysis.py`, `experiments/analyze.py`,
      `experiments/report.py`, everything under `benchmarks/`, `visual/`, `diversity/`.
  - **DO NOT COMMIT and DO NOT PUSH.** Leave the working tree for review.
  - Produce a report containing:
    1. every file changed and **why**, mapped to its area;
    2. per area, the **previous failure mode** (the exact pre-fix error text or the silent consequence)
       and **why the new test proves it is fixed**;
    3. any pre-fix expectation from design.md's failure table that did **not** reproduce, and what was
       re-hypothesized;
    4. the release-note line for the intended behavior change: configurations that previously ran
       silently now raise (`bugfix.md`, "Intended Consequences").
  - Ask the user if questions arise; do not resolve a pending decision unilaterally.
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 3.10, 3.11, 3.12, 3.13, 3.14, 3.15, 3.16, 3.17, 3.18, 3.19, 3.20_
