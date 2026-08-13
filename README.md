# Clage

A custom NEAT (NeuroEvolution of Augmenting Topologies) implementation paired
with a 2D artificial-life environment, plus reproducible experiments, behavioral
diversity metrics, and a pure-data visualization layer. The whole thing is
built from scratch — no `neat-python` — and validated against controlled
benchmark problems before use in the simulation.

```
neat/          custom NEAT engine: genome, phenotype, innovation ledger, mutation,
               crossover, speciation, population lifecycle
world/         2D walled grid: organisms, food, energy, movement, death, in-world
               reproduction, food regeneration
experiments/   config-driven one-factor-at-a-time environmental sweeps + analysis
diversity/     behavioral metrics (action entropy, transition entropy, coverage,
               food alignment, behavioral diversity index)
visual/        pure-data viewer: world replay, condition-comparison analytics,
               neural-network inspector (no evolutionary logic)
benchmarks/    NEAT validation suite (OR / AND / XOR / sin)
```

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e .          # installs matplotlib
.venv/bin/pip install pytest
.venv/bin/python -m pytest tests/   # 184 tests
```

## Quick usage

**Validate the engine** on the controlled benchmarks:

```bash
python -m benchmarks.run --problems or,and,xor,sin --trials 5 --generations 300
```

**Run an environmental experiment** (one factor changed at a time, control
included, 5 seeds each):

```bash
python -m experiments.run --config experiments/configs/food_abundance.json --out results
python -m experiments.run --config experiments/configs/food_abundance.json \
       --out results --record-generation 9     # also record gen 9 for replay
```

Configs shipped: `base.json`, `food_abundance`, `food_regeneration`,
`population_density`, `available_space`, `reproduction_cost`. Each condition
changes exactly one parameter (registry in `experiments/config.py`).

**Analyze and compare conditions:**

```bash
python -m experiments.analyze --results results/food_abundance \
       --report progress/report.md
```

**Visualize:**

```bash
python -m visual tui        --recording results/food_abundance/recordings/food_high/0.json
python -m visual tui        --recording <file> --export-tick 150 --export-out frame.txt
python -m visual analytics  --results results/food_abundance                    # interactive
python -m visual analytics  --results results/food_abundance --export results/plots  # PNGs
python -m visual replay     --recording <file>                                   # animated window
python -m visual network    --recording <file> --organism 3 --export net.png
```

`tui` is a boxed terminal replay (world on the left, selected-organism info +
an ASCII neural network on the right, generation/population/species/survival/
average-fitness in the footer). Controls: arrows/WASD move a cursor, Enter
selects the organism under it (opening its network panel), Space toggles
play/pause, `q` quits. The matplotlib `replay` window has a tick slider,
play/pause/step, and click-to-inspect.

## Programmatic use

**Evolve against the world** (shared world, one run per generation):

```python
from neat import Population
from world import EnvironmentConfig, make_evaluator

config = EnvironmentConfig(width=20, height=20, ticks=200, initial_food=50, food_target=50)
pop = Population(
    fitness_fn=None,
    evaluator=make_evaluator(config),
    population_size=30,
    input_ids=list(range(9)),      # 9 observations
    output_ids=[10, 11, 12, 13],   # 4 actions
    seed=0,
)
pop.run(10)
print(pop.statistics[-1])          # best/mean fitness, species count
print(pop.best_genome)             # champion genome
```

**Evolve against a plain fitness function** (no world):

```python
from neat import Population

def fitness(genome, generation):
    return float(len(genome.connections))

pop = Population(fitness, population_size=50, seed=0)
pop.run(20)
```

**Decode a genome into a network and run it:**

```python
from neat import Network

net = Network(pop.best_genome)
outputs = net.activate([0.5, -0.2, 0.1, 0.0, 0.9, 0.3, 0.2, 1.0, 0.0])  # 9 obs -> 4 outputs
```

**Run one world generation directly:**

```python
from world import run_generation

organisms = run_generation(population_genomes, config, generation=3)
for org in organisms:
    org.x, org.y, org.energy, org.food_eaten, org.age, org.offspring, org.alive
```

## Design notes you should know

- **Randomness is explicit.** The engine rng comes from `seed`; the world rng is
  derived per `(trial seed, generation)` via `EnvironmentConfig.world_rng_seed`.
  Same seed ⇒ identical results. Different trials are independent.
- **Fitness is world-relative per generation.** Each generation reseeds the
  world, so `best_fitness` is a champion snapshot from one generation's world;
  only `best_fitness >= current evaluated best` is guaranteed. Compare across
  seeds/conditions, not as a single absolute number.
- **No hard-coded behaviors.** Organisms move/turn/eat purely from
  observation → network → argmax action. Avoidance of other organisms is not
  even expressible (no directional organism sensor), and none of the diversity
  metrics claim cooperation, competition, aggression, or avoidance.
- **`food_alignment` is excluded** from automatic analytics charts — it is
  base-rate confounded by food density. View it only within a fixed food
  condition.
- **The engine is agnostic.** It only sees genomes and `fitness_fn`/
  `evaluator`. The `visual/` package contains no evolutionary logic; it consumes
  recorded JSON only.

## Layout

- `progress/` — build notes and analysis for each layer (genome, phenotype,
  mutation, crossover, speciation, population, world, experiments, diversity,
  visualization, validation, review fixes).
- `tests/` — 184 tests (`pytest`).
- `results/` — raw experiment output (gitignored).
