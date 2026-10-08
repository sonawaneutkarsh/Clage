# Clage NEAT Validation Report


Engine defaults; no parameter tuning. Each trial is an independent,
seeded run of `Population` with default `MutationConfig`/`SpeciationConfig`.


## or


```
trial | solved | gens | best fit | nodes | conns | max sp
---------------------------------------------------------
    0 |   True |    9 |   0.9513 |     4 |     4 |      3
    1 |   True |    6 |   0.9469 |     4 |     4 |      2
    2 |   True |    5 |   0.9068 |     4 |     4 |      2
    3 |   True |    6 |   0.9325 |     3 |     2 |      2
    4 |   True |    8 |   0.9565 |     3 |     2 |      2
```

- Success: **5/5** (100%)
- Generations to solve (min/median/max): 5 / 6 / 9
- Mean best fitness at end: 0.9388
- Mean best-genome nodes: 3.6
- Mean best-genome connections: 3.2
- Mean max species: 2.2


## and


```
trial | solved | gens | best fit | nodes | conns | max sp
---------------------------------------------------------
    0 |   True |    6 |   0.8972 |     4 |     4 |      2
    1 |   True |    8 |   0.9091 |     3 |     2 |      2
    2 |   True |    7 |   0.9042 |     4 |     2 |      2
    3 |   True |    8 |   0.9070 |     3 |     2 |      4
    4 |   True |   17 |   0.9077 |     4 |     4 |      2
```

- Success: **5/5** (100%)
- Generations to solve (min/median/max): 6 / 8 / 17
- Mean best fitness at end: 0.9051
- Mean best-genome nodes: 3.6
- Mean best-genome connections: 2.8
- Mean max species: 2.4


## xor


```
trial | solved | gens | best fit | nodes | conns | max sp
---------------------------------------------------------
    0 |   True |  212 |   0.8807 |    19 |    17 |     22
    1 |   True |  125 |   0.9017 |     9 |    15 |     17
    2 |   True |   68 |   0.8952 |     7 |     9 |     18
    3 |   True |  143 |   0.8927 |    20 |    20 |     21
    4 |   True |  102 |   0.8876 |     8 |    18 |     16
```

- Success: **5/5** (100%)
- Generations to solve (min/median/max): 68 / 125 / 212
- Mean best fitness at end: 0.8916
- Mean best-genome nodes: 12.6
- Mean best-genome connections: 15.8
- Mean max species: 18.8


## sin


```
trial | solved | gens | best fit | nodes | conns | max sp
---------------------------------------------------------
    0 |  False | >300 |   0.9112 |    38 |    19 |     21
    1 |  False | >300 |   0.9115 |    22 |    16 |     16
    2 |  False | >300 |   0.9093 |    15 |    25 |     19
    3 |  False | >300 |   0.9536 |    11 |    24 |     17
    4 |  False | >300 |   0.9117 |    21 |    14 |     17
```

- Success: **0/5** (0%)
- Generations to solve (min/median/max): None / None / None
- Mean best fitness at end: 0.9195
- Mean best-genome nodes: 21.4
- Mean best-genome connections: 19.6
- Mean max species: 18.0

### Diagnosis (failed)

| layer | result | detail |
|---|---|---|
| network execution | PASS | Network.activate matches a hand-computed forward pass (tanh(2x+0.5)) |
| genome representation | PASS | genome builds, validates, copies, and preserves ids/enabled/innovation |
| innovation tracking | PASS | innovation ledger reuses identical inventions, mints distinct ones |
| mutation | PASS | all mutation operators leave a valid, acyclic genome |
| crossover | PASS | crossover children are valid and only inherit parent genes |
| speciation | PASS | compatibility distance (1.00 for one extra connection) and species assignment behave correctly |
| fitness design | PASS | fitness fn returns 0.6774 in (0,1] for the empty genome |

Parameterization probe (3 extra seeds 100-102): 0/3 solved.

**Hypothesis (not established cause):** search/parameterization is a hypothesis, not a diagnosis. These smoke checks 


## Verdict

**MIXED — some benchmarks failed with default parameters.**
