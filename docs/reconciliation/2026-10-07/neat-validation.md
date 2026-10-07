# Clage NEAT Validation Report


Engine defaults; no parameter tuning. Each trial is an independent,
seeded run of `Population` with default `MutationConfig`/`SpeciationConfig`.


## xor


```
trial | solved | gens | best fit | nodes | conns | max sp
---------------------------------------------------------
    0 |   True |  243 |   0.9114 |    23 |    14 |     15
    1 |  False | >300 |   0.8000 |    16 |    19 |     15
    2 |   True |  204 |   0.8818 |    24 |    17 |     19
    3 |   True |  147 |   0.8328 |    14 |    17 |     20
    4 |   True |  165 |   0.9021 |    10 |    17 |     20
```

- Success: **4/5** (80%)
- Generations to solve (min/median/max): 147 / 184.5 / 243
- Mean best fitness at end: 0.8656
- Mean best-genome nodes: 17.4
- Mean best-genome connections: 16.8
- Mean max species: 17.8

### Diagnosis (failed)

| layer | result | detail |
|---|---|---|
| network execution | PASS | Network.activate matches a hand-computed forward pass (tanh(2x+0.5)) |
| genome representation | PASS | genome builds, validates, copies, and preserves ids/enabled/innovation |
| innovation tracking | PASS | innovation ledger reuses identical inventions, mints distinct ones |
| mutation | PASS | all mutation operators leave a valid, acyclic genome |
| crossover | PASS | crossover children are valid and only inherit parent genes |
| speciation | PASS | compatibility distance (1.00 for one extra connection) and species assignment behave correctly |
| fitness design | PASS | hand-built solution solves XOR (fitness 1.0000), empty genome fitness 0.6667 |

Parameterization probe (3 extra seeds 100-102): 2/3 solved.

**Most likely suspect:** borderline parameterization / seed luck (some seeds solve).


## sin


```
trial | solved | gens | best fit | nodes | conns | max sp
---------------------------------------------------------
    0 |  False | >300 |   0.9602 |    35 |    18 |     20
    1 |  False | >300 |   0.8877 |     9 |    12 |     19
    2 |  False | >300 |   0.8818 |    56 |    19 |     20
    3 |  False | >300 |   0.9326 |    10 |    19 |     21
    4 |  False | >300 |   0.8836 |    31 |    20 |     18
```

- Success: **0/5** (0%)
- Generations to solve (min/median/max): None / None / None
- Mean best fitness at end: 0.9092
- Mean best-genome nodes: 28.2
- Mean best-genome connections: 17.6
- Mean max species: 19.6

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

**Most likely suspect:** search/parameterization: the layers work; solutions exist but aren't found 


## Verdict

**MIXED — some benchmarks failed with default parameters.**
