from neat.population import Population
from neat.mutation import MutationConfig
from neat.speciation import SpeciationConfig


def signature(genome):
    return ([tuple(vars(node).items()) for node in genome.nodes.values()],
            [tuple(vars(edge).items()) for edge in genome.connections], genome.fitness)


def test_recording_parentage_and_deltas_does_not_change_evolution_or_rng():
    def score(genome, generation):
        return 1.0 + len(genome.connections) + sum(abs(node.bias) for node in genome.nodes.values())

    options = dict(population_size=20, seed=31,
                   mutation_config=MutationConfig(add_connection_prob=.9, add_node_prob=.7))
    plain, recorded = Population(score, **options), Population(score, record_reproduction=True, **options)
    for _ in range(8):
        parents = set(recorded.population)
        plain.run(1)
        recorded.run(1)
        assert [signature(genome) for genome in plain.population] == [signature(genome) for genome in recorded.population]
        assert plain.statistics == recorded.statistics
        assert plain.rng.getstate() == recorded.rng.getstate()
        assert len(recorded.reproduction_records) == recorded.population_size
        for record in recorded.reproduction_records.values():
            if record['kind'] not in {'champion_copy', 'champion_rescue'}:
                assert all(parent in parents for parent in record['parents'])
            if record['deltas'] is not None:
                assert 'added_innovations' in record['deltas']


def test_recording_stagnation_rescues_retains_actual_champion_origin():
    population = Population(lambda genome, generation: 1.0, population_size=6,
                            speciation_config=SpeciationConfig(stagnation_threshold=1),
                            record_reproduction=True, seed=0)
    population.run(4)
    assert len(population.reproduction_records) == 6
    assert population._best_origin is not None
    assert all(record['parents'] for record in population.reproduction_records.values())
