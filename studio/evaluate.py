"""Evaluate an archived champion/baselines without training or web time limits."""

import argparse
import gzip
import json
from pathlib import Path

from neat.genome import ConnectionGene, Genome, NodeGene, NodeType

from .core import RunConfig, evaluate_policies, validate_replay
from .contracts import GenomeRecord


def evaluate_bundle(bundle):
    champion = None
    if bundle.get('schema') == 'clage-champion' and type(bundle.get('version')) is int and bundle['version'] == 1:
        GenomeRecord.model_validate(bundle.get('genome'))
        archived = bundle['genome']
    else:
        validate_replay(bundle)
        archived = None
    if archived is None and bundle.get('history'):
        best = max(bundle['history'], key=lambda row: row['best_fitness'])
        archived = bundle['genomes'][best['champion']]
    if archived is not None:
        champion = Genome(
            nodes={node['id']: NodeGene(node['id'], NodeType[node['type']], node['bias']) for node in archived['nodes']},
            connections=[ConnectionGene(edge['in'], edge['out'], edge['weight'], edge['enabled'], edge['innovation']) for edge in archived['connections']])
        champion.fitness = archived['fitness']
        if len(champion.nodes) != len(archived['nodes']) or len(champion.inputs) != 9 or len(champion.outputs) != 4:
            raise ValueError('Champion must have unique nodes and the world interface')
    result = evaluate_policies(RunConfig.model_validate(bundle['config']), champion)
    result['training_provenance'] = bundle.get('metadata')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('replay', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    raw = args.replay.read_bytes()
    if args.replay.suffix == '.gz':
        raw = gzip.decompress(raw)
    result = evaluate_bundle(json.loads(raw))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as output:
        json.dump(result, output, indent=2)
    print(f"Saved {len(result['results'])} prespecified policy/world trials to {args.out}")
