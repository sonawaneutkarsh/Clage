"""Rerun a saved configuration from seed and compare recorded observations."""

import argparse
import gzip
import json
from pathlib import Path

from .core import Experiment, RunConfig, validate_replay


def reproduce(bundle):
    validate_replay(bundle)
    config = RunConfig.model_validate(bundle['config']).model_copy(update={'record': False})
    experiment = Experiment(config)
    expected = {frame['sequence']: frame for frame in json.loads(json.dumps(bundle['frames']))}
    final_sequence = max(expected)
    checked = 0
    while True:
        frame = experiment.current
        if frame['sequence'] in expected:
            actual = json.loads(json.dumps(frame))
            for body, recorded in zip(actual['organisms'], expected[frame['sequence']]['organisms']):
                if body['inference'] is not None and recorded['inference'] is not None and 'world_tick' not in recorded['inference']:
                    body['inference'].pop('world_tick', None)
            if actual != expected[frame['sequence']]:
                raise ValueError(f"Deterministic replay mismatch at sequence {frame['sequence']}")
            checked += 1
        if frame['sequence'] >= final_sequence:
            break
        if not experiment.advance():
            raise ValueError("Run completed before the recorded final sequence")
    return {'recorded_frames_verified': checked, 'last_sequence': final_sequence,
            'method': 'rerun from frozen config/seed, NOT resume from replay',
            'source_commit': bundle.get('metadata', {}).get('commit'),
            'current_commit': experiment.metadata['commit'],
            'current_dirty': experiment.metadata['dirty']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('replay', type=Path)
    args = parser.parse_args()
    raw = args.replay.read_bytes()
    if args.replay.suffix == '.gz':
        raw = gzip.decompress(raw)
    print(json.dumps(reproduce(json.loads(raw)), indent=2))
