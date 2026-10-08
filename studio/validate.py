"""Exercise a complete run, validate its archive and rerun recorded observations."""

import argparse
import json
import platform
import resource
import time
from pathlib import Path

from .core import Experiment, RunConfig, validate_replay
from .reproduce import reproduce


def validate_run(config):
    started = time.perf_counter()
    experiment = Experiment(config)
    calls = 0
    while not experiment.complete:
        experiment.advance()
        calls += 1
    duration = time.perf_counter() - started
    bundle = experiment.bundle()
    validate_replay(bundle)
    verification = reproduce(bundle)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        'config': config.model_dump(), 'metadata': experiment.metadata,
        'world_ticks': config.ticks * len(experiment.history),
        'advancement_calls': calls, 'duration_s': duration,
        'completed_generations': len(experiment.history),
        'retained_frames': len(experiment.frames), 'dropped_frames': experiment.dropped,
        'recording_json_bytes': experiment.recorded_bytes,
        'process_peak_rss_mib': peak / (1024 * 1024 if platform.system() == 'Darwin' else 1024),
        'reproduction': verification,
        'limitations': 'One complete run plus a deterministic rerun. Peak RSS includes the interpreter and both runs; not a leak test or learning claim.',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    arguments = parser.parse_args()
    result = validate_run(RunConfig())
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({key: result[key] for key in ('world_ticks', 'duration_s', 'retained_frames', 'reproduction')}, indent=2))
