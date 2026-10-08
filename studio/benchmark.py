"""Predeclared local engine/recording workloads; measured, not research outcomes."""

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

from .core import Experiment, RunConfig, provenance

WORKLOADS = [(72, 40, 32, 180), (512, 80, 64, 500), (1000, 96, 96, 900)]


def benchmark(repeats=3, ticks=30):
    rows = []
    for population, width, height, food in WORKLOADS:
        for recording in (False, True):
            trials = []
            for _ in range(repeats):
                config = RunConfig(population=population, width=width, height=height,
                                   food=food, ticks=ticks + 1, generations=1, record=recording,
                                   seed=42, repro_threshold=2.0)
                start = time.perf_counter()
                experiment = Experiment(config)
                initialization = time.perf_counter() - start
                start = time.perf_counter()
                for _ in range(ticks):
                    experiment.advance()
                duration = time.perf_counter() - start
                start = time.perf_counter()
                encoded = json.dumps(experiment.snapshot(), separators=(",", ":"))
                serialization = time.perf_counter() - start
                trials.append({"initialization_s": initialization, "engine_ticks_s": ticks / duration,
                               "serialization_ms": serialization * 1000, "snapshot_bytes": len(encoded),
                               "recording_bytes": experiment.recorded_bytes,
                               "living_bodies": experiment.current["metrics"]["population"]})
            rows.append({"population": population, "width": width, "height": height, "food": food,
                         "recording": recording, "ticks": ticks, "seed": 42, "reproduction": "disabled (threshold 2)",
                         "trials": trials,
                         "median": {key: statistics.median(trial[key] for trial in trials) for key in trials[0]}})
    return {"schema": "clage-studio-benchmark", "version": 1,
            "machine": {"platform": platform.platform(), "python": platform.python_version(), "processor": platform.processor()},
            "metadata": provenance(), "workloads": rows,
            "limitations": "30-tick warm workload, no reproduction, no evolution boundary. Includes inference telemetry/capture; recording disabled still produces the current live snapshot. Not a long-duration memory or browser benchmark."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("docs/studio/backend-benchmark.json"))
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    results = benchmark(args.repeats)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    for row in results["workloads"]:
        print(f"N={row['population']} recording={row['recording']}: {row['median']['engine_ticks_s']:.1f} ticks/s, init {row['median']['initialization_s']:.3f}s")
