"""Run outside the repository, with PYTHONPATH set to a fresh wheel install."""

import gzip
import json
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

import httpx
import experiments
import neat
import studio
from experiments.config import load_experiment
from neat.genome import Genome
from studio.reproduce import reproduce
from visual.app import main
from world import EnvironmentConfig, GenerationRecorder, run_generation

root = Path(sys.argv[1]).resolve()
for module in (neat, studio, experiments):
    assert Path(module.__file__).is_relative_to(root)
assert version("clage") == "0.4.1"
configs = sorted((root / "experiments/configs").glob("*.json"))
assert len(configs) == 6
for path in configs:
    load_experiment(path)
assert len(list((root / "studio/static").glob("*"))) == 4
config = EnvironmentConfig(width=5, height=7, ticks=3, initial_food=2, food_target=2)
genomes = [Genome.minimal(range(9), range(10, 14))]
recorder = GenerationRecorder(genomes, config, 0)
run_generation(genomes, config, 0, recorder=recorder)
path = Path("/private/tmp/clage-semantic-installed-legacy.json")
path.write_text(json.dumps(recorder.to_dict()))
main(["replay", "--recording", str(path), "--export-tick", "0", "--export-out",
      "/private/tmp/clage-semantic-legacy-world.png"])
main(["network", "--recording", str(path), "--genome", "0", "--export",
      "/private/tmp/clage-semantic-legacy-network.png"])
with open("/private/tmp/clage-semantic-installed-server.log", "w") as log:
    process = subprocess.Popen([sys.executable, "-m", "studio", "--port", "8879"],
                               stdout=log, stderr=log)
    try:
        url = "http://127.0.0.1:8879"
        for attempt in range(100):
            try:
                if httpx.get(url + "/api/state").status_code == 200:
                    break
            except httpx.ConnectError:
                pass
            time.sleep(.1)
        else:
            raise RuntimeError("installed server did not start")
        for route in ("/", "/app.js", "/openapi.json"):
            assert httpx.get(url + route).status_code == 200
        httpx.post(url + "/api/runs", json={"population": 4, "generations": 2, "ticks": 8,
                   "width": 6, "height": 6, "food": 4, "seed": 17}).raise_for_status()
        for _ in range(17):
            httpx.post(url + "/api/control", json={"action": "step"}).raise_for_status()
        replay = httpx.get(url + "/api/replay").json()
        compressed = httpx.get(url + "/api/export").content
        assert json.loads(gzip.decompress(compressed)) == replay
        assert replay["metadata"]["commit"] is None
        assert replay["metadata"]["engine_version"] == "0.4.1"
        verified = reproduce(replay)
        assert verified["recorded_frames_verified"] == len(replay["frames"])
        print(json.dumps({"version": version("clage"), "installed_from": str(root),
              "configs": [path.name for path in configs], "assets": 4,
              "http_smoke": "static/OpenAPI/run/step/JSON/gzip export",
              "reproduction": verified}, indent=2))
    finally:
        process.terminate()
        process.wait(timeout=5)
