"""Same-origin local API and WebSocket snapshots; no engine work on event loop."""

from __future__ import annotations

import asyncio
import copy
import gzip
import json
import os
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .core import Manager, RunConfig, evaluate_policies, validate_replay

MAX_REPLAY_BYTES = 24 * 1024 * 1024


def reject_nonfinite(value):
    raise ValueError(f"Nonfinite JSON value: {value}")


def decode_replay(raw):
    return validate_replay(json.loads(raw, parse_constant=reject_nonfinite))


class Command(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["pause", "resume", "reset", "step", "speed"]
    speed: float | None = Field(default=None, ge=1, le=120, allow_inf_nan=False)


def create_app(manager=None, worker=True, artifact_dir=None):
    manager = manager or Manager()
    directory = Path(artifact_dir or os.environ.get("CLAGE_ARTIFACTS", ".studio-runs"))

    @asynccontextmanager
    async def lifespan(app):
        if worker:
            manager.start_worker()
        yield
        manager.close()

    app = FastAPI(title="Clage Studio", version="1.0-slice", lifespan=lifespan,
                  docs_url=None, redoc_url=None)
    app.state.manager = manager
    hosts = ["localhost", "127.0.0.1"] + (["testserver"] if not worker else [])
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=hosts)

    def valid_origin(origin, host):
        return origin is None or origin in {f"http://{host}", f"https://{host}"}

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        if request.method not in {"GET", "HEAD"} and not valid_origin(request.headers.get("origin"), request.headers.get("host")):
            return Response("Cross-origin mutation denied", status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob: data:; connect-src 'self' ws://127.0.0.1:* ws://localhost:*; object-src 'none'; frame-ancestors 'none'"
        return response

    @app.get("/api/state")
    def state():
        return manager.snapshot()

    @app.post("/api/runs")
    def start(config: RunConfig):
        return manager.create(config)

    @app.post("/api/control")
    def command(command: Command):
        try:
            return manager.command(command.action, command.speed)
        except ValueError as error:
            raise HTTPException(409, str(error)) from error

    @app.get("/api/genomes")
    def genomes():
        with manager.lock:
            return copy.deepcopy(manager.experiment.genomes) if manager.experiment else {}

    @app.get("/api/lineage")
    def lineage():
        with manager.lock:
            return copy.deepcopy(manager.experiment.lineage) if manager.experiment else {}

    @app.get("/api/champion")
    def champion():
        with manager.lock:
            experiment = manager.experiment
            if not experiment or not experiment.history:
                raise HTTPException(409, "Complete an evaluated generation first")
            best = max(experiment.history, key=lambda row: row["best_fitness"])
            return copy.deepcopy({"schema": "clage-champion", "version": 1,
                                  "config": experiment.config.model_dump(),
                                  "generation": best["world_generation"],
                                  "genome": experiment.genomes[best["champion"]],
                                  "metadata": experiment.metadata})

    def get_bundle():
        with manager.lock:
            if not manager.experiment:
                raise HTTPException(409, "No active experiment")
            if not manager.experiment.frames:
                raise HTTPException(409, "Recording is disabled; enable it in a new run to save/export replay")
            bundle = copy.deepcopy(manager.experiment.bundle())
        trimmed = 0
        while len(json.dumps(bundle, separators=(",", ":"), allow_nan=False)) > MAX_REPLAY_BYTES:
            if len(bundle["frames"]) <= 1:
                raise HTTPException(413, "Replay frame/genome table exceeds 24 MiB; use a smaller population")
            drop = max(1, len(bundle["frames"]) // 8)
            bundle["frames"] = bundle["frames"][drop:]
            bundle["dropped_frames"] += drop
            trimmed += drop
            retained = {body["genome"] for frame in bundle["frames"] for body in frame["organisms"]}
            retained.update(row["champion"] for row in bundle["history"])
            bundle["genomes"] = {key: genome for key, genome in bundle["genomes"].items() if key in retained}
            bundle["species"] = {key: species for key, species in bundle["species"].items() if key in retained}
            bundle["metadata"]["portable_export_trimmed_frames"] = trimmed
        return bundle

    @app.get("/api/replay")
    def replay():
        return get_bundle()

    @app.get("/api/export")
    def export():
        encoded = json.dumps(get_bundle(), separators=(",", ":"), allow_nan=False).encode()
        return Response(gzip.compress(encoded, mtime=0), media_type="application/gzip",
                        headers={"Content-Disposition": 'attachment; filename="clage-replay.json.gz"'})

    @app.post("/api/replay/validate")
    async def import_replay(request: Request):
        chunks, length = [], 0
        async for chunk in request.stream():
            length += len(chunk)
            if length > MAX_REPLAY_BYTES:
                raise HTTPException(413, "Replay upload exceeds 24 MiB")
            chunks.append(chunk)
        try:
            return await asyncio.to_thread(decode_replay, b"".join(chunks))
        except (ValueError, TypeError, KeyError, RecursionError, OverflowError) as error:
            raise HTTPException(422, str(error)) from error

    @app.post("/api/artifacts")
    def save():
        bundle = get_bundle()
        directory.mkdir(parents=True, exist_ok=True)
        identity = uuid.uuid4().hex
        path = directory / f"{identity}.json.gz"
        temporary = directory / f".{identity}.tmp"
        temporary.write_bytes(gzip.compress(json.dumps(bundle, allow_nan=False).encode(), mtime=0))
        temporary.replace(path)
        return {"id": identity, "filename": path.name}

    @app.get("/api/artifacts")
    def artifacts():
        return [{"id": path.stem.split(".")[0], "bytes": path.stat().st_size}
                for path in sorted(directory.glob("*.json.gz"))]

    @app.get("/api/artifacts/{identity}")
    def artifact(identity: str):
        if len(identity) != 32 or any(character not in "0123456789abcdef" for character in identity):
            raise HTTPException(404, "Artifact not found")
        path = directory / f"{identity}.json.gz"
        if not path.is_file():
            raise HTTPException(404, "Artifact not found")
        return FileResponse(path, media_type="application/gzip", filename=path.name)

    evaluation_lock = threading.Lock()

    @app.post("/api/evaluate")
    def evaluate():
        if not evaluation_lock.acquire(blocking=False):
            raise HTTPException(409, "Evaluation already running")
        try:
            with manager.lock:
                if not manager.experiment:
                    raise HTTPException(409, "Start an experiment first")
                config = manager.experiment.config.model_copy(deep=True)
                if config.width * config.height > 4096:
                    raise HTTPException(409, "Interactive evaluation is limited to 4,096 cells; export replay and use python -m studio.evaluate for larger runs")
                champion = manager.experiment.population.best_genome
                champion = champion.copy() if champion else None
            try:
                return evaluate_policies(config, champion)
            except ValueError as error:
                raise HTTPException(409, str(error)) from error
        finally:
            evaluation_lock.release()

    @app.websocket("/api/stream")
    async def stream(websocket: WebSocket):
        if not valid_origin(websocket.headers.get("origin"), websocket.headers.get("host")):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        signature = None
        idle = 0
        try:
            while True:
                def encode_snapshot():
                    current_signature, snapshot = manager.stream_snapshot(signature)
                    return current_signature, json.dumps(snapshot, separators=(",", ":"), allow_nan=False) if snapshot is not None else None

                signature, state = await asyncio.to_thread(encode_snapshot)
                if state is not None:
                    await websocket.send_text(state)
                    idle = 0
                else:
                    idle += 1
                    if idle >= 20:
                        await websocket.send_json({"heartbeat": True})
                        idle = 0
                await asyncio.sleep(0.1)
        except (WebSocketDisconnect, RuntimeError):
            return

    app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="studio")
    return app
