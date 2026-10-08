"""Run with python -m studio (install .[studio] first)."""

import argparse

import uvicorn

from .server import create_app

parser = argparse.ArgumentParser(description="Local-first Clage Studio")
parser.add_argument("--port", type=int, default=8765)
arguments = parser.parse_args()
uvicorn.run(create_app(), host="127.0.0.1", port=arguments.port)
