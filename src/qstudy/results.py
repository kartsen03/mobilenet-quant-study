"""Result files: every JSON written here embeds the environment that produced it."""
from __future__ import annotations

import json
from pathlib import Path

from .env import REPO_ROOT, collect

RESULTS = REPO_ROOT / "results"
CKPT = REPO_ROOT / "checkpoints"


def write(name: str, payload: dict) -> Path:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps({"env": collect(), **payload}, indent=2))
    return path


def read(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text())
