"""Result files: every JSON written here embeds the environment that produced it.

The environment (versions, hardware, power plan, git commit and whether the code
had uncommitted changes) is captured when this module is first imported, i.e. at
the start of the run, so editing code while a long run is in progress cannot
change what the run reports about itself. The git state at write time is added
as `git_at_write`.
"""
from __future__ import annotations

import json
from pathlib import Path

from .env import REPO_ROOT, collect, git_info

RESULTS = REPO_ROOT / "results"
CKPT = REPO_ROOT / "checkpoints"

START_ENV = collect()


def write(name: str, payload: dict) -> Path:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{name}.json"
    env = START_ENV | {"git_at_write": git_info()}
    path.write_text(json.dumps({"env": env, **payload}, indent=2))
    return path


def read(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text())
