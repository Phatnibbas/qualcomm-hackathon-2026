"""HALO SafeShift — local preparation pipeline.

Scope of this package on this machine: freeze the station history, apply QC,
build leakage-safe windows, define retrospective folds, provide baselines and
portable export/inference types, and assemble an immutable Colab packet.

What this package deliberately does NOT do here:

* select a model family, parameterization, feature set or threshold;
* report any local number as model performance evidence;
* deploy anything to, or measure anything on, an Arduino UNO Q.

Every scientific-contract value lives in ``config/experiment.v1.json``. Code
loads it; code does not restate it.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]
DEFAULT_CONFIG_PATH = PACKAGE_DIR / "config" / "experiment.v1.json"

__all__ = [
    "PACKAGE_DIR",
    "REPO_ROOT",
    "DEFAULT_CONFIG_PATH",
    "load_config",
    "sha256_file",
    "sha256_bytes",
    "utc_now_stamp",
    "git_head_short",
    "make_run_id",
    "write_json",
]


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load the versioned experiment configuration.

    The returned mapping is the single source of scientific-contract values.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{config_path}: configuration root must be an object")
    if config.get("config_id") != "experiment.v1":
        raise ValueError(f"{config_path}: unexpected config_id {config.get('config_id')!r}")
    return config


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now_stamp() -> str:
    """Compact UTC stamp used inside the run identity."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def git_head_short(root: Path | str | None = None) -> str:
    """Short git HEAD, or ``nogit`` when unavailable.

    Read-only. Never changes identity, index or refs.
    """
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(root) if root is not None else REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "nogit"
    head = completed.stdout.strip()
    return head if completed.returncode == 0 and head else "nogit"


def make_run_id(root: Path | str | None = None) -> str:
    """``YYYYMMDDTHHMMSSZ-<short-git-head-or-nogit>``, generated once per run."""
    return f"{utc_now_stamp()}-{git_head_short(root)}"


def write_json(path: Path | str, payload: Any) -> str:
    """Write deterministic UTF-8 JSON and return the file SHA-256."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False) + "\n"
    target.write_text(text, encoding="utf-8", newline="\n")
    return sha256_file(target)
