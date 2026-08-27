#!/usr/bin/env python3
"""Verify every artifact listed in edge/manifest.json against its recorded SHA-256.

Run from the repository root:

    python edge/verify_manifest.py

Exits non-zero if any file is missing or its hash does not match, so this is
usable as a CI gate.

Note on the three figures: `heldout_replay.png`, `pruning_tradeoff.png` and
`training_history.png` were produced by the training run into the same output
directory, and are stored in `images/` in this repository so the README can
display them. Their hashes are still checked, from that location.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "edge" / "manifest.json"

# Artifacts relocated for presentation. Hash is unchanged and is still verified.
RELOCATED = {
    "heldout_replay.png": "images/heldout-replay.png",
    "pruning_tradeoff.png": "images/pruning-tradeoff.png",
    "training_history.png": "images/training-history.png",
}


def resolve(name: str) -> Path:
    return ROOT / RELOCATED.get(name, f"edge/{name}")


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    print(f"run_id: {manifest['run_id']}\n")

    ok = missing = mismatch = 0
    for name, entry in sorted(manifest["files"].items()):
        path = resolve(name)
        rel = path.relative_to(ROOT).as_posix()
        if not path.exists():
            print(f"MISSING   {rel}")
            missing += 1
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            print(f"MISMATCH  {rel}\n            expected {entry['sha256']}\n            actual   {digest}")
            mismatch += 1
            continue
        print(f"OK        {rel}  ({entry['bytes']:,} bytes)")
        ok += 1

    total = ok + missing + mismatch
    print(f"\n{ok}/{total} verified, {missing} missing, {mismatch} mismatched")
    return 0 if missing == mismatch == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
