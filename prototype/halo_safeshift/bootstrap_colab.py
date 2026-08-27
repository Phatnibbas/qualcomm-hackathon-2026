"""Standalone Colab bootstrap: verify, extract, verify again, then hand over.

This file is deliberately dependency-free and import-free with respect to its
own package. It runs *before* the package exists in the training environment,
so it may use nothing but the Python standard library. It is shipped both as a
package module and as a copy inside the packet (``bootstrap.py``), because a
notebook that has only the packet has no other way to obtain it.

Order of operations, and why it is this order:

1. read ``code_manifest.json`` from the packet and hash the source archive;
2. compare that hash **before** opening the archive — extracting first and
   checking afterwards would already have written unverified files to disk;
3. extract into the target root, refusing any member whose path escapes it;
4. re-hash every extracted file against the code manifest, which catches a
   truncated or partially-written extraction;
5. run the full packet verification, which is the package's own check;
6. invoke ``colab_train --help`` as a smoke test that the extracted package
   imports and its entry point is wired.

Step 6 starts no training. It proves the module loads.

Linux / Colab:

    python colab-input/bootstrap.py --packet colab-input --root .
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

ARCHIVE_NAME = "halo_safeshift-source.zip"


class BootstrapError(RuntimeError):
    """Raised when the environment cannot be prepared from a verified packet."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive(packet: Path) -> tuple[Path, dict]:
    """Hash the archive and compare it to the code manifest. No extraction yet."""
    manifest_path = packet / "code_manifest.json"
    if not manifest_path.is_file():
        raise BootstrapError(f"{manifest_path}: code manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = manifest.get("source_archive") or {}
    expected = record.get("sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise BootstrapError(f"{manifest_path}: no valid source-archive SHA-256 recorded")

    archive = packet / record.get("path", ARCHIVE_NAME)
    if not archive.is_file():
        raise BootstrapError(f"{archive}: source archive is missing")
    actual = sha256_file(archive)
    if actual != expected:
        raise BootstrapError(
            f"{archive}: source archive SHA-256 mismatch\n"
            f"  expected {expected}\n  actual   {actual}\n"
            f"Refusing to extract unverified code."
        )
    return archive, manifest


def extract(archive: Path, root: Path) -> list[str]:
    """Extract into ``root``, rejecting any member that escapes it."""
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    extracted: list[str] = []
    with zipfile.ZipFile(archive) as handle:
        corrupt = handle.testzip()
        if corrupt is not None:
            raise BootstrapError(f"{archive}: corrupt member {corrupt!r}")
        for name in handle.namelist():
            destination = (root / name).resolve()
            if root not in destination.parents and destination != root:
                raise BootstrapError(
                    f"{archive}: member {name!r} would write outside the extraction root"
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(handle.read(name))
            extracted.append(name)
    return sorted(extracted)


def verify_extracted(manifest: dict, root: Path) -> int:
    """Re-hash what actually landed on disk against the code manifest."""
    members = (manifest.get("source_archive") or {}).get("members") or {}
    if not members:
        raise BootstrapError("code manifest records no archive members to verify")
    for name, expected in sorted(members.items()):
        path = root / name
        if not path.is_file():
            raise BootstrapError(f"{path}: expected extracted file is missing")
        actual = sha256_file(path)
        if actual != expected:
            raise BootstrapError(
                f"{path}: extracted file SHA-256 mismatch\n"
                f"  expected {expected}\n  actual   {actual}"
            )
    return len(members)


def run_module(root: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", *args],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify a HALO SafeShift packet, extract its source archive and smoke-test it."
    )
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="skip the colab_train --help entry-point check",
    )
    args = parser.parse_args(argv)

    packet = args.packet.resolve()
    root = args.root.resolve()
    steps: list[str] = []

    archive, manifest = verify_archive(packet)
    steps.append(f"source archive hash verified: {archive.name}")

    extracted = extract(archive, root)
    steps.append(f"extracted {len(extracted)} members into {root.as_posix()}")

    n_verified = verify_extracted(manifest, root)
    steps.append(f"re-hashed {n_verified} extracted files against the code manifest")

    verification = run_module(
        root, ["prototype.halo_safeshift.verify_packet", "--packet", str(packet)]
    )
    if verification.returncode != 0:
        raise BootstrapError(
            f"packet verification failed with exit {verification.returncode}:\n"
            f"{verification.stderr.strip()}"
        )
    steps.append("packet verification passed")

    if not args.skip_smoke:
        smoke = run_module(root, ["prototype.halo_safeshift.colab_train", "--help"])
        if smoke.returncode != 0:
            raise BootstrapError(
                f"colab_train --help failed with exit {smoke.returncode}:\n"
                f"{smoke.stderr.strip()}"
            )
        steps.append("colab_train entry point imports and responds to --help")

    print(
        json.dumps(
            {
                "bootstrap": "ok",
                "packet": packet.as_posix(),
                "root": root.as_posix(),
                "steps": steps,
                "boundary": [
                    "Environment preparation only.",
                    "No training was started; --help proves the entry point imports, nothing more.",
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BootstrapError as exc:
        print(f"BOOTSTRAP FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
