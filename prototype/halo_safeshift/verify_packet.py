"""Verify a frozen Colab packet before anything reads its data.

The packet is the only thing that crosses from the preparation machine into a
training environment, and once it is there nobody can re-derive what it was
supposed to contain. So it carries its own attestation and this module checks
it, in an order that matters:

1. the packet manifest is read from *inside* the packet;
2. every declared file is hashed and compared;
3. the packet is checked for files nobody declared;
4. the independent records — source manifest, code manifest, data manifest —
   are cross-checked against each other and against the bytes on disk;
5. the source archive's members are hashed *out of the archive*, not off the
   preparation machine's disk;
6. only then may a caller load the CSV.

Every failure raises before the data is opened. A packet that fails here is not
partially usable: a hash mismatch means the bytes are not the bytes that were
prepared, and nothing downstream can compensate for that.

Usage (Linux / Colab):

    python -m prototype.halo_safeshift.verify_packet --packet colab-input
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import load_config, sha256_file
from .source_archive import archive_member_hashes

__all__ = ["PacketVerificationError", "verify_packet", "main"]

MANIFEST_FILENAME = "colab_packet_manifest.json"


class PacketVerificationError(RuntimeError):
    """Raised when a packet cannot be proven to be the packet that was prepared."""


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise PacketVerificationError(f"{path}: {label} is missing")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PacketVerificationError(f"{path}: {label} is not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise PacketVerificationError(f"{path}: {label} root must be an object")
    return payload


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PacketVerificationError(message)


def _compare(label: str, expected: Any, actual: Any) -> None:
    if expected != actual:
        raise PacketVerificationError(
            f"{label} mismatch\n  expected {expected!r}\n  actual   {actual!r}"
        )


def verify_packet(
    packet_dir: Path | str,
    config: dict[str, Any] | None = None,
    *,
    packet_manifest: Path | str | None = None,
) -> dict[str, Any]:
    """Verify every attestation in a packet. Returns the verification record."""
    resolved = config or load_config()
    settings = resolved["packet"]
    directory = Path(packet_dir)
    _require(directory.is_dir(), f"{directory}: packet directory does not exist")

    checks: list[str] = []

    # ---------------------------------------------------------------- #
    # 1. Packet manifest
    # ---------------------------------------------------------------- #
    manifest_path = (
        Path(packet_manifest) if packet_manifest is not None else directory / MANIFEST_FILENAME
    )
    manifest = _read_json(manifest_path, "packet manifest")
    _compare("packet manifest config_id", resolved["config_id"], manifest.get("config_id"))
    run_id = manifest.get("run_id")
    _require(bool(run_id), f"{manifest_path}: packet manifest declares no run_id")
    checks.append("packet manifest present and readable")

    declared = manifest.get("files")
    _require(
        isinstance(declared, dict) and bool(declared),
        f"{manifest_path}: packet manifest declares no files",
    )
    assert isinstance(declared, dict)

    # ---------------------------------------------------------------- #
    # 2. Declared file hashes
    # ---------------------------------------------------------------- #
    for name, expected in sorted(declared.items()):
        path = directory / name
        _require(path.is_file(), f"{directory}: declared packet file {name} is missing")
        _require(
            isinstance(expected, str) and len(expected) == 64,
            f"{manifest_path}: no valid SHA-256 declared for {name}",
        )
        actual = sha256_file(path)
        if actual != expected:
            raise PacketVerificationError(
                f"{directory}: SHA-256 mismatch for {name}\n"
                f"  declared {expected}\n  actual   {actual}"
            )
    checks.append(f"{len(declared)} declared packet files hash-verified")

    # ---------------------------------------------------------------- #
    # 3. Nothing undeclared
    # ---------------------------------------------------------------- #
    allowed = set(declared) | {manifest_path.name} | set(settings["allowed_extra_files"])
    present = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file()
    }
    undeclared = sorted(present - allowed)
    if undeclared:
        raise PacketVerificationError(
            f"{directory}: the packet contains undeclared file(s) {undeclared}. "
            f"{settings['allowed_extra_files_note']}"
        )
    missing = sorted(set(declared) - present)
    _require(not missing, f"{directory}: declared but absent: {missing}")
    checks.append("no undeclared file in the packet")

    # ---------------------------------------------------------------- #
    # 4. Config identity
    # ---------------------------------------------------------------- #
    packet_config = _read_json(directory / "experiment.v1.json", "packet config")
    _compare("packet config config_id", resolved["config_id"], packet_config.get("config_id"))
    packet_config_sha = sha256_file(directory / "experiment.v1.json")
    checks.append("packet config identity verified")

    # ---------------------------------------------------------------- #
    # 5. Independent records, cross-checked
    # ---------------------------------------------------------------- #
    source = _read_json(directory / "source_manifest.json", "source manifest")
    code = _read_json(directory / "code_manifest.json", "code manifest")
    data = _read_json(directory / "data_manifest.json", "data manifest")

    integrity = data.get("integrity")
    _require(isinstance(integrity, dict), "data_manifest.json carries no integrity block")
    assert isinstance(integrity, dict)

    csv_sha = sha256_file(directory / "station_raw.csv")
    metadata_sha = sha256_file(directory / "channel_metadata.json")
    _compare("station_raw.csv vs data_manifest.integrity.csv_sha256", integrity.get("csv_sha256"), csv_sha)
    _compare(
        "channel_metadata.json vs data_manifest.integrity.metadata_sha256",
        integrity.get("metadata_sha256"),
        metadata_sha,
    )
    _compare(
        "source_manifest.station_raw_csv.sha256 vs data_manifest",
        integrity.get("csv_sha256"),
        (source.get("station_raw_csv") or {}).get("sha256"),
    )
    _compare(
        "source_manifest.channel_metadata.sha256 vs data_manifest",
        integrity.get("metadata_sha256"),
        (source.get("channel_metadata") or {}).get("sha256"),
    )
    checks.append("raw CSV and channel metadata verified against the data manifest")

    puller = source.get("puller") or {}
    _compare("puller path", integrity.get("puller_path"), puller.get("path"))
    _compare("puller sha256", integrity.get("puller_sha256"), puller.get("sha256"))
    checks.append("puller path and hash agree across the source and data manifests")

    _compare(
        "channel id",
        int(resolved["source"]["thingspeak_channel_id"]),
        int((data.get("source") or {}).get("channel_id", -1)),
    )
    checks.append("channel id matches the configured station")

    # ---------------------------------------------------------------- #
    # 6. Source archive
    # ---------------------------------------------------------------- #
    archive_name = settings["source_archive"]
    archive_record = code.get("source_archive") or {}
    _require(
        archive_name in declared,
        f"{manifest_path}: the packet manifest does not declare {archive_name}; "
        f"without it Colab has no attested copy of the code that produced this packet",
    )
    _compare(
        f"{archive_name} hash in code_manifest vs packet manifest",
        declared[archive_name],
        archive_record.get("sha256"),
    )
    stored = archive_member_hashes(directory / archive_name)
    recorded_members = archive_record.get("members") or {}
    _compare(
        "source archive member set",
        sorted(recorded_members),
        sorted(stored),
    )
    differing = sorted(name for name, sha in stored.items() if recorded_members.get(name) != sha)
    _require(
        not differing,
        f"{archive_name}: member content differs from code_manifest for {differing}",
    )
    _compare(
        "code_manifest python_files vs archive members",
        sorted(name for name in code.get("python_files", {})),
        sorted(name for name in stored if name.endswith(".py")),
    )
    for name, sha in (code.get("python_files") or {}).items():
        _compare(f"code_manifest python_files[{name}]", stored.get(name), sha)
    for name, sha in (code.get("config_files") or {}).items():
        _compare(f"code_manifest config_files[{name}]", stored.get(name), sha)
    checks.append(f"source archive verified: {len(stored)} members hashed from inside the archive")

    packet_config_member = next(
        (name for name in stored if name.endswith("config/experiment.v1.json")), None
    )
    if packet_config_member is not None:
        _compare(
            "packet experiment.v1.json vs the config inside the source archive",
            stored[packet_config_member],
            packet_config_sha,
        )
        checks.append("packet config matches the config inside the source archive")

    # ---------------------------------------------------------------- #
    # 7. Run identity consistency
    # ---------------------------------------------------------------- #
    run_ids = {"colab_packet_manifest.json": run_id}
    for name in ("qc_report.json", "recovery_partition.json"):
        payload = _read_json(directory / name, name)
        run_ids[name] = payload.get("run_id")
    for name in ("source_manifest.json", "code_manifest.json"):
        payload = source if name.startswith("source") else code
        if payload.get("run_id") is not None:
            run_ids[name] = payload.get("run_id")
    disagreeing = {k: v for k, v in run_ids.items() if v != run_id}
    _require(
        not disagreeing,
        f"packet run_id is {run_id!r} but these artifacts disagree: {disagreeing}",
    )
    checks.append(f"run identity consistent across {len(run_ids)} artifacts: {run_id}")

    return {
        "artifact": "packet_verification",
        "verified": True,
        "packet_dir": directory.as_posix(),
        "run_id": run_id,
        "config_id": resolved["config_id"],
        "n_declared_files": len(declared),
        "n_source_archive_members": len(stored),
        "checks": checks,
        "boundary": [
            "Integrity verification only. It proves the packet is the packet that was prepared.",
            "It says nothing about data quality, model quality or any performance claim.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify a frozen HALO SafeShift Colab packet before any data is read."
    )
    parser.add_argument("--packet", type=Path, required=True, help="the colab-input directory")
    parser.add_argument(
        "--packet-manifest",
        type=Path,
        default=None,
        help="explicit manifest path; by default the manifest inside the packet is used",
    )
    args = parser.parse_args(argv)

    try:
        report = verify_packet(args.packet, packet_manifest=args.packet_manifest)
    except PacketVerificationError as exc:
        print(f"PACKET VERIFICATION FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
