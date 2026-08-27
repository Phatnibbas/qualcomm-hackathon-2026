"""Deterministic source archive: transport the exact package into Colab.

The problem this solves. The repository working tree carries uncommitted
SafeShift source, so a Colab notebook that clones git HEAD does **not** get the
code that produced the packet. It would import a different pipeline, produce
results attributed to this run, and the code manifest's per-file hashes would
have nothing to verify against. Shipping the bytes and attesting their hash is
the only way "Colab ran this exact code" can be checked rather than assumed.

Why the determinism work is not optional. A ZIP entry stores a modification
time and a creator-platform byte. Left alone, building the archive twice from
byte-identical source yields two different archive hashes, which makes the hash
useless as an identity: a mismatch would prove nothing and a match could not be
produced. Every varying field is therefore pinned from the configuration:
entries are emitted in sorted path order with a fixed timestamp, fixed mode,
fixed creator system and a fixed compression level.

The archive contains source only — no data, no model, no metric.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

from . import PACKAGE_DIR, REPO_ROOT, load_config, sha256_bytes, sha256_file

__all__ = [
    "SourceArchiveError",
    "collect_source_files",
    "build_source_archive",
    "archive_member_hashes",
]


class SourceArchiveError(RuntimeError):
    """Raised when the source archive cannot be built deterministically."""


def collect_source_files(config: dict[str, Any] | None = None) -> list[tuple[str, Path]]:
    """``(archive_member_name, absolute_path)`` pairs, in sorted member order.

    Member names are repository-relative POSIX paths so that extracting the
    archive at a repository root reproduces the package in place and
    ``python -m prototype.halo_safeshift...`` resolves without a path shim.
    """
    resolved = config or load_config()
    excludes = set(resolved["packet"]["source_archive_excludes"])

    candidates: list[Path] = []
    candidates.extend(PACKAGE_DIR.rglob("*.py"))
    candidates.extend((PACKAGE_DIR / "config").glob("*.json"))
    candidates.extend(PACKAGE_DIR.glob("requirements-colab*.txt"))

    members: dict[str, Path] = {}
    for path in candidates:
        if not path.is_file():
            continue
        parts = set(path.parts)
        if parts & excludes or path.suffix == ".pyc":
            continue
        members[path.resolve().relative_to(REPO_ROOT).as_posix()] = path.resolve()

    if not members:
        raise SourceArchiveError(f"{PACKAGE_DIR}: no source files matched the archive contract")
    return sorted(members.items())


def build_source_archive(
    destination: Path | str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the archive and return its manifest entry.

    Building twice from identical source produces byte-identical output, so the
    returned ``sha256`` is a usable identity for "this exact package".
    """
    resolved = config or load_config()
    settings = resolved["packet"]["source_archive_determinism"]
    members = collect_source_files(resolved)

    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    date_time = tuple(int(v) for v in settings["zip_date_time"])

    with zipfile.ZipFile(
        target,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=int(settings["compresslevel"]),
    ) as archive:
        for name, path in members:
            info = zipfile.ZipInfo(filename=name, date_time=date_time)  # type: ignore[arg-type]
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = int(settings["external_attr"]) << 16
            info.create_system = int(settings["create_system"])
            archive.writestr(info, path.read_bytes())

    return {
        "path": target.name,
        "sha256": sha256_file(target),
        "bytes": target.stat().st_size,
        "n_members": len(members),
        "members": {name: sha256_file(path) for name, path in members},
        "determinism": dict(settings),
        "why": resolved["packet"]["why_an_archive"],
        "boundary": "Source code only. No data, no model, no metric.",
    }


def archive_member_hashes(archive_path: Path | str) -> dict[str, str]:
    """SHA-256 of each member as stored, read back out of the archive itself.

    Hashing the files on disk proves what was *intended*; hashing the archive's
    own contents proves what was actually shipped. The two are compared during
    packet verification.
    """
    with zipfile.ZipFile(Path(archive_path)) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise SourceArchiveError(f"{archive_path}: corrupt archive member {bad!r}")
        return {name: sha256_bytes(archive.read(name)) for name in sorted(archive.namelist())}
