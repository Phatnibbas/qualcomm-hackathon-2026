"""Fail-closed verifier for a downloaded full Colab artifact ZIP."""

from __future__ import annotations

import hashlib
import json
import argparse
import tempfile
import zipfile
from pathlib import Path
from typing import Any


class BundleValidationError(RuntimeError):
    """The Colab artifact cannot be safely considered for board deployment."""


BASE_REQUIRED = {
    "metrics.json",
    "model_comparison.csv",
    "ablation_metrics.json",
    "feature_schema.json",
    "satellite_feature_manifest.json",
    "decoder_validation.json",
    "split.json",
    "qc_report.json",
    "replay_samples.json",
    "manifest.json",
    "full_runtime.py",
    "full_dashboard.py",
    "runtime_source_manifest.json",
    "station_only_bundle/model.json",
    "operational_bundle/model.json",
    "satellite_only_bundle",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_full_zip(path: Path | str) -> dict[str, Any]:
    """Verify required artifact topology and every manifest-declared member hash."""
    archive = Path(path)
    if not archive.is_file():
        raise BundleValidationError(f"full Colab ZIP is missing: {archive}")
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        with zipfile.ZipFile(archive) as bundle:
            members = {item.filename for item in bundle.infolist() if not item.is_dir()}
            unsafe = [name for name in members if name.startswith(("/", "\\")) or ".." in Path(name).parts]
            if unsafe:
                raise BundleValidationError(f"ZIP contains unsafe member path(s): {unsafe}")
            # Directory entries are optional in ZIPs, so test an actual member
            # below instead of requiring a directory record.
            missing = sorted((BASE_REQUIRED - {"satellite_only_bundle"}) - members)
            if missing:
                raise BundleValidationError(f"full ZIP is missing required artifacts: {missing}")
            if not any(name.startswith("satellite_only_bundle/") for name in members):
                raise BundleValidationError("full ZIP lacks satellite_only_bundle artifact")
            bundle.extractall(root)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        declared = manifest.get("files")
        if not isinstance(declared, dict):
            raise BundleValidationError("manifest.files must be an object")
        for relative, expected in declared.items():
            artifact = root / relative
            if not artifact.is_file() or not isinstance(expected, str) or _sha256(artifact) != expected:
                raise BundleValidationError(f"manifest hash verification failed: {relative}")
        satellite_gate = (manifest.get("satellite_gate") or {}).get("status")
        fusion_pass = bool((manifest.get("fusion_gate") or {}).get("pass"))
        fused_required = satellite_gate == "pass" and fusion_pass
        if fused_required:
            for required in ("fused_bundle/model.json",):
                if required not in members:
                    raise BundleValidationError("fusion gate passed but fused bundle is missing")
        return {
            "zip": str(archive),
            "zip_sha256": _sha256(archive),
            "manifest_members_verified": len(declared),
            "satellite_gate": satellite_gate,
            "fusion_gate": fusion_pass,
            "fused_bundle_required": fused_required,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a HALO SafeShift full Colab ZIP")
    parser.add_argument("artifact_zip", type=Path)
    print(json.dumps(validate_full_zip(parser.parse_args().artifact_zip), indent=2))


if __name__ == "__main__":
    main()
