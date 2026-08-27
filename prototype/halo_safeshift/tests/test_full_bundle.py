"""Validation of a completed full-Colab ZIP before any board deployment."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from prototype.halo_safeshift.full_bundle import BundleValidationError, validate_full_zip


class TestFullZip(unittest.TestCase):
    def build_zip(self, root: Path, *, corrupt=False) -> Path:
        run = root / "run"; run.mkdir()
        for name in ("metrics.json", "model_comparison.csv", "ablation_metrics.json", "feature_schema.json", "satellite_feature_manifest.json", "decoder_validation.json", "split.json", "qc_report.json", "replay_samples.json"):
            (run / name).write_text("{}" if name.endswith("json") else "trial_id\n", encoding="utf-8")
        (run / "station_only_bundle").mkdir(); (run / "operational_bundle").mkdir()
        (run / "satellite_only_bundle").mkdir()
        (run / "station_only_bundle" / "model.json").write_text("{}", encoding="utf-8")
        (run / "operational_bundle" / "model.json").write_text("{}", encoding="utf-8")
        (run / "satellite_only_bundle" / "BLOCKED.json").write_text("{}", encoding="utf-8")
        (run / "full_runtime.py").write_text("# embedded runtime\n", encoding="utf-8")
        (run / "full_dashboard.py").write_text("# embedded dashboard\n", encoding="utf-8")
        (run / "runtime_source_manifest.json").write_text("{}", encoding="utf-8")
        files = {str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest() for p in run.rglob("*") if p.is_file()}
        (run / "manifest.json").write_text(json.dumps({"files": files, "satellite_gate": {"status": "blocked"}, "fusion_gate": {"pass": False}}), encoding="utf-8")
        if corrupt:
            (run / "metrics.json").write_text('{"tampered":true}', encoding="utf-8")
        archive = root / "full.zip"
        with zipfile.ZipFile(archive, "w") as out:
            for path in run.rglob("*"):
                if path.is_file(): out.write(path, path.relative_to(run))
        return archive

    def test_validates_required_files_and_manifest_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = validate_full_zip(self.build_zip(Path(tmp)))
            self.assertFalse(report["fused_bundle_required"])
            self.assertEqual(report["satellite_gate"], "blocked")

    def test_rejects_tampered_member(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(BundleValidationError):
                validate_full_zip(self.build_zip(Path(tmp), corrupt=True))


if __name__ == "__main__":
    unittest.main()
