"""P1 dashboard's replay and board-origin API contract."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from prototype.halo_safeshift.full_dashboard import DashboardState


class TestFullDashboard(unittest.TestCase):
    def make_run(self, root: Path) -> Path:
        bundle = root / "operational_bundle"; bundle.mkdir()
        names = ["station_at_now"]
        (bundle / "feature_schema.json").write_text(json.dumps({"n_features": 1, "feature_names": names}))
        (bundle / "model.json").write_text(json.dumps({"format": "halo-safeshift-portable-v2", "model_type": "persistence", "n_features": 1, "runtime_input_names": names, "at_now_index": 0, "target": "direct", "preprocessing": {"mean": [0.0], "scale": [1.0]}}))
        np.savez(bundle / "model.npz", marker=np.array([0]))
        (bundle / "manifest.json").write_text(json.dumps({"files": {name: hashlib.sha256((bundle / name).read_bytes()).hexdigest() for name in ("model.json", "feature_schema.json", "model.npz")}}))
        replay = [{"issue_time_utc": f"2026-08-01T00:{i*5:02d}:00+00:00", "target_time_utc": f"2026-08-01T00:{30+i*5:02d}:00+00:00", "current_at_degc": 30+i, "persistence_degc": 30+i, "observed_at_degc": 29+i} for i in range(8)]
        (root / "replay_samples.json").write_text(json.dumps(replay))
        (root / "metrics.json").write_text("{}")
        (root / "manifest.json").write_text(json.dumps({"satellite_gate": {"status": "blocked"}, "fusion_gate": {"pass": False}, "learned_gate": {"pass": False}}))
        return root

    def test_observed_target_is_hidden_until_six_five_minute_steps_later(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = DashboardState(self.make_run(Path(tmp)))
            for _ in range(6):
                self.assertIsNone(state.next()["observed"])
            reveal = state.next()
            self.assertTrue(reveal["target_revealed"])
            self.assertEqual(reveal["observed"], 29.0)

    def test_status_never_calls_station_only_bundle_fused(self):
        with tempfile.TemporaryDirectory() as tmp:
            status = DashboardState(self.make_run(Path(tmp))).status()
            self.assertEqual(status["mode"], "station-only")
            self.assertEqual(status["board"]["hostname"], __import__("platform").node())


if __name__ == "__main__":
    unittest.main()
