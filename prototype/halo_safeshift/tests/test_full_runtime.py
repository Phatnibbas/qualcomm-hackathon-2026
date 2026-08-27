"""Runtime contract for P1 portable full-pipeline bundles."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from prototype.halo_safeshift.full_runtime import FullRuntimeError, PortablePredictor


class TestPortablePredictor(unittest.TestCase):
    def make_bundle(self, directory: Path, *, feature_names=None, model_type="linear") -> Path:
        feature_names = feature_names or ["station_x", "satellite_b13_p50_latest"]
        bundle = directory / "bundle"; bundle.mkdir()
        schema = {"n_features": len(feature_names), "feature_names": feature_names}
        (bundle / "feature_schema.json").write_text(json.dumps(schema), encoding="utf-8")
        payload = {"format": "halo-safeshift-portable-v2", "model_type": model_type, "n_features": len(feature_names), "runtime_input_names": feature_names, "at_now_index": 0, "parameterization": "direct"}
        (bundle / "model.json").write_text(json.dumps(payload), encoding="utf-8")
        np.savez(bundle / "model.npz", coef=np.array([2.0, 3.0]), intercept=np.array([1.0]), mean=np.zeros(2), scale=np.ones(2))
        files = {name: hashlib.sha256((bundle / name).read_bytes()).hexdigest() for name in ("model.json", "feature_schema.json", "model.npz")}
        (bundle / "manifest.json").write_text(json.dumps({"files": files}), encoding="utf-8")
        return bundle

    def test_colab_to_edge_parity_for_linear_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            predictor = PortablePredictor.load(self.make_bundle(Path(tmp)))
            np.testing.assert_allclose(predictor.predict(np.array([[2.0, 4.0]])), [17.0])

    def test_fused_mode_rejects_bundle_without_satellite_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = self.make_bundle(Path(tmp), feature_names=["station_x"])
            with self.assertRaises(FullRuntimeError):
                PortablePredictor.load(bundle, require_fused=True)

    def test_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = self.make_bundle(Path(tmp))
            (bundle / "model.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(FullRuntimeError):
                PortablePredictor.load(bundle)

    def test_persistence_bundle_uses_declared_at_now_feature(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = self.make_bundle(Path(tmp), model_type="persistence")
            model_path = bundle / "model.json"
            model = json.loads(model_path.read_text(encoding="utf-8"))
            model["at_now_index"] = 0
            model_path.write_text(json.dumps(model), encoding="utf-8")
            manifest_path = bundle / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["files"]["model.json"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            predictor = PortablePredictor.load(bundle)
            np.testing.assert_allclose(predictor.predict(np.array([[29.5, 4.0]])), [29.5])

    def test_portable_tree_ensemble_matches_declared_split_and_mean_aggregation(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = self.make_bundle(Path(tmp), feature_names=["station_x"], model_type="tree_ensemble")
            model_path = bundle / "model.json"; model = json.loads(model_path.read_text(encoding="utf-8"))
            model.update({"aggregation": "mean", "learning_rate": 1.0, "base_score": 0.0, "tree_count": 1})
            model_path.write_text(json.dumps(model), encoding="utf-8")
            np.savez(bundle / "model.npz", tree_offsets=np.array([0, 3]), tree_feature=np.array([0, -1, -1]), tree_threshold=np.array([1.0, 0.0, 0.0]), tree_left=np.array([1, -1, -1]), tree_right=np.array([2, -1, -1]), tree_value=np.array([0.0, 10.0, 20.0]), mean=np.zeros(1), scale=np.ones(1))
            manifest_path = bundle / "manifest.json"; manifest = json.loads(manifest_path.read_text(encoding="utf-8")); manifest["files"] = {name: hashlib.sha256((bundle / name).read_bytes()).hexdigest() for name in ("model.json", "feature_schema.json", "model.npz")}; manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            predictor = PortablePredictor.load(bundle)
            np.testing.assert_allclose(predictor.predict(np.array([[0.5], [1.5]])), [10.0, 20.0])


if __name__ == "__main__":
    unittest.main()
