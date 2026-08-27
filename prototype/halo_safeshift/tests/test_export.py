"""Bundle export: reference parity, pickle-free storage and manifest integrity."""

from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from prototype.halo_safeshift import load_config
from prototype.halo_safeshift.export import (
    SIBLING_FILES,
    ExportError,
    Preprocessing,
    export_estimator,
    export_linear,
    export_mlp_relu,
    export_persistence,
    export_tree_ensemble,
)
from prototype.halo_safeshift.features import build_feature_schema
from prototype.halo_safeshift.inference import SafeShiftPredictor
from prototype.halo_safeshift.models import build_estimator
from prototype.halo_safeshift.tests import FIELD_MAPPING
from prototype.halo_safeshift.tree_export import export_sklearn_gradient_boosting

CONFIG = load_config()
SCHEMA = build_feature_schema(FIELD_MAPPING, CONFIG)
N_FEATURES = SCHEMA["n_features"]
TOL = CONFIG["tolerances"]


def synthetic(n_samples: int = 200, seed: int = 20260815):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_samples, N_FEATURES))
    y = 0.8 * x[:, 0] - 0.3 * x[:, 5] + 0.2 * x[:, -1] + rng.normal(scale=0.05, size=n_samples)
    return x, y


def train_only_preprocessing(x: np.ndarray) -> Preprocessing:
    """Fit standardisation on the supplied block only."""
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale <= 0] = 1.0
    return Preprocessing(
        kind="standardize", mean=mean, scale=scale, fit_scope="synthetic train block only"
    )


class TestLinearParity(unittest.TestCase):
    def test_float64_parity_within_the_configured_tolerance(self):
        x, y = synthetic()
        pre = train_only_preprocessing(x)
        scaled = (x - pre.mean) / pre.scale
        estimator = build_estimator("ridge", {"alpha": 1.0}, CONFIG).fit(scaled, y)
        with tempfile.TemporaryDirectory() as tmp:
            bundle = export_linear(Path(tmp) / "b", estimator, SCHEMA, pre, dtype="float64", config=CONFIG)
            predictor = SafeShiftPredictor.load(bundle)
            error = float(np.max(np.abs(predictor.predict_features(x) - estimator.predict(scaled))))
        self.assertLessEqual(error, float(TOL["linear_fp64_max_abs_error"]), f"error {error:.3e}")

    def test_float32_parity_within_the_configured_tolerance(self):
        x, y = synthetic()
        pre = train_only_preprocessing(x)
        scaled = (x - pre.mean) / pre.scale
        estimator = build_estimator("ridge", {"alpha": 1.0}, CONFIG).fit(scaled, y)
        with tempfile.TemporaryDirectory() as tmp:
            bundle = export_linear(Path(tmp) / "b", estimator, SCHEMA, pre, dtype="float32", config=CONFIG)
            predictor = SafeShiftPredictor.load(bundle)
            error = float(np.max(np.abs(predictor.predict_features(x) - estimator.predict(scaled))))
        self.assertLessEqual(error, float(TOL["linear_fp32_max_abs_error"]), f"error {error:.3e}")

    def test_coefficient_width_mismatch_is_refused(self):
        x, y = synthetic()
        estimator = build_estimator("ridge", {"alpha": 1.0}, CONFIG).fit(x[:, :10], y)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ExportError):
                export_linear(Path(tmp) / "b", estimator, SCHEMA, Preprocessing(), config=CONFIG)


class TestMlpParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x, cls.y = synthetic(n_samples=300)
        cls.pre = train_only_preprocessing(cls.x)
        cls.scaled = (cls.x - cls.pre.mean) / cls.pre.scale
        cls.estimator = build_estimator(
            "mlp_relu",
            {"hidden_layer_sizes": (16, 8), "max_iter": 60, "learning_rate_init": 0.01},
            CONFIG,
        ).fit(cls.scaled, cls.y)

    def test_float64_parity(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = export_mlp_relu(
                Path(tmp) / "b", self.estimator, SCHEMA, self.pre, dtype="float64", config=CONFIG
            )
            predictor = SafeShiftPredictor.load(bundle)
            error = float(
                np.max(np.abs(predictor.predict_features(self.x) - self.estimator.predict(self.scaled)))
            )
        self.assertLessEqual(error, float(TOL["mlp_fp32_max_abs_error"]), f"error {error:.3e}")

    def test_float32_parity(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = export_mlp_relu(
                Path(tmp) / "b", self.estimator, SCHEMA, self.pre, dtype="float32", config=CONFIG
            )
            predictor = SafeShiftPredictor.load(bundle)
            error = float(
                np.max(np.abs(predictor.predict_features(self.x) - self.estimator.predict(self.scaled)))
            )
        self.assertLessEqual(error, float(TOL["mlp_fp32_max_abs_error"]), f"error {error:.3e}")

    def test_non_relu_activation_is_refused(self):
        estimator = build_estimator(
            "mlp_relu", {"hidden_layer_sizes": (4,), "max_iter": 5}, CONFIG
        )
        estimator.activation = "tanh"
        estimator.fit(self.scaled[:60], self.y[:60])
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ExportError):
                export_mlp_relu(Path(tmp) / "b", estimator, SCHEMA, self.pre, config=CONFIG)


class TestTreeBundle(unittest.TestCase):
    def test_gradient_boosting_bundle_matches_sklearn(self):
        x, y = synthetic(n_samples=200)
        estimator = build_estimator(
            "gradient_boosting", {"n_estimators": 20, "max_depth": 2}, CONFIG
        ).fit(x, y)
        neutral = export_sklearn_gradient_boosting(estimator)
        with tempfile.TemporaryDirectory() as tmp:
            bundle = export_tree_ensemble(
                Path(tmp) / "b", neutral, SCHEMA, Preprocessing(), config=CONFIG
            )
            predictor = SafeShiftPredictor.load(bundle)
            error = float(np.max(np.abs(predictor.predict_features(x) - estimator.predict(x))))
        self.assertLessEqual(error, float(TOL["tree_ensemble_fp64_max_abs_error"]))

    def test_dispatch_by_estimator_type(self):
        x, y = synthetic(n_samples=150)
        estimator = build_estimator("extra_trees", {"n_estimators": 8, "max_depth": 5}, CONFIG).fit(x, y)
        with tempfile.TemporaryDirectory() as tmp:
            bundle = export_estimator(
                Path(tmp) / "b", estimator, SCHEMA, Preprocessing(), config=CONFIG
            )
            manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["model_type"], "tree_ensemble")
            predictor = SafeShiftPredictor.load(bundle)
            error = float(np.max(np.abs(predictor.predict_features(x) - estimator.predict(x))))
        self.assertLessEqual(error, float(TOL["tree_ensemble_fp64_max_abs_error"]))


class TestBundleIntegrity(unittest.TestCase):
    def _bundle(self, tmp: str) -> Path:
        return export_persistence(Path(tmp) / "b", SCHEMA, run_id="test-run", config=CONFIG)

    def test_every_sibling_is_written_and_hashed(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = self._bundle(tmp)
            manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
            for name in SIBLING_FILES:
                self.assertTrue((bundle / name).is_file(), f"{name} was not written")
                self.assertEqual(len(manifest["files"][name]), 64)

    def test_model_npz_contains_no_pickled_objects(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = self._bundle(tmp)
            with zipfile.ZipFile(bundle / "model.npz") as archive:
                names = archive.namelist()
            self.assertTrue(all(name.endswith(".npy") for name in names), names)
            with np.load(bundle / "model.npz", allow_pickle=False) as handle:
                self.assertIn("at_now_index", handle.files)

    def test_persistence_bundle_predicts_the_current_apparent_temperature(self):
        with tempfile.TemporaryDirectory() as tmp:
            predictor = SafeShiftPredictor.load(self._bundle(tmp))
            index = SCHEMA["index"]["apparent_temperature_now"]
            x = np.zeros((3, N_FEATURES))
            x[:, index] = [30.0, 31.0, 32.0]
            np.testing.assert_allclose(predictor.predict_features(x), [30.0, 31.0, 32.0])

    def test_residual_parameterization_adds_back_the_current_value(self):
        x, y = synthetic(n_samples=120)
        at_now_index = SCHEMA["index"]["apparent_temperature_now"]
        residual = y - x[:, at_now_index]
        estimator = build_estimator("ridge", {"alpha": 1.0}, CONFIG).fit(x, residual)
        with tempfile.TemporaryDirectory() as tmp:
            bundle = export_linear(
                Path(tmp) / "b",
                estimator,
                SCHEMA,
                Preprocessing(),
                parameterization="residual",
                config=CONFIG,
            )
            manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["output_transform"]["kind"], "residual_plus_feature")
            self.assertIn("post-hoc", str(manifest["parameterization_provenance"]).lower())
            predictor = SafeShiftPredictor.load(bundle)
            expected = estimator.predict(x) + x[:, at_now_index]
            error = float(np.max(np.abs(predictor.predict_features(x) - expected)))
        self.assertLessEqual(error, float(TOL["linear_fp64_max_abs_error"]))

    def test_disallowed_model_type_is_refused(self):
        x, y = synthetic(n_samples=80)
        estimator = build_estimator("ridge", {"alpha": 1.0}, CONFIG).fit(x, y)
        narrowed = json.loads(json.dumps(CONFIG))
        narrowed["export"]["allowed_model_types"] = ["persistence"]
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ExportError):
                export_linear(Path(tmp) / "b", estimator, SCHEMA, Preprocessing(), config=narrowed)

    def test_preprocessing_requires_positive_scale(self):
        with self.assertRaises(ExportError):
            Preprocessing(
                kind="standardize",
                mean=np.zeros(N_FEATURES),
                scale=np.zeros(N_FEATURES),
                fit_scope="synthetic",
            )

    def test_manifest_records_the_claim_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = json.loads((self._bundle(tmp) / "manifest.json").read_text(encoding="utf-8"))
            joined = " ".join(manifest["claim_boundary"]).lower()
            self.assertIn("no performance claim", joined)
            self.assertIn("not a safety classification", joined)


if __name__ == "__main__":
    unittest.main()
