"""Runtime integrity: the predictor must fail closed on every corruption path."""

from __future__ import annotations

import json
import pickle
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from prototype.halo_safeshift import load_config
from prototype.halo_safeshift.export import Preprocessing, export_linear, export_persistence
from prototype.halo_safeshift.features import build_feature_schema
from prototype.halo_safeshift.inference import InferenceError, SafeShiftPredictor
from prototype.halo_safeshift.models import build_estimator
from prototype.halo_safeshift.tests import FIELD_MAPPING, reseal

CONFIG = load_config()
SCHEMA = build_feature_schema(FIELD_MAPPING, CONFIG)
N_FEATURES = SCHEMA["n_features"]


def make_linear_bundle(directory: Path) -> Path:
    rng = np.random.default_rng(int(CONFIG["seed"]))
    x = rng.normal(size=(150, N_FEATURES))
    y = 0.5 * x[:, 3] - 0.2 * x[:, 11] + rng.normal(scale=0.01, size=150)
    estimator = build_estimator("ridge", {"alpha": 1.0}, CONFIG).fit(x, y)
    return export_linear(directory, estimator, SCHEMA, Preprocessing(), config=CONFIG)


class BundleCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.bundle = make_linear_bundle(self.root / "bundle")

    def tearDown(self):
        self._tmp.cleanup()

    def rewrite_manifest(self, mutate, *, reseal_envelope=True):
        """Mutate manifest.json and, by default, reseal the envelope.

        The envelope is verified before the manifest is read, so without a
        reseal every one of these tests would only ever prove the envelope
        works. Resealing isolates the inner check each test is about; the
        un-resealed case has its own dedicated tests in test_corrections.
        """
        path = self.bundle / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        mutate(manifest)
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        if reseal_envelope:
            reseal(self.bundle)


class TestHappyPath(BundleCase):
    def test_loads_and_predicts(self):
        predictor = SafeShiftPredictor.load(self.bundle)
        x = np.zeros((5, N_FEATURES))
        prediction = predictor.predict_features(x)
        self.assertEqual(prediction.shape, (5,))
        self.assertTrue(bool(np.all(np.isfinite(prediction))))

    def test_predictions_are_deterministic_across_loads(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(40, N_FEATURES))
        first = SafeShiftPredictor.load(self.bundle).predict_features(x)
        second = SafeShiftPredictor.load(self.bundle).predict_features(x)
        np.testing.assert_array_equal(first, second)

    def test_a_single_row_is_accepted(self):
        predictor = SafeShiftPredictor.load(self.bundle)
        self.assertEqual(predictor.predict_features(np.zeros(N_FEATURES)).shape, (1,))

    def test_runtime_provenance_reports_hashes_and_no_classification(self):
        provenance = SafeShiftPredictor.load(self.bundle).runtime_provenance()
        self.assertEqual(len(provenance["model_npz_sha256"]), 64)
        self.assertEqual(len(provenance["feature_schema_sha256"]), 64)
        joined = " ".join(provenance["claim_boundary"]).lower()
        self.assertIn("no medical or safety classification", joined)

    def test_matching_schema_is_accepted(self):
        SafeShiftPredictor.load(self.bundle).assert_schema(SCHEMA)


class TestFailClosed(BundleCase):
    def test_missing_manifest(self):
        (self.bundle / "manifest.json").unlink()
        with self.assertRaises(InferenceError) as caught:
            SafeShiftPredictor.load(self.bundle)
        self.assertIn("manifest.json", str(caught.exception))
        self.assertIn("missing", str(caught.exception))

    def test_missing_sibling_file(self):
        (self.bundle / "feature_schema.json").unlink()
        with self.assertRaises(InferenceError) as caught:
            SafeShiftPredictor.load(self.bundle)
        self.assertIn("feature_schema.json", str(caught.exception))
        self.assertIn("missing", str(caught.exception))

    def test_sibling_hash_mismatch_is_rejected(self):
        path = self.bundle / "feature_schema.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["schema_version"] = "tampered"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        with self.assertRaises(InferenceError) as caught:
            SafeShiftPredictor.load(self.bundle)
        self.assertIn("SHA-256 mismatch", str(caught.exception))

    def test_tampered_weights_are_rejected_by_the_hash(self):
        with np.load(self.bundle / "model.npz", allow_pickle=False) as handle:
            arrays = {key: np.array(handle[key]) for key in handle.files}
        arrays["linear_coef"] = arrays["linear_coef"] + 1.0
        with (self.bundle / "model.npz").open("wb") as handle:
            np.savez(handle, allow_pickle=False, **arrays)
        with self.assertRaises(InferenceError) as caught:
            SafeShiftPredictor.load(self.bundle)
        self.assertIn("model.npz", str(caught.exception))

    def test_unsupported_model_type_is_rejected(self):
        self.rewrite_manifest(lambda manifest: manifest.update({"model_type": "random_forest"}))
        with self.assertRaises(InferenceError) as caught:
            SafeShiftPredictor.load(self.bundle)
        self.assertIn("unsupported model type", str(caught.exception))

    def test_manifest_schema_width_disagreement_is_rejected(self):
        self.rewrite_manifest(lambda manifest: manifest.update({"n_features": N_FEATURES + 1}))
        with self.assertRaises(InferenceError) as caught:
            SafeShiftPredictor.load(self.bundle)
        self.assertIn("n_features", str(caught.exception))

    def test_foreign_schema_is_rejected(self):
        predictor = SafeShiftPredictor.load(self.bundle)
        foreign = json.loads(json.dumps(SCHEMA))
        foreign["feature_names"][0] = "not_the_same_feature"
        with self.assertRaises(InferenceError) as caught:
            predictor.assert_schema(foreign)
        self.assertIn("feature schema mismatch", str(caught.exception))

    def test_wrong_feature_count_is_rejected(self):
        predictor = SafeShiftPredictor.load(self.bundle)
        with self.assertRaises(InferenceError) as caught:
            predictor.predict_features(np.zeros((3, N_FEATURES - 2)))
        self.assertIn(f"expected {N_FEATURES} features", str(caught.exception))

    def test_non_finite_input_is_rejected(self):
        predictor = SafeShiftPredictor.load(self.bundle)
        for bad in (np.nan, np.inf, -np.inf):
            with self.subTest(bad=bad):
                x = np.zeros((2, N_FEATURES))
                x[1, 4] = bad
                with self.assertRaises(InferenceError) as caught:
                    predictor.predict_features(x)
                self.assertIn("non-finite", str(caught.exception))

    def test_three_dimensional_input_is_rejected(self):
        predictor = SafeShiftPredictor.load(self.bundle)
        with self.assertRaises(InferenceError):
            predictor.predict_features(np.zeros((2, 3, N_FEATURES)))

    def test_corrupted_npz_is_rejected(self):
        (self.bundle / "model.npz").write_bytes(b"not a zip archive at all")
        with self.assertRaises(InferenceError):
            SafeShiftPredictor.load(self.bundle)

    def test_pickle_requiring_artifact_is_rejected(self):
        """An object-dtype array can only load with allow_pickle=True."""
        payload = np.empty(2, dtype=object)
        payload[0] = {"weights": [1, 2, 3]}
        payload[1] = "arbitrary python object"
        with (self.bundle / "model.npz").open("wb") as handle:
            np.savez(handle, allow_pickle=True, linear_coef=payload)
        with self.assertRaises(InferenceError):
            SafeShiftPredictor.load(self.bundle)

    def test_pickled_estimator_is_never_an_acceptable_bundle(self):
        estimator = build_estimator("ridge", {"alpha": 1.0}, CONFIG)
        (self.bundle / "model.npz").write_bytes(pickle.dumps(estimator))
        with self.assertRaises(InferenceError):
            SafeShiftPredictor.load(self.bundle)

    def test_missing_weight_array_is_rejected(self):
        with np.load(self.bundle / "model.npz", allow_pickle=False) as handle:
            arrays = {key: np.array(handle[key]) for key in handle.files if key != "linear_coef"}
        with (self.bundle / "model.npz").open("wb") as handle:
            np.savez(handle, allow_pickle=False, **arrays)
        self.rewrite_manifest(
            lambda manifest: manifest["files"].update({"model.npz": _sha(self.bundle / "model.npz")})
        )
        with self.assertRaises(InferenceError) as caught:
            SafeShiftPredictor.load(self.bundle)
        self.assertIn("linear_coef", str(caught.exception))

    def test_empty_directory_is_rejected(self):
        empty = self.root / "empty"
        empty.mkdir()
        with self.assertRaises(InferenceError):
            SafeShiftPredictor.load(empty)

    def test_out_of_range_output_transform_index_is_rejected(self):
        self.rewrite_manifest(
            lambda manifest: manifest.update(
                {"output_transform": {"kind": "residual_plus_feature", "feature_index": 10_000}}
            )
        )
        predictor = SafeShiftPredictor.load(self.bundle)
        with self.assertRaises(InferenceError):
            predictor.predict_features(np.zeros((2, N_FEATURES)))

    def test_unknown_output_transform_is_rejected(self):
        self.rewrite_manifest(
            lambda manifest: manifest.update({"output_transform": {"kind": "exp_then_scale"}})
        )
        predictor = SafeShiftPredictor.load(self.bundle)
        with self.assertRaises(InferenceError):
            predictor.predict_features(np.zeros((2, N_FEATURES)))


class TestPersistenceBundleRuntime(unittest.TestCase):
    def test_persistence_bundle_survives_a_directory_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            original = export_persistence(Path(tmp) / "a", SCHEMA, run_id="copy-test", config=CONFIG)
            copied = Path(tmp) / "b"
            shutil.copytree(original, copied)
            predictor = SafeShiftPredictor.load(copied)
            index = SCHEMA["index"]["apparent_temperature_now"]
            x = np.zeros((2, N_FEATURES))
            x[:, index] = [29.0, 33.0]
            np.testing.assert_allclose(predictor.predict_features(x), [29.0, 33.0])


def _sha(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


if __name__ == "__main__":
    unittest.main()
