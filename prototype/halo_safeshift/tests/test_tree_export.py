"""Neutral tree-ensemble conversion and reference parity against scikit-learn.

Parity is checked in float64 on both sides, so the tolerance is the exact-arithmetic
tolerance from the configuration, not a loose eyeball threshold.
"""

from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from prototype.halo_safeshift import load_config
from prototype.halo_safeshift.models import build_estimator
from prototype.halo_safeshift.tree_export import (
    LEAF,
    TreeExportError,
    evaluate_tree_ensemble,
    export_sklearn_extra_trees,
    export_sklearn_gradient_boosting,
    flatten_tree_ensemble,
    unflatten_tree_ensemble,
)

CONFIG = load_config()
TOLERANCE = float(CONFIG["tolerances"]["tree_ensemble_fp64_max_abs_error"])


def synthetic_regression(n_samples: int = 240, n_features: int = 8, seed: int = 20260815):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_samples, n_features))
    y = (
        1.7 * x[:, 0]
        - 0.9 * x[:, 1] * x[:, 2]
        + 0.4 * np.sin(3.0 * x[:, 3])
        + rng.normal(scale=0.05, size=n_samples)
    )
    return x, y


class TestGradientBoostingParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x, cls.y = synthetic_regression()
        cls.estimator = build_estimator(
            "gradient_boosting",
            {"n_estimators": 40, "max_depth": 3, "learning_rate": 0.1},
            CONFIG,
        ).fit(cls.x, cls.y)
        cls.neutral = export_sklearn_gradient_boosting(cls.estimator)

    def test_neutral_form_matches_sklearn_within_tolerance(self):
        reference = np.asarray(self.estimator.predict(self.x), dtype=np.float64)
        produced = evaluate_tree_ensemble(self.neutral, self.x)
        error = float(np.max(np.abs(reference - produced)))
        self.assertLessEqual(error, TOLERANCE, f"max abs error {error:.3e} exceeds {TOLERANCE:.0e}")

    def test_metadata_is_recorded_for_the_runtime(self):
        self.assertEqual(self.neutral["aggregation"], "sum")
        self.assertEqual(self.neutral["decision_rule"], "left_if_le")
        self.assertAlmostEqual(self.neutral["learning_rate"], 0.1, places=12)
        self.assertEqual(len(self.neutral["trees"]), 40)

    def test_base_score_is_the_fitted_initial_prediction(self):
        self.assertAlmostEqual(self.neutral["base_score"], float(np.mean(self.y)), places=8)

    def test_leaf_nodes_are_marked_consistently(self):
        for tree in self.neutral["trees"]:
            for position, feature in enumerate(tree["feature"]):
                if feature == LEAF:
                    self.assertEqual(tree["left"][position], LEAF)
                    self.assertEqual(tree["right"][position], LEAF)
                else:
                    self.assertNotEqual(tree["left"][position], LEAF)
                    self.assertNotEqual(tree["right"][position], LEAF)


class TestExtraTreesParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x, cls.y = synthetic_regression()
        cls.estimator = build_estimator(
            "extra_trees",
            {"n_estimators": 25, "max_depth": 8, "min_samples_leaf": 2},
            CONFIG,
        ).fit(cls.x, cls.y)
        cls.neutral = export_sklearn_extra_trees(cls.estimator)

    def test_neutral_form_matches_sklearn_within_tolerance(self):
        reference = np.asarray(self.estimator.predict(self.x), dtype=np.float64)
        produced = evaluate_tree_ensemble(self.neutral, self.x)
        error = float(np.max(np.abs(reference - produced)))
        self.assertLessEqual(error, TOLERANCE, f"max abs error {error:.3e} exceeds {TOLERANCE:.0e}")

    def test_aggregation_is_the_mean_over_trees(self):
        self.assertEqual(self.neutral["aggregation"], "mean")
        self.assertEqual(self.neutral["base_score"], 0.0)
        self.assertEqual(self.neutral["learning_rate"], 1.0)


class TestFlatteningRoundTrip(unittest.TestCase):
    def test_flatten_unflatten_preserves_predictions(self):
        x, y = synthetic_regression(n_samples=120)
        estimator = build_estimator(
            "gradient_boosting", {"n_estimators": 12, "max_depth": 2}, CONFIG
        ).fit(x, y)
        neutral = export_sklearn_gradient_boosting(estimator)
        arrays = flatten_tree_ensemble(neutral)
        rebuilt = unflatten_tree_ensemble(
            arrays,
            neutral["aggregation"],
            neutral["base_score"],
            neutral["learning_rate"],
            neutral["decision_rule"],
        )
        np.testing.assert_allclose(
            evaluate_tree_ensemble(neutral, x), evaluate_tree_ensemble(rebuilt, x), atol=0.0, rtol=0.0
        )

    def test_offsets_span_the_node_arrays_exactly(self):
        x, y = synthetic_regression(n_samples=120)
        estimator = build_estimator("extra_trees", {"n_estimators": 5, "max_depth": 4}, CONFIG).fit(x, y)
        arrays = flatten_tree_ensemble(export_sklearn_extra_trees(estimator))
        self.assertEqual(int(arrays["tree_offsets"][0]), 0)
        self.assertEqual(int(arrays["tree_offsets"][-1]), arrays["tree_node_feature"].shape[0])
        self.assertEqual(arrays["tree_offsets"].shape[0], 6)

    def test_flat_arrays_are_pickle_free_dtypes(self):
        x, y = synthetic_regression(n_samples=80)
        estimator = build_estimator("extra_trees", {"n_estimators": 3, "max_depth": 3}, CONFIG).fit(x, y)
        for name, array in flatten_tree_ensemble(export_sklearn_extra_trees(estimator)).items():
            with self.subTest(array=name):
                self.assertNotEqual(array.dtype, object)


class TestRejections(unittest.TestCase):
    def test_unfitted_estimator_is_refused(self):
        with self.assertRaises(TreeExportError):
            export_sklearn_gradient_boosting(build_estimator("gradient_boosting", {}, CONFIG))
        with self.assertRaises(TreeExportError):
            export_sklearn_extra_trees(build_estimator("extra_trees", {}, CONFIG))

    def test_unsupported_decision_rule_is_refused_at_evaluation(self):
        x, y = synthetic_regression(n_samples=60)
        estimator = build_estimator("extra_trees", {"n_estimators": 2, "max_depth": 2}, CONFIG).fit(x, y)
        neutral = export_sklearn_extra_trees(estimator)
        neutral["decision_rule"] = "left_if_gt"
        with self.assertRaises(TreeExportError):
            evaluate_tree_ensemble(neutral, x)


class TestOptionalXgboost(unittest.TestCase):
    """XGBoost is not installed here and must not be installed by this suite."""

    def test_xgboost_parity_is_skipped_with_an_explicit_reason(self):
        if importlib.util.find_spec("xgboost") is None:
            self.skipTest(
                "xgboost_local_status: absent. XGBoost is not installed on this "
                "preparation machine and installing it is out of scope. The "
                "XGBoost -> neutral-format parity check therefore did NOT run and "
                "must not be reported as passing. It is the Colab worker's gate "
                "before any XGBoost model may be selected for the UNO Q."
            )
        from prototype.halo_safeshift.tree_export import export_xgboost_regressor

        x, y = synthetic_regression(n_samples=200)
        estimator = build_estimator(
            "xgboost_optional", {"n_estimators": 20, "max_depth": 3}, CONFIG
        ).fit(x, y)
        neutral = export_xgboost_regressor(estimator)
        self.assertEqual(neutral["decision_rule"], "left_if_lt")
        reference = np.asarray(estimator.predict(x), dtype=np.float64)
        produced = evaluate_tree_ensemble(neutral, x)
        self.assertLessEqual(float(np.max(np.abs(reference - produced))), 1e-5)


if __name__ == "__main__":
    unittest.main()
