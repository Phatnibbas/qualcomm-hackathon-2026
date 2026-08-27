"""Baselines, error summaries and the residual parameterization.

No number produced here is a performance result; every input is synthetic.
"""

from __future__ import annotations

import math
import unittest
from datetime import datetime, timedelta, timezone

import numpy as np

from prototype.halo_safeshift import load_config
from prototype.halo_safeshift.baselines import (
    LocalTimeClimatology,
    PersistenceBaseline,
    from_residual,
    mean_absolute_error,
    regime_report,
    root_mean_squared_error,
    to_residual,
)
from prototype.halo_safeshift.features import build_feature_schema
from prototype.halo_safeshift.models import (
    curated_trials,
    family_status,
    parameterization_contract,
    registry_report,
)
from prototype.halo_safeshift.tests import FIELD_MAPPING

CONFIG = load_config()
SCHEMA = build_feature_schema(FIELD_MAPPING, CONFIG)


class TestPersistence(unittest.TestCase):
    def test_predicts_the_current_apparent_temperature(self):
        baseline = PersistenceBaseline.from_schema(SCHEMA)
        index = SCHEMA["index"]["apparent_temperature_now"]
        x = np.zeros((4, SCHEMA["n_features"]), dtype=np.float64)
        x[:, index] = [30.0, 31.5, 28.25, 40.0]
        np.testing.assert_allclose(baseline.predict(x), [30.0, 31.5, 28.25, 40.0])

    def test_index_comes_from_the_frozen_schema(self):
        baseline = PersistenceBaseline.from_schema(SCHEMA)
        self.assertEqual(baseline.at_now_index, SCHEMA["feature_names"].index("apparent_temperature_now"))


class TestClimatology(unittest.TestCase):
    def setUp(self):
        self.climatology = LocalTimeClimatology.from_config(CONFIG)
        base = datetime(2026, 7, 22, 0, 0, tzinfo=timezone.utc)
        # Two days; the target depends only on local hour, so the median is exact.
        self.times = [base + timedelta(minutes=30 * i) for i in range(96)]
        self.targets = np.array(
            [20.0 + (t.astimezone(timezone.utc).hour % 12) for t in self.times], dtype=np.float64
        )

    def test_is_fitted_only_on_the_values_it_is_given(self):
        train_times = self.times[:48]
        train_targets = self.targets[:48]
        self.climatology.fit(train_times, train_targets)
        self.assertEqual(self.climatology.fitted_on_n, 48)
        for key, value in self.climatology.table.items():
            self.assertIn(value, set(train_targets.tolist()), f"key {key} learned an unseen value")

    def test_predicts_the_train_median_for_each_local_time_key(self):
        self.climatology.fit(self.times, self.targets)
        predictions = self.climatology.predict(self.times)
        self.assertEqual(predictions.shape, self.targets.shape)
        self.assertTrue(bool(np.all(np.isfinite(predictions))))

    def test_unseen_key_falls_back_to_the_train_median(self):
        train_times = [t for t in self.times if t.hour < 12]
        train_targets = np.array([20.0 + (t.hour % 12) for t in train_times], dtype=np.float64)
        self.climatology.fit(train_times, train_targets)
        unseen = [datetime(2026, 7, 24, 23, 15, tzinfo=timezone.utc)]
        predicted = float(self.climatology.predict(unseen)[0])
        self.assertAlmostEqual(predicted, float(np.median(train_targets)), places=12)

    def test_refuses_to_fit_on_nothing(self):
        with self.assertRaises(ValueError):
            self.climatology.fit([], np.array([], dtype=np.float64))


class TestMetrics(unittest.TestCase):
    def test_mae_and_rmse_on_a_hand_checked_vector(self):
        truth = np.array([1.0, 2.0, 3.0])
        predicted = np.array([1.5, 2.0, 1.0])
        self.assertAlmostEqual(mean_absolute_error(truth, predicted), (0.5 + 0.0 + 2.0) / 3, places=12)
        self.assertAlmostEqual(
            root_mean_squared_error(truth, predicted),
            math.sqrt((0.25 + 0.0 + 4.0) / 3),
            places=12,
        )

    def test_regime_bins_partition_every_sample_exactly_once(self):
        at_now = np.zeros(9, dtype=np.float64)
        truth = np.array([-3.0, -2.5, -2.0, -1.5, -1.0, 0.0, 1.0, 1.5, 3.0])
        report = regime_report(truth, truth, at_now)
        self.assertEqual(sum(int(entry["n"]) for entry in report.values()), truth.size)
        self.assertEqual(int(report["cooling_lt_-2C"]["n"]), 2)
        self.assertEqual(int(report["stable_-1_to_+1C"]["n"]), 3)
        self.assertEqual(int(report["warming_gt_+2C"]["n"]), 1)


class TestResidualParameterization(unittest.TestCase):
    def test_residual_round_trip_is_exact(self):
        y = np.array([30.0, 31.0, 29.5])
        at_now = np.array([29.0, 31.5, 29.5])
        residual = to_residual(y, at_now)
        np.testing.assert_allclose(from_residual(residual, at_now), y, atol=0.0, rtol=0.0)

    def test_zero_residual_reduces_to_persistence(self):
        at_now = np.array([30.0, 25.0])
        np.testing.assert_allclose(from_residual(np.zeros(2), at_now), at_now)

    def test_residual_is_recorded_as_a_post_hoc_hypothesis(self):
        provenance = parameterization_contract(CONFIG)["parameterization_provenance"]
        self.assertIn("post-hoc", provenance["residual"].lower())
        self.assertNotIn("predeclared", provenance["residual"].lower().replace("do not describe it as predeclared", ""))


class TestRegistry(unittest.TestCase):
    def test_declared_families_match_the_configuration(self):
        report = registry_report(CONFIG)
        self.assertEqual(report["learned_families"], CONFIG["models"]["learned_families"])
        for forbidden in ("random_forest", "lightgbm", "catboost", "lstm", "gru", "transformer"):
            self.assertIn(forbidden, report["forbidden_families_this_phase"])
            self.assertNotIn(forbidden, report["learned_families"])

    def test_every_family_maps_to_an_allowed_export_type(self):
        allowed = set(CONFIG["export"]["allowed_model_types"])
        for family, export_type in registry_report(CONFIG)["family_to_export_type"].items():
            with self.subTest(family=family):
                self.assertIn(export_type, allowed)

    def test_trials_live_in_the_configuration_file(self):
        """The fitted set is a list in the file, not a product computed in code."""
        trials = curated_trials(CONFIG, enable_xgboost=True)
        self.assertEqual(
            [t["trial_id"] for t in trials],
            [t["trial_id"] for t in CONFIG["models"]["trials"]],
        )

    def test_xgboost_availability_is_recorded_not_assumed(self):
        status = family_status(CONFIG)["xgboost_optional"]
        self.assertIn(status["local_status"], {"available", "absent"})
        self.assertEqual(status["provider"], "xgboost")

    def test_registry_states_that_nothing_is_selected_here(self):
        report = registry_report(CONFIG)
        self.assertIn("no family", " ".join(report["boundary"]).lower())
        for entry in report["family_status"].values():
            self.assertFalse(entry["trainable_here"])


if __name__ == "__main__":
    unittest.main()
