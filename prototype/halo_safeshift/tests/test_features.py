"""Leakage-safety of the window construction.

The sentinel tests are the load-bearing ones: a value that exists only in the
future must never appear in a feature at issue time.
"""

from __future__ import annotations

import math
import unittest
from datetime import timedelta

import numpy as np
import pandas as pd

from prototype.halo_safeshift import load_config
from prototype.halo_safeshift.data import apply_row_qc, resample_five_minute
from prototype.halo_safeshift.features import build_feature_schema, build_windows
from prototype.halo_safeshift.tests import EPOCH, FIELD_MAPPING, dense_rows, raw_frame

CONFIG = load_config()
SENTINEL = 999999.0


def bins_from(rows):
    admitted, _counts = apply_row_qc(raw_frame(rows), FIELD_MAPPING, CONFIG)
    return resample_five_minute(admitted, FIELD_MAPPING, CONFIG)


class TestSchema(unittest.TestCase):
    def test_feature_count_and_order_are_deterministic(self):
        schema = build_feature_schema(FIELD_MAPPING, CONFIG)
        per_bin = len(CONFIG["features"]["per_bin_order"])
        issue = len(CONFIG["features"]["issue_time_order"])
        lookback = CONFIG["windows"]["lookback_bins"]
        self.assertEqual(schema["n_features"], lookback * per_bin + issue)
        self.assertEqual(len(schema["feature_names"]), schema["n_features"])
        self.assertEqual(len(set(schema["feature_names"])), schema["n_features"])
        self.assertEqual(schema["feature_names"][0], "t_minus_055min__temperature")
        self.assertEqual(schema["feature_names"][per_bin * (lookback - 1)], "t_minus_000min__temperature")
        self.assertEqual(schema["feature_names"][-1], "window_all_required_valid")

    def test_schema_records_the_timezone_name(self):
        schema = build_feature_schema(FIELD_MAPPING, CONFIG)
        self.assertEqual(schema["timezone"]["name"], "Asia/Ho_Chi_Minh")
        self.assertIn(schema["timezone"]["resolution_source"], {"zoneinfo", "fixed_offset"})

    def test_input_and_target_intervals_do_not_overlap(self):
        lookback = int(CONFIG["windows"]["lookback_minutes"])
        horizon = int(CONFIG["windows"]["horizon_minutes"])
        bin_minutes = int(CONFIG["resample"]["bin_minutes"])
        # inputs cover (t - lookback, t]; target covers (t + horizon - bin, t + horizon]
        self.assertLess(0, horizon - bin_minutes, "target bin must start after the issue time")
        self.assertGreater(lookback, 0)


class TestWindowConstruction(unittest.TestCase):
    def test_target_is_exactly_the_bin_thirty_minutes_ahead(self):
        windows = build_windows(bins_from(dense_rows(40)), FIELD_MAPPING, CONFIG)
        self.assertGreater(len(windows), 0)
        horizon = np.timedelta64(int(CONFIG["windows"]["horizon_minutes"]), "m")
        self.assertTrue(
            bool(np.all(windows.target_times - windows.issue_times == horizon)),
            "every target must sit exactly one horizon after its issue time",
        )

    def test_window_start_is_exactly_one_lookback_before_the_issue_time(self):
        windows = build_windows(bins_from(dense_rows(40)), FIELD_MAPPING, CONFIG)
        lookback = np.timedelta64(int(CONFIG["windows"]["lookback_minutes"]), "m")
        self.assertTrue(bool(np.all(windows.issue_times - windows.window_start_times == lookback)))

    def test_features_are_finite_and_correctly_shaped(self):
        windows = build_windows(bins_from(dense_rows(40)), FIELD_MAPPING, CONFIG)
        self.assertEqual(windows.x.shape[1], windows.schema["n_features"])
        self.assertEqual(windows.x.shape[0], windows.y.shape[0])
        self.assertTrue(bool(np.all(np.isfinite(windows.x))))
        self.assertTrue(bool(np.all(np.isfinite(windows.y))))

    def test_at_now_feature_equals_the_issue_bin_apparent_temperature(self):
        binned = bins_from(dense_rows(40))
        windows = build_windows(binned, FIELD_MAPPING, CONFIG)
        index = windows.schema["index"]["apparent_temperature_now"]
        by_time = {pd.Timestamp(row["bin_end_utc"]): row for _, row in binned.iterrows()}
        for position in range(min(5, len(windows))):
            issue = pd.Timestamp(windows.issue_times[position]).tz_localize("UTC")
            expected = float(by_time[issue]["apparent_temperature_c"])
            self.assertAlmostEqual(float(windows.x[position, index]), expected, places=12)


class TestSentinelLeakage(unittest.TestCase):
    """A value planted only in the future must never reach a feature at t."""

    def _windows_with_future_sentinel(self):
        binned = bins_from(dense_rows(40))
        self.assertGreater(len(binned), 20)
        clean = build_windows(binned, FIELD_MAPPING, CONFIG)
        self.assertGreater(len(clean), 0)

        # Plant the sentinel strictly after the last issue time we will check.
        cutoff_position = 0
        cutoff_issue = pd.Timestamp(clean.issue_times[cutoff_position]).tz_localize("UTC")
        poisoned = binned.copy()
        future = poisoned["bin_end_utc"] > cutoff_issue
        for column in ("temperature", "humidity", "wind_speed", "light", "noise", "pm25", "pressure"):
            poisoned.loc[future, column] = SENTINEL
        return poisoned, cutoff_position, cutoff_issue

    def test_sentinel_in_future_bins_never_enters_features_at_issue_time(self):
        poisoned, position, _issue = self._windows_with_future_sentinel()
        windows = build_windows(poisoned, FIELD_MAPPING, CONFIG)
        vector = windows.x[position]
        self.assertFalse(
            bool(np.any(np.isclose(vector, SENTINEL))),
            "a future-only sentinel reached the issue-time feature vector",
        )
        self.assertFalse(
            bool(np.any(np.isclose(vector, math.log1p(SENTINEL)))),
            "a future-only sentinel reached a log-transformed feature",
        )

    def test_sentinel_does_reach_the_target_when_planted_at_the_target_bin(self):
        """Control: the plumbing must be able to see the target bin at all."""
        binned = bins_from(dense_rows(40))
        windows = build_windows(binned, FIELD_MAPPING, CONFIG)
        target_time = pd.Timestamp(windows.target_times[0]).tz_localize("UTC")
        poisoned = binned.copy()
        poisoned.loc[poisoned["bin_end_utc"] == target_time, "apparent_temperature_c"] = SENTINEL
        repeated = build_windows(poisoned, FIELD_MAPPING, CONFIG)
        self.assertAlmostEqual(float(repeated.y[0]), SENTINEL, places=6)
        self.assertFalse(bool(np.any(np.isclose(repeated.x[0], SENTINEL))))


class TestOutageHandling(unittest.TestCase):
    def test_no_window_bridges_a_missing_bin(self):
        rows = dense_rows(40, skip_bins=(15, 16, 17))
        binned = bins_from(rows)
        windows = build_windows(binned, FIELD_MAPPING, CONFIG)
        present = {pd.Timestamp(value) for value in binned["bin_end_utc"]}
        lookback_bins = int(CONFIG["windows"]["lookback_bins"])
        bin_minutes = int(CONFIG["resample"]["bin_minutes"])
        horizon = timedelta(minutes=int(CONFIG["windows"]["horizon_minutes"]))
        for position in range(len(windows)):
            issue = pd.Timestamp(windows.issue_times[position]).tz_localize("UTC")
            for step in range(lookback_bins):
                needed = issue - timedelta(minutes=bin_minutes * step)
                self.assertIn(needed, present, "window bridged a missing input bin")
            self.assertIn(issue + horizon, present, "window bridged a missing target bin")

    def test_no_window_uses_an_invalid_bin(self):
        rows = dense_rows(40)
        binned = bins_from(rows)
        binned.loc[binned.index[20], "valid"] = False
        binned.loc[binned.index[20], "invalid_reason"] = "synthetic_invalid"
        windows = build_windows(binned, FIELD_MAPPING, CONFIG)
        invalid_time = pd.Timestamp(binned.loc[binned.index[20], "bin_end_utc"])
        lookback_bins = int(CONFIG["windows"]["lookback_bins"])
        bin_minutes = int(CONFIG["resample"]["bin_minutes"])
        horizon = timedelta(minutes=int(CONFIG["windows"]["horizon_minutes"]))
        for position in range(len(windows)):
            issue = pd.Timestamp(windows.issue_times[position]).tz_localize("UTC")
            used = {issue - timedelta(minutes=bin_minutes * step) for step in range(lookback_bins)}
            used.add(issue + horizon)
            self.assertNotIn(invalid_time, used, "an invalid bin was used by a window")

    def test_outage_reduces_the_sample_count(self):
        clean = build_windows(bins_from(dense_rows(40)), FIELD_MAPPING, CONFIG)
        holed = build_windows(bins_from(dense_rows(40, skip_bins=(15, 16, 17))), FIELD_MAPPING, CONFIG)
        self.assertLess(len(holed), len(clean))
        self.assertGreater(holed.diagnostics["rejected_input_bin_missing"], 0)

    def test_empty_bins_produce_an_empty_window_set(self):
        admitted, _counts = apply_row_qc(
            raw_frame(dense_rows(1, start=EPOCH)), FIELD_MAPPING, CONFIG
        )
        empty = resample_five_minute(admitted.iloc[0:0], FIELD_MAPPING, CONFIG)
        windows = build_windows(empty, FIELD_MAPPING, CONFIG)
        self.assertEqual(len(windows), 0)
        self.assertEqual(windows.x.shape[1], windows.schema["n_features"])


if __name__ == "__main__":
    unittest.main()
