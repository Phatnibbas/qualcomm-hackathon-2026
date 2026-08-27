"""Blocked-fold chronology, embargo, and the no-random-split guarantee."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from prototype.halo_safeshift import load_config
from prototype.halo_safeshift.data import apply_row_qc, resample_five_minute
from prototype.halo_safeshift.features import build_windows
from prototype.halo_safeshift.splits import (
    assert_fold_chronology,
    build_folds,
    fold_report,
)
from prototype.halo_safeshift.tests import FIELD_MAPPING, dense_rows, raw_frame

CONFIG = load_config()
DAY_START = datetime(2026, 7, 22, 0, 0, 0, tzinfo=timezone.utc)


def windows_over_days(n_days: int):
    """Dense synthetic coverage across ``n_days`` whole UTC days."""
    bins_per_day = 24 * 60 // int(CONFIG["resample"]["bin_minutes"])
    rows = dense_rows(bins_per_day * n_days, start=DAY_START, per_bin=6, seconds_between=45)
    admitted, _counts = apply_row_qc(raw_frame(rows), FIELD_MAPPING, CONFIG)
    binned = resample_five_minute(admitted, FIELD_MAPPING, CONFIG)
    return build_windows(binned, FIELD_MAPPING, CONFIG)


class TestFoldConstruction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        settings = CONFIG["splits"]
        cls.n_days = int(settings["min_train_days"]) + int(settings["validation_days"]) * 2
        cls.windows = windows_over_days(cls.n_days)
        cls.folds = build_folds(cls.windows, CONFIG)

    def test_at_least_one_fold_is_produced(self):
        self.assertGreater(len(self.folds), 0)
        for fold in self.folds:
            self.assertGreater(fold.n_train, 0)
            self.assertGreater(fold.n_validation, 0)

    def test_every_train_timestamp_precedes_every_validation_timestamp(self):
        issue = pd.DatetimeIndex(self.windows.issue_times)
        target = pd.DatetimeIndex(self.windows.target_times)
        start = pd.DatetimeIndex(self.windows.window_start_times)
        for fold in self.folds:
            with self.subTest(fold=fold.index):
                self.assertLess(issue[fold.train_mask].max(), issue[fold.validation_mask].min())
                self.assertLess(target[fold.train_mask].max(), start[fold.validation_mask].min())

    def test_embargo_is_respected_on_both_sides_of_each_boundary(self):
        embargo = timedelta(minutes=float(CONFIG["splits"]["embargo_minutes"]))
        target = pd.DatetimeIndex(self.windows.target_times).tz_localize(timezone.utc)
        start = pd.DatetimeIndex(self.windows.window_start_times).tz_localize(timezone.utc)
        for fold in self.folds:
            with self.subTest(fold=fold.index):
                self.assertLessEqual(target[fold.train_mask].max(), fold.boundary - embargo)
                self.assertGreaterEqual(start[fold.validation_mask].min(), fold.boundary + embargo)

    def test_train_and_validation_masks_never_overlap(self):
        for fold in self.folds:
            self.assertFalse(bool((fold.train_mask & fold.validation_mask).any()))

    def test_train_block_expands_monotonically(self):
        for previous, current in zip(self.folds, self.folds[1:]):
            self.assertEqual(previous.train_start, current.train_start)
            self.assertGreater(current.boundary, previous.boundary)
            self.assertGreaterEqual(current.n_train, previous.n_train)

    def test_chronology_assertion_passes_on_well_formed_folds(self):
        assert_fold_chronology(self.folds, self.windows)

    def test_chronology_assertion_catches_a_corrupted_fold(self):
        fold = self.folds[0]
        corrupted = np.array(fold.train_mask, copy=True)
        corrupted |= fold.validation_mask
        fold_copy = type(fold)(
            index=fold.index,
            train_start=fold.train_start,
            boundary=fold.boundary,
            validation_end=fold.validation_end,
            train_mask=corrupted,
            validation_mask=fold.validation_mask,
            embargo_minutes=fold.embargo_minutes,
        )
        with self.assertRaises(AssertionError):
            assert_fold_chronology([fold_copy], self.windows)

    def test_report_records_exact_dates_and_issue_counts(self):
        report = fold_report(self.folds, self.windows, CONFIG)
        self.assertEqual(len(report["folds"]), len(self.folds))
        for entry, fold in zip(report["folds"], self.folds):
            self.assertEqual(entry["n_train_issues"], fold.n_train)
            self.assertEqual(entry["n_validation_issues"], fold.n_validation)
            self.assertTrue(entry["boundary_utc"].endswith("Z"))
            self.assertIsNotNone(entry["train_issue_range_utc"]["first"])
        self.assertIn("random_split", report["forbidden"])
        self.assertIn("shuffled_cross_validation", report["forbidden"])

    def test_report_does_not_describe_any_block_as_an_untouched_test(self):
        text = str(fold_report(self.folds, self.windows, CONFIG)).lower()
        self.assertNotIn("untouched test", text.replace("do not call any block here an untouched test.", ""))


class TestDegenerateInput(unittest.TestCase):
    def test_too_short_a_record_produces_no_folds(self):
        settings = CONFIG["splits"]
        short = windows_over_days(max(1, int(settings["min_train_days"]) - 1))
        self.assertEqual(build_folds(short, CONFIG), [])

    def test_empty_windows_produce_no_folds(self):
        empty = windows_over_days(1)
        empty.x = empty.x[:0]
        empty.y = empty.y[:0]
        empty.issue_times = empty.issue_times[:0]
        empty.target_times = empty.target_times[:0]
        empty.window_start_times = empty.window_start_times[:0]
        self.assertEqual(build_folds(empty, CONFIG), [])


if __name__ == "__main__":
    unittest.main()
