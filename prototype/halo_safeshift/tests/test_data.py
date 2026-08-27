"""Field mapping, row QC, five-minute resampling and the recovery partition."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

import pandas as pd

from prototype.halo_safeshift import load_config
from prototype.halo_safeshift.data import (
    FieldMappingError,
    apply_row_qc,
    derive_recovery_partition,
    resample_five_minute,
    resolve_field_mapping,
)
from prototype.halo_safeshift.tests import (
    CHANNEL_METADATA,
    EPOCH,
    FIELD_MAPPING,
    dense_rows,
    raw_frame,
    raw_row,
)

CONFIG = load_config()


class TestFieldMapping(unittest.TestCase):
    def test_maps_semantics_from_metadata_labels(self):
        self.assertEqual(resolve_field_mapping(CHANNEL_METADATA, CONFIG), FIELD_MAPPING)

    def test_mapping_follows_labels_not_positions(self):
        """The same labels in different slots must produce the same semantics."""
        shuffled = dict(CHANNEL_METADATA)
        shuffled["field1"], shuffled["field3"] = shuffled["field3"], shuffled["field1"]
        mapping = resolve_field_mapping(shuffled, CONFIG)
        self.assertEqual(mapping["field1"], "temperature")
        self.assertEqual(mapping["field3"], "wind_speed")

    def test_missing_required_channel_is_a_hard_error(self):
        broken = dict(CHANNEL_METADATA)
        broken["field3"] = None
        with self.assertRaises(FieldMappingError) as caught:
            resolve_field_mapping(broken, CONFIG)
        self.assertIn("temperature", str(caught.exception))

    def test_unknown_label_is_refused_rather_than_guessed(self):
        broken = dict(CHANNEL_METADATA)
        broken["field5"] = "Rainfall"
        with self.assertRaises(FieldMappingError) as caught:
            resolve_field_mapping(broken, CONFIG)
        self.assertIn("Rainfall", str(caught.exception))

    def test_duplicate_semantic_is_refused(self):
        broken = dict(CHANNEL_METADATA)
        broken["field5"] = "Temperature"
        with self.assertRaises(FieldMappingError):
            resolve_field_mapping(broken, CONFIG)


class TestRowQc(unittest.TestCase):
    def test_rows_before_the_wind_scale_fix_epoch_are_discarded(self):
        epoch = pd.Timestamp(CONFIG["usable_epoch"]["wind_scale_fix_utc"])
        before = epoch.to_pydatetime() - timedelta(minutes=1)
        after = epoch.to_pydatetime() + timedelta(minutes=1)
        frame = raw_frame([raw_row(1, before), raw_row(2, after)])
        admitted, counts = apply_row_qc(frame, FIELD_MAPPING, CONFIG)
        self.assertEqual(counts["rejected_before_wind_scale_fix_epoch"], 1)
        self.assertEqual(list(admitted["entry_id"]), ["2"])

    def test_rejects_the_configured_physical_violations(self):
        rows = [
            raw_row(1, EPOCH + timedelta(seconds=10)),
            raw_row(2, EPOCH + timedelta(seconds=20), temperature=0.0),
            raw_row(3, EPOCH + timedelta(seconds=30), humidity=0.0),
            raw_row(4, EPOCH + timedelta(seconds=40), humidity=101.0),
            raw_row(5, EPOCH + timedelta(seconds=50), wind_speed=-0.5),
        ]
        _admitted, counts = apply_row_qc(raw_frame(rows), FIELD_MAPPING, CONFIG)
        self.assertEqual(counts["rejected_temperature_le_threshold"], 1)
        self.assertEqual(counts["rejected_humidity_le_threshold"], 1)
        self.assertEqual(counts["rejected_humidity_gt_threshold"], 1)
        self.assertEqual(counts["rejected_wind_speed_lt_threshold"], 1)
        self.assertEqual(counts["rows_admitted"], 1)

    def test_non_numeric_required_value_rejects_the_row(self):
        rows = [raw_row(1, EPOCH + timedelta(seconds=10)), raw_row(2, EPOCH + timedelta(seconds=20))]
        frame = raw_frame(rows)
        frame.loc[1, "field3"] = "n/a"
        _admitted, counts = apply_row_qc(frame, FIELD_MAPPING, CONFIG)
        self.assertEqual(counts["rejected_required_non_numeric_or_non_finite"], 1)
        self.assertEqual(counts["rows_admitted"], 1)

    def test_non_physical_pressure_nulls_the_channel_but_keeps_the_row(self):
        rows = [raw_row(1, EPOCH + timedelta(seconds=10), pressure=0.0)]
        admitted, counts = apply_row_qc(raw_frame(rows), FIELD_MAPPING, CONFIG)
        self.assertEqual(counts["nulled_pressure_le_threshold"], 1)
        self.assertEqual(counts["rows_admitted"], 1)
        self.assertTrue(pd.isna(admitted.loc[0, "pressure"]))

    def test_duplicate_entry_ids_are_dropped(self):
        rows = [
            raw_row(7, EPOCH + timedelta(seconds=10)),
            raw_row(7, EPOCH + timedelta(seconds=40)),
        ]
        _admitted, counts = apply_row_qc(raw_frame(rows), FIELD_MAPPING, CONFIG)
        self.assertEqual(counts["rejected_duplicate_entry_id"], 1)


class TestResampling(unittest.TestCase):
    MINIMUM = int(CONFIG["resample"]["min_raw_observations_per_bin"])

    def _bins(self, rows):
        admitted, _counts = apply_row_qc(raw_frame(rows), FIELD_MAPPING, CONFIG)
        return resample_five_minute(admitted, FIELD_MAPPING, CONFIG)

    def test_six_row_bin_is_valid_and_five_row_bin_is_not(self):
        """The configured minimum is 6; 6 passes and 5 fails."""
        self.assertEqual(self.MINIMUM, 6)
        rows_six = [
            raw_row(i + 1, EPOCH + timedelta(seconds=10 + 30 * i)) for i in range(self.MINIMUM)
        ]
        rows_five = [
            raw_row(i + 1, EPOCH + timedelta(seconds=10 + 30 * i)) for i in range(self.MINIMUM - 1)
        ]
        six = self._bins(rows_six)
        five = self._bins(rows_five)
        self.assertEqual(len(six), 1)
        self.assertTrue(bool(six.loc[0, "valid"]))
        self.assertEqual(len(five), 1)
        self.assertFalse(bool(five.loc[0, "valid"]))
        self.assertIn("lt_6", str(five.loc[0, "invalid_reason"]))

    def test_bins_are_right_labelled_and_use_medians(self):
        base = datetime(2026, 7, 22, 6, 0, 0, tzinfo=timezone.utc)
        temperatures = [28.0, 29.0, 30.0, 31.0, 32.0, 40.0]
        rows = [
            raw_row(i + 1, base + timedelta(seconds=10 + 30 * i), temperature=value)
            for i, value in enumerate(temperatures)
        ]
        binned = self._bins(rows)
        self.assertEqual(len(binned), 1)
        label = pd.Timestamp(binned.loc[0, "bin_end_utc"])
        self.assertEqual(label, pd.Timestamp("2026-07-22T06:05:00Z"))
        self.assertAlmostEqual(float(binned.loc[0, "temperature"]), 30.5, places=9)

    def test_no_interpolation_and_no_bin_invented_across_an_outage(self):
        """Bins 3..8 are absent, so no bin may exist over that interval."""
        rows = dense_rows(12, skip_bins=(3, 4, 5, 6, 7, 8))
        binned = self._bins(rows)
        labels = {pd.Timestamp(value) for value in binned["bin_end_utc"]}
        for missing_bin in range(3, 9):
            expected = pd.Timestamp(EPOCH + timedelta(minutes=5 * (missing_bin + 1)))
            self.assertNotIn(expected, labels, f"bin {expected} was invented across the outage")
        self.assertEqual(len(binned), 6)
        self.assertTrue(bool(binned["valid"].all()))

    def test_raw_counts_and_staleness_are_preserved(self):
        rows = dense_rows(1, per_bin=7, seconds_between=30)
        binned = self._bins(rows)
        self.assertEqual(int(binned.loc[0, "n_raw"]), 7)
        self.assertEqual(int(binned.loc[0, "n_valid_temperature"]), 7)
        self.assertGreaterEqual(float(binned.loc[0, "staleness_seconds"]), 0.0)

    def test_empty_input_yields_an_empty_table_not_a_crash(self):
        admitted, _counts = apply_row_qc(raw_frame([raw_row(1, EPOCH - timedelta(days=30))]), FIELD_MAPPING, CONFIG)
        binned = resample_five_minute(admitted, FIELD_MAPPING, CONFIG)
        self.assertEqual(len(binned), 0)


class TestRecoveryPartition(unittest.TestCase):
    def test_latest_long_outage_defines_the_recovery_start(self):
        hours = float(CONFIG["outage"]["long_outage_hours"])
        base = datetime(2026, 8, 1, 0, 0, 0, tzinfo=timezone.utc)
        times = [
            base,
            base + timedelta(minutes=1),
            base + timedelta(hours=hours + 2),          # first long outage
            base + timedelta(hours=hours + 2, minutes=1),
            base + timedelta(hours=3 * hours + 10),     # latest long outage
            base + timedelta(hours=3 * hours + 10, minutes=1),
        ]
        partition = derive_recovery_partition(pd.Series(times), CONFIG)
        self.assertEqual(len(partition["long_outages"]), 2)
        self.assertEqual(
            partition["recovery_start_utc"],
            times[4].isoformat().replace("+00:00", "Z"),
        )

    def test_no_long_outage_leaves_the_recovery_start_undefined(self):
        base = datetime(2026, 8, 1, tzinfo=timezone.utc)
        times = [base + timedelta(minutes=5 * i) for i in range(20)]
        partition = derive_recovery_partition(pd.Series(times), CONFIG)
        self.assertEqual(partition["long_outages"], [])
        self.assertIsNone(partition["recovery_start_utc"])

    def test_recovery_context_does_not_claim_the_firmware_is_fixed(self):
        partition = derive_recovery_partition(pd.Series([], dtype="datetime64[ns, UTC]"), CONFIG)
        context = partition["recovery_context"]
        self.assertEqual(context["cause"], "user-confirmed manual station reset")
        self.assertEqual(context["firmware_root_cause"], "unresolved")
        self.assertNotIn("fixed", context["cause"].lower())


if __name__ == "__main__":
    unittest.main()
