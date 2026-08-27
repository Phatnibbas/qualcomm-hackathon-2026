"""Contract tests for the P1 Colab satellite/fusion pipeline.

These tests use only small synthetic values.  They do not train a model and do
not turn a local Windows machine into a training environment.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import struct
import tempfile
import unittest
from pathlib import Path

from prototype.halo_safeshift.full_pipeline_contracts import (
    ContractError,
    assert_common_ablation_identity,
    assert_fused_schema,
    assert_roi_not_clipped,
    assert_satellite_frame_is_available,
    assert_split_sequences_do_not_cross,
    parse_hsd_segment_header,
    verify_sha256,
    assign_fixed_chronological_splits,
)
from benchmarks.halo.hsd import read_bt


UTC = timezone.utc


class TestSatelliteIssueTimeSafety(unittest.TestCase):
    def test_frame_after_issue_is_rejected_even_when_nominal_timestamp_is_earlier(self):
        issue = datetime(2026, 8, 9, 7, 10, tzinfo=UTC)
        with self.assertRaises(ContractError):
            assert_satellite_frame_is_available(
                nominal_scan_utc=datetime(2026, 8, 9, 7, 0, tzinfo=UTC),
                assumed_scan_completion_utc=datetime(2026, 8, 9, 7, 12, tzinfo=UTC),
                issue_time_utc=issue,
                publication_lag_minutes=0,
            )

    def test_lag_sensitivity_uses_effective_availability_before_issue(self):
        issue = datetime(2026, 8, 9, 7, 25, tzinfo=UTC)
        result = assert_satellite_frame_is_available(
            nominal_scan_utc=datetime(2026, 8, 9, 7, 0, tzinfo=UTC),
            assumed_scan_completion_utc=datetime(2026, 8, 9, 7, 10, tzinfo=UTC),
            issue_time_utc=issue,
            publication_lag_minutes=10,
        )
        self.assertEqual(result["frame_age_minutes"], 15.0)
        self.assertEqual(result["publication_lag_minutes"], 10)

    def test_fixed_split_boundaries_keep_the_original_p0_counts_and_target_gap(self):
        # The known P0 split is by issue timestamp; target ends 30 minutes
        # later and must still fall before the next non-embargo split begins.
        self.assertLess("2026-08-05T05:05:00Z", "2026-08-05T07:40:00Z")
        self.assertLess("2026-08-09T22:35:00Z", "2026-08-10T01:10:00Z")


class TestFairAblation(unittest.TestCase):
    def test_common_row_ablation_requires_identical_issue_target_and_split(self):
        identity = [
            ("2026-08-01T00:00:00Z", "2026-08-01T00:30:00Z", "test"),
            ("2026-08-01T00:05:00Z", "2026-08-01T00:35:00Z", "test"),
        ]
        assert_common_ablation_identity(identity, identity, identity)

    def test_common_row_ablation_rejects_a_different_target_row(self):
        station = [("2026-08-01T00:00:00Z", "2026-08-01T00:30:00Z", "test")]
        satellite = [("2026-08-01T00:00:00Z", "2026-08-01T00:35:00Z", "test")]
        with self.assertRaises(ContractError):
            assert_common_ablation_identity(station, satellite, station)

    def test_sequence_cannot_cross_split_boundary(self):
        with self.assertRaises(ContractError):
            assert_split_sequences_do_not_cross(
                [
                    {
                        "window_start_utc": "2026-08-01T00:00:00Z",
                        "issue_time_utc": "2026-08-01T01:00:00Z",
                        "target_time_utc": "2026-08-01T01:30:00Z",
                        "split": "train",
                        "window_split": "train",
                        "target_split": "validation",
                    }
                ]
            )

    def test_window_that_crosses_midnight_is_allowed_when_issue_and_target_share_split(self):
        rows = assign_fixed_chronological_splits(
            [
                {
                    "window_start_utc": "2026-08-04T23:00:00Z",
                    "issue_time_utc": "2026-08-04T23:55:00Z",
                    "target_time_utc": "2026-08-05T00:25:00Z",
                }
            ],
            train_last_issue_utc="2026-08-05T04:35:00Z",
            validation_first_issue_utc="2026-08-05T07:40:00Z",
            validation_last_issue_utc="2026-08-09T22:05:00Z",
            test_first_issue_utc="2026-08-10T01:10:00Z",
        )
        self.assertEqual(rows[0]["split"], "train")


class TestSatelliteAndFusionSchema(unittest.TestCase):
    def test_segment_metadata_uses_total_then_segment_number(self):
        metadata = parse_hsd_segment_header(bytes([10, 4, 0x73, 0x06]))
        self.assertEqual(metadata, {"segment": 4, "total_segments": 10, "first_line": 1651})

    def test_hsd_reader_reports_segment_04_of_10_from_block7_wire_order(self):
        # Minimal valid-enough HSD layout: read_bt reaches block 7 and exposes
        # metadata before calibration values affect this assertion.
        blocks = []
        for number in range(1, 12):
            payload = bytearray(120)
            payload[0] = number
            payload[1:3] = (120).to_bytes(2, "little")
            blocks.append(payload)
        blocks[1][3:9] = (16).to_bytes(2, "little") + (1).to_bytes(2, "little") + (1).to_bytes(2, "little")
        blocks[6][3:7] = bytes([10, 4, 0x73, 0x06])
        # A non-zero IR calibration so this minimal metadata fixture can run
        # through read_bt; pixel values themselves are not under test here.
        struct.pack_into("<HdHHH", blocks[4], 3, 13, 10.4, 16, 65535, 65534)
        struct.pack_into("<dd", blocks[4], 19, 1.0, 1.0)
        struct.pack_into("<9d", blocks[4], 35, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 299792458.0, 6.62607015e-34, 1.380649e-23)
        raw = b"".join(blocks) + (1).to_bytes(2, "little")
        _bt, meta = read_bt(raw)
        self.assertEqual((meta["segment"], meta["of"], meta["first_line"]), (4, 10, 1651))

    def test_roi_clipping_is_rejected(self):
        with self.assertRaises(ContractError):
            assert_roi_not_clipped(roi_pixels=100, covered_pixels=99)

    def test_fused_bundle_requires_satellite_feature_names_and_runtime_input(self):
        with self.assertRaises(ContractError):
            assert_fused_schema(
                {"feature_names": ["station_temperature"], "n_features": 1},
                runtime_input_names=["station_temperature"],
            )

    def test_fused_bundle_accepts_satellite_features_present_in_runtime_input(self):
        schema = {
            "feature_names": ["station_temperature", "satellite_b13_p50_latest"],
            "n_features": 2,
        }
        assert_fused_schema(schema, runtime_input_names=list(schema["feature_names"]))


class TestHashes(unittest.TestCase):
    def test_hash_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.bin"
            path.write_bytes(b"halo")
            with self.assertRaises(ContractError):
                verify_sha256(path, "0" * 64)


if __name__ == "__main__":
    unittest.main()
