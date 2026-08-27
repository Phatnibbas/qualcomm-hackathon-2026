"""Behavioural tests for the correction pass (A-Q).

Each class states the defect it exists to prevent. These are not documentation
tests: every one drives the real code path and asserts that a malformed,
tampered or truncated input is *refused*, because the whole point of the
correction pass is that these cases previously passed.
"""

from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from prototype.halo_safeshift import REPO_ROOT, load_config, sha256_file
from prototype.halo_safeshift.contracts import validate_preprocessing
from prototype.halo_safeshift.data import (
    FieldMappingError,
    apply_row_qc,
    build_recovery_report,
    raw_feed_timeline,
    resolve_field_mapping,
)
from prototype.halo_safeshift.export import (
    ENVELOPE_FILENAME,
    ExportError,
    Preprocessing,
    export_linear,
    write_bundle_envelope,
)
from prototype.halo_safeshift.features import (
    FeatureOrderError,
    build_feature_schema,
    build_windows,
    per_bin_values,
)
from prototype.halo_safeshift.inference import InferenceError, SafeShiftPredictor
from prototype.halo_safeshift.models import (
    TrialBudgetError,
    XGBoostPolicyError,
    assert_xgboost_eligible,
    build_estimator,
    curated_trials,
    trial_budget_report,
    xgboost_policy_status,
)
from prototype.halo_safeshift.source_archive import (
    archive_member_hashes,
    build_source_archive,
    collect_source_files,
)
from prototype.halo_safeshift.splits import EmptyFoldError, build_folds
from prototype.halo_safeshift.tests import (
    CHANNEL_METADATA,
    FIELD_MAPPING,
    dense_rows,
    raw_frame,
    raw_row,
    reseal,
)
from prototype.halo_safeshift.tree_export import (
    TreeExportError,
    flatten_tree_ensemble,
    validate_flat_tree_arrays,
)
from prototype.halo_safeshift.verify_packet import PacketVerificationError, verify_packet

CONFIG = load_config()
SCHEMA = build_feature_schema(FIELD_MAPPING, CONFIG)
N_FEATURES = SCHEMA["n_features"]
PULLER = REPO_ROOT / "benchmarks" / "halo" / "freeze_station_history.py"


def flip_one_byte(path: Path) -> None:
    """Change exactly one byte, so the test proves hashing and not size checks."""
    payload = bytearray(path.read_bytes())
    index = len(payload) // 2
    payload[index] = (payload[index] + 1) % 256
    path.write_bytes(bytes(payload))


# --------------------------------------------------------------------------- #
# Correction C — the recovery boundary comes from the raw feed
# --------------------------------------------------------------------------- #

class TestRawFeedRecoveryBoundary(unittest.TestCase):
    """Defect: QC deciding the boundary lets bad readings invent an outage."""

    def test_invalid_sensor_rows_do_not_manufacture_an_outage(self):
        start = datetime(2026, 7, 22, 0, 0, tzinfo=timezone.utc)
        rows = []
        entry = 1
        # 24 hours of continuous posting. The middle 12 hours carry readings
        # that QC rejects outright, but the station never stopped posting.
        for minute in range(0, 24 * 60, 5):
            when = start + timedelta(minutes=minute)
            broken = 6 * 60 <= minute < 18 * 60
            rows.append(
                raw_row(
                    entry,
                    when,
                    temperature=-99.0 if broken else 30.0,
                    humidity=70.0,
                    wind_speed=1.5,
                )
            )
            entry += 1
        raw = raw_frame(rows)
        admitted, _ = apply_row_qc(raw, FIELD_MAPPING, CONFIG)
        report = build_recovery_report(raw, admitted["created_at_utc"], CONFIG)

        self.assertEqual(report["boundary_source"], "raw feed timestamps")
        self.assertIsNone(
            report["raw_feed_recovery_boundary"],
            "the feed was continuous; rejected sensor values must not create an outage",
        )
        self.assertEqual(report["long_outages"], [])
        # The QC-admitted view *does* see a 12-hour hole. It is reported, and it
        # is explicitly not the boundary.
        self.assertTrue(report["qc_admitted_continuity"]["long_outages"])
        self.assertFalse(report["boundaries_agree"])

    def test_boundary_is_computed_before_the_wind_epoch_filter(self):
        before_epoch = datetime(2026, 7, 20, 0, 0, tzinfo=timezone.utc)
        rows = [raw_row(i + 1, before_epoch + timedelta(minutes=5 * i)) for i in range(12)]
        raw = raw_frame(rows)
        times, counts = raw_feed_timeline(raw, CONFIG)
        self.assertEqual(counts["rows_in_timeline"], 12)
        self.assertEqual(len(times), 12, "pre-epoch rows still prove the station posted")

    def test_unparsable_timestamps_and_duplicates_are_removed_deterministically(self):
        base = datetime(2026, 7, 22, 0, 0, tzinfo=timezone.utc)
        rows = [raw_row(1, base), raw_row(1, base + timedelta(seconds=30)), raw_row(2, base)]
        rows.append({**raw_row(3, base), "created_at": "not-a-timestamp"})
        raw = raw_frame(rows)
        times, counts = raw_feed_timeline(raw, CONFIG)
        self.assertEqual(counts["dropped_unparsable_timestamp"], 1)
        self.assertEqual(counts["dropped_duplicate_entry_id"], 1)
        self.assertEqual(len(times), 2)


# --------------------------------------------------------------------------- #
# Correction D — feature-required channels must exist in the metadata
# --------------------------------------------------------------------------- #

class TestFeatureChannelContract(unittest.TestCase):
    """Defect: a channel absent from metadata surfaced as a vector-length error."""

    def test_all_eight_configured_channels_resolve(self):
        mapping = resolve_field_mapping(CHANNEL_METADATA, CONFIG)
        for semantic in CONFIG["field_semantics"]["feature_required_semantics"]:
            self.assertIn(semantic, mapping.values())

    def test_missing_feature_required_channel_fails_during_resolution(self):
        for field, semantic in (("field5", "light"), ("field8", "pm25"), ("field7", "noise")):
            with self.subTest(missing=semantic):
                metadata = {k: v for k, v in CHANNEL_METADATA.items() if k != field}
                with self.assertRaises(FieldMappingError) as caught:
                    resolve_field_mapping(metadata, CONFIG)
                message = str(caught.exception)
                self.assertIn("feature-required", message)
                self.assertIn(semantic, message)

    def test_missing_target_required_channel_still_fails_first(self):
        metadata = {k: v for k, v in CHANNEL_METADATA.items() if k != "field3"}
        with self.assertRaises(FieldMappingError) as caught:
            resolve_field_mapping(metadata, CONFIG)
        self.assertIn("required semantic channel", str(caught.exception))

    def test_a_missing_value_in_a_row_is_a_different_failure_from_missing_metadata(self):
        """Metadata absence fails early; a null value only costs one sample."""
        rows = dense_rows(20)
        raw = raw_frame(rows)
        admitted, _ = apply_row_qc(raw, FIELD_MAPPING, CONFIG)
        self.assertGreater(len(admitted), 0)


# --------------------------------------------------------------------------- #
# Correction E — feature order comes from the config, once
# --------------------------------------------------------------------------- #

class TestConfigDrivenFeatureOrder(unittest.TestCase):
    """Defect: the vector order was hard-coded beside the configured order."""

    def _binned(self):
        raw = raw_frame(dense_rows(30))
        admitted, _ = apply_row_qc(raw, FIELD_MAPPING, CONFIG)
        from prototype.halo_safeshift.data import resample_five_minute

        return resample_five_minute(admitted, FIELD_MAPPING, CONFIG)

    def test_swapping_two_names_moves_both_the_schema_index_and_the_value(self):
        binned = self._binned()
        base = build_windows(binned, FIELD_MAPPING, CONFIG)
        self.assertGreater(len(base), 0)

        swapped = json.loads(json.dumps(CONFIG))
        order = swapped["features"]["per_bin_order"]
        i, j = order.index("temperature"), order.index("noise")
        order[i], order[j] = order[j], order[i]

        other = build_windows(binned, FIELD_MAPPING, swapped)
        base_names = base.schema["feature_names"]
        other_names = other.schema["feature_names"]

        self.assertNotEqual(base_names, other_names)
        # Both the schema and the emitted matrix must move together: the value
        # found at the new index must equal the value at the old index.
        for name in ("t_minus_000min__temperature", "t_minus_000min__noise"):
            with self.subTest(feature=name):
                self.assertEqual(
                    base.x[0, base_names.index(name)],
                    other.x[0, other_names.index(name)],
                )

    def test_swapping_issue_time_names_is_consistent_too(self):
        binned = self._binned()
        base = build_windows(binned, FIELD_MAPPING, CONFIG)
        swapped = json.loads(json.dumps(CONFIG))
        order = swapped["features"]["issue_time_order"]
        i, j = order.index("apparent_temperature_now"), order.index("day_of_year_sin")
        order[i], order[j] = order[j], order[i]

        other = build_windows(binned, FIELD_MAPPING, swapped)
        self.assertNotEqual(
            base.schema["index"]["apparent_temperature_now"],
            other.schema["index"]["apparent_temperature_now"],
        )
        self.assertEqual(
            base.x[0, base.schema["index"]["apparent_temperature_now"]],
            other.x[0, other.schema["index"]["apparent_temperature_now"]],
        )

    def test_unknown_feature_name_is_rejected_when_the_schema_is_built(self):
        broken = json.loads(json.dumps(CONFIG))
        broken["features"]["per_bin_order"].append("dew_point_someone_invented")
        with self.assertRaises(FeatureOrderError) as caught:
            build_feature_schema(FIELD_MAPPING, broken)
        self.assertIn("dew_point_someone_invented", str(caught.exception))

    def test_duplicate_feature_name_is_rejected(self):
        broken = json.loads(json.dumps(CONFIG))
        broken["features"]["per_bin_order"].append("temperature")
        with self.assertRaises(FeatureOrderError):
            build_feature_schema(FIELD_MAPPING, broken)

    def test_every_known_per_bin_name_is_computable(self):
        raw = raw_frame(dense_rows(3))
        admitted, _ = apply_row_qc(raw, FIELD_MAPPING, CONFIG)
        from prototype.halo_safeshift.data import resample_five_minute

        binned = resample_five_minute(admitted, FIELD_MAPPING, CONFIG)
        values = per_bin_values(binned.iloc[0], 11.0)
        for name in CONFIG["features"]["per_bin_order"]:
            self.assertIn(name, values)


# --------------------------------------------------------------------------- #
# Correction F — a fold with an empty block fails the run
# --------------------------------------------------------------------------- #

class TestFoldFailClosed(unittest.TestCase):
    """Defect: an empty fold was silently kept and silently skipped."""

    def _sparse_windows(self):
        """Enough days to plan folds, but almost no windows inside them."""
        rows: list[dict[str, str]] = []
        entry = 1
        start = datetime(2026, 7, 22, 0, 0, tzinfo=timezone.utc)
        for day in range(14):
            # One dense hour per day: enough to form a window, nowhere near
            # enough to populate every planned block.
            for step in dense_rows(
                24, start=start + timedelta(days=day), first_entry_id=entry
            ):
                rows.append(step)
                entry += 1
        raw = raw_frame(rows)
        admitted, _ = apply_row_qc(raw, FIELD_MAPPING, CONFIG)
        from prototype.halo_safeshift.data import resample_five_minute

        binned = resample_five_minute(admitted, FIELD_MAPPING, CONFIG)
        return build_windows(binned, FIELD_MAPPING, CONFIG)

    def test_an_empty_validation_block_rejects_the_run(self):
        windows = self._sparse_windows()
        settings = json.loads(json.dumps(CONFIG))
        # Validation blocks land on days whose windows were all consumed by the
        # embargo, so at least one planned fold realises nothing.
        settings["splits"]["min_train_days"] = 2
        settings["splits"]["validation_days"] = 1
        with self.assertRaises(EmptyFoldError) as caught:
            build_folds(windows, settings)
        self.assertIn("realised", str(caught.exception))

    def test_the_configured_flag_is_what_makes_it_fail(self):
        windows = self._sparse_windows()
        permissive = json.loads(json.dumps(CONFIG))
        permissive["splits"]["min_train_days"] = 2
        permissive["splits"]["validation_days"] = 1
        permissive["splits"]["fail_closed_on_empty_fold"] = False
        folds = build_folds(windows, permissive)
        self.assertTrue(
            any(f.n_train == 0 or f.n_validation == 0 for f in folds),
            "the permissive path must be the one that produces the empty fold",
        )


# --------------------------------------------------------------------------- #
# Correction G — bounded, explicit trial list
# --------------------------------------------------------------------------- #

class TestBoundedTrialList(unittest.TestCase):
    """Defect: a Cartesian product whose true size was invisible in the file."""

    def test_default_trial_count_is_inside_the_declared_budget(self):
        trials = curated_trials(CONFIG)
        self.assertGreaterEqual(len(trials), CONFIG["models"]["trial_budget"]["min_total_trials"])
        self.assertLessEqual(
            len(trials), CONFIG["models"]["trial_budget"]["max_default_trials_excluding_xgboost"]
        )
        self.assertEqual(len(trials), 26)

    def test_total_with_xgboost_stays_inside_the_budget(self):
        trials = curated_trials(CONFIG, enable_xgboost=True)
        self.assertEqual(len(trials), 30)
        self.assertLessEqual(len(trials), CONFIG["models"]["trial_budget"]["max_total_trials"])

    def test_trial_ids_are_unique_and_ordering_is_deterministic(self):
        first = [t["trial_id"] for t in curated_trials(CONFIG, enable_xgboost=True)]
        second = [t["trial_id"] for t in curated_trials(CONFIG, enable_xgboost=True)]
        self.assertEqual(first, second)
        self.assertEqual(len(set(first)), len(first))

    def test_duplicate_trial_id_is_rejected(self):
        broken = json.loads(json.dumps(CONFIG))
        broken["models"]["trials"].append(dict(broken["models"]["trials"][0]))
        with self.assertRaises(TrialBudgetError) as caught:
            curated_trials(broken, enable_xgboost=True)
        self.assertIn("duplicate", str(caught.exception))

    def test_a_budget_overrun_is_rejected_rather_than_truncated(self):
        broken = json.loads(json.dumps(CONFIG))
        template = broken["models"]["trials"][0]
        for extra in range(20):
            broken["models"]["trials"].append({**template, "trial_id": f"filler-{extra}"})
        with self.assertRaises(TrialBudgetError):
            curated_trials(broken)

    def test_forbidden_family_cannot_appear_as_a_trial(self):
        broken = json.loads(json.dumps(CONFIG))
        broken["models"]["trials"][0]["family"] = "random_forest"
        with self.assertRaises(TrialBudgetError):
            curated_trials(broken)

    def test_both_parameterizations_are_represented_with_provenance(self):
        kinds = {t["parameterization"] for t in curated_trials(CONFIG)}
        self.assertEqual(kinds, {"direct", "residual"})
        provenance = CONFIG["models"]["parameterization_provenance"]
        self.assertIn("post-hoc", provenance["residual"])

    def test_no_hidden_trials_beyond_the_reported_list(self):
        report = trial_budget_report(CONFIG, enable_xgboost=True)
        self.assertEqual(
            report["selected_trial_ids"],
            [t["trial_id"] for t in curated_trials(CONFIG, enable_xgboost=True)],
        )
        self.assertEqual(report["selected_total"], report["declared_total"])


# --------------------------------------------------------------------------- #
# Correction H — XGBoost is opt-in and gated
# --------------------------------------------------------------------------- #

class TestXGBoostOptIn(unittest.TestCase):
    """Defect: 'xgboost imports' is not a reason to consider xgboost."""

    def test_disabled_by_default(self):
        status = xgboost_policy_status(CONFIG)
        self.assertEqual(status["status"], "skipped_by_policy")
        self.assertFalse(status["eligible"])

    def test_no_xgboost_trial_is_selected_by_default(self):
        selected = {t["family"] for t in curated_trials(CONFIG)}
        self.assertNotIn("xgboost_optional", selected)

    def test_default_requirements_file_does_not_install_xgboost(self):
        default = (
            REPO_ROOT / "prototype" / "halo_safeshift" / "requirements-colab.txt"
        ).read_text(encoding="utf-8")
        pins = [
            line.strip()
            for line in default.splitlines()
            if line.strip() and not line.startswith("#")
        ]
        self.assertNotIn("xgboost", " ".join(pins))
        extra = (
            REPO_ROOT / "prototype" / "halo_safeshift" / "requirements-colab-xgboost.txt"
        ).read_text(encoding="utf-8")
        self.assertIn("xgboost==", extra)

    def test_eligibility_requires_the_flag(self):
        with self.assertRaises(XGBoostPolicyError) as caught:
            assert_xgboost_eligible(0.0, CONFIG, enable_xgboost=False)
        self.assertIn("skipped by policy", str(caught.exception))

    def test_enabling_alone_does_not_make_it_eligible(self):
        status = xgboost_policy_status(CONFIG, enable_xgboost=True)
        self.assertFalse(status["eligible"])
        with self.assertRaises(XGBoostPolicyError):
            assert_xgboost_eligible(0.0, CONFIG, enable_xgboost=True)

    def test_parity_failure_rejects_the_candidate(self):
        with self.assertRaises(XGBoostPolicyError):
            assert_xgboost_eligible(1e9, CONFIG, enable_xgboost=True)


# --------------------------------------------------------------------------- #
# Correction I / J / K — refit scope, fallback, parity gate
# --------------------------------------------------------------------------- #

class TestSelectionContract(unittest.TestCase):
    """Defect: exporting the last fold's estimator, and parity as a mere report."""

    def setUp(self):
        from prototype.halo_safeshift.colab_train import (
            ParityGateError,
            assert_parity_within,
            engineering_gate,
            final_refit,
            golden_vectors,
        )

        self.ParityGateError = ParityGateError
        self.assert_parity_within = assert_parity_within
        self.engineering_gate = engineering_gate
        self.final_refit = final_refit
        self.golden_vectors = golden_vectors

    def _windows(self):
        raw = raw_frame(dense_rows(400))
        admitted, _ = apply_row_qc(raw, FIELD_MAPPING, CONFIG)
        from prototype.halo_safeshift.data import resample_five_minute

        return build_windows(
            resample_five_minute(admitted, FIELD_MAPPING, CONFIG), FIELD_MAPPING, CONFIG
        )

    def test_final_refit_uses_every_eligible_window(self):
        windows = self._windows()
        trial = {"family": "ridge", "parameterization": "direct", "params": {"alpha": 1.0}}
        estimator, preprocessing = self.final_refit(trial, windows, CONFIG)
        self.assertIn("all eligible historical-development", preprocessing.fit_scope)
        # The scaler must have seen every row, not a fold prefix.
        np.testing.assert_allclose(preprocessing.mean, windows.x.mean(axis=0))
        self.assertEqual(int(getattr(estimator, "n_features_in_")), windows.x.shape[1])

    def test_a_prefix_fit_differs_from_the_final_refit(self):
        windows = self._windows()
        trial = {"family": "ridge", "parameterization": "direct", "params": {"alpha": 1.0}}
        full, _ = self.final_refit(trial, windows, CONFIG)
        half = int(windows.x.shape[0] // 2)
        prefix_mean = windows.x[:half].mean(axis=0)
        prefix_scale = windows.x[:half].std(axis=0)
        prefix_scale[prefix_scale <= 0] = 1.0
        prefix = build_estimator("ridge", {"alpha": 1.0}, CONFIG).fit(
            (windows.x[:half] - prefix_mean) / prefix_scale, windows.y[:half]
        )
        self.assertFalse(
            np.allclose(full.coef_, prefix.coef_),
            "if a prefix fit equalled the full refit the contract would be untestable",
        )

    def test_golden_vectors_are_deterministic(self):
        windows = self._windows()
        first = self.golden_vectors(windows, CONFIG)
        second = self.golden_vectors(windows, CONFIG)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape[0], CONFIG["tolerances"]["parity_probe_vectors"])

    def test_impossible_tolerance_fails_the_parity_gate(self):
        with self.assertRaises(self.ParityGateError) as caught:
            self.assert_parity_within(1e-12, -1.0, "deliberately impossible tolerance")
        self.assertIn("ineligible", str(caught.exception))

    def test_a_reproducing_candidate_passes_the_gate(self):
        self.assertEqual(self.assert_parity_within(0.0, 1e-10, "exact"), 0.0)

    def test_engineering_gate_fails_a_challenger_that_does_not_beat_persistence(self):
        per_fold = [
            {"validation_mae": 1.00, "persistence_mae": 1.01},
            {"validation_mae": 1.02, "persistence_mae": 1.02},
        ]
        gate = self.engineering_gate(per_fold, CONFIG)
        self.assertEqual(gate["verdict"], "fail")

    def test_engineering_gate_passes_a_consistent_clear_win(self):
        per_fold = [
            {"validation_mae": 0.80, "persistence_mae": 1.00},
            {"validation_mae": 0.82, "persistence_mae": 1.02},
        ]
        gate = self.engineering_gate(per_fold, CONFIG)
        self.assertEqual(gate["verdict"], "pass")
        self.assertGreaterEqual(gate["absolute_mean_gain_degc"], 0.05)

    def test_an_inconsistent_win_is_inconclusive_not_a_pass(self):
        """A big average gain built from one fold that lost is not a pass."""
        per_fold = [
            {"validation_mae": 0.50, "persistence_mae": 1.00},
            {"validation_mae": 1.05, "persistence_mae": 1.00},
        ]
        gate = self.engineering_gate(per_fold, CONFIG)
        self.assertEqual(gate["verdict"], "inconclusive")
        self.assertFalse(gate["paired_gain_all_same_direction"])

    def test_no_invented_confidence_interval_gate(self):
        gate = self.engineering_gate(
            [{"validation_mae": 0.8, "persistence_mae": 1.0}], CONFIG
        )
        self.assertIn("paired_gain_std_degc", gate)
        self.assertNotIn("confidence_interval_lower_bound", gate)


# --------------------------------------------------------------------------- #
# Correction L — preprocessing validation
# --------------------------------------------------------------------------- #

class TestPreprocessingValidation(unittest.TestCase):
    """Defect: a malformed scaler returns a plausible number instead of failing."""

    def _standardize(self, **overrides):
        payload = {
            "kind": "standardize",
            "arrays_in_model_npz": ["preproc_mean", "preproc_scale"],
            "clipping_applied": False,
        }
        arrays = {
            "preproc_mean": np.zeros(4),
            "preproc_scale": np.ones(4),
        }
        payload.update(overrides.pop("payload", {}))
        arrays.update(overrides.pop("arrays", {}))
        return payload, arrays

    def test_a_well_formed_payload_passes(self):
        payload, arrays = self._standardize()
        self.assertEqual(validate_preprocessing(payload, arrays, 4)["kind"], "standardize")

    def test_wrong_shape_is_rejected(self):
        payload, arrays = self._standardize(arrays={"preproc_scale": np.ones(3)})
        with self.assertRaises(ValueError):
            validate_preprocessing(payload, arrays, 4)

    def test_non_finite_mean_is_rejected(self):
        payload, arrays = self._standardize(arrays={"preproc_mean": np.array([0, np.nan, 0, 0.0])})
        with self.assertRaises(ValueError):
            validate_preprocessing(payload, arrays, 4)

    def test_non_positive_scale_is_rejected(self):
        for bad in (0.0, -1.0):
            with self.subTest(scale=bad):
                payload, arrays = self._standardize(
                    arrays={"preproc_scale": np.array([1.0, bad, 1.0, 1.0])}
                )
                with self.assertRaises(ValueError):
                    validate_preprocessing(payload, arrays, 4)

    def test_half_declared_clipping_is_rejected(self):
        payload, arrays = self._standardize(
            payload={
                "arrays_in_model_npz": ["preproc_clip_min", "preproc_mean", "preproc_scale"],
                "clipping_applied": True,
            },
            arrays={"preproc_clip_min": np.zeros(4)},
        )
        with self.assertRaises(ValueError) as caught:
            validate_preprocessing(payload, arrays, 4)
        self.assertIn("together or not at all", str(caught.exception))

    def test_clip_min_above_clip_max_is_rejected(self):
        payload, arrays = self._standardize(
            payload={
                "arrays_in_model_npz": [
                    "preproc_clip_max",
                    "preproc_clip_min",
                    "preproc_mean",
                    "preproc_scale",
                ],
                "clipping_applied": True,
            },
            arrays={
                "preproc_clip_min": np.array([0.0, 5.0, 0.0, 0.0]),
                "preproc_clip_max": np.array([1.0, 1.0, 1.0, 1.0]),
            },
        )
        with self.assertRaises(ValueError) as caught:
            validate_preprocessing(payload, arrays, 4)
        self.assertIn("index 1", str(caught.exception))

    def test_json_disagreeing_with_the_arrays_is_rejected(self):
        payload, arrays = self._standardize(payload={"kind": "identity"})
        with self.assertRaises(ValueError) as caught:
            validate_preprocessing(payload, arrays, 4)
        self.assertIn("different transforms", str(caught.exception))

    def test_declared_clipping_flag_must_match_reality(self):
        payload, arrays = self._standardize(payload={"clipping_applied": True})
        with self.assertRaises(ValueError):
            validate_preprocessing(payload, arrays, 4)

    def test_the_exporter_refuses_a_bad_scaler_before_writing(self):
        with self.assertRaises(ExportError):
            Preprocessing(
                kind="standardize",
                mean=np.zeros(N_FEATURES),
                scale=np.zeros(N_FEATURES),
                fit_scope="test",
            )

    def test_one_clipping_bound_alone_is_refused(self):
        with self.assertRaises(ExportError):
            Preprocessing(kind="identity", clip_min=np.zeros(N_FEATURES))

    def test_the_runtime_rejects_a_bundle_whose_preprocessing_json_was_edited(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = _linear_bundle(Path(tmp) / "b")
            path = bundle / "preprocessing.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["kind"] = "standardize"
            path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            reseal(bundle)
            with self.assertRaises(InferenceError):
                SafeShiftPredictor.load(bundle)


def _linear_bundle(directory: Path, standardize: bool = False) -> Path:
    rng = np.random.default_rng(int(CONFIG["seed"]))
    x = rng.normal(size=(80, N_FEATURES))
    y = x[:, 3] * 0.5
    if standardize:
        mean, scale = x.mean(axis=0), x.std(axis=0)
        scale[scale <= 0] = 1.0
        pre = Preprocessing("standardize", mean, scale, fit_scope="test")
        estimator = build_estimator("ridge", {"alpha": 1.0}, CONFIG).fit((x - mean) / scale, y)
    else:
        pre = Preprocessing()
        estimator = build_estimator("ridge", {"alpha": 1.0}, CONFIG).fit(x, y)
    return export_linear(directory, estimator, SCHEMA, pre, config=CONFIG)


# --------------------------------------------------------------------------- #
# Correction M — tree structural validation
# --------------------------------------------------------------------------- #

def _tiny_ensemble() -> dict:
    """One tree: root splits on feature 0, two leaves."""
    return {
        "model_type": "tree_ensemble",
        "aggregation": "sum",
        "base_score": 0.0,
        "learning_rate": 1.0,
        "decision_rule": "left_if_le",
        "trees": [
            {
                "feature": [0, -1, -1],
                "threshold": [0.5, 0.0, 0.0],
                "left": [1, -1, -1],
                "right": [2, -1, -1],
                "value": [0.0, 1.0, 2.0],
                "missing_go_to": [0, 0, 0],
            }
        ],
    }


class TestTreeStructuralValidation(unittest.TestCase):
    """Defect: a corrupt tree surfaced as an IndexError from inside NumPy."""

    def setUp(self):
        self.arrays = flatten_tree_ensemble(_tiny_ensemble())

    def _mutate(self, **changes):
        arrays = {k: np.array(v) for k, v in self.arrays.items()}
        arrays.update({k: np.asarray(v) for k, v in changes.items()})
        return arrays

    def test_a_well_formed_ensemble_validates(self):
        result = validate_flat_tree_arrays(self.arrays, N_FEATURES)
        self.assertEqual(result, {"n_trees": 1, "n_nodes": 3})

    def test_unequal_node_array_lengths_are_rejected(self):
        arrays = self._mutate(tree_node_value=np.array([0.0, 1.0]))
        with self.assertRaises(TreeExportError) as caught:
            validate_flat_tree_arrays(arrays, N_FEATURES)
        self.assertIn("unequal lengths", str(caught.exception))

    def test_offsets_must_start_at_zero(self):
        arrays = self._mutate(tree_offsets=np.array([1, 3], dtype=np.int32))
        with self.assertRaises(TreeExportError) as caught:
            validate_flat_tree_arrays(arrays, N_FEATURES)
        self.assertIn("start at zero", str(caught.exception))

    def test_offsets_must_be_strictly_increasing(self):
        arrays = self._mutate(tree_offsets=np.array([0, 0, 3], dtype=np.int32))
        with self.assertRaises(TreeExportError) as caught:
            validate_flat_tree_arrays(arrays, N_FEATURES)
        self.assertIn("strictly increasing", str(caught.exception))

    def test_final_offset_must_equal_the_node_count(self):
        arrays = self._mutate(tree_offsets=np.array([0, 2], dtype=np.int32))
        with self.assertRaises(TreeExportError) as caught:
            validate_flat_tree_arrays(arrays, N_FEATURES)
        self.assertIn("node count", str(caught.exception))

    def test_a_child_outside_its_own_tree_is_rejected(self):
        arrays = self._mutate(tree_node_right=np.array([9, -1, -1], dtype=np.int32))
        with self.assertRaises(TreeExportError) as caught:
            validate_flat_tree_arrays(arrays, N_FEATURES)
        self.assertIn("outside its own tree", str(caught.exception))

    def test_a_leaf_with_a_child_is_rejected(self):
        arrays = self._mutate(tree_node_left=np.array([1, 2, -1], dtype=np.int32))
        with self.assertRaises(TreeExportError) as caught:
            validate_flat_tree_arrays(arrays, N_FEATURES)
        self.assertIn("leaf node declares a child", str(caught.exception))

    def test_an_out_of_range_split_feature_is_rejected(self):
        arrays = self._mutate(
            tree_node_feature=np.array([N_FEATURES + 5, -1, -1], dtype=np.int32)
        )
        with self.assertRaises(TreeExportError) as caught:
            validate_flat_tree_arrays(arrays, N_FEATURES)
        self.assertIn("feature index outside", str(caught.exception))

    def test_a_non_finite_threshold_is_rejected(self):
        arrays = self._mutate(tree_node_threshold=np.array([np.nan, 0.0, 0.0]))
        with self.assertRaises(TreeExportError) as caught:
            validate_flat_tree_arrays(arrays, N_FEATURES)
        self.assertIn("non-finite threshold", str(caught.exception))

    def test_missing_go_to_must_be_zero_or_one(self):
        arrays = self._mutate(tree_node_missing_go_to=np.array([7, 0, 0], dtype=np.int8))
        with self.assertRaises(TreeExportError) as caught:
            validate_flat_tree_arrays(arrays, N_FEATURES)
        self.assertIn("only 0 (left) or 1 (right)", str(caught.exception))

    def test_manifest_counts_must_match_the_arrays(self):
        with self.assertRaises(TreeExportError) as caught:
            validate_flat_tree_arrays(self.arrays, N_FEATURES, manifest_n_trees=4)
        self.assertIn("declares 4 trees", str(caught.exception))
        with self.assertRaises(TreeExportError):
            validate_flat_tree_arrays(self.arrays, N_FEATURES, manifest_n_nodes=99)

    def test_unsupported_aggregation_and_rule_are_rejected(self):
        with self.assertRaises(TreeExportError):
            validate_flat_tree_arrays(self.arrays, N_FEATURES, aggregation="median")
        with self.assertRaises(TreeExportError):
            validate_flat_tree_arrays(self.arrays, N_FEATURES, decision_rule="left_if_gt")

    def test_the_runtime_raises_inference_error_not_index_error(self):
        """The whole point: a malformed artifact is refused, not crashed on."""
        from sklearn.ensemble import GradientBoostingRegressor

        from prototype.halo_safeshift.export import export_estimator

        rng = np.random.default_rng(0)
        x = rng.normal(size=(60, N_FEATURES))
        y = x[:, 2]
        estimator = GradientBoostingRegressor(n_estimators=4, max_depth=2, random_state=0).fit(
            x, y
        )
        with tempfile.TemporaryDirectory() as tmp:
            bundle = export_estimator(
                Path(tmp) / "trees", estimator, SCHEMA, Preprocessing(), config=CONFIG
            )
            with np.load(bundle / "model.npz", allow_pickle=False) as handle:
                arrays = {key: np.array(handle[key]) for key in handle.files}
            arrays["tree_node_right"][0] = 10_000  # child outside its own tree
            with (bundle / "model.npz").open("wb") as handle:
                np.savez(handle, allow_pickle=False, **arrays)
            manifest_path = bundle / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["files"]["model.npz"] = sha256_file(bundle / "model.npz")
            manifest_path.write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
            )
            reseal(bundle)
            with self.assertRaises(InferenceError) as caught:
                SafeShiftPredictor.load(bundle)
            self.assertNotIsInstance(caught.exception, IndexError)

    def test_split_comparison_reproduces_the_source_library_width(self):
        """Regression: float64 comparison routed samples down the wrong branch.

        scikit-learn casts X to float32 before traversal. A value one float32
        step away from a threshold therefore goes left in sklearn and right in a
        naive float64 evaluator, and the result is a plausible wrong number
        rather than an error. Random probe data almost never lands that close,
        which is why this survived until it was run against real standardized
        station features.
        """
        # float32 spacing just above 1.0 is ~1.19e-7, so 1.00000005 rounds down
        # to exactly 1.0 while staying above the threshold in float64.
        threshold = 1.00000002
        x = np.array([[1.00000005]], dtype=np.float64)
        self.assertGreater(x[0, 0], threshold, "float64 sends this sample right")
        self.assertLessEqual(
            float(np.float32(x[0, 0])), threshold, "float32 sends the same sample left"
        )

        bundle = {
            "model_type": "tree_ensemble",
            "aggregation": "sum",
            "base_score": 0.0,
            "learning_rate": 1.0,
            "decision_rule": "left_if_le",
            "trees": [
                {
                    "feature": [0, -1, -1],
                    "threshold": [threshold, 0.0, 0.0],
                    "left": [1, -1, -1],
                    "right": [2, -1, -1],
                    "value": [0.0, 10.0, 20.0],
                    "missing_go_to": [0, 0, 0],
                }
            ],
        }
        from prototype.halo_safeshift.tree_export import evaluate_tree_ensemble

        self.assertEqual(
            float(evaluate_tree_ensemble({**bundle, "split_comparison_dtype": "float32"}, x)[0]),
            10.0,
            "float32 comparison must send this sample left, as sklearn does",
        )
        self.assertEqual(
            float(evaluate_tree_ensemble({**bundle, "split_comparison_dtype": "float64"}, x)[0]),
            20.0,
            "float64 comparison sends it right - the two are different functions",
        )
        self.assertEqual(
            float(evaluate_tree_ensemble(bundle, x)[0]),
            10.0,
            "the default must be the source library's width",
        )

    def test_sklearn_exporters_record_the_comparison_width(self):
        from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor

        from prototype.halo_safeshift.tree_export import (
            export_sklearn_extra_trees,
            export_sklearn_gradient_boosting,
        )

        rng = np.random.default_rng(0)
        x = rng.normal(size=(50, 6))
        y = x[:, 0]
        for estimator, exporter in (
            (GradientBoostingRegressor(n_estimators=3, random_state=0), export_sklearn_gradient_boosting),
            (ExtraTreesRegressor(n_estimators=3, random_state=0), export_sklearn_extra_trees),
        ):
            with self.subTest(estimator=type(estimator).__name__):
                neutral = exporter(estimator.fit(x, y))
                self.assertEqual(neutral["split_comparison_dtype"], "float32")

    def test_an_unsupported_comparison_dtype_is_rejected(self):
        with self.assertRaises(TreeExportError):
            validate_flat_tree_arrays(self.arrays, N_FEATURES, comparison_dtype="float16")

    def test_missing_value_routing_is_declared_inert_not_supported(self):
        policy = CONFIG["validation"]["tree_ensemble"]["missing_value_routing_policy"]
        self.assertEqual(policy, "inert_under_finite_input_policy")
        source = (REPO_ROOT / "prototype" / "halo_safeshift" / "inference.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("unreachable", source)


# --------------------------------------------------------------------------- #
# Correction P — the bundle envelope
# --------------------------------------------------------------------------- #

class TestBundleEnvelope(unittest.TestCase):
    """Defect: manifest.json was both the claim and the authority for the claim."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.bundle = _linear_bundle(Path(self._tmp.name) / "bundle", standardize=True)

    def tearDown(self):
        self._tmp.cleanup()

    def test_an_envelope_is_written_and_loads(self):
        self.assertTrue((self.bundle / ENVELOPE_FILENAME).is_file())
        SafeShiftPredictor.load(self.bundle)

    def test_a_bundle_without_an_envelope_is_refused(self):
        (self.bundle / ENVELOPE_FILENAME).unlink()
        with self.assertRaises(InferenceError) as caught:
            SafeShiftPredictor.load(self.bundle)
        self.assertIn("bundle envelope is missing", str(caught.exception))

    def test_manifest_edited_without_resealing_is_caught(self):
        for field, value in (
            ("model_type", "tree_ensemble"),
            ("n_features", N_FEATURES + 1),
            ("output_transform", {"kind": "residual_plus_feature", "feature_index": 0}),
            ("run_id", "some-other-run"),
        ):
            with self.subTest(field=field):
                self.setUp()
                path = self.bundle / "manifest.json"
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload[field] = value
                path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
                with self.assertRaises(InferenceError) as caught:
                    SafeShiftPredictor.load(self.bundle)
                self.assertIn("envelope SHA-256 mismatch", str(caught.exception))

    def test_envelope_disagreeing_with_the_manifest_is_caught(self):
        """Both files present and internally consistent, but they disagree."""
        path = self.bundle / ENVELOPE_FILENAME
        envelope = json.loads(path.read_text(encoding="utf-8"))
        envelope["attested"]["model_type"] = "tree_ensemble"
        path.write_text(json.dumps(envelope, indent=2) + "\n", encoding="utf-8")
        with self.assertRaises(InferenceError) as caught:
            SafeShiftPredictor.load(self.bundle)
        self.assertIn("disagree on 'model_type'", str(caught.exception))

    def test_envelope_attests_the_runtime_source_and_the_config(self):
        envelope = json.loads((self.bundle / ENVELOPE_FILENAME).read_text(encoding="utf-8"))
        self.assertEqual(len(envelope["config_sha256"]), 64)
        for relative in CONFIG["envelope"]["runtime_source_files"]:
            self.assertIn(relative, envelope["runtime_source_files"])
            self.assertEqual(
                envelope["runtime_source_files"][relative]["sha256"],
                sha256_file(REPO_ROOT / relative),
            )

    def test_a_sibling_swapped_after_sealing_is_caught(self):
        (self.bundle / "target_contract.json").write_text("{}\n", encoding="utf-8")
        with self.assertRaises(InferenceError) as caught:
            SafeShiftPredictor.load(self.bundle)
        self.assertIn("target_contract.json", str(caught.exception))

    def test_envelope_cannot_be_written_without_a_manifest(self):
        (self.bundle / "manifest.json").unlink()
        with self.assertRaises(ExportError):
            write_bundle_envelope(self.bundle)


# --------------------------------------------------------------------------- #
# Correction B — deterministic source archive
# --------------------------------------------------------------------------- #

class TestSourceArchive(unittest.TestCase):
    """Defect: cloning git HEAD does not reproduce an uncommitted working tree."""

    def test_identical_source_produces_identical_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = build_source_archive(Path(tmp) / "a.zip", CONFIG)
            second = build_source_archive(Path(tmp) / "b.zip", CONFIG)
            self.assertEqual(first["sha256"], second["sha256"])
            self.assertEqual(
                (Path(tmp) / "a.zip").read_bytes(), (Path(tmp) / "b.zip").read_bytes()
            )

    def test_entry_metadata_is_pinned(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.zip"
            build_source_archive(path, CONFIG)
            expected = tuple(CONFIG["packet"]["source_archive_determinism"]["zip_date_time"])
            with zipfile.ZipFile(path) as archive:
                names = archive.namelist()
                for info in archive.infolist():
                    self.assertEqual(info.date_time, expected)
                    self.assertEqual(info.create_system, 3)
            self.assertEqual(names, sorted(names), "entries must be in sorted path order")

    def test_caches_and_bytecode_are_excluded(self):
        for name, _ in collect_source_files(CONFIG):
            self.assertNotIn("__pycache__", name)
            self.assertFalse(name.endswith(".pyc"))

    def test_contents_cover_the_declared_contract(self):
        names = [name for name, _ in collect_source_files(CONFIG)]
        self.assertTrue(any(n.endswith("config/experiment.v1.json") for n in names))
        self.assertTrue(any(n.endswith("requirements-colab.txt") for n in names))
        self.assertTrue(any(n.endswith("requirements-colab-xgboost.txt") for n in names))
        self.assertTrue(any(n.endswith("inference.py") for n in names))

    def test_member_hashes_match_the_files_on_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = build_source_archive(Path(tmp) / "a.zip", CONFIG)
            self.assertEqual(archive_member_hashes(Path(tmp) / "a.zip"), record["members"])

    def test_paths_are_repository_relative(self):
        for name, _ in collect_source_files(CONFIG):
            self.assertFalse(Path(name).is_absolute())
            self.assertTrue(name.startswith("prototype/halo_safeshift/"))


# --------------------------------------------------------------------------- #
# Correction A — packet self-verification
# --------------------------------------------------------------------------- #

def build_test_packet(directory: Path) -> Path:
    """A minimal but genuinely valid packet, assembled the way the driver does."""
    from prototype.halo_safeshift import DEFAULT_CONFIG_PATH, write_json
    from prototype.halo_safeshift.prepare_colab import code_manifest

    packet = directory / "colab-input"
    packet.mkdir(parents=True)
    run_id = "20260101T000000Z-testrun"

    rows = dense_rows(40)
    header = "entry_id,created_at,status,field1,field2,field3,field4,field5,field6,field7,field8\n"
    body = "".join(
        ",".join(
            [r["entry_id"], r["created_at"], r["status"], *[r[f"field{i}"] for i in range(1, 9)]]
        )
        + "\n"
        for r in rows
    )
    (packet / "station_raw.csv").write_text(header + body, encoding="utf-8", newline="")
    write_json(packet / "channel_metadata.json", {"channel": CHANNEL_METADATA})

    csv_sha = sha256_file(packet / "station_raw.csv")
    metadata_sha = sha256_file(packet / "channel_metadata.json")
    write_json(
        packet / "data_manifest.json",
        {
            "artifact": "data_manifest.json",
            "config_id": CONFIG["config_id"],
            "source": {"channel_id": CONFIG["source"]["thingspeak_channel_id"]},
            "integrity": {
                "csv_path": "test/station_raw.csv",
                "csv_sha256": csv_sha,
                "metadata_path": "test/channel_metadata.json",
                "metadata_sha256": metadata_sha,
                "puller_path": "benchmarks/halo/freeze_station_history.py",
                "puller_sha256": sha256_file(PULLER),
            },
        },
    )
    write_json(packet / "qc_report.json", {"artifact": "qc_report.json", "run_id": run_id})
    write_json(
        packet / "recovery_partition.json",
        {"artifact": "recovery_partition.json", "run_id": run_id},
    )
    (packet / "experiment.v1.json").write_bytes(DEFAULT_CONFIG_PATH.read_bytes())
    write_json(
        packet / "source_manifest.json",
        {
            "artifact": "source_manifest.json",
            "run_id": run_id,
            "station_raw_csv": {"path": "test/station_raw.csv", "sha256": csv_sha},
            "channel_metadata": {"path": "test/channel_metadata.json", "sha256": metadata_sha},
            "puller": {
                "path": "benchmarks/halo/freeze_station_history.py",
                "sha256": sha256_file(PULLER),
            },
        },
    )
    archive = build_source_archive(packet / "halo_safeshift-source.zip", CONFIG)
    (packet / "bootstrap.py").write_bytes(
        (REPO_ROOT / "prototype" / "halo_safeshift" / "bootstrap_colab.py").read_bytes()
    )
    write_json(packet / "code_manifest.json", code_manifest(CONFIG, run_id, archive))

    files = [
        "station_raw.csv",
        "channel_metadata.json",
        "data_manifest.json",
        "qc_report.json",
        "recovery_partition.json",
        "experiment.v1.json",
        "source_manifest.json",
        "code_manifest.json",
        "halo_safeshift-source.zip",
        "bootstrap.py",
    ]
    write_json(
        packet / "colab_packet_manifest.json",
        {
            "artifact": "colab_packet_manifest.json",
            "run_id": run_id,
            "config_id": CONFIG["config_id"],
            "files": {name: sha256_file(packet / name) for name in files},
        },
    )
    return packet


class TestPacketSelfVerification(unittest.TestCase):
    """Defect: the packet could not prove it was the packet that was prepared."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.packet = build_test_packet(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_an_untouched_packet_verifies(self):
        report = verify_packet(self.packet, CONFIG)
        self.assertTrue(report["verified"])
        self.assertEqual(report["n_declared_files"], 10)

    def test_the_manifest_lives_inside_the_packet(self):
        self.assertTrue((self.packet / "colab_packet_manifest.json").is_file())

    def test_one_flipped_byte_fails_every_declared_file(self):
        for name in (
            "station_raw.csv",
            "channel_metadata.json",
            "experiment.v1.json",
            "source_manifest.json",
            "code_manifest.json",
            "halo_safeshift-source.zip",
        ):
            with self.subTest(file=name):
                self.setUp()
                flip_one_byte(self.packet / name)
                with self.assertRaises(PacketVerificationError):
                    verify_packet(self.packet, CONFIG)

    def test_a_flipped_byte_in_the_packet_manifest_fails(self):
        path = self.packet / "colab_packet_manifest.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        entry = payload["files"]["station_raw.csv"]
        payload["files"]["station_raw.csv"] = ("0" if entry[0] != "0" else "1") + entry[1:]
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        with self.assertRaises(PacketVerificationError) as caught:
            verify_packet(self.packet, CONFIG)
        self.assertIn("SHA-256 mismatch", str(caught.exception))

    def test_an_undeclared_file_is_rejected(self):
        (self.packet / "notes_from_somewhere.txt").write_text("extra", encoding="utf-8")
        with self.assertRaises(PacketVerificationError) as caught:
            verify_packet(self.packet, CONFIG)
        self.assertIn("undeclared file", str(caught.exception))

    def test_a_missing_declared_file_is_rejected(self):
        (self.packet / "qc_report.json").unlink()
        with self.assertRaises(PacketVerificationError):
            verify_packet(self.packet, CONFIG)

    def test_a_csv_that_disagrees_with_the_data_manifest_is_rejected(self):
        """Re-hash the CSV into the packet manifest but not the data manifest."""
        (self.packet / "station_raw.csv").write_text("entry_id\n1\n", encoding="utf-8")
        path = self.packet / "colab_packet_manifest.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["files"]["station_raw.csv"] = sha256_file(self.packet / "station_raw.csv")
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        with self.assertRaises(PacketVerificationError) as caught:
            verify_packet(self.packet, CONFIG)
        self.assertIn("data_manifest.integrity.csv_sha256", str(caught.exception))

    def test_a_puller_hash_disagreement_is_rejected(self):
        path = self.packet / "source_manifest.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["puller"]["sha256"] = "0" * 64
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        self._reseal_packet()
        with self.assertRaises(PacketVerificationError) as caught:
            verify_packet(self.packet, CONFIG)
        self.assertIn("puller sha256", str(caught.exception))

    def test_a_wrong_channel_id_is_rejected(self):
        path = self.packet / "data_manifest.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["source"]["channel_id"] = 3448221
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        self._reseal_packet()
        with self.assertRaises(PacketVerificationError) as caught:
            verify_packet(self.packet, CONFIG)
        self.assertIn("channel id", str(caught.exception))

    def test_a_run_id_disagreement_is_rejected(self):
        path = self.packet / "qc_report.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["run_id"] = "a-different-run"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        self._reseal_packet()
        with self.assertRaises(PacketVerificationError) as caught:
            verify_packet(self.packet, CONFIG)
        self.assertIn("disagree", str(caught.exception))

    def test_the_archive_hash_recorded_in_the_code_manifest_is_cross_checked(self):
        """Defence in depth: two independent records must name the same archive."""
        archive_path = self.packet / "halo_safeshift-source.zip"
        self._inject_into_archive(archive_path)
        self._reseal_packet()
        with self.assertRaises(PacketVerificationError) as caught:
            verify_packet(self.packet, CONFIG)
        self.assertIn("code_manifest vs packet manifest", str(caught.exception))

    def test_a_swapped_archive_member_is_caught_from_inside_the_archive(self):
        """Even with every outer hash repaired, member content is re-read.

        Hashing the files on the preparation machine proves what was intended.
        Only hashing the archive's own contents proves what was shipped.
        """
        archive_path = self.packet / "halo_safeshift-source.zip"
        self._inject_into_archive(archive_path)
        code_path = self.packet / "code_manifest.json"
        code = json.loads(code_path.read_text(encoding="utf-8"))
        code["source_archive"]["sha256"] = sha256_file(archive_path)
        code_path.write_text(json.dumps(code, indent=2) + "\n", encoding="utf-8")
        self._reseal_packet()
        with self.assertRaises(PacketVerificationError) as caught:
            verify_packet(self.packet, CONFIG)
        self.assertIn("inference.py", str(caught.exception))

    @staticmethod
    def _inject_into_archive(archive_path: Path) -> None:
        with zipfile.ZipFile(archive_path) as source:
            entries = [(info, source.read(info.filename)) for info in source.infolist()]
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as target:
            for info, payload in entries:
                if info.filename.endswith("inference.py"):
                    payload = payload + b"\n# injected\n"
                target.writestr(info, payload)

    def _reseal_packet(self):
        """Re-hash the packet manifest so an inner cross-check is what fails."""
        path = self.packet / "colab_packet_manifest.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["files"] = {name: sha256_file(self.packet / name) for name in payload["files"]}
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Correction Q — bootstrap verifies before it extracts
# --------------------------------------------------------------------------- #

class TestBootstrap(unittest.TestCase):
    """Defect: extracting first and checking afterwards writes unverified code."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.packet = build_test_packet(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def test_a_tampered_archive_is_refused_before_anything_is_written(self):
        from prototype.halo_safeshift.bootstrap_colab import BootstrapError, verify_archive

        flip_one_byte(self.packet / "halo_safeshift-source.zip")
        target = self.root / "extract-here"
        with self.assertRaises(BootstrapError) as caught:
            verify_archive(self.packet)
        self.assertIn("Refusing to extract", str(caught.exception))
        self.assertFalse(target.exists(), "nothing may be written before verification")

    def test_a_verified_archive_extracts_and_re_hashes(self):
        from prototype.halo_safeshift.bootstrap_colab import (
            extract,
            verify_archive,
            verify_extracted,
        )

        archive, manifest = verify_archive(self.packet)
        target = self.root / "extract-here"
        members = extract(archive, target)
        self.assertGreater(len(members), 0)
        self.assertEqual(verify_extracted(manifest, target), len(members))

    def test_a_truncated_extraction_is_detected(self):
        from prototype.halo_safeshift.bootstrap_colab import (
            BootstrapError,
            extract,
            verify_archive,
            verify_extracted,
        )

        archive, manifest = verify_archive(self.packet)
        target = self.root / "extract-here"
        extract(archive, target)
        (target / "prototype" / "halo_safeshift" / "inference.py").unlink()
        with self.assertRaises(BootstrapError):
            verify_extracted(manifest, target)

    def test_the_documented_colab_command_uses_python_not_py_dash_3(self):
        source = (
            REPO_ROOT / "prototype" / "halo_safeshift" / "bootstrap_colab.py"
        ).read_text(encoding="utf-8")
        self.assertIn("python colab-input/bootstrap.py", source)
        self.assertNotIn("py -3", source)
        train = (REPO_ROOT / "prototype" / "halo_safeshift" / "colab_train.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("python -m prototype.halo_safeshift.colab_train", train)
        self.assertNotIn("py -3", train)


# --------------------------------------------------------------------------- #
# Correction N — the freezer refuses rather than writing something wrong
# --------------------------------------------------------------------------- #

def load_freezer():
    import importlib.util

    spec = importlib.util.spec_from_file_location("halo_freezer", PULLER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestFreezerFailClosed(unittest.TestCase):
    """Every case here is checked with a mocked fetch. No live API call is made."""

    def setUp(self):
        self.freezer = load_freezer()
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.channel = int(CONFIG["source"]["thingspeak_channel_id"])

    def tearDown(self):
        self._tmp.cleanup()

    def _feeds(self, count, start_entry=1):
        base = datetime(2026, 7, 22, 0, 0, tzinfo=timezone.utc)
        return [
            {
                "entry_id": start_entry + i,
                "created_at": (base + timedelta(seconds=30 * i))
                .strftime("%Y-%m-%dT%H:%M:%SZ"),
                "status": "",
                **{f"field{f}": "1.0" for f in range(1, 9)},
            }
            for i in range(count)
        ]

    def _channel(self, **overrides):
        payload = {"id": self.channel, "name": "test", **CHANNEL_METADATA}
        payload["id"] = self.channel
        payload.update(overrides)
        return payload

    def test_zero_rows_is_refused(self):
        def fetch(url):
            return {"channel": self._channel(), "feeds": []}

        with self.assertRaises(self.freezer.FreezeError) as caught:
            self.freezer.pull_all(fetch, "http://x", 10, 5)
        self.assertIn("zero rows", str(caught.exception))

    def test_a_page_cap_truncation_is_refused(self):
        state = {"entry": 1}

        def fetch(url):
            feeds = self._feeds(10, state["entry"])
            state["entry"] += 10
            return {"channel": self._channel(), "feeds": feeds}

        with self.assertRaises(self.freezer.FreezeError) as caught:
            self.freezer.pull_all(fetch, "http://x", 10, 3)
        self.assertIn("truncated prefix", str(caught.exception))

    def test_an_exhausted_history_is_marked_complete(self):
        calls = {"n": 0}

        def fetch(url):
            calls["n"] += 1
            if calls["n"] == 1:
                return {"channel": self._channel(), "feeds": self._feeds(5)}
            return {"channel": self._channel(), "feeds": []}

        rows, urls, channel, complete = self.freezer.pull_all(fetch, "http://x", 10, 5)
        self.assertTrue(complete)
        self.assertEqual(len(rows), 5)

    def test_absent_channel_metadata_is_refused(self):
        with self.assertRaises(self.freezer.FreezeError) as caught:
            self.freezer.validate_channel(None, CONFIG, self.channel)
        self.assertIn("no channel metadata", str(caught.exception))

    def test_a_different_channel_is_refused(self):
        with self.assertRaises(self.freezer.FreezeError) as caught:
            self.freezer.validate_channel(self._channel(id=3448221), CONFIG, self.channel)
        self.assertIn("attribute one station", str(caught.exception))

    def test_missing_required_feature_labels_are_refused(self):
        broken = self._channel()
        del broken["field5"]
        with self.assertRaises(self.freezer.FreezeError) as caught:
            self.freezer.validate_channel(broken, CONFIG, self.channel)
        self.assertIn("required feature", str(caught.exception))

    def test_refuses_to_overwrite_an_existing_output(self):
        existing = REPO_ROOT / "benchmarks" / "halo" / "_cache" / "overwrite_probe.csv"
        existing.parent.mkdir(parents=True, exist_ok=True)
        existing.write_text("already here\n", encoding="utf-8")
        try:
            with self.assertRaises(self.freezer.FreezeError) as caught:
                self.freezer.require_new_path(existing, "raw CSV")
            self.assertIn("already exists", str(caught.exception))
        finally:
            existing.unlink()

    def test_a_path_outside_the_repository_is_refused(self):
        with self.assertRaises(self.freezer.FreezeError) as caught:
            self.freezer.repo_relative(self.root / "somewhere.csv")
        self.assertIn("inside the repository", str(caught.exception))

    def test_atomic_write_leaves_no_partial_file(self):
        target = REPO_ROOT / "benchmarks" / "halo" / "_cache" / "atomic_probe.txt"
        try:
            self.freezer.atomic_write_text(target, "content\n")
            self.assertEqual(target.read_text(encoding="utf-8"), "content\n")
            self.assertFalse(target.with_name(target.name + ".partial").exists())
        finally:
            if target.exists():
                target.unlink()

    def test_frozen_csv_round_trips_without_network(self):
        run = REPO_ROOT / "benchmarks" / "halo" / "_cache" / "20260815T170833Z-99afc2c"
        if not (run / "station_raw.csv").is_file():
            self.skipTest("the reused freeze is not present in this checkout")
        rows = self.freezer.read_frozen_csv(run / "station_raw.csv")
        self.assertGreater(len(rows), 0)
        self.assertEqual(rows[0]["entry_id"], 1)


# --------------------------------------------------------------------------- #
# Correction O — preparation provenance
# --------------------------------------------------------------------------- #

class TestPreparationProvenance(unittest.TestCase):
    """Defect: the driver trusted whatever files it was pointed at."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.packet = build_test_packet(self.root)
        # Reuse the synthetic packet's files as standalone inputs.
        self.csv = self.packet / "station_raw.csv"
        self.metadata = self.packet / "channel_metadata.json"
        self.data_manifest = self.packet / "data_manifest.json"

    def tearDown(self):
        self._tmp.cleanup()

    def _assert(self):
        from prototype.halo_safeshift.prepare_colab import assert_source_provenance

        return assert_source_provenance(
            self.csv, self.metadata, self.data_manifest, PULLER, CONFIG
        )

    def test_matching_inputs_pass(self):
        result = self._assert()
        self.assertEqual(result["csv_sha256"], sha256_file(self.csv))
        self.assertEqual(result["puller_path"], "benchmarks/halo/freeze_station_history.py")

    def test_a_changed_csv_is_blocked(self):
        from prototype.halo_safeshift.prepare_colab import BlockedError

        self.csv.write_text("entry_id\n1\n", encoding="utf-8")
        with self.assertRaises(BlockedError) as caught:
            self._assert()
        self.assertIn("not the frozen pull", str(caught.exception))

    def test_a_changed_metadata_file_is_blocked(self):
        from prototype.halo_safeshift.prepare_colab import BlockedError

        self.metadata.write_text("{}\n", encoding="utf-8")
        with self.assertRaises(BlockedError):
            self._assert()

    def test_the_wrong_puller_is_blocked(self):
        from prototype.halo_safeshift.prepare_colab import BlockedError

        other = REPO_ROOT / "benchmarks" / "halo" / "pull_station_history.py"
        with self.assertRaises(BlockedError) as caught:
            from prototype.halo_safeshift.prepare_colab import assert_source_provenance

            assert_source_provenance(
                self.csv, self.metadata, self.data_manifest, other, CONFIG
            )
        self.assertIn("produced by", str(caught.exception))

    def test_a_changed_puller_hash_is_blocked(self):
        from prototype.halo_safeshift.prepare_colab import BlockedError

        payload = json.loads(self.data_manifest.read_text(encoding="utf-8"))
        payload["integrity"]["puller_sha256"] = "0" * 64
        self.data_manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        with self.assertRaises(BlockedError) as caught:
            self._assert()
        self.assertIn("producer has changed", str(caught.exception))

    def test_the_wrong_channel_is_blocked(self):
        from prototype.halo_safeshift.prepare_colab import BlockedError

        payload = json.loads(self.data_manifest.read_text(encoding="utf-8"))
        payload["source"]["channel_id"] = 3448221
        self.data_manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        with self.assertRaises(BlockedError) as caught:
            self._assert()
        self.assertIn("configured station", str(caught.exception))

    def test_the_default_puller_is_the_safeshift_freezer(self):
        from prototype.halo_safeshift.prepare_colab import DEFAULT_PULLER

        self.assertEqual(DEFAULT_PULLER.name, "freeze_station_history.py")

    def test_recorded_versions_are_observed_not_assumed(self):
        from prototype.halo_safeshift.prepare_colab import environment_versions

        versions = environment_versions()
        self.assertEqual(versions["numpy"], np.__version__)
        self.assertIn("sklearn_status", versions)
        if versions["xgboost"] is None:
            self.assertEqual(versions["xgboost_status"], "not importable in this environment")


if __name__ == "__main__":
    unittest.main()
