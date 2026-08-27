"""Contract tests for zero-training satellite context and its dashboard.

No test here touches the network. The download boundary is the only thing
mocked, and it is mocked at ``download_object`` so that everything above it -
timing admission, ROI geometry, statistics, schema, degradation - runs for real.
"""

from __future__ import annotations

import json
import tempfile
import unittest
import unittest.mock
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from prototype.halo_safeshift import satellite_context as sc
from prototype.halo_safeshift import satellite_context_dashboard as dash


NOMINAL = "2026-08-09T07:00:00Z"
ISSUE = "2026-08-09T07:45:00Z"


def synthetic_navigation(rows: int = 220, cols: int = 220, *, span_deg: float = 3.0):
    """A lon/lat grid centred on the site, wide enough that a 25 km disc is interior."""
    lat = np.linspace(sc.SITE_LATITUDE + span_deg / 2, sc.SITE_LATITUDE - span_deg / 2, rows)
    lon = np.linspace(sc.SITE_LONGITUDE - span_deg / 2, sc.SITE_LONGITUDE + span_deg / 2, cols)
    return np.meshgrid(lon, lat)


def make_context(**overrides):
    """A minimal admissible context document, before overrides."""
    timing = sc.resolve_timing(nominal_scan_utc=NOMINAL, issue_time_utc=ISSUE)
    document = {
        "schema_version": sc.SCHEMA_VERSION,
        "mode": sc.MODE,
        "source": {"platform": "Himawari-9", "bands": ["B13", "B08"], "segments": [4]},
        "location": {"latitude": sc.SITE_LATITUDE, "longitude": sc.SITE_LONGITUDE, "roi_km": 25.0},
        "timing": timing,
        "features": {
            "B13": {"p50": 285.0, "mean": 284.2, "std": 3.1, "roi_pixels": 200, "valid_fraction": 1.0},
            "B08": {"p50": 240.0, "mean": 239.5, "std": 2.0, "roi_pixels": 200, "valid_fraction": 1.0},
        },
        "objects": [],
        "decoder": {"reader": "satpy.ahi_hsd", "satpy_version": "0.60.0"},
        "satellite_used_in_prediction": False,
        "claim_boundary": sc.CLAIM_BOUNDARY,
    }
    document.update(overrides)
    return document


def make_station_input(tmp: Path, issue: str = ISSUE) -> Path:
    path = tmp / "station_input.json"
    path.write_text(
        json.dumps(
            {
                "issue_time_utc": issue,
                "temperature_degc": 28.2,
                "humidity_percent": 86.3,
                "wind_mps": 3.4,
                "raw_rows_in_bin": 11,
                "aggregation": "median of raw rows in the right-labelled closed 5-minute bin",
                "source": {"path": "frozen.csv", "name": "frozen.csv", "sha256": "0" * 64},
            }
        ),
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# 1-3: timing admission
# ---------------------------------------------------------------------------


class TestTimingAdmission(unittest.TestCase):
    def test_frame_available_after_issue_is_rejected(self):
        """10 min completion + 20 min lag = 07:30 available, which is after 07:20."""
        with self.assertRaises(sc.SatelliteContextError) as caught:
            sc.resolve_timing(nominal_scan_utc=NOMINAL, issue_time_utc="2026-08-09T07:20:00Z")
        self.assertIn("not usable", str(caught.exception))

    def test_availability_equal_to_issue_is_rejected(self):
        """Equality must fail: the contract is strictly before, not at-or-before."""
        nominal = datetime(2026, 8, 9, 7, 0, tzinfo=timezone.utc)
        available = nominal + timedelta(
            minutes=sc.ASSUMED_SCAN_COMPLETION_MINUTES + sc.PUBLICATION_LAG_MINUTES
        )
        with self.assertRaises(sc.SatelliteContextError):
            sc.resolve_timing(nominal_scan_utc=nominal, issue_time_utc=available)

    def test_timezone_naive_timestamp_is_rejected(self):
        with self.assertRaises(sc.SatelliteContextError) as caught:
            sc.resolve_timing(nominal_scan_utc="2026-08-09T07:00:00", issue_time_utc=ISSUE)
        self.assertIn("timezone", str(caught.exception))

    def test_admitted_frame_reports_age_and_sensitivity_without_lead_time_claim(self):
        timing = sc.resolve_timing(nominal_scan_utc=NOMINAL, issue_time_utc=ISSUE)
        self.assertTrue(timing["available_strictly_before_issue"])
        self.assertAlmostEqual(timing["frame_age_minutes"], 35.0)
        self.assertEqual(
            [entry["publication_lag_minutes"] for entry in timing["publication_lag_sensitivity"]],
            list(sc.LAG_SENSITIVITY_MINUTES),
        )
        self.assertNotIn("lead_time", json.dumps(timing).lower())


# ---------------------------------------------------------------------------
# 4: ROI geometry
# ---------------------------------------------------------------------------


class TestRoi(unittest.TestCase):
    def test_roi_far_from_site_is_empty_and_rejected(self):
        lons, lats = synthetic_navigation()
        with self.assertRaises(sc.SatelliteContextError) as caught:
            sc.roi_mask(lats + 40.0, lons, radius_km=25.0)
        self.assertIn("empty", str(caught.exception))

    def test_roi_reaching_the_segment_edge_is_rejected_as_clipped(self):
        """This is the defect that invalidated the retracted 50 km statistic."""
        lons, lats = synthetic_navigation(span_deg=0.05)  # grid narrower than the disc
        with self.assertRaises(sc.SatelliteContextError) as caught:
            sc.roi_mask(lats, lons, radius_km=25.0)
        self.assertIn("clipped", str(caught.exception))

    def test_interior_roi_is_accepted_and_bounded_by_radius(self):
        lons, lats = synthetic_navigation()
        mask = sc.roi_mask(lats, lons, radius_km=25.0)
        self.assertGreater(int(mask.sum()), 0)
        distance = sc.great_circle_km(lats[mask], lons[mask], sc.SITE_LATITUDE, sc.SITE_LONGITUDE)
        self.assertLessEqual(float(distance.max()), 25.0 + 1e-9)


# ---------------------------------------------------------------------------
# 5: required bands
# ---------------------------------------------------------------------------


class TestRequiredBands(unittest.TestCase):
    def test_either_required_band_missing_fails_closed(self):
        """Losing B13 *or* B08 must abort; a partial context is never written."""
        for absent in ("B13", "B08"):
            with self.subTest(absent=absent):

                def fake_download(key, cache, _absent=absent, **kwargs):
                    if _absent in key:
                        raise sc.SatelliteContextError(f"simulated missing {_absent} object")
                    return {
                        "key": key,
                        "cached_path": str(cache / "present.bz2"),
                        "bytes": 1,
                        "sha256": "a" * 64,
                    }

                with tempfile.TemporaryDirectory() as tmp, unittest.mock.patch.object(
                    sc, "download_object", side_effect=fake_download
                ):
                    output = Path(tmp) / "satellite_context.json"
                    with self.assertRaises(sc.SatelliteContextError):
                        sc.build_context(
                            nominal_scan_utc=NOMINAL,
                            issue_time_utc=ISSUE,
                            cache=Path(tmp) / "cache",
                            workdir=Path(tmp) / "work",
                        )
                    self.assertFalse(output.exists(), "no context file may survive a failed band")

    def test_empty_band_list_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(sc.SatelliteContextError):
                sc.build_context(
                    nominal_scan_utc=NOMINAL,
                    issue_time_utc=ISSUE,
                    cache=Path(tmp),
                    workdir=Path(tmp),
                    bands=[],
                )

    def test_object_key_matches_the_published_hsd_layout(self):
        key = sc.object_key("B13", 4, datetime(2026, 8, 9, 7, 0, tzinfo=timezone.utc))
        self.assertEqual(
            key, "AHI-L1b-FLDK/2026/08/09/0700/HS_H09_20260809_0700_B13_FLDK_R20_S0410.DAT.bz2"
        )


# ---------------------------------------------------------------------------
# 6-7: statistics carry no classification
# ---------------------------------------------------------------------------


class TestStatistics(unittest.TestCase):
    def build_values(self):
        lons, lats = synthetic_navigation()
        mask = sc.roi_mask(lats, lons, radius_km=25.0)
        values = np.full(lats.shape, 285.0)
        # A physically impossible decode artefact plus a genuinely cold pixel.
        indices = np.argwhere(mask)
        values[tuple(indices[0])] = 120.0   # below QC floor -> removed as decode QC
        values[tuple(indices[1])] = 210.0   # cold but physical -> retained, unlabelled
        values[tuple(indices[2])] = np.nan
        return values, mask

    def test_physical_qc_removes_only_impossible_values_and_labels_nothing(self):
        values, mask = self.build_values()
        stats = sc.band_statistics(values, mask)
        self.assertEqual(stats["physical_qc_removed_pixels"], 1)
        self.assertEqual(stats["finite_pixels"], int(mask.sum()) - 1)
        # The 210 K pixel survives and is described only as a minimum, not a class.
        self.assertAlmostEqual(stats["min"], 210.0)
        # Scan keys and numeric fields only. ``physical_qc_note`` is excluded on
        # purpose: it is a *denial* ("not a cloud threshold"), and a naive
        # substring ban would flag the very sentence that prevents the overclaim.
        measured = {key: value for key, value in stats.items() if key != "physical_qc_note"}
        blob = json.dumps(measured).lower()
        for banned in ("cloud", "storm", "rain", "convective", "deep_pct", "alert", "warning"):
            self.assertNotIn(banned, blob, f"statistics must not classify: found {banned!r}")
        self.assertIn("not a cloud threshold", stats["physical_qc_note"])

    def test_statistics_expose_no_threshold_and_no_deep_pct(self):
        values, mask = self.build_values()
        stats = sc.band_statistics(values, mask)
        self.assertNotIn("deep_pct", stats)
        self.assertNotIn("220", json.dumps({k: v for k, v in stats.items() if k != "physical_qc_note"}))
        self.assertEqual(stats["physical_qc_range_kelvin"], [sc.QC_MIN_KELVIN, sc.QC_MAX_KELVIN])
        self.assertIn("not a cloud threshold", stats["physical_qc_note"])

    def test_all_pixels_outside_physical_range_fails_closed(self):
        lons, lats = synthetic_navigation()
        mask = sc.roi_mask(lats, lons, radius_km=25.0)
        with self.assertRaises(sc.SatelliteContextError):
            sc.band_statistics(np.full(lats.shape, 50.0), mask)

    def test_missing_prior_frames_produce_no_fabricated_deltas(self):
        current = {"B13": {"p50": 285.0}}
        self.assertEqual(sc.temporal_deltas(current, [], nominal_scan_utc=NOMINAL), {})

    def test_deltas_appear_only_for_priors_that_really_exist(self):
        prior30 = make_context()
        prior30["timing"]["nominal_scan_utc"] = "2026-08-09T06:30:00+00:00"
        prior30["features"] = {"B13": {"p50": 280.0}}
        deltas = sc.temporal_deltas(
            {"B13": {"p50": 285.0}}, [prior30], nominal_scan_utc="2026-08-09T07:00:00+00:00"
        )
        self.assertEqual(deltas["B13"]["delta30"], 5.0)
        self.assertNotIn("delta60", deltas["B13"])
        self.assertNotIn("trend90", deltas["B13"])


# ---------------------------------------------------------------------------
# 8-11: dashboard contract
# ---------------------------------------------------------------------------


class TestDashboard(unittest.TestCase):
    def state_with(self, tmp: Path, context: dict | None, **kwargs) -> dash.DashboardState:
        path = None
        if context is not None:
            path = tmp / "satellite_context.json"
            path.write_text(json.dumps(context), encoding="utf-8")
        return dash.DashboardState(
            satellite_context=path, station_input=make_station_input(tmp), **kwargs
        )

    def test_mode_says_not_fused_whenever_context_is_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            status = self.state_with(Path(tmp), make_context()).status()
            self.assertTrue(status["satellite_context_available"])
            self.assertIn("not fused", status["mode"])
            self.assertEqual(status["mode"], dash.MODE_FUSED_FREE)

    def test_satellite_is_never_used_in_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            for context in (make_context(), None):
                status = self.state_with(Path(tmp), context).status()
                self.assertFalse(status["satellite_used_in_prediction"])
                self.assertEqual(status["forecast_method"], "persistence")

    def test_forecast_is_identical_with_and_without_satellite_context(self):
        """The strongest available proof that nothing is fused."""
        with tempfile.TemporaryDirectory() as tmp:
            with_context = self.state_with(Path(tmp), make_context()).forecast()
            without = self.state_with(Path(tmp), None).forecast()
            self.assertEqual(with_context, without)
            self.assertEqual(
                with_context["predicted_at_plus30_degc"], with_context["current_at_degc"]
            )

    def test_missing_context_still_serves_persistence_and_says_why(self):
        with tempfile.TemporaryDirectory() as tmp:
            status = self.state_with(Path(tmp), None).status()
            self.assertEqual(status["mode"], dash.MODE_PERSISTENCE_ONLY)
            self.assertFalse(status["satellite_context_available"])
            self.assertIn("absent", status["degraded_reason"])
            self.assertIsInstance(status["forecast"]["predicted_at_plus30_degc"], float)

    def test_malformed_context_degrades_with_an_explicit_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "satellite_context.json"
            path.write_text("{not json", encoding="utf-8")
            state = dash.DashboardState(
                satellite_context=path, station_input=make_station_input(Path(tmp))
            )
            self.assertIsNone(state.context)
            self.assertIn("unreadable", state.status()["degraded_reason"])

    def test_wrong_schema_version_degrades(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = self.state_with(Path(tmp), make_context(schema_version="something-else"))
            self.assertIn("schema_version", state.status()["degraded_reason"])

    def test_stale_frame_is_refused_rather_than_shown(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = self.state_with(Path(tmp), make_context(), max_frame_age_minutes=10.0)
            self.assertIsNone(state.context)
            self.assertIn("stale", state.status()["degraded_reason"])

    def test_context_not_available_before_issue_is_refused(self):
        context = make_context()
        context["timing"]["effective_available_utc"] = "2026-08-09T08:00:00+00:00"
        with tempfile.TemporaryDirectory() as tmp:
            state = self.state_with(Path(tmp), context)
            self.assertIn("strictly before", state.status()["degraded_reason"])

    def test_context_carrying_a_forbidden_derived_field_is_refused(self):
        context = make_context()
        context["features"]["B13"]["deep_pct"] = 68.9
        with tempfile.TemporaryDirectory() as tmp:
            state = self.state_with(Path(tmp), context)
            self.assertIn("deep_pct", state.status()["degraded_reason"])

    def test_context_claiming_to_be_used_in_prediction_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = self.state_with(Path(tmp), make_context(satellite_used_in_prediction=True))
            self.assertIn("satellite_used_in_prediction", state.status()["degraded_reason"])

    def test_rendered_page_shows_not_fused_and_never_claims_lead_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = self.state_with(Path(tmp), make_context())
            page = dash.render_page(state.status()).replace(
                "__BANDS__", dash._band_rows(state.status(), state.context)
            )
            self.assertIn("NOT FUSED", page)
            self.assertIn(dash.DESIGN_BANNER, page)
            lowered = page.lower()

            # The page *must* say "not a warning lead time". A blunt substring ban
            # would fail on that denial, so ban the affirmative constructions only.
            self.assertIn("not a warning lead time", lowered)
            for banned in (
                "provides lead time",
                "minutes of warning",
                "minute warning",
                "early warning",
                "satellite ai",
                "fused inference",
                "predicts rain",
                "improves the forecast",
            ):
                self.assertNotIn(banned, lowered, f"page must not claim {banned!r}")

    def test_degraded_page_renders_the_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = self.state_with(Path(tmp), None)
            page = dash.render_page(state.status()).replace("__BANDS__", "")
            self.assertIn("degraded_reason", page)
            self.assertIn("persistence only", page)


class TestApparentTemperatureParity(unittest.TestCase):
    def test_stdlib_mirror_matches_the_repository_target_authority(self):
        """The board copy has no NumPy, so it uses a mirror; it must not drift."""
        from prototype.halo_safeshift.target import apparent_temperature_c

        for temp in (18.0, 25.5, 31.8, 39.0):
            for humidity in (35.0, 66.8, 92.0):
                for wind in (0.0, 2.3, 7.5):
                    self.assertEqual(
                        dash._apparent_temperature_stdlib(temp, humidity, wind),
                        apparent_temperature_c(temp, humidity, wind),
                        f"mirror diverged at Ta={temp} RH={humidity} ws={wind}",
                    )


class TestNoNetworkInUnitTests(unittest.TestCase):
    def test_urlopen_is_never_reached_by_the_tested_surface(self):
        """Guard the mock boundary: nothing above download_object may open a socket."""
        with unittest.mock.patch.object(sc.urllib.request, "urlopen") as opener:
            sc.resolve_timing(nominal_scan_utc=NOMINAL, issue_time_utc=ISSUE)
            sc.object_key("B13", 4, datetime(2026, 8, 9, 7, 0, tzinfo=timezone.utc))
            lons, lats = synthetic_navigation()
            sc.band_statistics(np.full(lats.shape, 285.0), sc.roi_mask(lats, lons))
            with tempfile.TemporaryDirectory() as tmp:
                dash.DashboardState(
                    satellite_context=None, station_input=make_station_input(Path(tmp))
                ).status()
        opener.assert_not_called()


if __name__ == "__main__":
    unittest.main()
