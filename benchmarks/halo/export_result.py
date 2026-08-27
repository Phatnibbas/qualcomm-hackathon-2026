"""Export bounded HALO probe artifacts without upgrading interpretation.

The satellite cache contains one HSD segment per timeline. Consumers therefore use
only the nominal 25 km mask, which fits inside segment 04 at the assumed coordinate;
larger masks are clipped by the segment boundary. Satellite samples are selected as
the latest cached observation strictly before the requested issue time. The artifact
records both the requested and actual offset instead of calling a nearby slot T-60/T+0.
"""

from __future__ import annotations

import json
import hashlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


HERE = Path(__file__).parent / "_cache"
EVIDENCE = Path(__file__).resolve().parents[2] / "evidence" / "halo-probe-2026-08-11"
ICT = "Asia/Ho_Chi_Minh"
WIND_EPOCH = "2026-07-21T15:42:00Z"
NOMINAL_RADIUS_KM = 25


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    head = completed.stdout.strip()
    if completed.returncode != 0 or not head:
        raise RuntimeError(f"cannot resolve git HEAD: {completed.stderr.strip()}")
    return head


def git_worktree_state() -> str:
    completed = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"cannot inspect git worktree: {completed.stderr.strip()}")
    return "dirty" if completed.stdout.strip() else "clean"


def latest_before(slots: list[pd.Timestamp], target: pd.Timestamp) -> pd.Timestamp | None:
    """Return the latest cached satellite slot strictly before issue time."""
    eligible = [slot for slot in slots if slot < target]
    return max(eligible) if eligible else None


def load_satellite_day(day: str) -> tuple[dict, dict[pd.Timestamp, str], dict]:
    path = EVIDENCE / f"sat_{day}.json"
    failure_path = EVIDENCE / f"sat_{day}_failures.json"
    failures = json.loads(failure_path.read_text(encoding="utf-8")) if failure_path.exists() else {}
    if not path.exists():
        return {}, {}, failures
    payload = json.loads(path.read_text(encoding="utf-8"))
    slots = {
        pd.Timestamp(
            f"{day[:4]}-{day[4:6]}-{day[6:8]}T{stamp[:2]}:{stamp[2:]}:00Z"
        ): stamp
        for stamp in payload
    }
    return payload, slots, failures


def build_lead_rows(table: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    cache: dict[str, tuple[dict, dict[pd.Timestamp, str], dict]] = {}
    labels = {-60: "m60", -30: "m30", 0: "m0"}

    for _, event in table.iterrows():
        peak = pd.Timestamp(event["peak"])
        day = peak.strftime("%Y%m%d")
        if day not in cache:
            cache[day] = load_satellite_day(day)
        satellite, slot_map, acquisition_failures = cache[day]
        available = sorted(slot_map)
        row = {
            "event_ict": peak.tz_convert(ICT).strftime("%Y-%m-%d %H:%M"),
            "event_peak_utc": peak.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "temp_drop_c": round(float(event["dT"]), 2),
            "metric_boundary": (
                "segment-04-only nominal 25 km mask; assumed coordinate; "
                "unvalidated decoder; 220 K threshold provenance unknown"
            ),
            "missing_reason": None if available else "no satellite cache for event UTC day",
        }
        for requested_offset, label in labels.items():
            target = peak + pd.Timedelta(minutes=requested_offset)
            slot = latest_before(available, target) if available else None
            row[f"requested_offset_{label}_min"] = requested_offset
            if slot is None:
                row[f"satellite_slot_{label}_utc"] = None
                row[f"actual_offset_{label}_min"] = None
                row[f"seg04_nominal25_below220k_{label}_pct"] = None
                row[f"missing_reason_{label}"] = (
                    (
                        "no satellite cache for event UTC day; "
                        f"{len(acquisition_failures)} acquisition failure(s) recorded"
                    )
                    if not available and acquisition_failures
                    else "no satellite cache for event UTC day"
                    if not available
                    else "no cached satellite slot strictly before requested issue time"
                )
                continue
            stamp = slot_map[slot]
            rings = satellite.get(stamp)
            ring = rings.get(str(NOMINAL_RADIUS_KM)) if isinstance(rings, dict) else None
            value = ring.get("deep_pct") if isinstance(ring, dict) else None
            row[f"satellite_slot_{label}_utc"] = slot.strftime("%Y-%m-%dT%H:%M:%SZ")
            row[f"actual_offset_{label}_min"] = round(
                (slot - peak).total_seconds() / 60, 1
            )
            row[f"seg04_nominal25_below220k_{label}_pct"] = (
                None if value is None else round(float(value), 1)
            )
            row[f"missing_reason_{label}"] = (
                None
                if value is not None
                else acquisition_failures.get(stamp, {}).get(
                    "message", "cached slot lacks nominal-25-km deep_pct"
                )
            )
        rows.append(row)
    return rows


def main() -> None:
    station = pd.read_pickle(HERE / "saigon.pkl")
    events = pd.read_pickle(HERE / "events.pkl")

    table = events.copy()
    table["peak_utc"] = table["peak"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    table["peak_ict"] = table["peak"].dt.tz_convert(ICT).dt.strftime("%Y-%m-%d %H:%M")
    table["hour_ict"] = table["peak"].dt.tz_convert(ICT).dt.hour
    out = table[
        [
            "peak_utc", "peak_ict", "hour_ict", "dur", "dT", "dRH", "dP",
            "gust", "lux_drop", "label_dependency_start", "label_dependency_end",
            "temp_interpolated_fraction", "rh_interpolated_fraction",
            "label_depends_on_interpolation", "peak_label_depends_on_interpolation",
        ]
    ].round(3)
    out.columns = [
        "peak_utc",
        "peak_ict",
        "hour_ict",
        "duration_min",
        "temp_drop_c",
        "rh_rise_pct",
        "press_rise_kpa",
        "gust_over_baseline_ms",
        "illuminance_drop_frac",
        "label_dependency_start_utc",
        "label_dependency_end_utc",
        "temp_interpolated_fraction",
        "rh_interpolated_fraction",
        "label_depends_on_interpolation",
        "peak_label_depends_on_interpolation",
    ]
    out.insert(0, "event", range(1, len(out) + 1))
    out.to_csv(EVIDENCE / "events.csv", index=False)

    lead = pd.DataFrame(build_lead_rows(table))
    lead.to_csv(EVIDENCE / "satellite_lead_time.csv", index=False)
    output_paths = [EVIDENCE / "events.csv", EVIDENCE / "satellite_lead_time.csv"]

    gaps = station["ts"].diff().dt.total_seconds().dropna()
    hours = table["peak"].dt.tz_convert(ICT).dt.hour
    count_strong = int(((table["dT"] <= -3.0) & (table["dRH"] >= 12.0)).sum())
    repo_root = Path(__file__).resolve().parents[2]
    generator_path = Path(__file__).resolve()
    input_paths = [
        HERE / "saigon.pkl",
        HERE / "events.pkl",
        EVIDENCE / "sat_20260809.json",
        EVIDENCE / "sat_20260810.json",
        EVIDENCE / "sat_20260809_failures.json",
        EVIDENCE / "sat_20260810_failures.json",
    ]
    missing_inputs = [path for path in input_paths if not path.is_file()]
    if missing_inputs:
        missing = ", ".join(str(path) for path in missing_inputs)
        raise FileNotFoundError(f"required probe input(s) missing: {missing}")
    producer_paths = [
        repo_root / "benchmarks/halo/pull_station_history.py",
        repo_root / "benchmarks/halo/detect_threshold_rule_occurrences.py",
        repo_root / "benchmarks/halo/fetch_satellite_day.py",
        repo_root / "benchmarks/halo/hsd.py",
        repo_root / "benchmarks/halo/himawari_geometry.py",
        generator_path,
    ]
    missing_producers = [path for path in producer_paths if not path.is_file()]
    if missing_producers:
        missing = ", ".join(str(path) for path in missing_producers)
        raise FileNotFoundError(f"required probe producer(s) missing: {missing}")

    summary = {
        "probe_date": "2026-08-11",
        "purpose": "Feasibility probe for HALO Phase 2. NOT a design decision.",
        "generator": "benchmarks/halo/export_result.py",
        "provenance": {
            "tag": "MEASURED — REFERENCE ONLY",
            "command": "py -3 benchmarks/halo/export_result.py",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_head_at_generation": git_head(),
            "git_worktree_state_at_generation": git_worktree_state(),
            "generator_sha256": sha256(generator_path),
            "python_version": sys.version.split()[0],
            "pandas_version": pd.__version__,
            "platform": platform.platform(),
            "environment": "Windows host; cached historical station and satellite inputs",
            "conditions": (
                "Artifact-only regeneration; no board or live sensor measurement. "
                "Original acquisition conditions remain incomplete."
            ),
            "input_sha256": {
                str(path.relative_to(repo_root)).replace("\\", "/"): sha256(path)
                for path in input_paths
            },
            "producer_sha256": {
                str(path.relative_to(repo_root)).replace("\\", "/"): sha256(path)
                for path in producer_paths
            },
            "output_sha256": {
                str(path.relative_to(repo_root)).replace("\\", "/"): sha256(path)
                for path in output_paths
            },
        },
        "station": {
            "channel": 3428136,
            "channel_name_observed": "MakerLab Giadinh Station",
            "rows": int(len(station)),
            "first_utc": station["ts"].iloc[0].isoformat(),
            "last_utc": station["ts"].iloc[-1].isoformat(),
            "span_days": round(
                (station["ts"].iloc[-1] - station["ts"].iloc[0]).total_seconds() / 86400,
                2,
            ),
            "cadence_median_s": float(gaps.median()),
            "cadence_p95_s": float(gaps.quantile(0.95)),
            "gaps_over_60s": int((gaps > 60).sum()),
            "gaps_over_900s": int((gaps > 900).sum()),
            "wind_usable_from_utc": WIND_EPOCH,
            "wind_usable_note": "Before this instant field1 used a /100 divisor and reads 10x low.",
            "pressure_quantisation_kpa": 0.1,
            "pressure_note": "Observed 0.1 kPa = 1 hPa steps; predictive value is not established.",
        },
        "events": {
            "criterion": (
                "exploratory chosen threshold: temperature falls >= 2.0 C AND humidity "
                "rises >= 8.0 %RH within a 20-minute window, on a 1-minute interpolated grid"
            ),
            "threshold_provenance": "UNKNOWN; no preregistration record located",
            "interpolation": "1-minute resample; interpolation bridges at most 10 consecutive minutes",
            "interpolation_dependency_columns": [
                "label_dependency_start_utc",
                "label_dependency_end_utc",
                "temp_interpolated_fraction",
                "rh_interpolated_fraction",
                "label_depends_on_interpolation",
                "peak_label_depends_on_interpolation",
            ],
            "count_core": int(len(table)),
            "count_strong": count_strong,
            "strong_criterion": "temperature falls >= 3.0 C AND humidity rises >= 12.0 %RH",
            "observation_window_span_days": round(
                (station["ts"].iloc[-1] - station["ts"].iloc[0]).total_seconds() / 86400,
                2,
            ),
            "hour_histogram_ict": {str(hour): int((hours == hour).sum()) for hour in range(24)},
            "events_0000_to_0659_ict": int(((hours >= 0) & (hours < 7)).sum()),
            "events_1300_to_1759_ict": int(((hours >= 13) & (hours < 18)).sum()),
            "hour_distribution_description": (
                "Descriptive rule output only; it does not validate a physical mechanism."
            ),
        },
        "satellite": {
            "source": "NOAA Open Data on AWS, bucket noaa-himawari9, AHI-L1b-FLDK",
            "archive_confirmed_from": "2026-07-15",
            "station_coordinate_status": "ASSUMED: 10.80 N, 106.70 E; channel metadata is 0.0/0.0",
            "station_pixel_2km_grid_derived": {"line": 2178.6, "column": 1059.7},
            "segment_containing_assumed_station": 4,
            "segment_coverage_boundary": (
                "Only segment 04 was downloaded. Masks >=50 km are clipped southward; "
                "accepted consumers use only the nominal 25 km segment-04 mask."
            ),
            "object_size_mib_reference": {"B13_S04": 2.81, "B08_S04": 1.19, "B03_S04": 33.69},
            "l2_clouds_size_mib_reference": {"CHGT": 749, "CMSK": 331, "CPHS": 82},
            "size_boundary": "Observed object sizes only; design feasibility remains undecided.",
            "reader": "benchmarks/halo/hsd.py; not validated against a reference reader",
            "internal_sanity_check_only": (
                "B13 segment-04 output median 269 K, max 298 K; this is not decoder validation."
            ),
            "lead_time_artifact": "evidence/halo-probe-2026-08-11/satellite_lead_time.csv",
            "lead_time_alignment": (
                "latest cached slot strictly before each requested issue time; actual slots "
                "and offsets are stored, so requested offsets are not presented as exact"
            ),
            "lead_rows_total": int(len(lead)),
            "lead_rows_with_satellite_cache": int(lead["missing_reason"].isna().sum()),
        },
        "open_questions": [
            "Physical cause and independent label for the 28 rule occurrences.",
            "Provenance and validity of the 220 K threshold.",
            "Reference-reader validation of hsd.py.",
            "True station coordinate, projection, parallax and full-ROI segment stitching.",
            "Satellite product selection and observation-to-download latency.",
            "Prediction target, horizon, model family and evaluation protocol.",
        ],
        "decided_by_user_2026_08_11": [
            "Use the weather station.",
            "Use the Arduino UNO Q.",
            "Use satellite data.",
        ],
        "explicitly_not_decided": [
            "Which satellite and which data product.",
            "The detailed problem formulation.",
            "Model family, feature set, and label definition.",
            "Whether Edge Impulse is in the pipeline.",
        ],
        "evidence_boundaries": {
            "mechanism": "UNKNOWN; canonical name is occurrences of our temperature-fall/humidity-rise threshold rule.",
            "hour_histogram": "Descriptive only; not mechanism validation.",
            "geometry": "DERIVED from an ASSUMED coordinate.",
            "provenance": "REFERENCE ONLY until V-1..V-6 rerun with conditions recorded.",
        },
    }

    result_path = EVIDENCE / "result.json"
    result_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest_paths = output_paths + [result_path]
    artifact_manifest = {
        "generated_by": "benchmarks/halo/export_result.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_sha256": {
            str(path.relative_to(repo_root)).replace("\\", "/"): sha256(path)
            for path in manifest_paths
        },
    }
    (EVIDENCE / "artifact_manifest.json").write_text(
        json.dumps(artifact_manifest, indent=2), encoding="utf-8"
    )
    print(f"events.csv               {len(out)} rows")
    print(f"satellite_lead_time.csv  {len(lead)} rows")
    print("result.json              written")
    print()
    print(lead.to_string(index=False))


if __name__ == "__main__":
    main()
