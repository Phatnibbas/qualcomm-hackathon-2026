"""Leakage-safe window construction.

For an issue time ``t``:

* features come from the twelve right-labelled five-minute bins covering
  ``(t - 60 min, t]`` — that is the bins labelled ``t-55 ... t``;
* the target is the apparent temperature in the right-labelled bin labelled
  ``t + 30``, covering ``(t + 25 min, t + 30 min]``.

The two intervals are disjoint, so no future value can reach a feature. A
window is only formed when every bin it needs actually exists and is valid,
which means an outage produces no sample rather than a bridged one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

from . import load_config

__all__ = [
    "WindowSet",
    "FeatureOrderError",
    "resolve_local_timezone",
    "build_feature_schema",
    "per_bin_values",
    "build_windows",
]


KNOWN_PER_BIN_FEATURES = frozenset(
    {
        "temperature",
        "humidity",
        "wind_speed",
        "wind_direction_sin",
        "wind_direction_cos",
        "light_log1p",
        "pressure",
        "pm25_log1p",
        "noise",
        "raw_count_normalised",
    }
)

KNOWN_ISSUE_TIME_FEATURES = frozenset(
    {
        "apparent_temperature_now",
        "local_time_of_day_sin",
        "local_time_of_day_cos",
        "day_of_year_sin",
        "day_of_year_cos",
        "issue_bin_staleness_normalised",
        "window_min_raw_count_normalised",
        "window_all_required_valid",
    }
)


@dataclass
class WindowSet:
    """Feature matrix and its aligned time bookkeeping."""

    x: np.ndarray
    y: np.ndarray
    issue_times: np.ndarray
    target_times: np.ndarray
    window_start_times: np.ndarray
    at_now: np.ndarray
    schema: dict[str, Any]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.x.shape[0])


def resolve_local_timezone(config: dict[str, Any] | None = None) -> tuple[Any, str]:
    """Return ``(tzinfo, source)`` for the configured local timezone.

    Prefers the IANA database; falls back to the configured fixed offset, which
    is exact for Vietnam over the data range (no DST since 1975). The source is
    recorded in the feature schema so a later runtime can tell which was used.
    """
    resolved = config or load_config()
    settings = resolved["timezone"]
    name = settings["local_timezone"]
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(name), "zoneinfo"
    except Exception:  # noqa: BLE001 - missing tz database is an environment fact
        offset = timedelta(hours=float(settings["local_utc_offset_hours"]))
        return timezone(offset), "fixed_offset"


def build_feature_schema(
    field_mapping: dict[str, str],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Deterministic, frozen feature order and its provenance."""
    resolved = config or load_config()
    windows = resolved["windows"]
    features = resolved["features"]
    lookback_bins = int(windows["lookback_bins"])
    bin_minutes = int(resolved["resample"]["bin_minutes"])
    _, tz_source = resolve_local_timezone(resolved)

    per_bin_order = list(features["per_bin_order"])
    issue_time_order = list(features["issue_time_order"])

    # The schema and the emitted vector must be generated from the same source
    # of truth. Validating the configured names here means an unknown name is
    # rejected when the schema is built, not discovered as a length mismatch
    # after a run has already produced a matrix.
    unknown_per_bin = [name for name in per_bin_order if name not in KNOWN_PER_BIN_FEATURES]
    unknown_issue = [name for name in issue_time_order if name not in KNOWN_ISSUE_TIME_FEATURES]
    if unknown_per_bin or unknown_issue:
        raise FeatureOrderError(
            f"configured feature order names unknown features: "
            f"per_bin={unknown_per_bin}, issue_time={unknown_issue}. "
            f"Known per-bin: {sorted(KNOWN_PER_BIN_FEATURES)}. "
            f"Known issue-time: {sorted(KNOWN_ISSUE_TIME_FEATURES)}."
        )
    if len(set(per_bin_order)) != len(per_bin_order) or len(set(issue_time_order)) != len(
        issue_time_order
    ):
        raise FeatureOrderError("configured feature order contains a duplicate name")

    names: list[str] = []
    for step in range(lookback_bins):
        offset_minutes = (lookback_bins - 1 - step) * bin_minutes
        suffix = f"t_minus_{offset_minutes:03d}min"
        for base in per_bin_order:
            names.append(f"{suffix}__{base}")
    names.extend(issue_time_order)

    return {
        "schema_version": "halo-safeshift-feature-schema-v1",
        "config_id": resolved["config_id"],
        "n_features": len(names),
        "feature_names": names,
        "per_bin_order": list(features["per_bin_order"]),
        "issue_time_order": list(features["issue_time_order"]),
        "lookback_bins": lookback_bins,
        "bin_minutes": bin_minutes,
        "lookback_minutes": int(windows["lookback_minutes"]),
        "horizon_minutes": int(windows["horizon_minutes"]),
        "input_interval": windows["input_interval"],
        "target_interval": windows["target_interval"],
        "field_mapping": dict(field_mapping),
        "timezone": {
            "name": resolved["timezone"]["local_timezone"],
            "resolution_source": tz_source,
            "fixed_offset_hours": resolved["timezone"]["local_utc_offset_hours"],
        },
        "eligibility": dict(features["eligibility"]),
        "no_leakage_rule": features["no_leakage_rule"],
        "target": {
            "name": "station-derived shade apparent-temperature estimate",
            "unit": "degC",
            "bin_label_offset_minutes": int(windows["horizon_minutes"]),
        },
        "index": {
            "apparent_temperature_now": names.index("apparent_temperature_now"),
        },
    }


class FeatureOrderError(ValueError):
    """Raised when the configured feature order names something unbuildable."""


def per_bin_values(row: pd.Series, expected_count: float) -> dict[str, float]:
    """Every per-bin feature this pipeline knows how to compute, keyed by name.

    The emitted *vector* is this map read in the configured order — the order is
    never restated in Python. Duplicating it here and in the config was how the
    schema and the vector could drift apart while both looked correct.
    """
    direction_rad = math.radians(float(row["wind_direction"]))
    return {
        "temperature": float(row["temperature"]),
        "humidity": float(row["humidity"]),
        "wind_speed": float(row["wind_speed"]),
        "wind_direction_sin": math.sin(direction_rad),
        "wind_direction_cos": math.cos(direction_rad),
        "light_log1p": math.log1p(max(float(row["light"]), 0.0)),
        "pressure": float(row["pressure"]),
        "pm25_log1p": math.log1p(max(float(row["pm25"]), 0.0)),
        "noise": float(row["noise"]),
        "raw_count_normalised": float(row["n_raw"]) / expected_count,
    }


def _ordered(values: dict[str, float], order: list[str], scope: str) -> list[float]:
    """Read ``values`` in ``order``; refuse a name with no computable value."""
    unknown = [name for name in order if name not in values]
    if unknown:
        raise FeatureOrderError(
            f"configured {scope} feature order names {unknown}, which this "
            f"pipeline does not know how to compute. Known names: "
            f"{sorted(values)}. Refusing to emit a vector that does not match "
            f"its own schema."
        )
    return [values[name] for name in order]


def build_windows(
    binned: pd.DataFrame,
    field_mapping: dict[str, str],
    config: dict[str, Any] | None = None,
) -> WindowSet:
    """Construct every eligible ``(features, target)`` pair from binned data."""
    resolved = config or load_config()
    windows_cfg = resolved["windows"]
    sampling = resolved["resample"]
    lookback_bins = int(windows_cfg["lookback_bins"])
    bin_minutes = int(sampling["bin_minutes"])
    horizon = timedelta(minutes=int(windows_cfg["horizon_minutes"]))
    lookback = timedelta(minutes=int(windows_cfg["lookback_minutes"]))
    expected_count = float(sampling["expected_raw_observations_per_bin"])
    per_bin_order = list(resolved["features"]["per_bin_order"])
    issue_time_order = list(resolved["features"]["issue_time_order"])
    schema = build_feature_schema(field_mapping, resolved)
    tzinfo, _ = resolve_local_timezone(resolved)

    optional = [
        name for name in field_mapping.values()
        if name not in set(resolved["field_semantics"]["required_semantics"])
    ]

    diagnostics: dict[str, Any] = {
        "candidate_issue_bins": 0,
        "rejected_input_bin_missing": 0,
        "rejected_input_bin_invalid": 0,
        "rejected_input_bin_not_feature_complete": 0,
        "rejected_target_bin_missing": 0,
        "rejected_target_bin_invalid": 0,
        "accepted_samples": 0,
    }

    empty = WindowSet(
        x=np.zeros((0, schema["n_features"]), dtype=np.float64),
        y=np.zeros((0,), dtype=np.float64),
        issue_times=np.array([], dtype="datetime64[ns]"),
        target_times=np.array([], dtype="datetime64[ns]"),
        window_start_times=np.array([], dtype="datetime64[ns]"),
        at_now=np.zeros((0,), dtype=np.float64),
        schema=schema,
        diagnostics=diagnostics,
    )
    if binned is None or len(binned) == 0:
        return empty

    frame = binned.copy()
    frame["bin_end_utc"] = pd.to_datetime(frame["bin_end_utc"], utc=True)
    frame = frame.sort_values("bin_end_utc").reset_index(drop=True)
    frame["feature_complete"] = (
        frame[optional].notna().all(axis=1) if optional else True
    )

    by_time: dict[pd.Timestamp, pd.Series] = {
        row["bin_end_utc"]: row for _, row in frame.iterrows()
    }

    rows: list[list[float]] = []
    targets: list[float] = []
    issue_times: list[pd.Timestamp] = []
    target_times: list[pd.Timestamp] = []
    window_starts: list[pd.Timestamp] = []
    at_now_values: list[float] = []

    for issue_time in frame["bin_end_utc"]:
        diagnostics["candidate_issue_bins"] += 1

        needed = [
            issue_time - timedelta(minutes=bin_minutes * (lookback_bins - 1 - step))
            for step in range(lookback_bins)
        ]
        input_rows: list[pd.Series] = []
        failure: str | None = None
        for stamp in needed:
            row = by_time.get(stamp)
            if row is None:
                failure = "rejected_input_bin_missing"
                break
            if not bool(row["valid"]):
                failure = "rejected_input_bin_invalid"
                break
            if not bool(row["feature_complete"]):
                failure = "rejected_input_bin_not_feature_complete"
                break
            input_rows.append(row)
        if failure is not None:
            diagnostics[failure] += 1
            continue

        target_row = by_time.get(issue_time + horizon)
        if target_row is None:
            diagnostics["rejected_target_bin_missing"] += 1
            continue
        if not bool(target_row["valid"]):
            diagnostics["rejected_target_bin_invalid"] += 1
            continue

        vector: list[float] = []
        for row in input_rows:
            vector.extend(
                _ordered(per_bin_values(row, expected_count), per_bin_order, "per_bin")
            )

        issue_row = input_rows[-1]
        local = issue_time.tz_convert(tzinfo)
        seconds_of_day = local.hour * 3600 + local.minute * 60 + local.second
        tod_angle = 2.0 * math.pi * seconds_of_day / 86400.0
        days_in_year = 366.0 if calendar_is_leap(local.year) else 365.0
        doy_angle = 2.0 * math.pi * (local.timetuple().tm_yday - 1) / days_in_year
        staleness = float(issue_row["staleness_seconds"]) / (bin_minutes * 60.0)
        min_raw_norm = min(float(row["n_raw"]) / expected_count for row in input_rows)

        issue_values = {
            "apparent_temperature_now": float(issue_row["apparent_temperature_c"]),
            "local_time_of_day_sin": math.sin(tod_angle),
            "local_time_of_day_cos": math.cos(tod_angle),
            "day_of_year_sin": math.sin(doy_angle),
            "day_of_year_cos": math.cos(doy_angle),
            "issue_bin_staleness_normalised": staleness,
            "window_min_raw_count_normalised": min_raw_norm,
            "window_all_required_valid": 1.0,
        }
        vector.extend(_ordered(issue_values, issue_time_order, "issue_time"))

        if len(vector) != schema["n_features"]:  # pragma: no cover - structural guard
            raise AssertionError(
                f"feature vector length {len(vector)} != schema {schema['n_features']}"
            )
        if not all(math.isfinite(value) for value in vector):  # pragma: no cover
            raise AssertionError(f"non-finite feature vector at issue time {issue_time}")

        rows.append(vector)
        targets.append(float(target_row["apparent_temperature_c"]))
        issue_times.append(issue_time)
        target_times.append(issue_time + horizon)
        window_starts.append(issue_time - lookback)
        at_now_values.append(float(issue_row["apparent_temperature_c"]))
        diagnostics["accepted_samples"] += 1

    if not rows:
        return empty

    return WindowSet(
        x=np.asarray(rows, dtype=np.float64),
        y=np.asarray(targets, dtype=np.float64),
        issue_times=_utc_naive_array(issue_times),
        target_times=_utc_naive_array(target_times),
        window_start_times=_utc_naive_array(window_starts),
        at_now=np.asarray(at_now_values, dtype=np.float64),
        schema=schema,
        diagnostics=diagnostics,
    )


def _utc_naive_array(stamps: list[pd.Timestamp]) -> np.ndarray:
    """UTC instants as naive ``datetime64[ns]``.

    Time arrays are stored tz-naive-in-UTC on purpose: a tz-aware
    ``DatetimeIndex.to_numpy()`` yields an object array of Timestamps, which
    silently degrades every downstream comparison.
    """
    return pd.DatetimeIndex(stamps).tz_convert(timezone.utc).tz_localize(None).to_numpy()


def calendar_is_leap(year: int) -> bool:
    return (year % 4 == 0 and year % 100 != 0) or year % 400 == 0


def utc_datetime(value: Any) -> datetime:
    return pd.Timestamp(value).tz_convert(timezone.utc).to_pydatetime()
