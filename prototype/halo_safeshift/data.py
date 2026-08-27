"""Metadata-aware field mapping, QC and five-minute resampling.

Contract, all of it loaded from ``config/experiment.v1.json``:

* semantic channels are resolved from the *frozen channel metadata*, never from
  positional memory;
* rows before the wind-scale fix epoch are discarded;
* a raw row is admitted only when every required channel is finite and inside
  its physical range;
* admitted rows are aggregated into right-labelled five-minute bins by median;
* a bin is valid only when every required channel carries at least the
  configured minimum number of raw observations;
* required channels are never interpolated, so an outage yields no bin and
  therefore no sample.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import load_config
from .target import apparent_temperature_array

__all__ = [
    "FieldMappingError",
    "resolve_field_mapping",
    "load_raw_csv",
    "apply_row_qc",
    "resample_five_minute",
    "raw_feed_timeline",
    "derive_recovery_partition",
    "build_recovery_report",
    "build_qc_report",
]


class FieldMappingError(ValueError):
    """Raised when channel metadata cannot be mapped onto the semantic contract."""


# --------------------------------------------------------------------------- #
# Field mapping
# --------------------------------------------------------------------------- #

def resolve_field_mapping(
    channel_metadata: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Map ``fieldN`` -> semantic name using the channel metadata labels.

    Returns e.g. ``{"field3": "temperature", ...}``. Raises when a label maps to
    two different fields, when a label has no semantic mapping, or when either
    requirement level is unsatisfied:

    * ``required_semantics`` — needed for the target and for bin validity;
    * ``feature_required_semantics`` — needed to *construct* the P0 feature
      schema at all. A channel the metadata never declares cannot be filled by
      any row, so its absence fails here, during metadata resolution, rather
      than surfacing later as a feature-vector length mismatch.
    """
    resolved = config or load_config()
    semantics = resolved["field_semantics"]
    label_to_semantic: dict[str, str] = semantics["label_to_semantic"]
    required: list[str] = list(semantics["required_semantics"])
    feature_required: list[str] = list(semantics["feature_required_semantics"])

    if not isinstance(channel_metadata, dict):
        raise FieldMappingError("channel metadata must be an object")

    mapping: dict[str, str] = {}
    unknown_labels: dict[str, str] = {}
    for index in range(1, 9):
        key = f"field{index}"
        label = channel_metadata.get(key)
        if label is None:
            continue
        label = str(label).strip()
        if not label:
            continue
        semantic = label_to_semantic.get(label)
        if semantic is None:
            unknown_labels[key] = label
            continue
        if semantic in mapping.values():
            duplicate = [k for k, v in mapping.items() if v == semantic][0]
            raise FieldMappingError(
                f"semantic channel {semantic!r} is claimed by both {duplicate} and {key}"
            )
        mapping[key] = semantic

    observed = {f"field{i}": channel_metadata.get(f"field{i}") for i in range(1, 9)}

    missing = [name for name in required if name not in mapping.values()]
    if missing:
        raise FieldMappingError(
            "channel metadata does not supply required semantic channel(s) "
            f"{missing}; observed labels {observed}"
        )
    missing_features = [name for name in feature_required if name not in mapping.values()]
    if missing_features:
        raise FieldMappingError(
            "channel metadata does not supply feature-required semantic channel(s) "
            f"{missing_features}. The P0 feature schema emits one column per "
            "semantic per bin, so a channel the metadata never declares cannot be "
            "constructed at all. This is a contract violation, not a missing "
            f"value, and it fails here rather than during feature-vector "
            f"construction. Observed labels {observed}"
        )
    if unknown_labels:
        raise FieldMappingError(
            "channel metadata contains labels with no semantic mapping in the "
            f"configuration: {unknown_labels}. Refusing to guess."
        )
    return mapping


# --------------------------------------------------------------------------- #
# Raw loading
# --------------------------------------------------------------------------- #

def load_raw_csv(csv_path: Path | str) -> pd.DataFrame:
    """Load the frozen station CSV without renaming or reordering raw columns."""
    frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False, na_values=[""])
    if "created_at" not in frame.columns or "entry_id" not in frame.columns:
        raise ValueError(f"{csv_path}: expected entry_id and created_at columns")
    frame["created_at_utc"] = pd.to_datetime(
        frame["created_at"], utc=True, format="ISO8601", errors="coerce"
    )
    frame["entry_id_int"] = pd.to_numeric(frame["entry_id"], errors="coerce").astype("Int64")
    return frame


# --------------------------------------------------------------------------- #
# Row-level QC
# --------------------------------------------------------------------------- #

def apply_row_qc(
    raw: pd.DataFrame,
    field_mapping: dict[str, str],
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return admitted rows with semantic float columns, plus exclusion counts.

    A row is rejected outright when any *required* channel is unusable; an
    unusable *optional* channel becomes ``nan`` for that channel only.
    """
    resolved = config or load_config()
    qc = resolved["qc"]
    semantics = resolved["field_semantics"]
    required: list[str] = list(semantics["required_semantics"])
    epoch = pd.Timestamp(resolved["usable_epoch"]["wind_scale_fix_utc"])

    counts: dict[str, int] = {
        "rows_input": int(len(raw)),
        "rejected_unparsable_timestamp": 0,
        "rejected_before_wind_scale_fix_epoch": 0,
        "rejected_duplicate_entry_id": 0,
        "rejected_required_non_numeric_or_non_finite": 0,
        "rejected_temperature_le_threshold": 0,
        "rejected_humidity_le_threshold": 0,
        "rejected_humidity_gt_threshold": 0,
        "rejected_wind_speed_lt_threshold": 0,
        "nulled_optional_non_numeric_or_non_finite": 0,
        "nulled_pressure_le_threshold": 0,
        "nulled_wind_direction_out_of_range": 0,
        "rows_admitted": 0,
    }

    frame = raw.copy()

    bad_time = frame["created_at_utc"].isna()
    counts["rejected_unparsable_timestamp"] = int(bad_time.sum())
    frame = frame[~bad_time]

    before_epoch = frame["created_at_utc"] < epoch
    counts["rejected_before_wind_scale_fix_epoch"] = int(before_epoch.sum())
    frame = frame[~before_epoch]

    duplicated = frame["entry_id_int"].duplicated(keep="first")
    counts["rejected_duplicate_entry_id"] = int(duplicated.sum())
    frame = frame[~duplicated]

    for field_key, semantic in field_mapping.items():
        frame[semantic] = pd.to_numeric(frame.get(field_key), errors="coerce")
        frame.loc[~np.isfinite(frame[semantic].to_numpy(dtype="float64")), semantic] = np.nan

    optional = [name for name in field_mapping.values() if name not in required]

    # Optional-channel physical rules: null the channel, keep the row.
    if "pressure" in optional:
        bad = frame["pressure"].notna() & (frame["pressure"] <= float(qc["reject_pressure_le_kpa"]))
        counts["nulled_pressure_le_threshold"] = int(bad.sum())
        frame.loc[bad, "pressure"] = np.nan
    if "wind_direction" in optional:
        low, high = (float(v) for v in qc["wind_direction_valid_range_deg"])
        bad = frame["wind_direction"].notna() & (
            (frame["wind_direction"] < low) | (frame["wind_direction"] > high)
        )
        counts["nulled_wind_direction_out_of_range"] = int(bad.sum())
        frame.loc[bad, "wind_direction"] = np.nan
    counts["nulled_optional_non_numeric_or_non_finite"] = int(
        sum(int(frame[name].isna().sum()) for name in optional)
        - counts["nulled_pressure_le_threshold"]
        - counts["nulled_wind_direction_out_of_range"]
    )

    # Required-channel rules: reject the whole row.
    non_finite_required = frame[required].isna().any(axis=1)
    counts["rejected_required_non_numeric_or_non_finite"] = int(non_finite_required.sum())
    frame = frame[~non_finite_required]

    bad_temp = frame["temperature"] <= float(qc["reject_temperature_le_c"])
    counts["rejected_temperature_le_threshold"] = int(bad_temp.sum())
    frame = frame[~bad_temp]

    bad_rh_low = frame["humidity"] <= float(qc["reject_humidity_le_percent"])
    counts["rejected_humidity_le_threshold"] = int(bad_rh_low.sum())
    frame = frame[~bad_rh_low]

    bad_rh_high = frame["humidity"] > float(qc["reject_humidity_gt_percent"])
    counts["rejected_humidity_gt_threshold"] = int(bad_rh_high.sum())
    frame = frame[~bad_rh_high]

    bad_wind = frame["wind_speed"] < float(qc["reject_wind_speed_lt_mps"])
    counts["rejected_wind_speed_lt_threshold"] = int(bad_wind.sum())
    frame = frame[~bad_wind]

    frame = frame.sort_values("created_at_utc").reset_index(drop=True)
    frame["apparent_temperature_c"] = apparent_temperature_array(
        frame["temperature"].to_numpy(dtype="float64"),
        frame["humidity"].to_numpy(dtype="float64"),
        frame["wind_speed"].to_numpy(dtype="float64"),
        config=resolved,
    )
    still_bad = frame["apparent_temperature_c"].isna()
    if int(still_bad.sum()):
        counts["rejected_required_non_numeric_or_non_finite"] += int(still_bad.sum())
        frame = frame[~still_bad].reset_index(drop=True)

    counts["rows_admitted"] = int(len(frame))
    return frame, counts


# --------------------------------------------------------------------------- #
# Five-minute resampling
# --------------------------------------------------------------------------- #

def resample_five_minute(
    admitted: pd.DataFrame,
    field_mapping: dict[str, str],
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Aggregate admitted rows into right-labelled five-minute median bins.

    No interpolation and no bin creation across an outage: only bins that
    actually contain admitted observations appear in the result.
    """
    resolved = config or load_config()
    sampling = resolved["resample"]
    semantics = resolved["field_semantics"]
    required: list[str] = list(semantics["required_semantics"])
    minimum = int(sampling["min_raw_observations_per_bin"])
    expected = int(sampling["expected_raw_observations_per_bin"])
    rule = f"{int(sampling['bin_minutes'])}min"
    channels = list(field_mapping.values())

    columns = [
        "bin_end_utc",
        "n_raw",
        *channels,
        "apparent_temperature_c",
        *[f"n_valid_{name}" for name in channels],
        "last_obs_utc",
        "staleness_seconds",
        "valid",
        "invalid_reason",
    ]
    if admitted.empty:
        empty = pd.DataFrame(columns=columns)
        empty["bin_end_utc"] = pd.to_datetime(empty["bin_end_utc"], utc=True)
        return empty

    frame = admitted.set_index("created_at_utc").sort_index()
    frame["last_obs_utc"] = frame.index
    grouper = frame.resample(rule, label=sampling["label"], closed=sampling["closed"])

    medians = grouper[[*channels, "apparent_temperature_c"]].median()
    valid_counts = grouper[channels].count()
    valid_counts.columns = [f"n_valid_{name}" for name in channels]
    n_raw = grouper.size().rename("n_raw")
    last_obs = grouper["last_obs_utc"].max()

    binned = pd.concat([n_raw, medians, valid_counts, last_obs], axis=1)
    binned = binned[binned["n_raw"] > 0].copy()
    binned.index.name = "bin_end_utc"
    binned = binned.reset_index()

    binned["staleness_seconds"] = (
        binned["bin_end_utc"] - binned["last_obs_utc"]
    ).dt.total_seconds()

    reasons: list[str] = []
    valid_flags: list[bool] = []
    for _, row in binned.iterrows():
        problems = [
            f"{name}_raw_count_{int(row[f'n_valid_{name}'])}_lt_{minimum}"
            for name in required
            if int(row[f"n_valid_{name}"]) < minimum
        ]
        if not math.isfinite(float(row["apparent_temperature_c"])):
            problems.append("apparent_temperature_non_finite")
        valid_flags.append(not problems)
        reasons.append(";".join(problems))
    binned["valid"] = valid_flags
    binned["invalid_reason"] = reasons
    binned["expected_raw_observations_per_bin"] = expected

    return binned[[*columns, "expected_raw_observations_per_bin"]]


# --------------------------------------------------------------------------- #
# Outage / recovery partition
# --------------------------------------------------------------------------- #

def raw_feed_timeline(
    raw: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> tuple[pd.Series, dict[str, int]]:
    """Feed instants after timestamp parsing and duplicate handling, nothing else.

    This is deliberately upstream of every QC rule. An outage means the station
    stopped *posting*; a row whose sensor values fail validation still proves it
    was alive and posting. Deriving the boundary from admitted rows would let a
    run of bad readings manufacture an outage that never happened, so the only
    operations permitted here are:

    * parse ``created_at`` to UTC;
    * drop rows whose timestamp will not parse;
    * drop duplicate ``entry_id`` values, keeping the first in ascending
      ``entry_id`` order.

    No wind-epoch filter, no required-channel check, no physical-range check and
    no sensor-field validation runs before this.
    """
    _ = config  # the rule takes no configured value; the signature stays uniform
    counts = {
        "rows_input": int(len(raw)),
        "dropped_unparsable_timestamp": 0,
        "dropped_duplicate_entry_id": 0,
        "rows_in_timeline": 0,
    }
    if len(raw) == 0:
        return pd.Series([], dtype="datetime64[ns, UTC]"), counts

    frame = raw[["entry_id_int", "created_at_utc"]].copy()

    unparsable = frame["created_at_utc"].isna()
    counts["dropped_unparsable_timestamp"] = int(unparsable.sum())
    frame = frame[~unparsable]

    frame = frame.sort_values("entry_id_int", kind="mergesort")
    duplicated = frame["entry_id_int"].duplicated(keep="first")
    counts["dropped_duplicate_entry_id"] = int(duplicated.sum())
    frame = frame[~duplicated]

    times = frame["created_at_utc"].sort_values().reset_index(drop=True)
    counts["rows_in_timeline"] = int(len(times))
    return times, counts


def build_recovery_report(
    raw: pd.DataFrame,
    admitted_times: pd.Series | list[pd.Timestamp],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The recovery boundary, plus the admitted-data continuity beside it.

    ``raw_feed_recovery_boundary`` is the split point and comes from the raw
    feed. ``qc_admitted_continuity`` describes the same record after QC and is
    reported for comparison only — it never decides the split. The two are
    separate quantities and are not interchangeable even when they agree.
    """
    resolved = config or load_config()
    settings = resolved["recovery_partition"]

    feed_times, feed_counts = raw_feed_timeline(raw, resolved)
    feed = derive_recovery_partition(feed_times, resolved, boundary_source="raw feed timestamps")
    admitted = derive_recovery_partition(
        admitted_times, resolved, boundary_source="QC-admitted rows"
    )

    payload = dict(feed)
    payload["boundary_source"] = settings["boundary_source"]
    payload["boundary_source_rule"] = settings["boundary_source_rule"]
    payload["boundary_source_reason"] = settings["boundary_source_reason"]
    payload["raw_feed_recovery_boundary"] = feed["recovery_start_utc"]
    payload["raw_feed_timeline_counts"] = feed_counts
    payload["qc_admitted_continuity"] = {
        "boundary_source": "QC-admitted rows",
        "role": "reported for comparison only; it does not decide the split",
        "recovery_start_utc": admitted["recovery_start_utc"],
        "first_observation_utc": admitted["first_observation_utc"],
        "last_observation_utc": admitted["last_observation_utc"],
        "observations": admitted["observations"],
        "long_outages": admitted["long_outages"],
    }
    payload["boundaries_agree"] = bool(
        feed["recovery_start_utc"] == admitted["recovery_start_utc"]
    )
    payload["boundaries_agree_note"] = (
        "Agreement between the raw-feed boundary and the QC-admitted boundary is "
        "a property of this particular record. It is NOT evidence that deriving "
        "the boundary from admitted rows would be correct in general, and it must "
        "not be cited as validating an earlier implementation."
    )
    return payload


def derive_recovery_partition(
    admitted_times: pd.Series | list[pd.Timestamp],
    config: dict[str, Any] | None = None,
    boundary_source: str = "raw feed timestamps",
) -> dict[str, Any]:
    """Locate the latest gap longer than the configured long-outage threshold.

    Returns the outage interval, the first observation after it, and the rule
    that produced them. ``recovery_start_utc`` is ``None`` when the record
    contains no qualifying outage.
    """
    resolved = config or load_config()
    threshold_hours = float(resolved["outage"]["long_outage_hours"])
    times = pd.to_datetime(pd.Series(list(admitted_times)), utc=True).sort_values().reset_index(drop=True)

    result: dict[str, Any] = {
        "rule": resolved["recovery_partition"]["rule"],
        "boundary_source": boundary_source,
        "long_outage_hours": threshold_hours,
        "observations": int(len(times)),
        "first_observation_utc": None,
        "last_observation_utc": None,
        "long_outages": [],
        "latest_long_outage": None,
        "recovery_start_utc": None,
        "recovery_context": {
            "cause": "user-confirmed manual station reset",
            "firmware_root_cause": "unresolved",
            "boundary": (
                "The station resuming after a manual reset is not evidence that "
                "the firmware hang is fixed. Only the outage interval, the first "
                "entry after recovery and the current data state are recorded."
            ),
        },
    }
    if len(times) == 0:
        return result

    result["first_observation_utc"] = times.iloc[0].isoformat().replace("+00:00", "Z")
    result["last_observation_utc"] = times.iloc[-1].isoformat().replace("+00:00", "Z")
    if len(times) < 2:
        return result

    deltas = times.diff().dt.total_seconds()
    threshold_seconds = threshold_hours * 3600.0
    long_indices = [i for i in range(1, len(times)) if float(deltas.iloc[i]) > threshold_seconds]
    for index in long_indices:
        result["long_outages"].append(
            {
                "gap_start_utc": times.iloc[index - 1].isoformat().replace("+00:00", "Z"),
                "gap_end_utc": times.iloc[index].isoformat().replace("+00:00", "Z"),
                "gap_hours": round(float(deltas.iloc[index]) / 3600.0, 4),
            }
        )
    if long_indices:
        latest = long_indices[-1]
        result["latest_long_outage"] = result["long_outages"][-1]
        result["recovery_start_utc"] = times.iloc[latest].isoformat().replace("+00:00", "Z")
    return result


# --------------------------------------------------------------------------- #
# QC report
# --------------------------------------------------------------------------- #

def build_qc_report(
    *,
    run_id: str,
    csv_path: Path | str,
    csv_sha256: str,
    metadata_sha256: str,
    field_mapping: dict[str, str],
    row_counts: dict[str, int],
    binned: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the QC evidence document."""
    required = list(config["field_semantics"]["required_semantics"])
    channels = list(field_mapping.values())
    valid_bins = binned[binned["valid"]] if len(binned) else binned

    reason_counts: dict[str, int] = {}
    for reason_text in binned.get("invalid_reason", pd.Series(dtype=str)):
        if not reason_text:
            continue
        for reason in str(reason_text).split(";"):
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    def _stamp(value: Any) -> str | None:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return pd.Timestamp(value).isoformat().replace("+00:00", "Z")

    return {
        "artifact": "qc_report.json",
        "run_id": run_id,
        "config_id": config["config_id"],
        "input": {
            "csv_path": str(Path(csv_path).as_posix()),
            "csv_sha256": csv_sha256,
            "channel_metadata_sha256": metadata_sha256,
        },
        "field_mapping": dict(field_mapping),
        "required_semantics": required,
        "row_level": dict(row_counts),
        "resampling": {
            "bin_minutes": config["resample"]["bin_minutes"],
            "label": config["resample"]["label"],
            "closed": config["resample"]["closed"],
            "aggregation": config["resample"]["aggregation"],
            "min_raw_observations_per_bin": config["resample"]["min_raw_observations_per_bin"],
            "expected_raw_observations_per_bin": config["resample"]["expected_raw_observations_per_bin"],
            "interpolation_applied": False,
            "wind_direction_aggregation_limitation": config["resample"][
                "wind_direction_aggregation_limitation"
            ],
            "apparent_temperature_aggregation": (
                "Apparent temperature is computed per admitted raw row and then "
                "median-aggregated inside the bin, matching the technical "
                "design's wording 'the median apparent-temperature estimate in "
                "the bin'. This is not identical to evaluating the equation on "
                "the binned medians."
            ),
        },
        "bins": {
            "total": int(len(binned)),
            "valid": int(len(valid_bins)),
            "invalid": int(len(binned) - len(valid_bins)),
            "first_bin_utc": _stamp(binned["bin_end_utc"].min()) if len(binned) else None,
            "last_bin_utc": _stamp(binned["bin_end_utc"].max()) if len(binned) else None,
            "invalid_reason_counts": reason_counts,
            "per_channel_null_bins": {
                name: int(binned[name].isna().sum()) if len(binned) else 0 for name in channels
            },
        },
        "boundary": [
            "These counts describe data preparation only.",
            "No model was trained, selected or evaluated to produce this report.",
        ],
    }


def load_channel_metadata(path: Path | str) -> dict[str, Any]:
    """Load the frozen channel-metadata JSON written by the puller."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    channel = payload.get("channel") if isinstance(payload, dict) else None
    if isinstance(channel, dict):
        return channel
    if isinstance(payload, dict) and "field1" in payload:
        return payload
    raise FieldMappingError(f"{path}: no channel metadata object found")


def utc_day(timestamp: pd.Timestamp) -> datetime:
    stamp = pd.Timestamp(timestamp).tz_convert(timezone.utc)
    return datetime(stamp.year, stamp.month, stamp.day, tzinfo=timezone.utc)


def next_utc_day(day: datetime) -> datetime:
    return day + timedelta(days=1)
