"""Pure, dependency-light guardrails shared by the full Colab pipeline.

The full notebook owns heavy decoding/training.  This module deliberately owns
only invariants that can be unit-tested on synthetic values locally.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


class ContractError(ValueError):
    """A P1 artifact or comparison violates a declared hard gate."""


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ContractError("timestamps must carry an explicit timezone")
    return value.astimezone(timezone.utc)


def assert_satellite_frame_is_available(
    *,
    nominal_scan_utc: datetime,
    assumed_scan_completion_utc: datetime,
    issue_time_utc: datetime,
    publication_lag_minutes: int,
) -> dict[str, float | int | str]:
    """Accept only a frame available strictly before its forecast issue time."""
    nominal = _as_utc(nominal_scan_utc)
    completed = _as_utc(assumed_scan_completion_utc)
    issue = _as_utc(issue_time_utc)
    if publication_lag_minutes < 0:
        raise ContractError("publication lag cannot be negative")
    available = completed.timestamp() + 60 * publication_lag_minutes
    if not available < issue.timestamp():
        raise ContractError(
            "satellite frame is not strictly available before issue time; "
            "nearest/post-issue matching is forbidden"
        )
    return {
        "nominal_scan_utc": nominal.isoformat(),
        "assumed_scan_completion_utc": completed.isoformat(),
        "issue_time_utc": issue.isoformat(),
        "publication_lag_minutes": publication_lag_minutes,
        "frame_age_minutes": (issue.timestamp() - completed.timestamp()) / 60.0,
    }


def assert_common_ablation_identity(
    station_rows: Sequence[tuple[str, str, str]],
    satellite_rows: Sequence[tuple[str, str, str]],
    fused_rows: Sequence[tuple[str, str, str]],
) -> None:
    """Require station/satellite/fused score the exact same issue/target/split rows."""
    expected = list(station_rows)
    if not expected:
        raise ContractError("common-row ablation cohort is empty")
    for label, observed in (("satellite", satellite_rows), ("fused", fused_rows)):
        if list(observed) != expected:
            raise ContractError(
                f"{label} ablation rows differ from station common cohort; metrics are incomparable"
            )


def assert_split_sequences_do_not_cross(rows: Iterable[dict[str, str]]) -> None:
    """Every feature window and its target must remain inside the declared split."""
    for row in rows:
        split = row.get("split")
        if not split or row.get("window_split") != split or row.get("target_split") != split:
            raise ContractError(
                "feature window or target crosses a split boundary; embargo/split isolation failed"
            )


def assign_fixed_chronological_splits(
    rows: Sequence[dict[str, str]],
    *,
    train_last_issue_utc: str,
    validation_first_issue_utc: str,
    validation_last_issue_utc: str,
    test_first_issue_utc: str,
) -> list[dict[str, str]]:
    """Assign fixed P0 splits by issue timestamp, allowing UTC-midnight windows.

    A 60-minute lookback may start on the prior calendar day.  The invariant is
    not same-day data; it is that an issue/target pair belongs to one split and
    respects the predeclared embargo boundaries.
    """
    boundaries = {
        "train_last": datetime.fromisoformat(train_last_issue_utc.replace("Z", "+00:00")),
        "validation_first": datetime.fromisoformat(validation_first_issue_utc.replace("Z", "+00:00")),
        "validation_last": datetime.fromisoformat(validation_last_issue_utc.replace("Z", "+00:00")),
        "test_first": datetime.fromisoformat(test_first_issue_utc.replace("Z", "+00:00")),
    }
    assigned: list[dict[str, str]] = []
    for original in rows:
        row = dict(original)
        issue = datetime.fromisoformat(row["issue_time_utc"].replace("Z", "+00:00"))
        target = datetime.fromisoformat(row["target_time_utc"].replace("Z", "+00:00"))
        if issue <= boundaries["train_last"] and target <= boundaries["train_last"] + (target - issue):
            row["split"] = "train"
        elif boundaries["validation_first"] <= issue <= boundaries["validation_last"] and target <= boundaries["validation_last"] + (target - issue):
            row["split"] = "validation"
        elif issue >= boundaries["test_first"]:
            row["split"] = "test"
        else:
            row["split"] = "embargo"
        assigned.append(row)
    return assigned


def parse_hsd_segment_header(raw: bytes) -> dict[str, int]:
    """Parse HSD block-7 fields: total segments, segment number, first line."""
    if len(raw) != 4:
        raise ContractError("segment header fixture must contain exactly four bytes")
    total_segments, segment, low, high = raw
    first_line = low | (high << 8)
    if not (1 <= segment <= total_segments):
        raise ContractError(f"invalid HSD segment {segment} of {total_segments}")
    return {"segment": segment, "total_segments": total_segments, "first_line": first_line}


def assert_roi_not_clipped(*, roi_pixels: int, covered_pixels: int) -> None:
    if roi_pixels <= 0:
        raise ContractError("ROI contains no navigated pixels")
    if covered_pixels != roi_pixels:
        raise ContractError(
            f"ROI is clipped: {covered_pixels} of {roi_pixels} required navigated pixels covered"
        )


def assert_fused_schema(schema: dict, *, runtime_input_names: Sequence[str]) -> None:
    names = schema.get("feature_names")
    if not isinstance(names, list) or int(schema.get("n_features", -1)) != len(names):
        raise ContractError("feature schema is malformed")
    satellite = [name for name in names if str(name).startswith("satellite_")]
    if not satellite:
        raise ContractError("fused bundle has no satellite feature names")
    missing = [name for name in satellite if name not in runtime_input_names]
    if missing:
        raise ContractError(f"fused runtime input omits satellite feature(s): {missing}")


def verify_sha256(path: Path | str, expected: str) -> str:
    candidate = Path(path)
    actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
    if actual != expected:
        raise ContractError(f"SHA-256 mismatch for {candidate.name}: expected {expected}, got {actual}")
    return actual
