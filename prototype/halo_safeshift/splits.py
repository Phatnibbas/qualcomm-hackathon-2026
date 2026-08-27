"""Retrospective blocked evaluation folds.

Random splits and shuffled cross-validation are forbidden. Folds are
expanding-window and strictly chronological:

    [ train block ] -- embargo -- | boundary B | -- embargo -- [ validation block ]

A train sample is admitted only when its *target* timestamp is at or before
``B - embargo``; a validation sample only when its *window start* is at or
after ``B + embargo``. Both bounds are checked against the sample's full span,
so no sample can straddle a boundary.

This is retrospective model development. It is not an untouched-test
certification: the historical final dates have already been examined by an
external verifier.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

from . import load_config
from .features import WindowSet

__all__ = ["Fold", "EmptyFoldError", "build_folds", "fold_report", "assert_fold_chronology"]


class EmptyFoldError(AssertionError):
    """Raised when a planned fold realises an empty train or validation block."""


class Fold:
    """One expanding-window train/validation block pair."""

    def __init__(
        self,
        index: int,
        train_start: datetime,
        boundary: datetime,
        validation_end: datetime,
        train_mask: np.ndarray,
        validation_mask: np.ndarray,
        embargo_minutes: float,
    ) -> None:
        self.index = index
        self.train_start = train_start
        self.boundary = boundary
        self.validation_end = validation_end
        self.train_mask = train_mask
        self.validation_mask = validation_mask
        self.embargo_minutes = embargo_minutes

    @property
    def n_train(self) -> int:
        return int(self.train_mask.sum())

    @property
    def n_validation(self) -> int:
        return int(self.validation_mask.sum())

    def to_dict(self, windows: WindowSet | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "fold_index": self.index,
            "train_block_start_utc": _iso(self.train_start),
            "boundary_utc": _iso(self.boundary),
            "validation_block_end_utc": _iso(self.validation_end),
            "embargo_minutes_each_side": self.embargo_minutes,
            "total_separation_minutes": 2 * self.embargo_minutes,
            "n_train_issues": self.n_train,
            "n_validation_issues": self.n_validation,
        }
        if windows is not None and len(windows):
            payload["train_issue_range_utc"] = _range(windows.issue_times, self.train_mask)
            payload["train_target_range_utc"] = _range(windows.target_times, self.train_mask)
            payload["validation_issue_range_utc"] = _range(
                windows.issue_times, self.validation_mask
            )
            payload["validation_target_range_utc"] = _range(
                windows.target_times, self.validation_mask
            )
        return payload


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    return pd.Timestamp(value).tz_convert(timezone.utc).isoformat().replace("+00:00", "Z")


def _range(times: np.ndarray, mask: np.ndarray) -> dict[str, str | None]:
    selected = times[mask]
    if selected.size == 0:
        return {"first": None, "last": None}
    return {
        "first": _iso(pd.Timestamp(selected.min()).tz_localize(timezone.utc)),
        "last": _iso(pd.Timestamp(selected.max()).tz_localize(timezone.utc)),
    }


def _utc_days(times: np.ndarray) -> list[datetime]:
    stamps = pd.DatetimeIndex(times)
    days = sorted({datetime(s.year, s.month, s.day, tzinfo=timezone.utc) for s in stamps})
    return days


def build_folds(
    windows: WindowSet,
    config: dict[str, Any] | None = None,
) -> list[Fold]:
    """Build expanding-window folds over the supplied windows.

    Day boundaries are UTC. Internal incomplete days are retained; only the
    per-window eligibility rules decide which issues become samples. No
    post-hoc completeness threshold is applied.
    """
    resolved = config or load_config()
    settings = resolved["splits"]
    if settings["policy"] != "expanding-window blocked folds over historical_development only":
        raise ValueError(f"unexpected split policy {settings['policy']!r}")

    min_train_days = int(settings["min_train_days"])
    validation_days = int(settings["validation_days"])
    max_folds = int(settings["max_folds"])
    embargo = float(settings["embargo_minutes"])
    embargo_delta = timedelta(minutes=embargo)
    fail_closed = bool(settings["fail_closed_on_empty_fold"])

    if len(windows) == 0:
        return []

    target_times = pd.DatetimeIndex(windows.target_times).tz_localize(timezone.utc)
    window_starts = pd.DatetimeIndex(windows.window_start_times).tz_localize(timezone.utc)

    days = _utc_days(windows.issue_times)
    if len(days) < min_train_days + validation_days:
        return []

    folds: list[Fold] = []
    train_start = days[0]
    cursor = min_train_days
    while cursor + validation_days <= len(days) and len(folds) < max_folds:
        boundary = days[cursor]
        validation_end = days[cursor + validation_days - 1] + timedelta(days=1)

        train_mask = np.asarray(
            (target_times <= boundary - embargo_delta) & (window_starts >= train_start)
        )
        validation_mask = np.asarray(
            (window_starts >= boundary + embargo_delta) & (target_times <= validation_end)
        )

        fold = Fold(
            index=len(folds),
            train_start=train_start,
            boundary=boundary,
            validation_end=validation_end,
            train_mask=train_mask,
            validation_mask=validation_mask,
            embargo_minutes=embargo,
        )
        # Fail closed. A planned fold that realises no train or no validation
        # sample is not a smaller fold, it is a different protocol: the mean
        # across folds would then be taken over a set nobody declared. Dropping
        # it silently would make the realised fold count stop describing what
        # was planned, so the run is rejected and the operator decides.
        if fail_closed and (fold.n_train == 0 or fold.n_validation == 0):
            raise EmptyFoldError(
                f"planned fold {fold.index} (train from {_iso(train_start)}, "
                f"boundary {_iso(boundary)}, validation end {_iso(validation_end)}) "
                f"realised n_train={fold.n_train} and n_validation={fold.n_validation}. "
                f"A fold with an empty block changes the evaluation protocol, so the "
                f"run is rejected rather than silently reshaped. Adjust "
                f"splits.min_train_days / splits.validation_days, or supply a record "
                f"with enough eligible windows."
            )
        folds.append(fold)
        cursor += validation_days

    return folds


def assert_fold_chronology(folds: list[Fold], windows: WindowSet) -> None:
    """Fail closed if any fold violates the chronological or embargo contract."""
    issue_times = pd.DatetimeIndex(windows.issue_times).tz_localize(timezone.utc)
    target_times = pd.DatetimeIndex(windows.target_times).tz_localize(timezone.utc)
    window_starts = pd.DatetimeIndex(windows.window_start_times).tz_localize(timezone.utc)

    for fold in folds:
        embargo = timedelta(minutes=fold.embargo_minutes)
        # An empty block is a protocol violation, not a fold to skip over.
        # Skipping it here is what let an empty fold reach a report unchallenged.
        if fold.n_train == 0 or fold.n_validation == 0:
            raise EmptyFoldError(
                f"fold {fold.index}: n_train={fold.n_train}, "
                f"n_validation={fold.n_validation}. An empty block cannot be "
                f"checked for chronology and must not be evaluated."
            )
        train_target_max = target_times[fold.train_mask].max()
        validation_start_min = window_starts[fold.validation_mask].min()
        validation_issue_min = issue_times[fold.validation_mask].min()
        train_issue_max = issue_times[fold.train_mask].max()

        if train_target_max > fold.boundary - embargo:
            raise AssertionError(
                f"fold {fold.index}: a train target at {train_target_max} violates the "
                f"left embargo at {fold.boundary - embargo}"
            )
        if validation_start_min < fold.boundary + embargo:
            raise AssertionError(
                f"fold {fold.index}: a validation window starts at {validation_start_min}, "
                f"before the right embargo at {fold.boundary + embargo}"
            )
        if train_issue_max >= validation_issue_min:
            raise AssertionError(
                f"fold {fold.index}: train issue {train_issue_max} does not precede "
                f"validation issue {validation_issue_min}"
            )
        if bool((fold.train_mask & fold.validation_mask).any()):
            raise AssertionError(f"fold {fold.index}: train and validation masks overlap")


def fold_report(
    folds: list[Fold],
    windows: WindowSet,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialisable description of the fold protocol and its realised counts."""
    resolved = config or load_config()
    settings = resolved["splits"]
    days = _utc_days(windows.issue_times) if len(windows) else []
    return {
        "artifact": "fold_definitions",
        "policy": settings["policy"],
        "forbidden": list(settings["forbidden"]),
        "day_boundary": settings["day_boundary"],
        "min_train_days": settings["min_train_days"],
        "validation_days": settings["validation_days"],
        "max_folds": settings["max_folds"],
        "embargo_minutes_each_side": settings["embargo_minutes"],
        "embargo_rule": settings["embargo_rule"],
        "fail_closed_on_empty_fold": settings["fail_closed_on_empty_fold"],
        "fail_closed_on_empty_fold_reason": settings["fail_closed_on_empty_fold_reason"],
        "incomplete_day_policy": settings["incomplete_day_policy"],
        "fit_scope": settings["fit_scope"],
        "status": settings["status"],
        "available_utc_days": [day.date().isoformat() for day in days],
        "n_windows": len(windows),
        "folds": [fold.to_dict(windows) for fold in folds],
        "boundary": [
            "Fold definitions only. No model was fitted or scored to produce this artifact.",
            "Do not call any block here an untouched test.",
        ],
    }
