"""Baselines that any learned candidate must be compared against.

Both baselines are fitted strictly inside a fold's train block. The climatology
never sees a validation timestamp, a validation target, or the recovery
quarantine.

Nothing here selects a model. These are reference predictors and error
summaries, not results.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from . import load_config
from .features import WindowSet, resolve_local_timezone

__all__ = [
    "PersistenceBaseline",
    "LocalTimeClimatology",
    "mean_absolute_error",
    "root_mean_squared_error",
    "regime_report",
    "to_residual",
    "from_residual",
]


class PersistenceBaseline:
    """``AT(t + horizon) = AT(t)``.

    Reads the apparent-temperature-now column straight out of the frozen
    feature schema, so it cannot drift away from the deployed feature order.
    """

    model_type = "persistence"

    def __init__(self, at_now_index: int) -> None:
        self.at_now_index = int(at_now_index)

    @classmethod
    def from_schema(cls, schema: dict[str, Any]) -> "PersistenceBaseline":
        names = schema["feature_names"]
        return cls(names.index("apparent_temperature_now"))

    def fit(self, x: np.ndarray, y: np.ndarray) -> "PersistenceBaseline":  # noqa: ARG002
        """Persistence has no parameters; ``fit`` exists for interface symmetry."""
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        matrix = np.asarray(x, dtype=np.float64)
        return matrix[:, self.at_now_index].copy()


class LocalTimeClimatology:
    """Train-only median target, keyed by local time of day.

    Unseen keys fall back to the train-block target median, so the predictor is
    total without ever reading a validation value.
    """

    model_type = "climatology_local_time"

    def __init__(self, bin_minutes: int, tzinfo: Any, statistic: str = "median") -> None:
        self.bin_minutes = int(bin_minutes)
        self.tzinfo = tzinfo
        self.statistic = statistic
        self.table: dict[int, float] = {}
        self.fallback: float = float("nan")
        self.fitted_on_n: int = 0

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> "LocalTimeClimatology":
        resolved = config or load_config()
        settings = resolved["models"]["climatology"]
        tzinfo, _ = resolve_local_timezone(resolved)
        return cls(int(settings["bin_minutes"]), tzinfo, str(settings["statistic"]))

    def key_for(self, timestamp: Any) -> int:
        local = pd.Timestamp(timestamp)
        if local.tzinfo is None:
            local = local.tz_localize("UTC")
        local = local.tz_convert(self.tzinfo)
        minutes_of_day = local.hour * 60 + local.minute
        return int(minutes_of_day // self.bin_minutes)

    def fit(self, issue_times: np.ndarray, y: np.ndarray) -> "LocalTimeClimatology":
        targets = np.asarray(y, dtype=np.float64)
        if targets.size == 0:
            raise ValueError("climatology cannot be fitted on an empty train block")
        keys = np.array([self.key_for(stamp) for stamp in issue_times], dtype=np.int64)
        table: dict[int, float] = {}
        for key in np.unique(keys):
            values = targets[keys == key]
            table[int(key)] = float(np.median(values) if self.statistic == "median" else np.mean(values))
        self.table = table
        self.fallback = float(np.median(targets))
        self.fitted_on_n = int(targets.size)
        return self

    def predict(self, issue_times: np.ndarray) -> np.ndarray:
        if not self.table and not math.isfinite(self.fallback):
            raise RuntimeError("climatology has not been fitted")
        return np.array(
            [self.table.get(self.key_for(stamp), self.fallback) for stamp in issue_times],
            dtype=np.float64,
        )


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64))))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    difference = np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean(difference * difference)))


def regime_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    at_now: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    """Error grouped by the *actual* 30-minute apparent-temperature change.

    This is error analysis. It is not a new event label and it names no
    mechanism.
    """
    truth = np.asarray(y_true, dtype=np.float64)
    predicted = np.asarray(y_pred, dtype=np.float64)
    change = truth - np.asarray(at_now, dtype=np.float64)

    bands: list[tuple[str, np.ndarray]] = [
        ("cooling_lt_-2C", change < -2.0),
        ("cooling_-2_to_-1C", (change >= -2.0) & (change < -1.0)),
        ("stable_-1_to_+1C", (change >= -1.0) & (change <= 1.0)),
        ("warming_+1_to_+2C", (change > 1.0) & (change <= 2.0)),
        ("warming_gt_+2C", change > 2.0),
    ]
    report: dict[str, dict[str, float | int]] = {}
    for name, mask in bands:
        count = int(mask.sum())
        report[name] = {
            "n": count,
            "mae": mean_absolute_error(truth[mask], predicted[mask]) if count else float("nan"),
        }
    return report


def to_residual(y: np.ndarray, at_now: np.ndarray) -> np.ndarray:
    """``AT(t+30) - AT(t)`` — the residual parameterization's learning target."""
    return np.asarray(y, dtype=np.float64) - np.asarray(at_now, dtype=np.float64)


def from_residual(residual_prediction: np.ndarray, at_now: np.ndarray) -> np.ndarray:
    """Undo :func:`to_residual`: ``AT(t) + predicted residual``."""
    return np.asarray(residual_prediction, dtype=np.float64) + np.asarray(at_now, dtype=np.float64)


def baseline_bundle(windows: WindowSet, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Descriptive record of the baselines that are available for a window set."""
    resolved = config or load_config()
    return {
        "baselines": list(resolved["models"]["baselines"]),
        "persistence": {
            "definition": "AT(t + horizon) = AT(t)",
            "at_now_feature_index": windows.schema["index"]["apparent_temperature_now"],
        },
        "climatology_local_time": {
            "definition": "train-only median target keyed by local time of day",
            "bin_minutes": resolved["models"]["climatology"]["bin_minutes"],
            "timezone": windows.schema["timezone"],
            "fit_scope": "fold train block only",
        },
        "boundary": "Baseline definitions only. No score is reported here.",
    }
