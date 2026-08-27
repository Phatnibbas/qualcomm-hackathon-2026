"""Target contract: the Australian Bureau of Meteorology non-radiation
apparent-temperature equation, and the boundaries on what it may be called.

The equation is:

    e  = RH/100 * 6.105 * exp(17.27 * Ta / (237.7 + Ta))
    AT = Ta + 0.33 * e - 0.70 * ws - 4.00

Constants are explicit here because they *are* the cited equation, but they are
also loaded from ``config/experiment.v1.json`` and serialised into
``target_contract.json`` so the artifact and the code cannot silently diverge.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from . import load_config

__all__ = [
    "TargetInputError",
    "vapour_pressure_hpa",
    "apparent_temperature_c",
    "apparent_temperature_array",
    "target_contract",
]


class TargetInputError(ValueError):
    """Raised when an input cannot be admitted to the apparent-temperature equation."""


def _target_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    return (config or load_config())["target"]


def _check_finite(name: str, value: float) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise TargetInputError(f"{name} must be finite, got {value!r}")
    return numeric


def _check_rh(rh_percent: float, cfg: dict[str, Any]) -> float:
    rh = _check_finite("rh_percent", rh_percent)
    if rh < 0.0 or rh > 100.0:
        raise TargetInputError(f"rh_percent must lie in [0, 100], got {rh!r}")
    guard = float(cfg["rh_fraction_guard_max_percent"])
    if 0.0 < rh <= guard:
        raise TargetInputError(
            f"rh_percent={rh!r} is at or below the unit-substitution guard "
            f"({guard} %RH). A relative humidity supplied as a 0-1 fraction "
            f"would land here and produce a plausible but wrong apparent "
            f"temperature. Supply percent, not fraction."
        )
    return rh


def vapour_pressure_hpa(temp_c: float, rh_percent: float) -> float:
    """Water-vapour pressure in hPa from dry-bulb temperature and relative humidity."""
    cfg = _target_config()
    temp = _check_finite("temp_c", temp_c)
    rh = _check_rh(rh_percent, cfg)
    constants = cfg["vapour_pressure_constants"]
    a = float(constants["a"])
    b = float(constants["b"])
    c = float(constants["c"])
    denominator = c + temp
    if denominator == 0.0:
        raise TargetInputError(f"temp_c={temp!r} makes the Magnus denominator zero")
    return (rh / 100.0) * a * math.exp(b * temp / denominator)


def apparent_temperature_c(temp_c: float, rh_percent: float, wind_mps: float) -> float:
    """Station-derived shade apparent-temperature estimate, in degrees Celsius.

    No wind-height correction is applied. See :func:`target_contract` for the
    transfer limit and the forbidden output names.
    """
    cfg = _target_config()
    temp = _check_finite("temp_c", temp_c)
    wind = _check_finite("wind_mps", wind_mps)
    if wind < 0.0:
        raise TargetInputError(f"wind_mps must be non-negative, got {wind!r}")
    vapour = vapour_pressure_hpa(temp, rh_percent)
    coefficients = cfg["apparent_temperature_coefficients"]
    return (
        temp
        + float(coefficients["vapour_pressure"]) * vapour
        + float(coefficients["wind_speed"]) * wind
        + float(coefficients["offset"])
    )


def apparent_temperature_array(
    temp_c: np.ndarray,
    rh_percent: np.ndarray,
    wind_mps: np.ndarray,
    config: dict[str, Any] | None = None,
) -> np.ndarray:
    """Vectorised apparent temperature.

    Invalid elements produce ``nan`` rather than raising, so a whole column can
    be evaluated and the invalid bins accounted for downstream. The scalar
    entry point remains fail-closed.
    """
    cfg = _target_config(config)
    constants = cfg["vapour_pressure_constants"]
    coefficients = cfg["apparent_temperature_coefficients"]
    guard = float(cfg["rh_fraction_guard_max_percent"])

    temp = np.asarray(temp_c, dtype=np.float64)
    rh = np.asarray(rh_percent, dtype=np.float64)
    wind = np.asarray(wind_mps, dtype=np.float64)

    valid = (
        np.isfinite(temp)
        & np.isfinite(rh)
        & np.isfinite(wind)
        & (rh >= 0.0)
        & (rh <= 100.0)
        & ~((rh > 0.0) & (rh <= guard))
        & (wind >= 0.0)
        & ((float(constants["c"]) + temp) != 0.0)
    )

    safe_temp = np.where(valid, temp, 0.0)
    safe_rh = np.where(valid, rh, 0.0)
    safe_wind = np.where(valid, wind, 0.0)

    vapour = (
        (safe_rh / 100.0)
        * float(constants["a"])
        * np.exp(float(constants["b"]) * safe_temp / (float(constants["c"]) + safe_temp))
    )
    result = (
        safe_temp
        + float(coefficients["vapour_pressure"]) * vapour
        + float(coefficients["wind_speed"]) * safe_wind
        + float(coefficients["offset"])
    )
    return np.where(valid, result, np.nan)


def target_contract(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Serialisable statement of what the target is, and what it is not."""
    resolved = config or load_config()
    cfg = resolved["target"]
    return {
        "name": "station-derived shade apparent-temperature estimate",
        "symbol": "AT_station",
        "unit": "degC",
        "formula_version": cfg["formula_version"],
        "equations": {
            "vapour_pressure_hpa": "e = RH/100 * 6.105 * exp(17.27 * Ta / (237.7 + Ta))",
            "apparent_temperature_c": "AT = Ta + 0.33 * e - 0.70 * ws - 4.00",
        },
        "constants": {
            "vapour_pressure": dict(cfg["vapour_pressure_constants"]),
            "apparent_temperature": dict(cfg["apparent_temperature_coefficients"]),
        },
        "inputs": {
            "Ta": "station dry-bulb air temperature, degC",
            "RH": "station relative humidity, percent (not a 0-1 fraction)",
            "ws": "station wind speed, m/s",
        },
        "primary_source": cfg["source"],
        "input_validation": {
            "reject_non_finite": True,
            "rh_range_percent": [0.0, 100.0],
            "rh_fraction_guard_max_percent": cfg["rh_fraction_guard_max_percent"],
            "rh_fraction_guard_reason": cfg["rh_fraction_guard_reason"],
            "reject_negative_wind": True,
        },
        "wind_height": {
            "station_sensor_height_m_agl": cfg["wind_height_m_agl"],
            "station_height_basis": "USER-CONFIRMED, Phat, 2026-08-15. Height above local ground, not surveyed elevation above mean sea level.",
            "bom_reference_wind_height_m": cfg["bom_reference_wind_height_m"],
            "wind_height_correction_applied": cfg["apply_wind_height_correction"],
            "why_no_correction": (
                "No site-validated roughness length or height-normalisation "
                "coefficient exists for this mast. Applying an assumed profile "
                "would manufacture precision. The observed station wind is used "
                "as observed."
            ),
        },
        "claim_boundary": {
            "is": (
                "A station-derived shade apparent-temperature estimate computed "
                "from this site's temperature, humidity and observed wind."
            ),
            "is_not": [
                "WBGT or an ISO 7243 measurement",
                "a QCVN-compliant heat measurement",
                "a heatstroke or illness probability",
                "a safe/unsafe workplace classification",
                "a legal work/rest limit",
                "a direct-sun apparent temperature",
                "a medical prediction",
            ],
            "required_subtitle": (
                "Shade estimate from site temperature, humidity and wind; not "
                "WBGT or a medical/legal safety limit."
            ),
        },
        "uncertainty": {
            "calibrated_uncertainty_degc": None,
            "reason": (
                "UNKNOWN. No calibrated reference instrument has been co-located "
                "with this station, so no uncertainty figure can be stated. "
                "Resolver: co-locate a calibrated reference and record paired "
                "observations. A fabricated value is worse than an absent one."
            ),
        },
    }
