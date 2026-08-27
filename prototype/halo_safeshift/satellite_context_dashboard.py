#!/usr/bin/env python3
"""Persistence forecast served beside descriptive satellite context.

The one thing this service exists to make impossible is a viewer believing the
satellite improved the forecast. The forecast is persistence,
``AT(t+30) = AT(t)``; it never reads a satellite value. So:

* ``mode`` is literally ``"persistence + satellite context (not fused)"``;
* ``satellite_used_in_prediction`` is a constant ``False``;
* the page carries a ``NOT FUSED`` badge and the design's mandated
  ``SATELLITE CONTEXT - NOT USED BY P0 MODEL`` line (technical design §10);
* when the context is missing, stale, malformed or inadmissible, the forecast
  still runs and the service says exactly why it degraded rather than quietly
  serving an old frame.

Dependency note
---------------
This module uses only the standard library, so it can be copied to the UNO Q,
which has neither NumPy nor pip. ``target.apparent_temperature_c`` is the
authority for the equation and is used when importable; the stdlib mirror below
exists only for the board, and a unit test pins the two to bit-equality so they
cannot drift.

Port 8766 is deliberate: the P0 emergency runtime owns 8765 and the live model
service owns 8080. Neither is touched.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import platform
import statistics
import threading
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse

SCHEMA_VERSION = "halo-satellite-context-v1"
MODE_FUSED_FREE = "persistence + satellite context (not fused)"
MODE_PERSISTENCE_ONLY = "persistence only"
FORECAST_METHOD = "persistence"
HORIZON_MINUTES = 30
DEFAULT_MAX_FRAME_AGE_MINUTES = 180

CLAIM_BOUNDARY = (
    "Station-derived shade apparent-temperature estimate; not WBGT, not direct-sun exposure, "
    "not a medical or legal safety limit. Satellite panels are descriptive context only: the "
    "forecast is persistence and consumes no satellite value. Frame age is a measured lag, "
    "not a warning lead time."
)
DESIGN_BANNER = "SATELLITE CONTEXT - NOT USED BY P0 MODEL"

# Right-labelled 5-minute binning, matching the P0 data contract.
BIN_SECONDS = 300
MIN_RAW_PER_BIN = 6


class DashboardError(RuntimeError):
    """The dashboard cannot be constructed from the supplied inputs."""


# --------------------------------------------------------------------------
# apparent temperature
# --------------------------------------------------------------------------


def _apparent_temperature_stdlib(temp_c: float, rh_percent: float, wind_mps: float) -> float:
    """Board-safe mirror of the BoM non-radiation apparent temperature."""
    vapour = (rh_percent / 100.0) * 6.105 * math.exp(17.27 * temp_c / (237.7 + temp_c))
    return temp_c + 0.33 * vapour - 0.70 * wind_mps - 4.00


def apparent_temperature(temp_c: float, rh_percent: float, wind_mps: float) -> float:
    """Use the repository's target authority when it is importable.

    On the UNO Q the import fails (no NumPy) and the stdlib mirror runs. A unit
    test asserts the two agree exactly, so the fallback cannot drift.
    """
    try:
        from .target import apparent_temperature_c
    except ImportError:
        try:
            from target import apparent_temperature_c  # type: ignore[no-redef]
        except ImportError:
            return _apparent_temperature_stdlib(temp_c, rh_percent, wind_mps)
    return float(apparent_temperature_c(temp_c, rh_percent, wind_mps))


AT_EQUATION = {
    "source": "Australian Bureau of Meteorology, non-radiation apparent temperature",
    "source_url": "https://www.bom.gov.au/info/thermal_stress/",
    "equations": {
        "vapour_pressure": "e = RH/100 * 6.105 * exp(17.27*Ta/(237.7+Ta))",
        "apparent_temperature": "AT = Ta + 0.33*e - 0.70*ws - 4.00",
    },
    "equation_status": "fixed; published equation, not fitted to our data",
    "constants": [
        {"symbol": "6.105", "meaning": "Magnus reference vapour pressure", "unit": "hPa"},
        {"symbol": "17.27", "meaning": "Magnus coefficient", "unit": "dimensionless"},
        {"symbol": "237.7", "meaning": "Magnus coefficient", "unit": "degC"},
        {"symbol": "0.33", "meaning": "Vapour-pressure weight", "unit": "degC per hPa"},
        {"symbol": "0.70", "meaning": "Wind-cooling weight", "unit": "degC per m/s"},
        {"symbol": "4.00", "meaning": "Offset", "unit": "degC"},
    ],
    "wind_height_note": (
        "Station wind is used as observed (~15 m AGL); the BoM equation references 10 m. "
        "No height correction is applied, because no site-validated roughness length exists."
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_utc(value: str, label: str) -> datetime:
    try:
        moment = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise DashboardError(f"{label} is not an ISO-8601 timestamp: {value!r}") from exc
    if moment.tzinfo is None:
        raise DashboardError(f"{label} must carry an explicit timezone")
    return moment.astimezone(timezone.utc)


# --------------------------------------------------------------------------
# station input
# --------------------------------------------------------------------------

_CSV_COLUMNS = {
    "timestamp": "timestamp_utc_iso",
    "temperature": "Temperature (°C)",
    "humidity": "Humidity (%RH)",
    "wind": "Wind speed (m/s)",
}


def build_station_input_from_csv(csv_path: Path, issue_time_utc: str) -> dict[str, Any]:
    """Median the frozen station rows in the closed bin ending at the issue time.

    Nothing is invented: if the real archive has fewer than six raw rows in that
    bin, this fails rather than interpolating a measurement.
    """
    issue = _parse_utc(issue_time_utc, "issue_time_utc")
    window_start = issue - timedelta(seconds=BIN_SECONDS)

    rows: list[dict[str, float]] = []
    stamps: list[str] = []
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        for record in csv.DictReader(handle):
            raw_stamp = (record.get(_CSV_COLUMNS["timestamp"]) or "").strip()
            if not raw_stamp:
                continue
            try:
                moment = datetime.fromisoformat(raw_stamp.replace("Z", "+00:00"))
            except ValueError:
                continue
            if moment.tzinfo is None:
                moment = moment.replace(tzinfo=timezone.utc)
            if not (window_start < moment <= issue):
                continue
            try:
                values = {
                    "temperature": float(record[_CSV_COLUMNS["temperature"]]),
                    "humidity": float(record[_CSV_COLUMNS["humidity"]]),
                    "wind_speed": float(record[_CSV_COLUMNS["wind"]]),
                }
            except (KeyError, TypeError, ValueError):
                continue
            rows.append(values)
            stamps.append(moment.isoformat())

    if len(rows) < MIN_RAW_PER_BIN:
        raise DashboardError(
            f"only {len(rows)} raw station rows in the 5-minute bin ending "
            f"{issue.isoformat()}; the P0 contract requires at least {MIN_RAW_PER_BIN}. "
            f"A station measurement must not be fabricated to fill this bin."
        )

    temperature = statistics.median(r["temperature"] for r in rows)
    humidity = statistics.median(r["humidity"] for r in rows)
    wind = statistics.median(r["wind_speed"] for r in rows)
    return {
        "issue_time_utc": issue.isoformat(),
        "bin_seconds": BIN_SECONDS,
        "raw_rows_in_bin": len(rows),
        "first_raw_utc": min(stamps),
        "last_raw_utc": max(stamps),
        "temperature_degc": round(temperature, 4),
        "humidity_percent": round(humidity, 4),
        "wind_mps": round(wind, 4),
        "current_at_degc": round(apparent_temperature(temperature, humidity, wind), 4),
        "aggregation": "median of raw rows in the right-labelled closed 5-minute bin",
        "source": {
            "path": str(csv_path),
            "name": csv_path.name,
            "sha256": _sha256(csv_path),
            "bytes": csv_path.stat().st_size,
        },
    }


def load_station_input(path: Path, issue_time_utc: str | None) -> dict[str, Any]:
    """Accept either a prepared station-input JSON or the frozen station CSV."""
    if not path.is_file():
        raise DashboardError(f"station input not found: {path}")
    if path.suffix.lower() == ".csv":
        if not issue_time_utc:
            raise DashboardError("--issue-time-utc is required when --station-input is a CSV")
        return build_station_input_from_csv(path, issue_time_utc)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DashboardError(f"cannot read station input {path}: {exc}") from exc
    for field in ("issue_time_utc", "temperature_degc", "humidity_percent", "wind_mps"):
        if field not in document:
            raise DashboardError(f"station input is missing required field {field!r}")
    document.setdefault(
        "current_at_degc",
        round(
            apparent_temperature(
                float(document["temperature_degc"]),
                float(document["humidity_percent"]),
                float(document["wind_mps"]),
            ),
            4,
        ),
    )
    document.setdefault("source", {"path": str(path), "name": path.name, "sha256": _sha256(path)})
    return document


# --------------------------------------------------------------------------
# satellite context admission
# --------------------------------------------------------------------------

_FORBIDDEN_KEYS = ("deep_pct", "cloud_class", "storm_class", "rain_class", "alert", "threshold_220k")


def evaluate_context(
    document: Any,
    *,
    issue_time_utc: str,
    max_frame_age_minutes: float = DEFAULT_MAX_FRAME_AGE_MINUTES,
) -> tuple[dict[str, Any] | None, str | None]:
    """Admit a context document, or return the exact reason it was refused.

    Returns ``(context, None)`` on admission and ``(None, reason)`` otherwise.
    The service never silently falls back to a stale frame.
    """
    if document is None:
        return None, "satellite context file is absent"
    if not isinstance(document, dict):
        return None, "satellite context is not a JSON object"
    if document.get("schema_version") != SCHEMA_VERSION:
        return None, f"unexpected schema_version {document.get('schema_version')!r}"
    if document.get("satellite_used_in_prediction") is not False:
        return None, "context does not declare satellite_used_in_prediction=false"

    features = document.get("features")
    if not isinstance(features, dict) or not features:
        return None, "satellite context carries no band features"

    blob = json.dumps(document).lower()
    for banned in _FORBIDDEN_KEYS:
        if banned in blob:
            return None, f"context contains forbidden derived field {banned!r}"
    if "220" in str(document.get("features", {}).get("threshold_kelvin", "")):
        return None, "context references the unproven 220 K threshold"

    timing = document.get("timing")
    if not isinstance(timing, dict):
        return None, "satellite context has no timing block"
    try:
        issue = _parse_utc(issue_time_utc, "issue_time_utc")
        available = _parse_utc(str(timing.get("effective_available_utc")), "effective_available_utc")
        nominal = _parse_utc(str(timing.get("nominal_scan_utc")), "nominal_scan_utc")
    except DashboardError as exc:
        return None, f"invalid timing: {exc}"

    if not available < issue:
        return None, (
            f"frame is not available strictly before the issue time "
            f"({available.isoformat()} >= {issue.isoformat()})"
        )
    if timing.get("available_strictly_before_issue") is not True:
        return None, "context does not assert availability strictly before issue"

    frame_age = (issue - nominal).total_seconds() / 60.0
    if frame_age > max_frame_age_minutes:
        return None, (
            f"frame is stale: {frame_age:.1f} min old exceeds the "
            f"{max_frame_age_minutes:.0f} min freshness limit"
        )

    for band, stats in features.items():
        if not isinstance(stats, dict) or stats.get("p50") is None:
            return None, f"band {band!r} has no usable p50 statistic"

    return document, None


# --------------------------------------------------------------------------
# state
# --------------------------------------------------------------------------


class DashboardState:
    def __init__(
        self,
        *,
        satellite_context: Path | None,
        station_input: Path,
        issue_time_utc: str | None = None,
        max_frame_age_minutes: float = DEFAULT_MAX_FRAME_AGE_MINUTES,
    ) -> None:
        self.lock = threading.Lock()
        self.max_frame_age_minutes = max_frame_age_minutes
        self.station = load_station_input(station_input, issue_time_utc)
        self.issue_time_utc = str(self.station["issue_time_utc"])
        self.satellite_path = satellite_context

        raw: Any = None
        self.context_source: dict[str, Any] | None = None
        if satellite_context is not None and satellite_context.is_file():
            try:
                raw = json.loads(satellite_context.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raw = {"__unreadable__": str(exc)}
            else:
                self.context_source = {
                    "path": str(satellite_context),
                    "name": satellite_context.name,
                    "sha256": _sha256(satellite_context),
                    "bytes": satellite_context.stat().st_size,
                }
        if isinstance(raw, dict) and "__unreadable__" in raw:
            self.context, self.degraded_reason = None, f"satellite context is unreadable: {raw['__unreadable__']}"
        else:
            self.context, self.degraded_reason = evaluate_context(
                raw, issue_time_utc=self.issue_time_utc, max_frame_age_minutes=max_frame_age_minutes
            )

    # -- forecast -----------------------------------------------------------

    def forecast(self) -> dict[str, Any]:
        """Persistence. Deliberately independent of every satellite value."""
        current = float(self.station["current_at_degc"])
        issue = _parse_utc(self.issue_time_utc, "issue_time_utc")
        return {
            "forecast_method": FORECAST_METHOD,
            "horizon_minutes": HORIZON_MINUTES,
            "issue_time_utc": issue.isoformat(),
            "target_time_utc": (issue + timedelta(minutes=HORIZON_MINUTES)).isoformat(),
            "current_at_degc": round(current, 3),
            "predicted_at_plus30_degc": round(current, 3),
            "rule": "AT(t+30) = AT(t)",
            "inputs": {
                "temperature_degc": self.station["temperature_degc"],
                "humidity_percent": self.station["humidity_percent"],
                "wind_mps": self.station["wind_mps"],
            },
        }

    def frame_age_minutes(self) -> float | None:
        """Age from the *nominal scan start* — deliberately the larger figure.

        The context document also carries a ``frame_age_minutes`` measured from
        assumed scan completion, which is 10 minutes smaller. Two different
        numbers under one name invites a quiet overclaim, so the dashboard
        publishes the conservative one and labels its basis in
        ``satellite_frame_age_basis``.
        """
        if not self.context:
            return None
        timing = self.context.get("timing", {})
        issue = _parse_utc(self.issue_time_utc, "issue_time_utc")
        nominal = _parse_utc(str(timing.get("nominal_scan_utc")), "nominal_scan_utc")
        return round((issue - nominal).total_seconds() / 60.0, 3)

    def status(self) -> dict[str, Any]:
        with self.lock:
            available = self.context is not None
            payload: dict[str, Any] = {
                "mode": MODE_FUSED_FREE if available else MODE_PERSISTENCE_ONLY,
                "forecast_method": FORECAST_METHOD,
                "satellite_used_in_prediction": False,
                "satellite_context_available": available,
                "satellite_frame_age_minutes": self.frame_age_minutes(),
                "satellite_frame_age_basis": (
                    "minutes from the nominal scan start to the issue time; the context "
                    "document's own frame_age_minutes is measured from assumed scan "
                    "completion and is therefore 10 minutes smaller"
                ),
                "design_banner": DESIGN_BANNER,
                "claim_boundary": CLAIM_BOUNDARY,
                "forecast": self.forecast(),
                "station_input": {
                    "issue_time_utc": self.issue_time_utc,
                    "raw_rows_in_bin": self.station.get("raw_rows_in_bin"),
                    "aggregation": self.station.get("aggregation"),
                    "source": self.station.get("source"),
                },
                "at_equation": AT_EQUATION,
                "host": {
                    "hostname": platform.node(),
                    "machine": platform.machine(),
                    "python": platform.python_version(),
                },
            }
            if available and self.context is not None:
                payload["satellite_context"] = {
                    "nominal_scan_utc": self.context["timing"]["nominal_scan_utc"],
                    "effective_available_utc": self.context["timing"]["effective_available_utc"],
                    "publication_lag_assumption_minutes": self.context["timing"].get(
                        "publication_lag_assumption_minutes"
                    ),
                    "bands": list(self.context.get("features", {})),
                    "decoder": self.context.get("decoder", {}).get("reader"),
                    "satpy_version": self.context.get("decoder", {}).get("satpy_version"),
                    "roi_km": self.context.get("location", {}).get("roi_km"),
                    "source_file": self.context_source,
                }
            else:
                payload["degraded_reason"] = self.degraded_reason or "satellite context unavailable"
            return payload

    def context_payload(self) -> dict[str, Any]:
        with self.lock:
            if self.context is None:
                return {
                    "satellite_context_available": False,
                    "satellite_used_in_prediction": False,
                    "mode": MODE_PERSISTENCE_ONLY,
                    "degraded_reason": self.degraded_reason or "satellite context unavailable",
                    "claim_boundary": CLAIM_BOUNDARY,
                }
            return {
                "satellite_context_available": True,
                "satellite_used_in_prediction": False,
                "mode": MODE_FUSED_FREE,
                "claim_boundary": CLAIM_BOUNDARY,
                "context": self.context,
                "source_file": self.context_source,
            }


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

_PAGE_HEAD = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>HALO SafeShift - persistence + satellite context</title>
<style>
*{box-sizing:border-box}
body{margin:0;background:#eef2f1;color:#16211d;font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif}
header{background:linear-gradient(135deg,#0f3830,#1b5c4c);color:#fff;padding:18px 22px}
header h1{margin:0;font-size:22px}header .sub{opacity:.85;font-size:13px;margin-top:3px}
main{max-width:1080px;margin:auto;padding:18px}
.badge{display:inline-block;background:#a8322d;color:#fff;font:700 11px ui-monospace,monospace;
  letter-spacing:.08em;padding:5px 10px;border-radius:3px;margin-top:8px}
.note{border-left:4px solid #d6632d;background:#fff6ec;padding:10px 13px;margin-bottom:14px;font-size:13px}
.warn{border-left-color:#a8322d;background:#fdefec}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin-bottom:16px}
.card{background:#fff;border:1px solid #dbe5e1;border-radius:6px;padding:12px 14px}
.k{font:600 10.5px ui-monospace,monospace;text-transform:uppercase;color:#5c7268;letter-spacing:.07em}
.v{font-size:26px;font-weight:700;margin-top:4px}.v small{font-size:14px;font-weight:400;color:#5c7268}
section{background:#fff;border:1px solid #dbe5e1;border-radius:6px;padding:14px;margin-bottom:14px}
h2{margin:0 0 10px;font-size:15px}
table{width:100%;border-collapse:collapse;font-size:13px}
th{background:#f4f8f6;font:600 10px ui-monospace,monospace;text-transform:uppercase;color:#5c7268;
   letter-spacing:.06em;padding:7px 9px;text-align:left;border-bottom:1px solid #dbe5e1}
td{padding:6px 9px;border-bottom:1px solid #f0f4f2}
td.n{text-align:right;font-family:ui-monospace,monospace}
td.s{font-family:ui-monospace,monospace;font-weight:700}
.eqline{font-family:ui-monospace,monospace;font-size:15px;margin-top:7px}
.mea{color:#0b6b50;font-weight:700;background:#e4f7ef;padding:1px 4px;border-radius:3px}
.con{color:#9a5312;font-weight:700;background:#fdf0e2;padding:1px 4px;border-radius:3px}
.cmp{color:#2d4f9e;font-weight:700;background:#e8eefb;padding:1px 4px;border-radius:3px}
.op{color:#8a9c94}
.dim{color:#5c7268;font-size:12px;margin-top:8px}
pre{background:#f7faf9;border:1px solid #dbe5e1;border-radius:5px;padding:10px;overflow-x:auto;
    font-size:11.5px;max-height:320px}
</style>
"""


def render_page(status: dict[str, Any]) -> str:
    """Server-side render, so the page states its mode even without JavaScript."""
    esc = html.escape
    available = bool(status["satellite_context_available"])
    forecast = status["forecast"]

    parts = [
        _PAGE_HEAD,
        "<header><h1>HALO SafeShift</h1>",
        f"<div class=\"sub\">{esc(status['mode'])}</div>",
        f"<div class=\"badge\">NOT FUSED &middot; {esc(DESIGN_BANNER)}</div></header><main>",
        f"<p class=\"note\">{esc(status['claim_boundary'])}</p>",
        "<div class=\"grid\">",
        f"<div class=\"card\"><div class=\"k\">Current AT</div><div class=\"v\">{forecast['current_at_degc']:.2f}<small> &deg;C</small></div></div>",
        f"<div class=\"card\"><div class=\"k\">Persistence forecast +{forecast['horizon_minutes']} min</div>"
        f"<div class=\"v\">{forecast['predicted_at_plus30_degc']:.2f}<small> &deg;C</small></div></div>",
        f"<div class=\"card\"><div class=\"k\">Forecast method</div><div class=\"v\" style=\"font-size:19px\">{esc(forecast['forecast_method'])}</div></div>",
        "<div class=\"card\"><div class=\"k\">Satellite in prediction</div><div class=\"v\" style=\"font-size:19px\">NO</div></div>",
        "</div>",
    ]

    # --- equation -------------------------------------------------------
    inputs = forecast["inputs"]
    vapour = (inputs["humidity_percent"] / 100.0) * 6.105 * math.exp(
        17.27 * inputs["temperature_degc"] / (237.7 + inputs["temperature_degc"])
    )
    parts.append(
        "<section><h2>Apparent temperature &mdash; exactly how this number is computed</h2>"
        "<div class=\"dim\">The equation is <b>fixed</b>: the published Bureau of Meteorology "
        "non-radiation apparent temperature, not something fitted to our data. Only the "
        "<span class=\"mea\">measured</span> symbols change.</div>"
        "<div class=\"eqline\"><span class=\"cmp\">e</span> <span class=\"op\">=</span> "
        "<span class=\"mea\">RH</span><span class=\"op\">/</span><span class=\"con\">100</span> "
        "<span class=\"op\">&times;</span> <span class=\"con\">6.105</span> "
        "<span class=\"op\">&times; exp(</span><span class=\"con\">17.27</span>"
        "<span class=\"op\">&middot;</span><span class=\"mea\">Ta</span><span class=\"op\"> / (</span>"
        "<span class=\"con\">237.7</span> <span class=\"op\">+</span> <span class=\"mea\">Ta</span>"
        "<span class=\"op\">))</span></div>"
        "<div class=\"eqline\"><span class=\"cmp\">AT</span> <span class=\"op\">=</span> "
        "<span class=\"mea\">Ta</span> <span class=\"op\">+</span> <span class=\"con\">0.33</span>"
        "<span class=\"op\">&middot;</span><span class=\"cmp\">e</span> <span class=\"op\">&minus;</span> "
        "<span class=\"con\">0.70</span><span class=\"op\">&middot;</span><span class=\"mea\">ws</span> "
        "<span class=\"op\">&minus;</span> <span class=\"con\">4.00</span></div>"
        "<table><thead><tr><th>Symbol</th><th>What it is</th><th>Unit</th><th>Kind</th>"
        "<th class=\"n\">Value now</th></tr></thead><tbody>"
    )
    rows = [
        ("Ta", "Air temperature (dry bulb)", "&deg;C", "mea", "measured &middot; SEN0658 station", f"{inputs['temperature_degc']:.2f}"),
        ("RH", "Relative humidity", "% RH", "mea", "measured &middot; SEN0658 station", f"{inputs['humidity_percent']:.2f}"),
        ("ws", "Wind speed", "m/s", "mea", "measured &middot; SEN0658 station", f"{inputs['wind_mps']:.2f}"),
        ("e", "Water-vapour pressure", "hPa", "cmp", "computed &middot; from Ta and RH", f"{vapour:.3f}"),
        ("AT", "Apparent temperature (shade estimate)", "&deg;C", "cmp", "computed &middot; the number this product reports", f"{forecast['current_at_degc']:.2f}"),
    ]
    for symbol, meaning, unit, kind, source, value in rows:
        parts.append(
            f"<tr><td class=\"s\"><span class=\"{kind}\">{symbol}</span></td><td>{esc(meaning)}</td>"
            f"<td>{unit}</td><td>{source}</td><td class=\"n\">{value}</td></tr>"
        )
    for constant in AT_EQUATION["constants"]:
        unit = str(constant["unit"]).replace("degC", "&deg;C")
        parts.append(
            f"<tr><td class=\"s\"><span class=\"con\">{constant['symbol']}</span></td>"
            f"<td>{esc(str(constant['meaning']))}</td><td>{unit}</td>"
            f"<td>constant &middot; fixed by BoM</td><td class=\"n\">{constant['symbol']}</td></tr>"
        )
    parts.append("</tbody></table>")
    parts.append(f"<div class=\"dim\">{esc(str(AT_EQUATION['wind_height_note']))}</div></section>")

    # --- satellite context ----------------------------------------------
    if available:
        context = status["satellite_context"]
        parts.append(
            "<section><h2>Satellite context &mdash; descriptive only, not used by the forecast</h2>"
            "<table><thead><tr><th>Band</th><th class=\"n\">p50 (K)</th><th class=\"n\">mean (K)</th>"
            "<th class=\"n\">std (K)</th><th class=\"n\">ROI px</th><th class=\"n\">valid</th></tr>"
            "</thead><tbody>"
        )
        parts.append("__BANDS__")
        parts.append(
            "</tbody></table>"
            f"<div class=\"dim\">Nominal scan <b>{esc(str(context['nominal_scan_utc']))}</b> &middot; "
            f"frame age <b>{status['satellite_frame_age_minutes']} min</b> &middot; "
            f"ROI {context['roi_km']} km &middot; decoder <b>{esc(str(context['decoder']))}</b> "
            f"(satpy {esc(str(context['satpy_version']))}) &middot; assumed publication lag "
            f"{context['publication_lag_assumption_minutes']} min.<br>"
            "Frame age is a measured lag. It is <b>not</b> a warning lead time, and no statistic "
            "here is a warning score, cloud class or rain indicator.</div></section>"
        )
    else:
        parts.append(
            "<section><h2>Satellite context unavailable</h2>"
            f"<p class=\"note warn\"><b>mode:</b> {esc(status['mode'])}<br>"
            f"<b>degraded_reason:</b> {esc(str(status.get('degraded_reason')))}</p>"
            "<div class=\"dim\">The persistence forecast above is unaffected: it never read a "
            "satellite value. A stale frame is refused rather than displayed.</div></section>"
        )

    parts.append(
        "<section><h2>Raw service state</h2><pre>"
        + esc(json.dumps(status, indent=2, ensure_ascii=False))
        + "</pre></section></main></html>"
    )
    return "".join(parts)


def _band_rows(status: dict[str, Any], context_document: dict[str, Any] | None) -> str:
    if not context_document:
        return ""
    rows = []
    for band, stats in (context_document.get("features") or {}).items():
        rows.append(
            f"<tr><td class=\"s\">{html.escape(str(band))}</td>"
            f"<td class=\"n\">{stats.get('p50')}</td><td class=\"n\">{stats.get('mean')}</td>"
            f"<td class=\"n\">{stats.get('std')}</td><td class=\"n\">{stats.get('roi_pixels')}</td>"
            f"<td class=\"n\">{stats.get('valid_fraction')}</td></tr>"
        )
    return "".join(rows)


def handler_for(state: DashboardState):
    class Handler(BaseHTTPRequestHandler):
        server_version = "HALOSatelliteContext/1.0"

        def reply(self, code: int, content_type: str, body: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            try:
                if path == "/healthz":
                    self.reply(200, "application/json", json.dumps({"ok": True}).encode())
                elif path == "/api/status":
                    self.reply(200, "application/json", json.dumps(state.status(), ensure_ascii=False).encode())
                elif path == "/api/context":
                    self.reply(200, "application/json", json.dumps(state.context_payload(), ensure_ascii=False).encode())
                elif path in {"/", "/index.html"}:
                    status = state.status()
                    page = render_page(status).replace("__BANDS__", _band_rows(status, state.context))
                    self.reply(200, "text/html; charset=utf-8", page.encode("utf-8"))
                else:
                    self.reply(404, "application/json", b'{"error":"not found"}')
            except Exception as exc:  # keep the service answering rather than dying mid-demo
                self.reply(500, "application/json", json.dumps({"error": repr(exc)}).encode())

        def log_message(self, *args: Any, **kwargs: Any) -> None:
            return

    return Handler


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="satellite_context_dashboard",
        description="Persistence forecast served beside descriptive satellite context (never fused).",
    )
    parser.add_argument("--satellite-context", type=Path, default=None)
    parser.add_argument("--station-input", required=True, type=Path)
    parser.add_argument("--issue-time-utc", default=None, help="required when --station-input is a CSV")
    parser.add_argument("--max-frame-age-minutes", type=float, default=DEFAULT_MAX_FRAME_AGE_MINUTES)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--dump-status", action="store_true", help="print status JSON and exit")
    args = parser.parse_args(argv)

    try:
        state = DashboardState(
            satellite_context=args.satellite_context,
            station_input=args.station_input,
            issue_time_utc=args.issue_time_utc,
            max_frame_age_minutes=args.max_frame_age_minutes,
        )
    except DashboardError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), flush=True)
        return 1

    if args.dump_status:
        print(json.dumps(state.status(), indent=2, ensure_ascii=False), flush=True)
        return 0

    server = ThreadingHTTPServer((args.host, args.port), handler_for(state))
    print(
        json.dumps(
            {
                "ready": True,
                "url": f"http://{args.host}:{args.port}",
                "mode": state.status()["mode"],
                "satellite_context_available": state.context is not None,
                "degraded_reason": state.degraded_reason,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
