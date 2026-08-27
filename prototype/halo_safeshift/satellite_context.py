"""Zero-training Himawari-9 satellite *context* for HALO SafeShift.

What this module is
-------------------
It downloads one Himawari-9 AHI HSD frame, decodes it with Satpy's ``ahi_hsd``
reference reader, and reduces a 25 km region of interest around the MakerLab
site to continuous per-band statistics.

What it is **not**
------------------
Nothing here enters a prediction. The forecast that ships alongside this
context is persistence, ``AT(t+30) = AT(t)``, and it does not read a single
satellite number. This module therefore produces *descriptive context*, and the
emitted JSON says so in both ``mode`` and ``claim_boundary``.

Three prohibitions are enforced by construction rather than by convention:

* **No 220 K threshold, no cloud/storm/rain class, no ``deep_pct``.** Only
  continuous order statistics are emitted. The 220 K threshold's provenance was
  never recorded (V-4), so nothing derived from it may appear.
* **No nearest-frame matching.** A frame is admitted only when its assumed
  availability falls *strictly* before the issue time, via the existing
  :func:`~.full_pipeline_contracts.assert_satellite_frame_is_available`.
* **No lead-time claim.** The timing block reports frame *age*, which is a
  measured lag, not an established warning horizon.

``benchmarks/halo/hsd.py`` is deliberately **not** used: that reader has never
passed a pixel-level comparison against a reference implementation, and it is
known to admit sub-180 K pixels that contaminate any statistic. Satpy is the
reference decoder for anything that reaches a dashboard.

Import cost
-----------
Satpy is imported lazily inside :func:`decode_band`, so the timing, ROI and
schema logic in this module can be unit-tested with the repository's ordinary
interpreter, which has NumPy but no Satpy.
"""

from __future__ import annotations

import argparse
import bz2
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

try:  # normal package execution
    from .full_pipeline_contracts import ContractError, assert_satellite_frame_is_available
except ImportError:  # pragma: no cover - standalone copy
    from full_pipeline_contracts import ContractError, assert_satellite_frame_is_available


__all__ = [
    "SatelliteContextError",
    "SCHEMA_VERSION",
    "SITE_LATITUDE",
    "SITE_LONGITUDE",
    "ROI_KM",
    "BANDS",
    "SEGMENTS",
    "object_key",
    "resolve_timing",
    "great_circle_km",
    "roi_mask",
    "band_statistics",
    "temporal_deltas",
    "build_context",
    "main",
]


SCHEMA_VERSION = "halo-satellite-context-v1"
MODE = "satellite-context-not-fused"

# docs/VERIFIED_FACTS.md §1: Google Maps feature for MakerLab.vn, resolved
# 2026-08-15. Not a GNSS survey at the sensor head.
SITE_LATITUDE = 10.7986848
SITE_LONGITUDE = 106.6961223

ROI_KM = 25.0
BANDS: tuple[str, ...] = ("B13", "B08")
RESOLUTION = "R20"
# Segment 04 contains the site (full-disk line ~2178.7 of 5500) and extends
# ~42 km south of it, so a nominal 25 km disc fits inside this one segment.
# A 50 km disc does NOT - that is the defect that invalidated the retracted
# 68.9 % / 0.0 % contrast (VERIFIED_FACTS §1, charter A-19).
SEGMENTS: tuple[int, ...] = (4,)
TOTAL_SEGMENTS = 10

BUCKET = "https://noaa-himawari9.s3.amazonaws.com"
PRODUCT = "AHI-L1b-FLDK"
PLATFORM = "Himawari-9"

# AHI sweeps the full disk within its 10-minute timeline, so the nominal file
# timestamp is not the instant our site was observed (JMA). We therefore assume
# the scan is complete this long after the nominal time before it can be used.
ASSUMED_SCAN_COMPLETION_MINUTES = 10
# Core assumption for the historical contract. V-5 measured ~11.8-12.0 min of
# real publication latency on the live path; 20 minutes is the conservative
# contract value, and the sensitivity list records what other choices would do.
PUBLICATION_LAG_MINUTES = 20
LAG_SENSITIVITY_MINUTES: tuple[int, ...] = (0, 10, 20, 30)

# Physical decode quality control, NOT a cloud mask and NOT a threshold rule.
# AHI infrared brightness temperatures outside this interval indicate a decode
# or calibration failure rather than a cold cloud top.
QC_MIN_KELVIN = 180.0
QC_MAX_KELVIN = 330.0

# Mean Earth radius (IUGG). Over a 25 km disc the difference between this
# great-circle distance and a WGS84 geodesic is under 0.3 % (~75 m), an order
# of magnitude below the 2 km pixel, so no projection library is required.
EARTH_RADIUS_KM = 6371.0088

CLAIM_BOUNDARY = (
    "Descriptive satellite context only; not fused inference or established warning lead time. "
    "The forecast shown beside it is persistence and does not read any satellite value."
)

class SatelliteContextError(RuntimeError):
    """A satellite context input, download, decode or ROI is unusable."""


# --------------------------------------------------------------------------
# timing
# --------------------------------------------------------------------------


def _parse_utc(value: str | datetime, label: str) -> datetime:
    """Parse an explicit-UTC timestamp. Timezone-naive input is rejected."""
    if isinstance(value, datetime):
        moment = value
    else:
        text = str(value).strip()
        try:
            moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError as exc:
            raise SatelliteContextError(f"{label} is not an ISO-8601 timestamp: {value!r}") from exc
    if moment.tzinfo is None:
        raise SatelliteContextError(
            f"{label} has no timezone. Satellite timing must be explicit UTC; a naive "
            f"timestamp silently becomes local time and would admit a post-issue frame."
        )
    return moment.astimezone(timezone.utc)


def resolve_timing(
    *,
    nominal_scan_utc: str | datetime,
    issue_time_utc: str | datetime,
    publication_lag_minutes: int = PUBLICATION_LAG_MINUTES,
    scan_completion_minutes: int = ASSUMED_SCAN_COMPLETION_MINUTES,
    lag_sensitivity_minutes: Sequence[int] = LAG_SENSITIVITY_MINUTES,
) -> dict[str, Any]:
    """Admit a frame only when it is available strictly before the issue time.

    Delegates the decision to the existing P1 contract helper so this module
    cannot drift from the rule the rest of the pipeline is held to.
    """
    nominal = _parse_utc(nominal_scan_utc, "nominal_scan_utc")
    issue = _parse_utc(issue_time_utc, "issue_time_utc")
    if scan_completion_minutes < 0:
        raise SatelliteContextError("scan completion offset cannot be negative")
    completed = nominal + timedelta(minutes=scan_completion_minutes)

    try:
        verdict = assert_satellite_frame_is_available(
            nominal_scan_utc=nominal,
            assumed_scan_completion_utc=completed,
            issue_time_utc=issue,
            publication_lag_minutes=publication_lag_minutes,
        )
    except ContractError as exc:
        raise SatelliteContextError(
            f"frame {nominal.isoformat()} is not usable for issue {issue.isoformat()}: {exc}"
        ) from exc

    effective = completed + timedelta(minutes=publication_lag_minutes)
    sensitivity = []
    for candidate in lag_sensitivity_minutes:
        available_at = completed + timedelta(minutes=int(candidate))
        sensitivity.append(
            {
                "publication_lag_minutes": int(candidate),
                "effective_available_utc": available_at.isoformat(),
                "available_strictly_before_issue": available_at < issue,
            }
        )

    return {
        "nominal_scan_utc": nominal.isoformat(),
        "assumed_scan_completion_minutes": scan_completion_minutes,
        "assumed_scan_completion_utc": completed.isoformat(),
        "issue_time_utc": issue.isoformat(),
        "publication_lag_assumption_minutes": int(publication_lag_minutes),
        "effective_available_utc": effective.isoformat(),
        "available_strictly_before_issue": True,
        "frame_age_minutes": round(float(verdict["frame_age_minutes"]), 3),
        "nominal_to_issue_minutes": round((issue - nominal).total_seconds() / 60.0, 3),
        "publication_lag_sensitivity": sensitivity,
        "scan_timing_note": (
            "AHI sweeps the disc within its 10-minute timeline, so the nominal timestamp is "
            "not the instant this site was observed. Frame age is a measured lag, not a "
            "warning lead time."
        ),
    }


# --------------------------------------------------------------------------
# download
# --------------------------------------------------------------------------


def object_key(band: str, segment: int, nominal: datetime) -> str:
    """S3 key for one HSD segment, e.g. ``.../HS_H09_..._B13_FLDK_R20_S0410.DAT.bz2``."""
    moment = _parse_utc(nominal, "nominal_scan_utc")
    day = moment.strftime("%Y%m%d")
    hhmm = moment.strftime("%H%M")
    return (
        f"{PRODUCT}/{moment:%Y/%m/%d}/{hhmm}/"
        f"HS_H09_{day}_{hhmm}_{band}_FLDK_{RESOLUTION}_S{segment:02d}{TOTAL_SEGMENTS:02d}.DAT.bz2"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_object(key: str, cache: Path, *, timeout: int = 180, attempts: int = 4) -> dict[str, Any]:
    """Fetch one object into ``cache``, resuming a partial download if present.

    The compressed object stays in the cache and never enters git; only its
    SHA-256 and byte size travel into the evidence record.
    """
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / key.replace("/", "_")
    partial = target.with_suffix(target.suffix + ".part")
    url = f"{BUCKET}/{key}"

    started = time.perf_counter()
    if not target.is_file():
        for attempt in range(1, attempts + 1):
            have = partial.stat().st_size if partial.is_file() else 0
            request = urllib.request.Request(url)
            if have:
                request.add_header("Range", f"bytes={have}-")
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    mode = "ab" if have and response.status == 206 else "wb"
                    if mode == "wb":
                        have = 0
                    with partial.open(mode) as handle:
                        while True:
                            chunk = response.read(1024 * 256)
                            if not chunk:
                                break
                            handle.write(chunk)
                break
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                if attempt == attempts:
                    raise SatelliteContextError(
                        f"cannot download {url} after {attempts} attempts: {exc!r}. "
                        f"This is a real blocker; a synthetic frame must not be substituted."
                    ) from exc
                time.sleep(min(2 ** attempt, 15))
        os.replace(partial, target)
    elapsed = time.perf_counter() - started

    size = target.stat().st_size
    if size <= 0:
        raise SatelliteContextError(f"{url} produced an empty object")
    return {
        "key": key,
        "url": url,
        "cached_path": str(target),
        "bytes": size,
        "sha256": _sha256_file(target),
        "download_seconds": round(elapsed, 3),
    }


def _decompress(record: dict[str, Any], workdir: Path) -> Path:
    """Expand a cached ``.bz2`` object to a ``.DAT`` Satpy can open.

    The cache flattens the S3 key by replacing separators, which destroys the
    ``HS_H09_<date>_<time>_<band>_FLDK_<res>_S<seg><tot>.DAT`` name that Satpy's
    ``ahi_hsd`` file pattern matches on. So the decompressed copy is written
    under the object's *original* basename, not the flattened cache name.
    """
    source = Path(record["cached_path"])
    basename = str(record["key"]).rsplit("/", 1)[-1]
    target = workdir / (basename[: -len(".bz2")] if basename.endswith(".bz2") else basename + ".DAT")
    if not target.is_file():
        try:
            target.write_bytes(bz2.decompress(source.read_bytes()))
        except (OSError, ValueError) as exc:
            raise SatelliteContextError(f"cannot decompress {source.name}: {exc}") from exc
    return target


# --------------------------------------------------------------------------
# decode + ROI
# --------------------------------------------------------------------------


def decode_band(paths: Sequence[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Decode HSD segments with Satpy and return ``(bt_kelvin, lons, lats, meta)``.

    Satpy's ``ahi_hsd`` reader is the reference decoder. It is imported here so
    the rest of this module stays testable without it.
    """
    try:
        from satpy import Scene
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise SatelliteContextError(
            "Satpy is not importable. It is the required reference decoder for any "
            "satellite value that reaches a dashboard; benchmarks/halo/hsd.py is not "
            "an acceptable substitute because it has never passed a pixel-level "
            "reference comparison."
        ) from exc

    scene = Scene(filenames=[str(p) for p in paths], reader="ahi_hsd")
    available = list(scene.available_dataset_names())
    if not available:
        raise SatelliteContextError(f"Satpy exposed no dataset for {[p.name for p in paths]}")
    name = available[0]
    scene.load([name], calibration="brightness_temperature")
    data = scene[name]
    values = np.asarray(data.values, dtype=np.float64)
    area = data.attrs.get("area")
    if area is None:
        raise SatelliteContextError(f"Satpy returned no navigation area for {name}")
    lons, lats = area.get_lonlats()
    meta = {
        "satpy_dataset": name,
        "calibration": "brightness_temperature",
        "units": str(data.attrs.get("units", "K")),
        "wavelength_um": _wavelength_of(data),
        "shape": [int(values.shape[0]), int(values.shape[1])],
        "start_time_utc": _isoformat_or_none(data.attrs.get("start_time")),
        "end_time_utc": _isoformat_or_none(data.attrs.get("end_time")),
    }
    return values, np.asarray(lons, dtype=np.float64), np.asarray(lats, dtype=np.float64), meta


def _wavelength_of(data: Any) -> float | None:
    wavelength = data.attrs.get("wavelength")
    central = getattr(wavelength, "central", None)
    if central is not None:
        return float(central)
    if isinstance(wavelength, (list, tuple)) and len(wavelength) == 3:
        return float(wavelength[1])
    return None


def _isoformat_or_none(value: Any) -> str | None:
    if isinstance(value, datetime):
        moment = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return moment.astimezone(timezone.utc).isoformat()
    return None


def great_circle_km(lats: np.ndarray, lons: np.ndarray, site_lat: float, site_lon: float) -> np.ndarray:
    """Great-circle distance in km from every pixel to the site (haversine)."""
    phi1 = np.radians(lats)
    phi2 = np.radians(site_lat)
    dphi = phi2 - phi1
    dlam = np.radians(site_lon - lons)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def roi_mask(
    lats: np.ndarray,
    lons: np.ndarray,
    *,
    site_lat: float = SITE_LATITUDE,
    site_lon: float = SITE_LONGITUDE,
    radius_km: float = ROI_KM,
) -> np.ndarray:
    """Boolean mask of pixels within ``radius_km``; fail closed if empty or clipped.

    "Clipped" means the disc reaches the edge of the loaded array, i.e. the
    downloaded segment does not contain the whole region. That is precisely the
    defect that invalidated the retracted 50 km statistic, so it is a hard
    error rather than a warning.
    """
    if lats.shape != lons.shape:
        raise SatelliteContextError("navigation latitude/longitude grids differ in shape")
    if radius_km <= 0:
        raise SatelliteContextError(f"ROI radius must be positive, got {radius_km!r}")

    finite_navigation = np.isfinite(lats) & np.isfinite(lons)
    distance = np.full(lats.shape, np.inf, dtype=np.float64)
    distance[finite_navigation] = great_circle_km(
        lats[finite_navigation], lons[finite_navigation], site_lat, site_lon
    )
    mask = distance <= radius_km

    count = int(mask.sum())
    if count == 0:
        raise SatelliteContextError(
            f"ROI is empty: no navigated pixel lies within {radius_km} km of "
            f"{site_lat}, {site_lon}. The requested segment does not cover the site."
        )
    if mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any():
        raise SatelliteContextError(
            f"ROI is clipped: the {radius_km} km disc reaches the edge of the loaded "
            f"segment, so part of the region was never downloaded. Statistics over a "
            f"clipped ring describe an area that does not exist in the data."
        )
    return mask


def band_statistics(values: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    """Continuous ROI statistics for one band. No thresholds, no classes."""
    roi_pixels = int(mask.sum())
    if roi_pixels == 0:
        raise SatelliteContextError("cannot summarise an empty ROI")

    selected = values[mask]
    finite = selected[np.isfinite(selected)]
    finite_pixels = int(finite.size)

    physical = finite[(finite >= QC_MIN_KELVIN) & (finite <= QC_MAX_KELVIN)]
    removed = finite_pixels - int(physical.size)
    if physical.size == 0:
        raise SatelliteContextError(
            f"every ROI pixel fell outside the physical decode range "
            f"[{QC_MIN_KELVIN}, {QC_MAX_KELVIN}] K; the decode is not trustworthy"
        )

    percentiles = np.percentile(physical, [10.0, 25.0, 50.0, 75.0, 90.0])
    ordered: dict[str, float] = {
        "min": float(np.min(physical)),
        "p10": float(percentiles[0]),
        "p25": float(percentiles[1]),
        "p50": float(percentiles[2]),
        "p75": float(percentiles[3]),
        "p90": float(percentiles[4]),
        "max": float(np.max(physical)),
        "mean": float(np.mean(physical)),
        "std": float(np.std(physical, ddof=0)),
    }
    stats: dict[str, Any] = {key: round(value, 4) for key, value in ordered.items()}
    stats.update(
        {
            "unit": "K",
            "roi_pixels": roi_pixels,
            "finite_pixels": finite_pixels,
            "physical_qc_removed_pixels": removed,
            "valid_fraction": round(float(physical.size) / float(roi_pixels), 6),
            "physical_qc_range_kelvin": [QC_MIN_KELVIN, QC_MAX_KELVIN],
            "physical_qc_note": (
                "Pixels outside this range are removed as a physical decode quality "
                "control. This is not a cloud threshold and yields no cloud, storm or "
                "rain classification."
            ),
        }
    )
    return stats


def temporal_deltas(
    current: dict[str, dict[str, Any]],
    priors: Sequence[dict[str, Any]],
    *,
    nominal_scan_utc: str,
) -> dict[str, dict[str, float]]:
    """Add ``delta30``/``delta60``/``trend90`` where a prior frame really exists.

    A missing prior yields a missing key, never a fabricated zero.
    """
    now = _parse_utc(nominal_scan_utc, "nominal_scan_utc")
    by_offset: dict[int, dict[str, dict[str, Any]]] = {}
    for prior in priors:
        stamp = (prior.get("timing") or {}).get("nominal_scan_utc")
        features = prior.get("features") or {}
        if not stamp or not features:
            continue
        offset = round((now - _parse_utc(stamp, "prior nominal_scan_utc")).total_seconds() / 60.0)
        if offset > 0:
            by_offset[int(offset)] = features

    deltas: dict[str, dict[str, float]] = {}
    for band, stats in current.items():
        p50 = stats.get("p50")
        if p50 is None:
            continue
        entry: dict[str, float] = {}
        for label, offset in (("delta30", 30), ("delta60", 60)):
            prior_stats = by_offset.get(offset, {}).get(band)
            if prior_stats and prior_stats.get("p50") is not None:
                entry[label] = round(float(p50) - float(prior_stats["p50"]), 4)
        prior90 = by_offset.get(90, {}).get(band)
        if prior90 and prior90.get("p50") is not None:
            entry["trend90"] = round((float(p50) - float(prior90["p50"])) / 90.0, 6)
        if entry:
            deltas[band] = entry
    return deltas


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------


def _satpy_version() -> str:
    try:
        import satpy

        return str(satpy.__version__)
    except ImportError:  # pragma: no cover
        return "unavailable"


def build_context(
    *,
    nominal_scan_utc: str,
    issue_time_utc: str,
    cache: Path,
    workdir: Path,
    bands: Sequence[str] = BANDS,
    segments: Sequence[int] = SEGMENTS,
    roi_km: float = ROI_KM,
    publication_lag_minutes: int = PUBLICATION_LAG_MINUTES,
    priors: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Download, decode and reduce one frame into a context document.

    Timing is resolved *before* any byte is downloaded, so an inadmissible
    frame costs nothing.
    """
    timing = resolve_timing(
        nominal_scan_utc=nominal_scan_utc,
        issue_time_utc=issue_time_utc,
        publication_lag_minutes=publication_lag_minutes,
    )
    nominal = _parse_utc(nominal_scan_utc, "nominal_scan_utc")

    if not bands:
        raise SatelliteContextError("at least one band is required")
    workdir.mkdir(parents=True, exist_ok=True)

    objects: list[dict[str, Any]] = []
    features: dict[str, dict[str, Any]] = {}
    decoders: dict[str, Any] = {}
    started = time.perf_counter()

    for band in bands:
        paths: list[Path] = []
        for segment in segments:
            record = download_object(object_key(band, segment, nominal), cache)
            record["band"] = band
            record["segment"] = int(segment)
            objects.append(record)
            paths.append(_decompress(record, workdir))

        decode_started = time.perf_counter()
        values, lons, lats, meta = decode_band(paths)
        meta["decode_seconds"] = round(time.perf_counter() - decode_started, 3)

        mask = roi_mask(lats, lons, radius_km=roi_km)
        features[band] = band_statistics(values, mask)
        decoders[band] = meta

    missing = [band for band in bands if band not in features]
    if missing:
        raise SatelliteContextError(f"required band(s) missing from the context: {missing}")

    deltas = temporal_deltas(features, priors, nominal_scan_utc=timing["nominal_scan_utc"])
    for band, entry in deltas.items():
        features[band].update(entry)

    return {
        "schema_version": SCHEMA_VERSION,
        "mode": MODE,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "platform": PLATFORM,
            "product": PRODUCT,
            "bucket": BUCKET,
            "bands": list(bands),
            "resolution": RESOLUTION,
            "segments": [int(s) for s in segments],
            "total_segments": TOTAL_SEGMENTS,
        },
        "location": {
            "latitude": SITE_LATITUDE,
            "longitude": SITE_LONGITUDE,
            "roi_km": roi_km,
            "coordinate_basis": (
                "Google Maps feature for MakerLab.vn resolved 2026-08-15; not a GNSS "
                "survey at the sensor head."
            ),
            "distance_model": (
                f"great-circle on a sphere of radius {EARTH_RADIUS_KM} km; under 0.3 % from a "
                f"WGS84 geodesic at {roi_km} km, far below the 2 km pixel"
            ),
        },
        "timing": timing,
        "features": features,
        "temporal_deltas_available": bool(deltas),
        "objects": objects,
        "decoder": {
            "reader": "satpy.ahi_hsd",
            "satpy_version": _satpy_version(),
            "numpy_version": np.__version__,
            "per_band": decoders,
            "why_not_repo_reader": (
                "benchmarks/halo/hsd.py has never passed a pixel-level reference "
                "comparison and admits sub-180 K pixels, so it may not produce any "
                "value that reaches a dashboard."
            ),
        },
        "total_seconds": round(time.perf_counter() - started, 3),
        "satellite_used_in_prediction": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def write_atomic(path: Path, payload: dict[str, Any]) -> str:
    """Write JSON atomically and return its SHA-256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_priors(paths: Iterable[Path]) -> list[dict[str, Any]]:
    priors: list[dict[str, Any]] = []
    for path in paths:
        try:
            document = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SatelliteContextError(f"cannot read prior context {path}: {exc}") from exc
        if document.get("schema_version") != SCHEMA_VERSION:
            raise SatelliteContextError(f"prior context {path} has an unexpected schema_version")
        priors.append(document)
    return priors


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="satellite_context",
        description="Zero-training Himawari-9 satellite context (never fused into a prediction).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    historical = sub.add_parser("historical", help="extract one historical frame")
    historical.add_argument("--nominal-scan-utc", required=True)
    historical.add_argument("--issue-time-utc", required=True)
    historical.add_argument("--output", required=True, type=Path)
    historical.add_argument("--cache", required=True, type=Path)
    historical.add_argument("--roi-km", type=float, default=ROI_KM)
    historical.add_argument("--publication-lag-minutes", type=int, default=PUBLICATION_LAG_MINUTES)
    historical.add_argument("--bands", nargs="+", default=list(BANDS))
    historical.add_argument("--segments", nargs="+", type=int, default=list(SEGMENTS))
    historical.add_argument("--prior", action="append", type=Path, default=[])

    args = parser.parse_args(argv)
    try:
        context = build_context(
            nominal_scan_utc=args.nominal_scan_utc,
            issue_time_utc=args.issue_time_utc,
            cache=args.cache,
            workdir=args.cache / "decompressed",
            bands=args.bands,
            segments=args.segments,
            roi_km=args.roi_km,
            publication_lag_minutes=args.publication_lag_minutes,
            priors=_load_priors(args.prior),
        )
    except SatelliteContextError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), flush=True)
        return 1

    digest = write_atomic(args.output, context)
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(args.output),
                "sha256": digest,
                "mode": context["mode"],
                "frame_age_minutes": context["timing"]["frame_age_minutes"],
                "bands": list(context["features"]),
                "total_seconds": context["total_seconds"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
