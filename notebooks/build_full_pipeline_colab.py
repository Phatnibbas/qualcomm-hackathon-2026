"""Generate the self-contained HALO SafeShift full Colab notebook.

The generator writes notebook JSON only.  It never downloads imagery or trains
on Windows; the generated notebook performs those operations in Colab.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "HALO_SafeShift_Full_Pipeline.ipynb"
RUNTIME_SOURCE = (ROOT / "prototype" / "halo_safeshift" / "full_runtime.py").read_text(encoding="utf-8")
DASHBOARD_SOURCE = (ROOT / "prototype" / "halo_safeshift" / "full_dashboard.py").read_text(encoding="utf-8")


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}


INTRO = r'''# HALO SafeShift — Full Colab Pipeline

This notebook is the **only** place for heavy satellite preprocessing, model fitting,
quantization and pruning. It must run in Colab, never on Windows.

## Fixed contract

- CSV: `VFCD_3rd_landscape_20260816-0336_74700rows.csv`, SHA-256
  `c79ddcbf039d85211ef24c99c17c6e4fd506422345b6e0016a8f2a44dbe82106`.
- target: station-derived shade apparent-temperature estimate at `t + 30 min`.
- chronological 60/20/20 split with a 90-minute embargo; no random split/shuffle.
- station-only, satellite-only and fused are scored on the same common-row cohort.
- Satellite source is NOAA public Himawari-9 AHI HSD, B13+B08, R20, S04+S05.
- Satpy `ahi_hsd` is the reference decoder. The historical custom decoder is not used
  for features. If Satpy validation fails, satellite/fused are BLOCKED; the full
  station-only matrix continues.

## Claim boundary

Retrospective chronological holdout only; **not prospective certification**. This is
not WBGT, medical prediction, a legal safety limit, or direct-sun worker exposure.
No NPU/QNN/Hexagon/GPU inference claim is produced.
'''


SETUP = r'''# 1. Environment, Drive cache and immutable station input
import ast, bz2, csv, hashlib, importlib.metadata, io, json, math, os, platform
import shutil, subprocess, sys, time, zipfile
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

EXPECTED_CSV = "VFCD_3rd_landscape_20260816-0336_74700rows.csv"
EXPECTED_SHA256 = "c79ddcbf039d85211ef24c99c17c6e4fd506422345b6e0016a8f2a44dbe82106"
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-full-colab"
SEED = 20260816
np_rng_seed = SEED
SATELLITE_GATE = {"status": "not_run", "reason": "Satpy reference validation is required"}

# NumPy/scikit-learn must be validated before pip touches the environment. A
# live upgrade can leave Python files from one NumPy version paired with an
# already-loaded C extension from another, producing missing numpy._core.umath
# symbols. Keep Colab's coherent scientific stack untouched.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from pyproj import Geod
try:
    import sklearn
    from sklearn.linear_model import Ridge
except (ImportError, AttributeError) as exc:
    raise RuntimeError(
        "The Colab NumPy/scikit-learn ABI is inconsistent. Restart the runtime, then rerun this notebook from Cell 1. "
        "Existing Drive HSD/cache outputs are resumable; do not pip-upgrade NumPy in this live kernel. Original error: %r" % (exc,)
    ) from exc

def pip_install_satellite_only(*packages):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade-strategy", "only-if-needed", *packages])

# These are satellite I/O/geometry packages only. Without --upgrade, pip does
# not replace an already-satisfied Colab NumPy or scikit-learn installation.
pip_install_satellite_only("pyarrow", "satpy", "pyresample", "pyproj")

np.random.seed(SEED)
try:
    from google.colab import drive, files
    drive.mount("/content/drive", force_remount=False)
    DRIVE_ROOT = Path("/content/drive/MyDrive/HALO_SafeShift")
except Exception:
    DRIVE_ROOT = Path("HALO_SafeShift")
DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
CACHE = DRIVE_ROOT / "himawari_cache"
CACHE.mkdir(parents=True, exist_ok=True)
SATELLITE_CHECKPOINT = DRIVE_ROOT / "satellite_feature_checkpoints" / (EXPECTED_SHA256[:12] + "-b13b08-s04-roi25-v1")
SATELLITE_CHECKPOINT.mkdir(parents=True, exist_ok=True)
OUT = DRIVE_ROOT / "runs" / RUN_ID
OUT.mkdir(parents=True, exist_ok=False)

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def locate_csv():
    found = list(Path(".").glob("**/" + EXPECTED_CSV)) + list(DRIVE_ROOT.glob("**/" + EXPECTED_CSV))
    unique = list(dict.fromkeys(found))
    if unique:
        return unique[0]
    if "files" not in globals():
        raise FileNotFoundError("Upload the exact frozen station CSV to Colab or Google Drive")
    uploaded = files.upload()
    if EXPECTED_CSV not in uploaded:
        raise RuntimeError("Upload exactly " + EXPECTED_CSV)
    return Path(EXPECTED_CSV)

CSV_PATH = locate_csv()
if sha256_file(CSV_PATH) != EXPECTED_SHA256:
    raise RuntimeError("Station CSV SHA-256 mismatch: refuse to train on another dataset")
ENV = {
    "run_id": RUN_ID, "csv": CSV_PATH.name, "csv_sha256": EXPECTED_SHA256,
    "python": sys.version.split()[0], "platform": platform.platform(),
    "packages": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scikit-learn", "xgboost", "satpy", "pyproj"]},
    "training_location": "Google Colab only",
    "claim_boundary": "retrospective chronological holdout; not prospective certification",
}
(OUT / "environment.json").write_text(json.dumps(ENV, indent=2) + "\n")
print(json.dumps(ENV, indent=2))
'''


STATION = r'''# 2. Station QC, target and fixed chronological cohort
ALIASES = {
    "timestamp": ["timestamp_utc_iso", "created_at", "timestamp"],
    "wind_speed": ["Wind speed (m/s)", "WindSpeed", "wind_speed"],
    "wind_direction": ["Wind direction (°)", "WindDirection", "wind_direction"],
    "temperature": ["Temperature (°C)", "Temperature", "temperature"],
    "pressure": ["Pressure (kPa)", "Pressure", "pressure"],
    "light": ["Light (lux)", "Light", "light"],
    "humidity": ["Humidity (%RH)", "Humidity", "humidity"],
    "noise": ["Sound level (dB)", "Noise", "noise"],
    "pm25": ["PM2.5 (µg/m³)", "PM2.5", "pm25"],
}

def resolve_columns(frame):
    result = {}
    for name, choices in ALIASES.items():
        hits = [choice for choice in choices if choice in frame.columns]
        if not hits:
            raise ValueError("missing %s; available=%s" % (name, list(frame.columns)))
        result[name] = hits[0]
    return result

raw = pd.read_csv(CSV_PATH, low_memory=False)
if len(raw) != 74700:
    raise ValueError("Expected exactly 74,700 raw rows, got %d" % len(raw))
columns = resolve_columns(raw)
work = pd.DataFrame({"timestamp": pd.to_datetime(raw[columns["timestamp"]], utc=True, errors="coerce")})
for name in ALIASES:
    if name != "timestamp":
        work[name] = pd.to_numeric(raw[columns[name]], errors="coerce")
work = work.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp")
valid = (
    work.temperature.between(0.01, 60.0) & work.humidity.between(1.01, 100.0)
    & work.wind_speed.between(0.0, 40.0) & work.pressure.between(80.0, 120.0)
    & work.wind_direction.between(0.0, 360.0, inclusive="left")
    & work[["light", "noise", "pm25"]].notna().all(axis=1)
)
qc = {"raw_rows": int(len(raw)), "timestamp_valid_unique_rows": int(len(work)), "rows_rejected": int((~valid).sum())}
if qc["rows_rejected"] != 116:
    raise RuntimeError("Expected 116 QC-rejected rows, got %d" % qc["rows_rejected"])
work = work.loc[valid].set_index("timestamp")
numeric = ["wind_speed", "wind_direction", "temperature", "pressure", "light", "humidity", "noise", "pm25"]
bins = work[numeric].resample("5min", label="right", closed="right").median()
bins["raw_count"] = work.temperature.resample("5min", label="right", closed="right").count()
bins = bins.loc[bins.raw_count >= 6].dropna()
if len(bins) != 6815:
    raise RuntimeError("Expected 6,815 valid 5-minute bins, got %d" % len(bins))

def apparent_temperature(temp_c, rh, wind):
    e = (rh / 100.0) * 6.105 * np.exp(17.27 * temp_c / (237.7 + temp_c))
    return temp_c + 0.33 * e - 0.70 * wind - 4.0

bins["at"] = apparent_temperature(bins.temperature, bins.humidity, bins.wind_speed)
rad = np.deg2rad(bins.wind_direction.to_numpy())
bins["wind_direction_sin"] = np.sin(rad)
bins["wind_direction_cos"] = np.cos(rad)
bins["light_log1p"] = np.log1p(np.maximum(bins.light, 0.0))
bins["pm25_log1p"] = np.log1p(np.maximum(bins.pm25, 0.0))
PER_STEP = ["temperature", "humidity", "wind_speed", "wind_direction_sin", "wind_direction_cos", "light_log1p", "pressure", "pm25_log1p", "noise", "at"]
STATION_FEATURE_NAMES = ["station_tminus_%02dmin_%s" % (55 - 5*i, name) for i in range(12) for name in PER_STEP]
STATION_FEATURE_NAMES += ["station_at_now", "station_local_time_sin", "station_local_time_cos", "station_day_of_year_sin", "station_day_of_year_cos"]

def build_station_windows(frame):
    rows = []
    for i in range(11, len(frame) - 6):
        span = frame.index[i + 6] - frame.index[i - 11]
        if span != pd.Timedelta(minutes=85):
            continue
        block = frame.iloc[i - 11:i + 1]
        # Do not compare DatetimeIndex.asi8 with Timedelta.value: pandas may
        # choose microsecond (rather than nanosecond) datetime resolution in
        # Colab.  Direct Timedelta comparison is resolution-independent.
        if not block.index.to_series().diff().iloc[1:].eq(pd.Timedelta(minutes=5)).all():
            continue
        issue = frame.index[i]
        target = frame.index[i + 6]
        local = issue.tz_convert("Asia/Ho_Chi_Minh")
        minute = local.hour * 60 + local.minute
        extras = [frame.iloc[i]["at"], math.sin(2*math.pi*minute/1440), math.cos(2*math.pi*minute/1440), math.sin(2*math.pi*local.dayofyear/366), math.cos(2*math.pi*local.dayofyear/366)]
        rows.append({"issue_time_utc": issue, "target_time_utc": target, "window_start_utc": frame.index[i-11], "y_direct": float(frame.iloc[i+6]["at"]), "at_now": float(frame.iloc[i]["at"]), "station_vector": np.concatenate([block[PER_STEP].to_numpy(dtype=np.float64).reshape(-1), extras])})
    return pd.DataFrame(rows)

windows = build_station_windows(bins)
if len(windows) != 6628:
    raise RuntimeError("Expected 6,628 eligible windows, got %d" % len(windows))
if any(len(v) != 125 for v in windows.station_vector):
    raise RuntimeError("Station schema must contain 125 features")
windows["y_residual"] = windows.y_direct - windows.at_now
# P0's fixed chronological boundaries.  Do not recalculate cut points by row
# count after satellite filtering: every arm inherits these dates, then common
# cohort filtering happens inside each already-labelled split.
P0_BOUNDARIES = {
    "train_last_issue_utc": pd.Timestamp("2026-08-05T04:35:00Z"),
    "validation_first_issue_utc": pd.Timestamp("2026-08-05T07:40:00Z"),
    "validation_last_issue_utc": pd.Timestamp("2026-08-09T22:05:00Z"),
    "test_first_issue_utc": pd.Timestamp("2026-08-10T01:10:00Z"),
}
windows["split"] = "embargo"
windows.loc[windows.issue_time_utc <= P0_BOUNDARIES["train_last_issue_utc"], "split"] = "train"
windows.loc[(windows.issue_time_utc >= P0_BOUNDARIES["validation_first_issue_utc"]) & (windows.issue_time_utc <= P0_BOUNDARIES["validation_last_issue_utc"]), "split"] = "validation"
windows.loc[windows.issue_time_utc >= P0_BOUNDARIES["test_first_issue_utc"], "split"] = "test"
for split in ("train", "validation", "test"):
    subset = windows[windows.split == split]
    if subset.empty:
        raise RuntimeError("empty split " + split)
    # UTC midnight is not a split boundary. A 60-minute history may begin on
    # the preceding day; the P0 embargo dates are the actual split boundary.
    if split == "train" and not (subset.target_time_utc < P0_BOUNDARIES["validation_first_issue_utc"]).all():
        raise RuntimeError("train target enters validation/embargo period")
    if split == "validation" and (not (subset.window_start_utc > P0_BOUNDARIES["train_last_issue_utc"]).all() or not (subset.target_time_utc < P0_BOUNDARIES["test_first_issue_utc"]).all()):
        raise RuntimeError("validation sequence crosses the fixed P0 boundary")
    if split == "test" and not (subset.window_start_utc > P0_BOUNDARIES["validation_last_issue_utc"]).all():
        raise RuntimeError("test sequence enters validation/embargo period")
split_report = {name: {"n": int((windows.split == name).sum()), "first_issue_utc": windows.loc[windows.split == name, "issue_time_utc"].min().isoformat(), "last_target_utc": windows.loc[windows.split == name, "target_time_utc"].max().isoformat()} for name in ("train", "validation", "test")}
qc.update({"valid_5min_bins": int(len(bins)), "eligible_windows": int(len(windows)), "station_feature_count": len(STATION_FEATURE_NAMES), "split": split_report})
(OUT / "qc_report.json").write_text(json.dumps(qc, indent=2, default=str) + "\n")
(OUT / "split.json").write_text(json.dumps({"policy": "fixed P0 chronological 60/20/20 with 90-minute embargo", "fixed_boundaries_utc": {key: value.isoformat() for key, value in P0_BOUNDARIES.items()}, "splits": split_report, "evaluation_status": "retrospective chronological holdout; not prospective certification"}, indent=2) + "\n")
print(json.dumps(qc, indent=2, default=str))
'''


SATELLITE = r'''# 3. NOAA cache, Satpy reference gate, exact-coordinate continuous features
LAT, LON = 10.7986848, 106.6961223
BANDS = ("B13", "B08")
REFERENCE_SEGMENTS = (4, 5)
HISTORICAL_SEGMENTS = (4,)
HISTORICAL_ROI_KM = 25
RESOLUTION = "R20"
SCAN_COMPLETION_MINUTES = 10  # declared timing assumption, not measured publication latency
LAG_SENSITIVITY_MINUTES = (0, 10, 20, 30)
NOAA = "https://noaa-himawari9.s3.amazonaws.com"
GEOD = Geod(ellps="WGS84")
OBJECTS = []

def hsd_key(stamp, band, segment):
    return "AHI-L1b-FLDK/{:%Y/%m/%d/%H%M}/HS_H09_{:%Y%m%d_%H%M}_{}_FLDK_{}_S{:02d}10.DAT.bz2".format(stamp, stamp, band, RESOLUTION, segment)

def download_resumable(key):
    target = CACHE / key
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size:
        return target
    partial = target.with_suffix(target.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": "bytes=%d-" % offset} if offset else {}
    response = requests.get(NOAA + "/" + key, headers=headers, stream=True, timeout=(20, 300))
    if response.status_code == 416 and partial.exists():
        partial.rename(target); return target
    response.raise_for_status()
    mode = "ab" if response.status_code == 206 and offset else "wb"
    with open(partial, mode) as handle:
        for chunk in response.iter_content(1024 * 1024):
            if chunk: handle.write(chunk)
    partial.rename(target)
    return target

def read_block7_metadata(path):
    raw = bz2.decompress(Path(path).read_bytes())
    offsets, cursor = {}, 0
    for _ in range(11):
        number = raw[cursor]; length = int.from_bytes(raw[cursor+1:cursor+3], "little")
        offsets[number] = cursor; cursor += length
    base = offsets[7]
    total, segment = raw[base+3], raw[base+4]
    first_line = int.from_bytes(raw[base+5:base+7], "little")
    return {"segment": int(segment), "total_segments": int(total), "first_line": int(first_line), "data_start": cursor}

def load_reference_scene(paths, bands):
    from satpy import Scene
    scene = Scene(filenames=[str(p) for p in paths], reader="ahi_hsd")
    scene.load(list(bands), calibration="brightness_temperature")
    return scene

def available_before(issue, frames, lag):
    possible = []
    for stamp in frames:
        completed = stamp + pd.Timedelta(minutes=SCAN_COMPLETION_MINUTES)
        if completed + pd.Timedelta(minutes=lag) < issue:
            possible.append(stamp)
    return max(possible) if possible else None

def roi_mask(data_array, radius_km):
    area = data_array.attrs["area"]
    lons, lats = area.get_lonlats()
    _, _, meters = GEOD.inv(np.full_like(lons, LON), np.full_like(lats, LAT), lons, lats)
    mask = np.isfinite(meters) & (meters <= radius_km * 1000.0)
    # The stitched navigation bounds must include four geodesic boundary points.
    bounds_lon, bounds_lat = [], []
    for azimuth in (0, 90, 180, 270):
        lon, lat, _ = GEOD.fwd(LON, LAT, azimuth, radius_km * 1000.0)
        bounds_lon.append(lon); bounds_lat.append(lat)
    finite_lon, finite_lat = lons[np.isfinite(lons)], lats[np.isfinite(lats)]
    if not mask.any() or min(finite_lon) > min(bounds_lon) or max(finite_lon) < max(bounds_lon) or min(finite_lat) > min(bounds_lat) or max(finite_lat) < max(bounds_lat):
        raise RuntimeError("ROI clipping gate failed for %dkm: S04+S05 coverage is insufficient" % radius_km)
    return mask

def continuous_features(values, mask, prefix):
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values) & mask
    out = {prefix + "_roi_pixels": int(mask.sum()), prefix + "_finite_pixels": int(finite.sum()), prefix + "_valid_fraction": float(finite.sum()/mask.sum())}
    physical_invalid = finite & ((values < 180.0) | (values > 330.0))
    out[prefix + "_physical_qc_removed_pixels"] = int(physical_invalid.sum())
    valid = values[finite & ~physical_invalid]
    if not len(valid):
        for stat in ("min", "p10", "p25", "p50", "p75", "p90", "max", "mean", "std"):
            out[prefix + "_" + stat] = np.nan
        return out
    for stat, value in {"min": valid.min(), "p10": np.percentile(valid, 10), "p25": np.percentile(valid, 25), "p50": np.median(valid), "p75": np.percentile(valid, 75), "p90": np.percentile(valid, 90), "max": valid.max(), "mean": valid.mean(), "std": valid.std()}.items():
        out[prefix + "_" + stat] = float(value)
    return out

# Historical 30-minute slots only; raw HSD remains in Drive cache and is not copied into git/output ZIP.
first_slot = windows.issue_time_utc.min().floor("30min")
last_slot = windows.issue_time_utc.max().ceil("30min")
frame_times = list(pd.date_range(first_slot, last_slot, freq="30min", tz="UTC"))
fixed_stamp = pd.Timestamp("2026-08-09T07:00:00Z")
fixed_b13_s04 = download_resumable(hsd_key(fixed_stamp, "B13", 4))
fixed_b13_s05 = download_resumable(hsd_key(fixed_stamp, "B13", 5))
fixed_meta = read_block7_metadata(fixed_b13_s04)
if fixed_meta["segment"] != 4 or fixed_meta["total_segments"] != 10:
    raise RuntimeError("HSD block-7 validation failed: expected segment 04 of 10, got %s" % fixed_meta)
try:
    fixed_scene = load_reference_scene([fixed_b13_s04, fixed_b13_s05], ["B13"])
    fixed_ref = fixed_scene["B13"]
    fixed_values = np.asarray(fixed_ref.values, dtype=np.float64)
    fixed_roi25 = roi_mask(fixed_ref, 25)
    reference_report = {"status": "pass", "reader": "satpy.ahi_hsd", "satpy_version": importlib.metadata.version("satpy"), "fixed_objects": [fixed_b13_s04.name, fixed_b13_s05.name], "segment_header": fixed_meta, "compared_pixel_count": int(np.isfinite(fixed_values).sum()), "impossible_pixels_lt180k_inside_roi25": int((np.isfinite(fixed_values) & fixed_roi25 & (fixed_values < 180.0)).sum()), "custom_reader_used": False, "custom_pixel_comparison": "not applicable; historical custom decoder is not used", "max_abs_bt_error_k": None, "mean_abs_bt_error_k": None}
except Exception as exc:
    reference_report = {"status": "blocked", "reader": "satpy.ahi_hsd", "fixed_object": fixed_b13_s04.name, "segment_header": fixed_meta, "reason": repr(exc)}
SATELLITE_GATE = reference_report
(OUT / "decoder_validation.json").write_text(json.dumps(reference_report, indent=2) + "\n")
print(json.dumps(reference_report, indent=2))
'''


FEATURES = r'''# 4. Fast historical B13/B08 S04 ROI-25km extraction and strict alignment
from concurrent.futures import ThreadPoolExecutor, as_completed
satellite_rows, object_manifest, alignment_rows = [], [], []
satellite_parquet = OUT / "satellite_features.parquet"
alignment_parquet = OUT / "issue_time_alignment.parquet"
checkpoint_parquet = SATELLITE_CHECKPOINT / "satellite_features.parquet"
checkpoint_alignment = SATELLITE_CHECKPOINT / "issue_time_alignment.parquet"
checkpoint_manifest = SATELLITE_CHECKPOINT / "satellite_feature_manifest.json"
if checkpoint_parquet.exists() and checkpoint_alignment.exists() and checkpoint_manifest.exists():
    # Recovery path after a Colab kernel restart: raw HSD and extracted features
    # live in Drive.  Do not re-download or re-decode the historical timeline.
    prior_manifest = json.loads(checkpoint_manifest.read_text())
    if prior_manifest.get("historical_segments") != list(HISTORICAL_SEGMENTS) or prior_manifest.get("historical_roi_km") != HISTORICAL_ROI_KM:
        raise RuntimeError("Existing satellite feature cache uses a different segment/ROI contract; refuse mixed cohorts")
    shutil.copy2(checkpoint_parquet, satellite_parquet)
    shutil.copy2(checkpoint_alignment, alignment_parquet)
    sat = pd.read_parquet(checkpoint_parquet)
    sat["nominal_scan_utc"] = pd.to_datetime(sat.nominal_scan_utc, utc=True)
    SATELLITE_GATE = prior_manifest.get("satellite_gate", {"status": "blocked", "reason": "missing persisted satellite gate"})
    SATELLITE_GATE = {**SATELLITE_GATE, "resumed_from_drive_feature_cache": True, "decoded_frames": int(len(sat))}
elif SATELLITE_GATE["status"] == "pass":
    # The 25km production ROI is reference-validated inside S04.  S05 is used
    # only by the fixed 50km geometry/reference gate above, not downloaded for
    # every history slot.  This halves historical HSD transfers without making
    # a clipped-ROI claim.  Eight bounded workers overlap NOAA/Drive I/O.
    historical_jobs = [(stamp, band, segment, hsd_key(stamp, band, segment)) for stamp in frame_times for band in BANDS for segment in HISTORICAL_SEGMENTS]
    downloaded = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        future_to_job = {pool.submit(download_resumable, key): (stamp, band, segment, key) for stamp, band, segment, key in historical_jobs}
        for future in as_completed(future_to_job):
            stamp, band, segment, key = future_to_job[future]
            try:
                local = future.result()
                downloaded[key] = local
                object_manifest.append({"key": key, "cache_path": str(local.relative_to(DRIVE_ROOT)), "sha256": sha256_file(local), "bytes": local.stat().st_size, "status": "downloaded_or_cached"})
            except Exception as exc:
                object_manifest.append({"key": key, "nominal_scan_utc": stamp.isoformat(), "band": band, "segment": segment, "status": "missing_or_download_failed", "reason": repr(exc)})
    for stamp in frame_times:
        files_for_frame = []
        try:
            for band in BANDS:
                for segment in HISTORICAL_SEGMENTS:
                    key = hsd_key(stamp, band, segment)
                    local = downloaded[key]
                    files_for_frame.append(local)
            scene = load_reference_scene(files_for_frame, BANDS)
            row = {"nominal_scan_utc": stamp.isoformat(), "assumed_scan_completion_utc": (stamp + pd.Timedelta(minutes=SCAN_COMPLETION_MINUTES)).isoformat()}
            for band in BANDS:
                array = scene[band]
                mask = roi_mask(array, HISTORICAL_ROI_KM)
                row.update(continuous_features(array.values, mask, "satellite_%s_roi%02d" % (band.lower(), HISTORICAL_ROI_KM)))
            satellite_rows.append(row)
        except Exception as exc:
            object_manifest.append({"nominal_scan_utc": stamp.isoformat(), "status": "missing_or_decode_failed", "reason": repr(exc)})
    sat = pd.DataFrame(satellite_rows)
    if sat.empty:
        SATELLITE_GATE = {"status": "blocked", "reason": "No historical B13/B08 S04 frame decoded"}
    else:
        sat["nominal_scan_utc"] = pd.to_datetime(sat.nominal_scan_utc, utc=True)
        sat = sat.sort_values("nominal_scan_utc")
        continuous = [c for c in sat.columns if c.endswith(("_min", "_p10", "_p25", "_p50", "_p75", "_p90", "_max", "_mean", "_std", "_valid_fraction"))]
        for column in continuous:
            sat[column + "_delta30"] = sat[column].diff(1)
            sat[column + "_delta60"] = sat[column].diff(2)
            sat[column + "_trend90"] = sat[column].diff(3) / 3.0
        sat.to_parquet(satellite_parquet, index=False)
        for issue in windows.issue_time_utc:
            for lag in LAG_SENSITIVITY_MINUTES:
                frame = available_before(issue, list(sat.nominal_scan_utc), lag)
                alignment_rows.append({"issue_time_utc": issue.isoformat(), "lag_assumption_minutes": lag, "matched_nominal_scan_utc": None if frame is None else frame.isoformat(), "actual_age_minutes": None if frame is None else float((issue-frame).total_seconds()/60), "available_strictly_before_issue": frame is not None})
        pd.DataFrame(alignment_rows).to_parquet(alignment_parquet, index=False)
        SATELLITE_GATE = {**SATELLITE_GATE, "status": "pass", "decoded_frames": int(len(sat)), "feature_columns": int(len(sat.columns)), "scan_completion_assumption_minutes": SCAN_COMPLETION_MINUTES, "publication_lag_sensitivity_minutes": list(LAG_SENSITIVITY_MINUTES), "historical_download_workers": 8}
else:
    sat = pd.DataFrame()

coverage = {"satellite_gate": SATELLITE_GATE, "object_count": len(object_manifest), "successful_feature_frames": int(len(satellite_rows)), "raw_hsd_cache": str(CACHE), "raw_hsd_in_git": False}
(OUT / "satellite_feature_manifest.json").write_text(json.dumps({"location": {"latitude": LAT, "longitude": LON, "station_height_agl_m": 15, "height_boundary": "not a surveyed elevation"}, "bands": list(BANDS), "resolution": RESOLUTION, "historical_segments": list(HISTORICAL_SEGMENTS), "historical_roi_km": HISTORICAL_ROI_KM, "reference_segments_for_roi50": list(REFERENCE_SEGMENTS), "cadence": "30 minutes", "download_workers": 8, "objects": object_manifest, "satellite_gate": SATELLITE_GATE, "qc_rule": "Values <180K or >330K are removed only as physical BT QC, never as a cloud threshold", "forbidden": ["No 220K deep_pct is used", "No satellite lead time is claimed"]}, indent=2, default=str) + "\n")
(OUT / "satellite_coverage_report.json").write_text(json.dumps(coverage, indent=2, default=str) + "\n")
(OUT / "decoder_validation.json").write_text(json.dumps(SATELLITE_GATE, indent=2, default=str) + "\n")
if SATELLITE_GATE.get("status") == "pass" and satellite_parquet.exists() and alignment_parquet.exists():
    shutil.copy2(satellite_parquet, checkpoint_parquet)
    shutil.copy2(alignment_parquet, checkpoint_alignment)
    shutil.copy2(OUT / "satellite_feature_manifest.json", checkpoint_manifest)
print(json.dumps(SATELLITE_GATE, indent=2, default=str))
'''


BENCHMARK = r'''# 5. Identical-cohort benchmark: persistence, climatology, station-only, satellite-only, fused
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

def make_estimator(family, gpu=True):
    if family == "ridge": return Ridge(alpha=1.0, random_state=SEED)
    if family == "elasticnet": return ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=SEED, max_iter=5000)
    if family == "gradient_boosting": return GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=SEED)
    if family == "extra_trees": return ExtraTreesRegressor(n_estimators=300, max_depth=16, min_samples_leaf=2, n_jobs=-1, random_state=SEED)
    if family == "mlp": return MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", alpha=0.001, learning_rate_init=0.001, max_iter=500, early_stopping=True, random_state=SEED)
    if family == "xgboost":
        import xgboost as xgb
        return xgb.XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=1.0, objective="reg:squarederror", tree_method="hist", device="cuda" if gpu else "cpu", random_state=SEED, n_jobs=-1)
    raise ValueError(family)

FAMILIES = ("ridge", "elasticnet", "gradient_boosting", "extra_trees", "mlp", "xgboost")
PARAMETERIZATIONS = ("direct", "residual")
TRIAL_DECLARATION = ([{"trial_id": "%s-%s-%s" % (arm, family, target), "arm": arm, "family": family, "target": target} for arm in ("station", "fused") for family in FAMILIES for target in PARAMETERIZATIONS] + [{"trial_id": "satellite-%s-direct" % family, "arm": "satellite", "family": family, "target": "direct"} for family in FAMILIES])
if len(TRIAL_DECLARATION) != 30:
    raise RuntimeError("Curated full benchmark must be exactly 30 declared fits")

def split_arrays(frame, vectors):
    masks = {name: frame.split.eq(name).to_numpy() for name in ("train", "validation", "test")}
    return {name: (np.asarray(vectors)[mask], frame.loc[mask, "y_direct"].to_numpy(), frame.loc[mask, "at_now"].to_numpy(), frame.loc[mask, "issue_time_utc"].to_numpy()) for name, mask in masks.items()}

def fit_predict(estimator, x_train, y_train, x_eval, family):
    scaler = StandardScaler().fit(x_train)  # train-only by construction
    try:
        estimator.fit(scaler.transform(x_train), y_train)
    except Exception as exc:
        if family != "xgboost": raise
        estimator = make_estimator("xgboost", gpu=False)
        estimator.fit(scaler.transform(x_train), y_train)
    return estimator, scaler, estimator.predict(scaler.transform(x_eval))

def metrics(y, p, at):
    delta = y - at
    regimes = {"cool_lt_minus2": delta < -2, "cool_minus2_to_minus1": (delta >= -2) & (delta < -1), "stable_minus1_to_plus1": (delta >= -1) & (delta <= 1), "warm_plus1_to_plus2": (delta > 1) & (delta <= 2), "warm_gt_plus2": delta > 2}
    return {"mae_degc": float(mean_absolute_error(y, p)), "rmse_degc": float(mean_squared_error(y, p, squared=False)), "bias_degc": float(np.mean(p-y)), "regime_mae_degc": {name: None if not mask.any() else float(mean_absolute_error(y[mask], p[mask])) for name, mask in regimes.items()}}

def climatology_predict(train_times, train_y, test_times):
    key = pd.DatetimeIndex(train_times).tz_convert("Asia/Ho_Chi_Minh").strftime("%H:%M")
    table = pd.Series(train_y).groupby(key).median(); fallback = float(np.median(train_y))
    keys = pd.DatetimeIndex(test_times).tz_convert("Asia/Ho_Chi_Minh").strftime("%H:%M")
    return np.asarray([table.get(value, fallback) for value in keys], dtype=float)

def select_arm(arm, frame, feature_names, vectors, parameterizations=PARAMETERIZATIONS):
    sets = split_arrays(frame, vectors)
    xt, yt, at, times = sets["train"]; xv, yv, av, _ = sets["validation"]
    scores = []
    for family in FAMILIES:
        for target in parameterizations:
            learn = yt if target == "direct" else yt-at
            estimator, scaler, raw = fit_predict(make_estimator(family), xt, learn, xv, family)
            prediction = raw if target == "direct" else av + raw
            scores.append({"arm": arm, "family": family, "target": target, "validation": metrics(yv, prediction, av), "estimator": estimator, "scaler": scaler})
    # XGBoost is benchmarked, but not eligible for portable deployment until a
    # separately validated neutral XGBoost tree exporter exists. This filter is
    # declared before any scores are read; it is not a post-test reselection.
    deployable_scores = [row for row in scores if row["family"] != "xgboost"]
    winner = min(deployable_scores, key=lambda row: row["validation"]["mae_degc"])
    return winner, scores

def refit_and_test(winner, frame, feature_names, vectors):
    sets = split_arrays(frame, vectors)
    x_train = np.vstack([sets["train"][0], sets["validation"][0]])
    y_train = np.concatenate([sets["train"][1], sets["validation"][1]])
    at_train = np.concatenate([sets["train"][2], sets["validation"][2]])
    x_test, y_test, at_test, test_times = sets["test"]
    learn = y_train if winner["target"] == "direct" else y_train-at_train
    estimator, scaler, raw = fit_predict(make_estimator(winner["family"]), x_train, learn, x_test, winner["family"])
    prediction = raw if winner["target"] == "direct" else at_test + raw
    return {**winner, "estimator": estimator, "scaler": scaler, "feature_names": feature_names, "test": metrics(y_test, prediction, at_test), "test_prediction": prediction, "test_y": y_test, "test_at": at_test, "test_times": test_times, "test_x": x_test}

station_vectors = np.vstack(windows.station_vector.to_numpy())
station_winner, station_trials = select_arm("station_only_full_coverage", windows, STATION_FEATURE_NAMES, station_vectors)
station_full = refit_and_test(station_winner, windows, STATION_FEATURE_NAMES, station_vectors)

if SATELLITE_GATE.get("status") == "pass":
    alignment = pd.read_parquet(OUT / "issue_time_alignment.parquet")
    core_lag = 20
    aligned = alignment[(alignment.lag_assumption_minutes == core_lag) & alignment.available_strictly_before_issue].copy()
    aligned.issue_time_utc = pd.to_datetime(aligned.issue_time_utc, utc=True); aligned.matched_nominal_scan_utc = pd.to_datetime(aligned.matched_nominal_scan_utc, utc=True)
    joined = windows.merge(aligned[["issue_time_utc", "matched_nominal_scan_utc", "actual_age_minutes"]], on="issue_time_utc", how="inner").merge(sat, left_on="matched_nominal_scan_utc", right_on="nominal_scan_utc", how="inner")
    satellite_feature_names = [c for c in sat.columns if c.startswith("satellite_") and not c.endswith(("_roi_pixels", "_finite_pixels", "_physical_qc_removed_pixels"))]
    joined = joined.dropna(subset=satellite_feature_names)
    common = joined.copy().reset_index(drop=True)
    # Preserve global split labels and require each full sequence to remain in that label.
    if common.empty or not common.split.isin(["train", "validation", "test"]).any(): raise RuntimeError("empty common satellite cohort")
    missing_common_splits = [name for name in ("train", "validation", "test") if not (common.split == name).any()]
    if missing_common_splits: raise RuntimeError("common satellite/station cohort lost required split(s): " + repr(missing_common_splits))
    station_common_vectors = np.vstack(common.station_vector.to_numpy())
    satellite_vectors = common[satellite_feature_names].to_numpy(dtype=float)
    fused_vectors = np.hstack([station_common_vectors, satellite_vectors])
    station_common = refit_and_test(station_winner, common, STATION_FEATURE_NAMES, station_common_vectors)
    satellite_winner, satellite_trials = select_arm("satellite_only_common", common, satellite_feature_names, satellite_vectors, parameterizations=("direct",))
    fused_winner, fused_trials = select_arm("fused_common", common, STATION_FEATURE_NAMES + satellite_feature_names, fused_vectors)
    satellite_final = refit_and_test(satellite_winner, common, satellite_feature_names, satellite_vectors)
    fused_final = refit_and_test(fused_winner, common, STATION_FEATURE_NAMES + satellite_feature_names, fused_vectors)
    common_identity = common[["issue_time_utc", "target_time_utc", "split"]].astype(str).to_dict("records")
    if len({json.dumps(v, sort_keys=True) for v in common_identity}) != len(common_identity): raise RuntimeError("duplicate common cohort identities")
    # same ordered data are the literal input to all three common-cohort arms
    ablation = {"lag_assumption_minutes": core_lag, "common_rows": int(len(common)), "common_identity_sha256": hashlib.sha256(json.dumps(common_identity, sort_keys=True).encode()).hexdigest(), "station_only_common": station_common["test"], "satellite_only": satellite_final["test"], "fused": fused_final["test"]}
else:
    common = pd.DataFrame(); station_common = satellite_final = fused_final = None; station_trials = station_trials; satellite_trials = fused_trials = []
    ablation = {"status": "blocked", "reason": SATELLITE_GATE.get("reason", "Satpy reference validation failed")}

test_sets = split_arrays(windows, station_vectors)
pt = test_sets["test"]
persistence_test = metrics(pt[1], pt[2], pt[2])
climatology_test = metrics(pt[1], climatology_predict(split_arrays(windows, station_vectors)["train"][3], split_arrays(windows, station_vectors)["train"][1], pt[3]), pt[2])
learned_gain_abs = persistence_test["mae_degc"] - station_full["test"]["mae_degc"]
learned_gain_rel = learned_gain_abs / persistence_test["mae_degc"]
learned_gate = {"absolute_gain_degc": learned_gain_abs, "relative_gain_fraction": learned_gain_rel, "absolute_threshold_degc": 0.05, "relative_threshold_fraction": 0.05, "pass": bool(learned_gain_abs >= .05 and learned_gain_rel >= .05)}
if fused_final is not None:
    fusion_gain_abs = station_common["test"]["mae_degc"] - fused_final["test"]["mae_degc"]
    fusion_gain_rel = fusion_gain_abs / station_common["test"]["mae_degc"]
    fusion_gate = {"absolute_gain_degc": fusion_gain_abs, "relative_gain_fraction": fusion_gain_rel, "absolute_threshold_degc": 0.05, "relative_threshold_fraction": .05, "pass": bool(fusion_gain_abs >= .05 and fusion_gain_rel >= .05), "fused_has_satellite_runtime_features": True}
else:
    fusion_gate = {"status": "blocked", "pass": False, "reason": ablation.get("reason", "satellite unavailable")}

model_rows = []
for group in (station_trials, satellite_trials, fused_trials):
    for row in group:
        model_rows.append({"trial_id": "%s-%s-%s" % (row["arm"], row["family"], row["target"]), "arm": row["arm"], "family": row["family"], "target": row["target"], "validation_mae_degc": row["validation"]["mae_degc"]})
pd.DataFrame(model_rows).to_csv(OUT / "model_comparison.csv", index=False)
(OUT / "ablation_metrics.json").write_text(json.dumps(ablation, indent=2, default=str) + "\n")
METRICS = {"evaluation_status": "retrospective chronological holdout; not prospective certification", "trial_count": len(model_rows), "persistence": persistence_test, "climatology": climatology_test, "station_only_full_coverage": station_full["test"], "station_winner": {k: station_full[k] for k in ("family", "target")}, "learned_vs_persistence_gate": learned_gate, "satellite_gate": SATELLITE_GATE, "fusion_gate": fusion_gate, "ablation": ablation}
(OUT / "metrics.json").write_text(json.dumps(METRICS, indent=2, default=str) + "\n")
print(json.dumps({"persistence": persistence_test, "station": station_full["test"], "learned_gate": learned_gate, "fusion_gate": fusion_gate}, indent=2))
'''


EXPORT = r'''# 6. Portable export, optimization artifacts, figures and final ZIP
def portable_model(final, name):
    model = final["estimator"]; scaler = final["scaler"]
    payload = {"format": "halo-safeshift-portable-v2", "name": name, "family": final["family"], "target": final["target"], "feature_names": final["feature_names"], "n_features": len(final["feature_names"]), "preprocessing": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(), "fit_scope": "train+validation only after validation selection"}, "runtime_input_names": final["feature_names"], "claim_boundary": "retrospective chronological holdout; not prospective certification"}
    if final["target"] == "residual":
        if "station_at_now" not in final["feature_names"]: raise RuntimeError("satellite-only residual target would require a station feature; direct target only is permitted")
        payload["at_now_index"] = final["feature_names"].index("station_at_now")
    arrays = {}
    if final["family"] in ("ridge", "elasticnet"):
        payload["model_type"] = "linear"; arrays["coef"] = np.asarray(model.coef_, dtype=np.float32); arrays["intercept"] = np.asarray([model.intercept_], dtype=np.float32)
    elif final["family"] == "mlp":
        payload["model_type"] = "mlp_relu"; payload["layers"] = len(model.coefs_)
        for i, (w, b) in enumerate(zip(model.coefs_, model.intercepts_)): arrays["W%d" % i] = np.asarray(w, dtype=np.float32); arrays["b%d" % i] = np.asarray(b, dtype=np.float32)
    elif final["family"] in ("gradient_boosting", "extra_trees"):
        source_trees = list(model.estimators_.ravel()) if final["family"] == "gradient_boosting" else list(model.estimators_)
        offsets, feature, threshold, left, right, value = [0], [], [], [], [], []
        for tree in source_trees:
            structure = tree.tree_
            feature.extend(structure.feature.astype(int).tolist()); threshold.extend(structure.threshold.astype(float).tolist())
            left.extend(structure.children_left.astype(int).tolist()); right.extend(structure.children_right.astype(int).tolist()); value.extend(structure.value[:, 0, 0].astype(float).tolist())
            offsets.append(len(feature))
        if final["family"] == "gradient_boosting":
            base_score = float(model.init_.predict(final["scaler"].transform(final["test_x"][:1]))[0]); aggregation = "sum"; learning_rate = float(model.learning_rate)
        else:
            base_score, aggregation, learning_rate = 0.0, "mean", 1.0
        payload.update({"model_type": "tree_ensemble", "aggregation": aggregation, "base_score": base_score, "learning_rate": learning_rate, "tree_count": len(source_trees)})
        arrays.update({"tree_offsets": np.asarray(offsets, dtype=np.int32), "tree_feature": np.asarray(feature, dtype=np.int32), "tree_threshold": np.asarray(threshold, dtype=np.float32), "tree_left": np.asarray(left, dtype=np.int32), "tree_right": np.asarray(right, dtype=np.int32), "tree_value": np.asarray(value, dtype=np.float32)})
    else:
        # XGBoost remains benchmarked. It is deployable only after a dedicated neutral-tree
        # exporter/parity proof; until then it cannot become the operational artifact.
        payload["model_type"] = "xgboost_export_blocked"; payload["source_family"] = final["family"]
    directory = OUT / (name + "_bundle"); directory.mkdir(exist_ok=True)
    (directory / "model.json").write_text(json.dumps(payload, indent=2) + "\n")
    np.savez_compressed(directory / "model.npz", **arrays)
    (directory / "feature_schema.json").write_text(json.dumps({"n_features": len(final["feature_names"]), "feature_names": final["feature_names"]}, indent=2) + "\n")
    manifest = {"files": {p.name: sha256_file(p) for p in directory.iterdir() if p.is_file()}, "model": payload}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if payload["model_type"] != "xgboost_export_blocked":
        # Exact Colab-to-standalone parity is mandatory before a portable candidate is eligible.
        import importlib.util
        runtime_path = OUT / "_portable_runtime_check.py"
        runtime_path.write_text("", encoding="utf-8")
        if payload["model_type"] == "tree_ensemble":
            def portable_tree(x):
                x = (x - np.asarray(payload["preprocessing"]["mean"])) / np.asarray(payload["preprocessing"]["scale"]); total = np.zeros(len(x))
                for tree_index in range(len(arrays["tree_offsets"]) - 1):
                    start, end = int(arrays["tree_offsets"][tree_index]), int(arrays["tree_offsets"][tree_index+1]); node = np.zeros(len(x), dtype=int); active = arrays["tree_feature"][start+node] >= 0
                    while active.any():
                        idx = np.flatnonzero(active); current = start + node[idx]; node[idx] = np.where(x[idx, arrays["tree_feature"][current]] <= arrays["tree_threshold"][current], arrays["tree_left"][current], arrays["tree_right"][current]); active = arrays["tree_feature"][start+node] >= 0
                    total += arrays["tree_value"][start+node]
                if payload["aggregation"] == "mean": total /= len(arrays["tree_offsets"]) - 1
                return payload["base_score"] + payload["learning_rate"] * total
            portable_prediction = portable_tree(final["test_x"][:8])
        elif payload["model_type"] == "linear":
            portable_prediction = ((final["test_x"][:8] - np.asarray(payload["preprocessing"]["mean"])) / np.asarray(payload["preprocessing"]["scale"])) @ arrays["coef"] + arrays["intercept"][0]
        else:
            hidden = (final["test_x"][:8] - np.asarray(payload["preprocessing"]["mean"])) / np.asarray(payload["preprocessing"]["scale"])
            for i in range(payload["layers"]):
                hidden = hidden @ arrays["W%d" % i] + arrays["b%d" % i]
                if i < payload["layers"] - 1: hidden = np.maximum(hidden, 0.0)
            portable_prediction = hidden.ravel()
        source_prediction = model.predict(final["scaler"].transform(final["test_x"][:8]))
        if final["target"] == "residual":
            at_index = payload["at_now_index"]; portable_prediction = portable_prediction + final["test_x"][:8, at_index]; source_prediction = source_prediction + final["test_x"][:8, at_index]
        parity = float(np.max(np.abs(np.asarray(portable_prediction) - np.asarray(source_prediction))))
        payload["portable_parity_max_abs_error"] = parity
        if parity > 1e-4: raise RuntimeError("portable export parity failed for %s: %g" % (name, parity))
        (directory / "model.json").write_text(json.dumps(payload, indent=2) + "\n")
        manifest = {"files": {p.name: sha256_file(p) for p in directory.iterdir() if p.is_file()}, "model": payload}
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return directory, payload, arrays

station_dir, station_payload, station_arrays = portable_model(station_full, "station_only")
def portable_persistence():
    directory = OUT / "persistence_bundle"; directory.mkdir(exist_ok=True)
    at_index = STATION_FEATURE_NAMES.index("station_at_now")
    payload = {"format": "halo-safeshift-portable-v2", "name": "persistence", "model_type": "persistence", "target": "direct", "n_features": len(STATION_FEATURE_NAMES), "feature_names": STATION_FEATURE_NAMES, "runtime_input_names": STATION_FEATURE_NAMES, "at_now_index": at_index, "preprocessing": {"mean": [0.0] * len(STATION_FEATURE_NAMES), "scale": [1.0] * len(STATION_FEATURE_NAMES), "fit_scope": "not applicable"}, "claim_boundary": "retrospective chronological holdout; not prospective certification"}
    (directory / "model.json").write_text(json.dumps(payload, indent=2) + "\n")
    np.savez_compressed(directory / "model.npz", marker=np.asarray([at_index], dtype=np.int32))
    (directory / "feature_schema.json").write_text(json.dumps({"n_features": len(STATION_FEATURE_NAMES), "feature_names": STATION_FEATURE_NAMES}, indent=2) + "\n")
    (directory / "manifest.json").write_text(json.dumps({"files": {p.name: sha256_file(p) for p in directory.iterdir() if p.is_file()}, "model": payload}, indent=2) + "\n")
    return directory
persistence_dir = portable_persistence()
if station_full["family"] == "mlp":
    weights = {key: value.copy() for key, value in station_arrays.items()}
    optimization = []
    for sparsity in (0.0, .30, .50, .70):
        transformed = {key: value.copy() for key, value in weights.items()}
        for key, value in transformed.items():
            if key.startswith("W") and sparsity:
                cutoff = np.quantile(np.abs(value), sparsity); value[np.abs(value) <= cutoff] = 0
        path = station_dir / ("mlp_pruned_%02d.npz" % int(sparsity*100)); np.savez_compressed(path, **transformed)
        optimization.append({"format": "fp32_pruned", "sparsity_target": sparsity, "bytes": path.stat().st_size})
    fp16 = {key: value.astype(np.float16) for key, value in weights.items()}; np.savez_compressed(station_dir / "mlp_fp16_storage.npz", **fp16)
    q = {}; scales = {}
    for key, value in weights.items():
        scale = max(float(np.max(np.abs(value))) / 127.0, 1e-12); q[key] = np.round(value/scale).astype(np.int8); scales[key] = scale
    np.savez_compressed(station_dir / "mlp_int8.npz", **q); (station_dir / "mlp_int8_scales.json").write_text(json.dumps(scales, indent=2))
    optimization += [{"format": "fp16_storage", "bytes": (station_dir / "mlp_fp16_storage.npz").stat().st_size}, {"format": "int8_weight_storage", "bytes": (station_dir / "mlp_int8.npz").stat().st_size}]
else:
    optimization = [{"format": "fp32_portable", "bytes": (station_dir / "model.npz").stat().st_size, "note": "Quantization/pruning applies only to compact MLP; tree model receives portable-tree-size optimization after parity."}]

if fused_final is not None and fusion_gate["pass"]:
    fused_dir, fused_payload, fused_arrays = portable_model(fused_final, "fused")
    if not any(name.startswith("satellite_") for name in fused_payload["runtime_input_names"]): raise RuntimeError("refuse fused export without satellite runtime inputs")
else:
    fused_dir = None
satellite_dir = None if satellite_final is None else portable_model(satellite_final, "satellite_only")[0]
if satellite_dir is None:
    satellite_dir = OUT / "satellite_only_bundle"; satellite_dir.mkdir(exist_ok=True)
    (satellite_dir / "BLOCKED.json").write_text(json.dumps({"status": "blocked", "reason": SATELLITE_GATE.get("reason", "Satpy reference validation failed")}, indent=2) + "\n")

if fused_dir is not None:
    operational_dir, operational_mode, operational_final = fused_dir, "fused", fused_final
elif learned_gate["pass"] and station_full["family"] in ("ridge", "elasticnet", "mlp"):
    operational_dir, operational_mode, operational_final = station_dir, "station-only", station_full
else:
    operational_dir, operational_mode, operational_final = persistence_dir, "persistence", station_full
shutil.copytree(operational_dir, OUT / "operational_bundle")

replay = []
for issue, y, at, prediction, feature in zip(operational_final["test_times"], operational_final["test_y"], operational_final["test_at"], operational_final["test_prediction"], operational_final["test_x"]):
    replay.append({"issue_time_utc": pd.Timestamp(issue).isoformat(), "target_time_utc": (pd.Timestamp(issue)+pd.Timedelta(minutes=30)).isoformat(), "current_at_degc": float(at), "persistence_degc": float(at), "operational_prediction_degc": float(prediction), "observed_at_degc": float(y), "features": np.asarray(feature, dtype=float).tolist(), "station_missingness": 0.0, "satellite_frame_age_minutes": None if operational_mode != "fused" else "see issue_time_alignment.parquet", "satellite_lag_assumption_minutes": None if operational_mode != "fused" else 20})
(OUT / "replay_samples.json").write_text(json.dumps(replay, indent=2) + "\n")
(OUT / "feature_schema.json").write_text(json.dumps({"station_only": STATION_FEATURE_NAMES, "satellite_only": [] if satellite_final is None else satellite_final["feature_names"], "fused": None if fused_final is None else fused_final["feature_names"], "exact_station_feature_count": len(STATION_FEATURE_NAMES)}, indent=2) + "\n")
(OUT / "optimization_tradeoff.json").write_text(json.dumps(optimization, indent=2) + "\n")

plt.figure(figsize=(12, 5)); table = pd.DataFrame(model_rows); table.groupby("arm").validation_mae_degc.min().sort_values().plot(kind="bar", color=["#2155cd", "#de7022", "#138a63"]); plt.ylabel("Validation MAE (°C)"); plt.title("Declared 36-trial model matrix"); plt.tight_layout(); plt.savefig(OUT / "figure_model_comparison.png", dpi=180); plt.close()
if fused_final is not None:
    plt.figure(figsize=(8, 4)); pd.Series({"station common": station_common["test"]["mae_degc"], "satellite": satellite_final["test"]["mae_degc"], "fused": fused_final["test"]["mae_degc"]}).plot(kind="bar", color=["#2155cd", "#8e8e8e", "#138a63"]); plt.ylabel("Test MAE (°C)"); plt.title("Identical common-row ablation cohort"); plt.tight_layout(); plt.savefig(OUT / "figure_ablation.png", dpi=180); plt.close()
plt.figure(figsize=(12, 4)); subset = replay[:200]; plt.plot([r["observed_at_degc"] for r in subset], label="later observed", color="#138a63"); plt.plot([r["persistence_degc"] for r in subset], label="persistence", color="#2155cd"); plt.plot([r["operational_prediction_degc"] for r in subset], label="operational model", color="#de7022"); plt.legend(); plt.title("Held-out replay (later target shown only after its target timestamp at runtime)"); plt.tight_layout(); plt.savefig(OUT / "figure_heldout_replay.png", dpi=180); plt.close()
if SATELLITE_GATE.get("status") == "pass":
    plt.figure(figsize=(12,4)); sat.set_index("nominal_scan_utc")["satellite_b13_roi25_p50"].plot(); plt.title("Satellite B13 ROI-25km feature coverage"); plt.tight_layout(); plt.savefig(OUT / "figure_satellite_coverage.png", dpi=180); plt.close()
else:
    plt.figure(figsize=(10, 2)); plt.axis("off"); plt.text(.02, .5, "SATELLITE/FUSION BLOCKED: " + str(SATELLITE_GATE.get("reason", "Satpy reference gate did not pass")), wrap=True, fontsize=12); plt.tight_layout(); plt.savefig(OUT / "figure_satellite_coverage.png", dpi=180); plt.close()
plt.figure(figsize=(8, 4)); opt = pd.DataFrame(optimization); plt.bar(opt["format"].astype(str), opt["bytes"], color="#5f7c6a"); plt.xticks(rotation=25, ha="right"); plt.ylabel("Artifact bytes"); plt.title("Optimization trade-off — quality is recorded separately; latency requires board measurement"); plt.tight_layout(); plt.savefig(OUT / "figure_optimization_tradeoff.png", dpi=180); plt.close()

manifest_files = {}
for path in OUT.rglob("*"):
    if path.is_file() and path.name != "manifest.json": manifest_files[str(path.relative_to(OUT))] = sha256_file(path)
manifest = {"run_id": RUN_ID, "csv_sha256": EXPECTED_SHA256, "operational_mode": operational_mode, "satellite_gate": SATELLITE_GATE, "learned_gate": learned_gate, "fusion_gate": fusion_gate, "files": manifest_files, "required_boundaries": ["retrospective chronological holdout; not prospective certification", "No WBGT/medical/legal/direct-sun claim", "No NPU/QNN/Hexagon/GPU claim", "Satellite frame timing uses declared assumptions, not measured publication latency"]}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
zip_path = OUT.with_suffix(".zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in OUT.rglob("*"):
        if path.is_file(): archive.write(path, path.relative_to(OUT))
print(json.dumps({"run": str(OUT), "zip": str(zip_path), "zip_sha256": sha256_file(zip_path), "operational_mode": operational_mode, "satellite_gate": SATELLITE_GATE["status"], "fusion_gate": fusion_gate.get("pass")}, indent=2, default=str))
'''


def build_notebook() -> dict:
    runtime_source_cell = """# 7. Embed board runtime/dashboard source in this self-contained output\n# These exact sources are written into the final ZIP. Board deployment must copy them from\n# the ZIP, never from a potentially different laptop checkout.\nruntime_source = %r\ndashboard_source = %r\n(OUT / 'full_runtime.py').write_text(runtime_source, encoding='utf-8')\n(OUT / 'full_dashboard.py').write_text(dashboard_source, encoding='utf-8')\n(OUT / 'runtime_source_manifest.json').write_text(json.dumps({'full_runtime.py': sha256_file(OUT / 'full_runtime.py'), 'full_dashboard.py': sha256_file(OUT / 'full_dashboard.py')}, indent=2) + '\\n', encoding='utf-8')\n""" % (RUNTIME_SOURCE, DASHBOARD_SOURCE)
    return {
        "cells": [markdown(INTRO), code(SETUP), code(STATION), code(SATELLITE), code(FEATURES), code(BENCHMARK), code(runtime_source_cell), code(EXPORT)],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    OUT.write_text(json.dumps(build_notebook(), indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
