"""Find occurrences of a temperature-fall/humidity-rise threshold rule.

Method: resample to a 1-minute grid, then work in rates of change. The thresholds
below were selected for this exploratory pass; no preregistration record exists.
Interpolation may bridge at most ten consecutive minutes. Later validation must
quantify sensitivity to both the interpolation and threshold choices.

The rule output is not ground truth and does not establish a physical mechanism.
Thunderstorm outflow and cloud shading are among the untested explanations.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ICT = "Asia/Ho_Chi_Minh"
HERE = Path(__file__).parent / "_cache"
WIND_EPOCH = pd.Timestamp("2026-07-21T15:42:00Z")  # /100 -> /10 divisor fix

frame = pd.read_pickle(HERE / "saigon.pkl")

# Boot records write literal zeros across the sensor block; they are not
# measurements of a 0 degree, 0 %RH, 0 kPa world.
bogus = (frame["temp_c"] <= 1) | (frame["rh_pct"] <= 1) | (frame["press_kpa"] <= 1)
print(f"dropping {int(bogus.sum())} boot/zero rows")
frame = frame.loc[~bogus].copy()

raw_grid = (
    frame.set_index("ts")[["temp_c", "rh_pct", "press_kpa", "lux", "wind_ms", "pm25"]]
    .resample("1min")
    .mean()
)
grid = raw_grid.interpolate(limit=10)  # bridge <=10 min; longer outages stay NaN
temp_interpolated = raw_grid["temp_c"].isna() & grid["temp_c"].notna()
rh_interpolated = raw_grid["rh_pct"].isna() & grid["rh_pct"].notna()
print(f"1-min grid: {len(grid):,} slots, {grid['temp_c'].isna().sum():,} still empty")

# --- indicators, all differential -----------------------------------------
# 20-minute change captures a cold pool arrival without smearing it away.
grid["dT20"] = grid["temp_c"].diff(20)
grid["dRH20"] = grid["rh_pct"].diff(20)
grid["dP20"] = grid["press_kpa"].diff(20)
# Gust = how far the current wind exceeds the preceding half hour's typical.
grid["wind_base"] = grid["wind_ms"].rolling(30, min_periods=10).median()
grid["gust"] = grid["wind_ms"] - grid["wind_base"]
grid["gust_max20"] = grid["gust"].rolling(20, min_periods=5).max()

print()
print("=" * 72)
print("INDICATOR DISTRIBUTIONS  (percentiles, before any threshold)")
print("=" * 72)
cols = ["dT20", "dRH20", "dP20", "gust_max20"]
print(
    grid[cols]
    .describe(percentiles=[0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999])
    .T.to_string(float_format=lambda v: f"{v:8.3f}")
)

# --- candidate events ------------------------------------------------------
# Require the two most reliable channels to agree, then use gust/pressure as
# corroboration rather than as gates (pressure is only 1 hPa-resolved).
have_wind = grid.index >= WIND_EPOCH

core = (grid["dT20"] <= -2.0) & (grid["dRH20"] >= 8.0)
strong = (grid["dT20"] <= -3.0) & (grid["dRH20"] >= 12.0)

print()
print("=" * 72)
print("CANDIDATE MINUTES")
print("=" * 72)
print(f"core   (dT<=-2.0 C & dRH>=+8)    {int(core.sum()):,} minutes")
print(f"strong (dT<=-3.0 C & dRH>=+12)   {int(strong.sum()):,} minutes")
print(f"  of core minutes, gust>=1.5 m/s over baseline: "
      f"{int((core & (grid['gust_max20'] >= 1.5) & have_wind).sum()):,}")
print(f"  of core minutes, pressure rose >=0.1 kPa:     "
      f"{int((core & (grid['dP20'] >= 0.1)).sum()):,}")


def group_occurrences(mask, gap_minutes=60):
    """Collapse contiguous/nearby flagged minutes into discrete rule occurrences."""
    stamps = grid.index[mask.fillna(False)]
    if not len(stamps):
        return []
    occurrences, start, prev = [], stamps[0], stamps[0]
    for stamp in stamps[1:]:
        if (stamp - prev).total_seconds() > gap_minutes * 60:
            occurrences.append((start, prev))
            start = stamp
        prev = stamp
    occurrences.append((start, prev))
    return occurrences


for label, mask in (("core", core), ("strong", strong)):
    occurrences = group_occurrences(mask)
    print(f"\n{label}: {len(occurrences)} discrete rule occurrences over "
          f"{(grid.index[-1] - grid.index[0]).days} days")

occurrences = group_occurrences(core)

print()
print("=" * 72)
print("RULE-OCCURRENCE LIST  (core criterion)")
print("=" * 72)
print(f"{'#':>3}  {'peak (ICT)':<17} {'dur':>5} {'dT':>6} {'dRH':>6} {'dP':>6} "
      f"{'gust':>6} {'lux drop':>9}")
rows = []
for index, (start, end) in enumerate(occurrences, 1):
    window = grid.loc[start:end]
    peak = window["dT20"].idxmin()
    pre = grid.loc[peak - pd.Timedelta(minutes=40):peak - pd.Timedelta(minutes=20), "lux"].mean()
    post = grid.loc[peak:peak + pd.Timedelta(minutes=10), "lux"].mean()
    drop = np.nan if (not np.isfinite(pre) or pre < 2000) else (1 - post / pre)
    gust = window["gust_max20"].max() if peak >= WIND_EPOCH else np.nan
    dependency_start = start - pd.Timedelta(minutes=20)
    temp_dependency = temp_interpolated.loc[dependency_start:end]
    rh_dependency = rh_interpolated.loc[dependency_start:end]
    peak_inputs = [peak - pd.Timedelta(minutes=20), peak]
    rows.append({
        "peak": peak, "dur": len(window), "dT": window["dT20"].min(),
        "dRH": window["dRH20"].max(), "dP": window["dP20"].max(),
        "gust": gust, "lux_drop": drop,
        "label_dependency_start": dependency_start,
        "label_dependency_end": end,
        "temp_interpolated_fraction": float(temp_dependency.mean()),
        "rh_interpolated_fraction": float(rh_dependency.mean()),
        "label_depends_on_interpolation": bool(
            temp_dependency.any() or rh_dependency.any()
        ),
        "peak_label_depends_on_interpolation": bool(
            temp_interpolated.reindex(peak_inputs, fill_value=False).any()
            or rh_interpolated.reindex(peak_inputs, fill_value=False).any()
        ),
    })
    print(f"{index:>3}  {peak.tz_convert(ICT):%m-%d %H:%M}     {len(window):>4}m "
          f"{window['dT20'].min():>6.1f} {window['dRH20'].max():>6.1f} "
          f"{window['dP20'].max():>6.2f} "
          f"{'   n/a' if not np.isfinite(gust) else f'{gust:>6.1f}'} "
          f"{'      n/a' if not np.isfinite(drop) else f'{drop:>8.0%}'}")

table = pd.DataFrame(rows)
print()
print("=" * 72)
print("DESCRIPTIVE CHECK: local-hour distribution of threshold-rule occurrences")
print("=" * 72)
hours = table["peak"].dt.tz_convert(ICT).dt.hour
histogram = hours.value_counts().sort_index()
for hour in range(24):
    count = int(histogram.get(hour, 0))
    print(f"  {hour:02d}h  {'#' * count}{'' if count else '.'}  {count}")

# Keep the historical cache filename for compatibility; its rows are rule occurrences,
# not independently labelled physical events.
table.to_pickle(HERE / "events.pkl")
print(f"\nsaved {len(table)} rule occurrences -> {HERE / 'events.pkl'}")
