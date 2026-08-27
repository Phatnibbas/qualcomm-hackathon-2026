"""Describe the Saigon station history before attempting any event detection.

Understand the data first: coverage, cadence, ranges, missingness, and the
known firmware epochs. No thresholds are applied here.
"""

from pathlib import Path

import numpy as np
import pandas as pd

CSV = Path(__file__).parent / "_cache" / "saigon_full.csv"
ICT = "Asia/Ho_Chi_Minh"

NAMES = {
    "field1": "wind_ms",
    "field2": "wind_deg",
    "field3": "temp_c",
    "field4": "press_kpa",
    "field5": "lux",
    "field6": "rh_pct",
    "field7": "noise_db",
    "field8": "pm25",
}

frame = pd.read_csv(CSV)
frame["ts"] = pd.to_datetime(frame["created_at"], utc=True)
frame = frame.rename(columns=NAMES).sort_values("ts").reset_index(drop=True)
for column in NAMES.values():
    frame[column] = pd.to_numeric(frame[column], errors="coerce")

print("=" * 72)
print("COVERAGE")
print("=" * 72)
span = frame["ts"].iloc[-1] - frame["ts"].iloc[0]
print(f"rows            {len(frame):,}")
print(f"first           {frame['ts'].iloc[0]}  ({frame['ts'].iloc[0].tz_convert(ICT)} ICT)")
print(f"last            {frame['ts'].iloc[-1]}  ({frame['ts'].iloc[-1].tz_convert(ICT)} ICT)")
print(f"span            {span.total_seconds() / 86400:.2f} days")

gaps = frame["ts"].diff().dt.total_seconds().dropna()
print(f"cadence         median {gaps.median():.0f}s  p95 {gaps.quantile(0.95):.0f}s  max {gaps.max():,.0f}s")
for limit in (60, 300, 900, 3600):
    print(f"  gaps > {limit:>4}s   {int((gaps > limit).sum())}")

big = frame.loc[gaps[gaps > 900].index]
if len(big):
    print("\n  outages > 15 min:")
    for idx in big.index[:15]:
        prev = frame['ts'].iloc[idx - 1].tz_convert(ICT)
        here = frame['ts'].iloc[idx].tz_convert(ICT)
        mins = (frame['ts'].iloc[idx] - frame['ts'].iloc[idx - 1]).total_seconds() / 60
        print(f"    {mins:7.1f} min   {prev:%m-%d %H:%M} -> {here:%m-%d %H:%M} ICT")

print()
print("=" * 72)
print("VALUE RANGES  (raw, no filtering)")
print("=" * 72)
described = frame[list(NAMES.values())].describe(
    percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]
).T
described["missing"] = frame[list(NAMES.values())].isna().sum()
print(described[["count", "missing", "min", "1%", "50%", "99%", "max"]].to_string(
    float_format=lambda v: f"{v:,.2f}"
))

print()
print("=" * 72)
print("FIRMWARE EPOCHS  (wind divisor changed /100 -> /10 around 2026-07-21 21:21 ICT)")
print("=" * 72)
cut = pd.Timestamp("2026-07-21T15:42:00Z")
before = frame[frame["ts"] < cut]["wind_ms"].dropna()
after = frame[frame["ts"] >= cut]["wind_ms"].dropna()


def grid_fraction(series):
    """Share of values landing exactly on a 0.1 grid (the /10 signature)."""
    if not len(series):
        return float("nan")
    return float(np.isclose((series * 10).round(), series * 10, atol=1e-6).mean())


print(f"before cut  n={len(before):,}  mean {before.mean():.3f}  max {before.max():.2f}"
      f"  on-0.1-grid {grid_fraction(before):.1%}")
print(f"after  cut  n={len(after):,}  mean {after.mean():.3f}  max {after.max():.2f}"
      f"  on-0.1-grid {grid_fraction(after):.1%}")

print()
print("=" * 72)
print("STATUS VOCABULARY")
print("=" * 72)
print(frame["status"].fillna("(none)").value_counts().head(15).to_string())

print()
print("=" * 72)
print("DIURNAL SHAPE  (mean by ICT hour — tells us what must be de-trended)")
print("=" * 72)
frame["hour"] = frame["ts"].dt.tz_convert(ICT).dt.hour
hourly = frame.groupby("hour")[["temp_c", "rh_pct", "press_kpa", "lux", "wind_ms", "pm25"]].mean()
print(hourly.to_string(float_format=lambda v: f"{v:,.2f}"))

frame.to_pickle(Path(__file__).parent / "_cache" / "saigon.pkl")
print(f"\ncached -> {Path(__file__).with_name('saigon.pkl')}")
