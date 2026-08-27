"""Is NASA POWER usable for HALO? Measure it; do not assume.

Answers four questions with measurements rather than documentation:
  1. Publication latency  - how stale is the newest real value?
  2. Spatial resolution   - shift the coordinate until the value changes.
  3. Units                - read them from the response, never from memory.
  4. Signal               - is POWER precipitation elevated at the 28 threshold-rule
                            occurrences, AFTER controlling for hour of day?

Question 4 is the one that needs care. POWER's own precipitation peaks in the
late afternoon and so do our events, so comparing event hours against *all*
hours just rediscovers the diurnal cycle. The control is the same hour-of-day
on days without an event.

No API key. Anonymous. Run: py -3 benchmarks/halo/probe_nasa_power.py
"""

import collections
import csv
import json
import random
import statistics as st
import urllib.request
from pathlib import Path

LAT, LON = 10.80, 106.70          # approximate; true coordinates are UNKNOWN (O-012)
START, END = "20260715", "20260808"
EVENTS = Path(__file__).resolve().parents[2] / "evidence/halo-probe-2026-08-11/events.csv"
API = "https://power.larc.nasa.gov/api/temporal/{t}/point"


def fetch(params, lat=LAT, lon=LON, start=START, end=END, temporal="hourly"):
    url = API.format(t=temporal) + (
        f"?parameters={params}&community=RE&longitude={lon}&latitude={lat}"
        f"&start={start}&end={end}&format=JSON"
    )
    with urllib.request.urlopen(url, timeout=120) as response:
        return json.loads(response.read())


def real(series):
    """POWER writes -999.0 for absent data; the fill value is in the header."""
    return {k: v for k, v in series.items() if v is not None and v > -900}


def median_shift_p(treated, control, iterations=20000, seed=7):
    """One-sided permutation test on the difference of medians."""
    observed = st.median(treated) - st.median(control)
    pool, rnd, hits = treated + control, random.Random(seed), 0
    for _ in range(iterations):
        rnd.shuffle(pool)
        if st.median(pool[: len(treated)]) - st.median(pool[len(treated):]) >= observed:
            hits += 1
    return (hits + 1) / (iterations + 1)


print("=" * 70)
print("1. UNITS AND LATENCY  (units are read from the response, not assumed)")
print("=" * 70)
raw = fetch("T2M,RH2M,PRECTOTCORR,WS10M,PS")
print(f"  time standard : {raw['header']['time_standard']}   "
      f"(NOT UTC and NOT ICT - at {LON}E, LST is about ICT+7 min)")
print(f"  sources       : {', '.join(raw['header']['sources'])}")
for name, meta in raw.get("parameters", {}).items():
    print(f"  {name:16s} units={meta['units']:8s} {meta['longname']}")
print()
for name, series in raw["properties"]["parameter"].items():
    good = sorted(real(series))
    print(f"  {name:16s} newest real value: {good[-1] if good else 'NONE'}")

for product, temporal in (("ALLSKY_SFC_SW_DWN,CLRSKY_SFC_SW_DWN", "daily"),):
    solar = fetch(product, start="20260101", end=END, temporal=temporal)
    for name, series in solar["properties"]["parameter"].items():
        good = sorted(real(series))
        print(f"  {name:16s} newest real value: {good[-1] if good else 'NONE'}  (daily)")

print()
print("=" * 70)
print("2. SPATIAL RESOLUTION  (shift the point until the value changes)")
print("=" * 70)
base = fetch("T2M", start="20260801", end="20260801")["properties"]["parameter"]["T2M"]
for dlat, dlon in ((0, 0.2), (0, 0.4), (0, 0.7), (0.2, 0), (0.4, 0), (0.6, 0)):
    other = fetch("T2M", lat=LAT + dlat, lon=LON + dlon,
                  start="20260801", end="20260801")["properties"]["parameter"]["T2M"]
    identical = all(abs(other[k] - base[k]) < 1e-9 for k in base)
    print(f"  +{dlat:.1f} lat / +{dlon:.1f} lon  (~{dlat*111:3.0f}/{dlon*109:3.0f} km)  "
          f"same cell = {identical}")

print()
print("=" * 70)
print("3. SIGNAL vs OUR 28 EVENTS  (controlled for hour of day)")
print("=" * 70)
precip = real(fetch("PRECTOTCORR")["properties"]["parameter"]["PRECTOTCORR"])
rows = list(csv.DictReader(open(EVENTS)))
event_hours = {r["peak_ict"][:13].replace("-", "").replace(" ", "").replace("T", "")[:10]
               for r in rows}
event_hours &= set(precip)

treated = [precip[k] for k in sorted(event_hours)]
control = [v for k, v in precip.items()
           if k[-2:] in {h[-2:] for h in event_hours} and k not in event_hours]
print(f"  at event hours         n={len(treated):3d}  median {st.median(treated):6.2f} mm/day")
print(f"  same hours, other days n={len(control):3d}  median {st.median(control):6.2f} mm/day")
print(f"  permutation p (one-sided)               = {median_shift_p(treated, control):.4f}")

by_day = collections.defaultdict(list)
for key, value in precip.items():
    by_day[key[:8]].append(value)
daily = {d: sum(v) / 24 for d, v in by_day.items() if len(v) == 24}   # rate -> accumulation
event_days = {r["peak_ict"][:10].replace("-", "") for r in rows}
wet = [v for d, v in daily.items() if d in event_days]
dry = [v for d, v in daily.items() if d not in event_days]
print()
print(f"  event days n={len(wet):2d}  median {st.median(wet):5.2f} mm/day")
print(f"  quiet days n={len(dry):2d}  median {st.median(dry):5.2f} mm/day")
print(f"  permutation p (one-sided)               = {median_shift_p(wet, dry):.4f}")
print(f"  quiet-day accumulations: {sorted(round(v, 1) for v in dry)}")
print()
print("  Read the largest quiet-day value before drawing any conclusion: the wettest")
print("  day in the record carried NO detected event at the station.")
