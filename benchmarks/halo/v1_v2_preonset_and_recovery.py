"""V-1 and V-2: does the station alone see anything BEFORE onset, and how does it recover?

Two questions the campaign says could change what this project is, both answerable from
data already on disk, with no network and no satellite.

V-2 — STRICTLY PRE-ONSET. Every illuminance and gust statistic recorded so far was
measured OVER the occurrence window, so it may be coincident with or after onset. Here,
for each frozen issue time (onset minus h), only data at or before that instant is used.
A 20-minute difference at issue time onset-5 reaches back to onset-25: still clean.

    The test is meaningless without a control. Illuminance falls every afternoon and wind
    gusts constantly. So each event's pre-onset indicator is scored as a PERCENTILE
    against a control sample of ordinary times drawn from the same record, excluding a
    guard band around every occurrence. A predictor that cannot beat that control is not
    a predictor.

V-1 — RECOVERY SHAPE. A cumulus shading the sensor passes in minutes; a cold pool sits.
Recovery time is not proof of mechanism, but if the 28 are one population it should look
like one, and if they split, that split is the finding.

Neither test uses satellite data, and neither can label an occurrence. They bound what
the station alone supports.

    py -3 benchmarks/halo/v1_v2_preonset_and_recovery.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ICT = "Asia/Ho_Chi_Minh"
HERE = Path(__file__).parent / "_cache"
OUT = Path(__file__).resolve().parents[2] / "evidence" / "halo-v1-v2-2026-08-14"
HORIZONS_MIN = [5, 10, 20, 30]
GUARD_MIN = 120          # control times must be this far from any occurrence
CONTROL_SEED = 20260814  # fixed so the control set is reproducible
WIND_EPOCH = pd.Timestamp("2026-07-21T15:42:00Z")


# --- rebuild the same grid the detector used -------------------------------
frame = pd.read_pickle(HERE / "saigon.pkl")
bogus = (frame["temp_c"] <= 1) | (frame["rh_pct"] <= 1) | (frame["press_kpa"] <= 1)
frame = frame.loc[~bogus].copy()

raw_grid = (frame.set_index("ts")[["temp_c", "rh_pct", "press_kpa", "lux", "wind_ms", "pm25"]]
            .resample("1min").mean())
grid = raw_grid.interpolate(limit=10)

grid["dT20"] = grid["temp_c"].diff(20)
grid["dRH20"] = grid["rh_pct"].diff(20)
grid["dP20"] = grid["press_kpa"].diff(20)
grid["wind_base"] = grid["wind_ms"].rolling(30, min_periods=10).median()
grid["gust"] = grid["wind_ms"] - grid["wind_base"]
grid["gust_max20"] = grid["gust"].rolling(20, min_periods=5).max()

# Illuminance as a FRACTIONAL drop against its own recent typical level: an absolute
# lux change is meaningless across dawn, noon and dusk.
grid["lux_base60"] = grid["lux"].rolling(60, min_periods=20).median()
grid["lux_drop_frac"] = 1.0 - (grid["lux"] / grid["lux_base60"].replace(0, np.nan))
grid["dPM25_20"] = grid["pm25"].diff(20)

INDICATORS = ["dT20", "dRH20", "dP20", "gust_max20", "lux_drop_frac", "dPM25_20"]
# Sign convention: for each indicator, is a LARGER value the "storm-like" direction?
LARGER_IS_STORMY = {"dT20": False, "dRH20": True, "dP20": True,
                    "gust_max20": True, "lux_drop_frac": True, "dPM25_20": True}

events = pd.read_csv(Path(__file__).resolve().parents[2] /
                     "evidence/halo-probe-2026-08-11/events.csv")
events["peak_utc"] = pd.to_datetime(events["peak_utc"], utc=True)
events["dep_start_utc"] = pd.to_datetime(events["label_dependency_start_utc"], utc=True)
events["end_utc"] = pd.to_datetime(events["label_dependency_end_utc"], utc=True)

# `label_dependency_start_utc` is the earliest minute the LABEL depended on, which is the
# occurrence start minus the detector's 20-minute difference window. The occurrence itself
# therefore starts 20 minutes later, and that is what a horizon must be measured from.
#
# Contamination boundary: at issue time t the indicators read [t-20, t]. That window is
# clean exactly when t <= occurrence start. So issuing at start-5 is legitimate; issuing
# from dep_start would test 20 minutes further back than asked and silently answer a
# different question.
events["onset_utc"] = events["dep_start_utc"] + pd.Timedelta(minutes=20)
print(f"{len(events)} occurrences, {len(grid):,} one-minute slots")


# --- control sample --------------------------------------------------------
occupied = pd.Series(False, index=grid.index)
for _, row in events.iterrows():
    lo = row["dep_start_utc"] - pd.Timedelta(minutes=GUARD_MIN)
    hi = row["end_utc"] + pd.Timedelta(minutes=GUARD_MIN)
    occupied.loc[lo:hi] = True

usable = grid[INDICATORS].notna().all(axis=1)
control_index = grid.index[(~occupied) & usable]
print(f"control pool: {len(control_index):,} clean minutes "
      f"({100 * len(control_index) / len(grid):.1f}% of the record)")
control = grid.loc[control_index, INDICATORS]


def percentile_of(value, series, larger_is_stormy):
    """Where does `value` sit in `series`, expressed as 'how extreme in the stormy direction'."""
    clean = series.dropna()
    if not len(clean) or pd.isna(value):
        return None
    if larger_is_stormy:
        return round(100.0 * (clean < value).mean(), 2)
    return round(100.0 * (clean > value).mean(), 2)


# --- V-2: strictly pre-onset ----------------------------------------------
v2_rows = []
for _, row in events.iterrows():
    for horizon in HORIZONS_MIN:
        issue = row["onset_utc"] - pd.Timedelta(minutes=horizon)
        if issue not in grid.index:
            v2_rows.append({"event": int(row["event"]), "horizon_min": horizon,
                            "issue_utc": issue.strftime("%Y-%m-%dT%H:%M:%SZ"),
                            "status": "issue time not on grid"})
            continue
        record = {"event": int(row["event"]), "horizon_min": horizon,
                  "issue_utc": issue.strftime("%Y-%m-%dT%H:%M:%SZ"),
                  "hour_ict": int(row["hour_ict"]), "status": "ok"}
        for name in INDICATORS:
            value = grid.at[issue, name]
            record[name] = None if pd.isna(value) else round(float(value), 4)
            record[f"{name}_pctile_vs_control"] = percentile_of(
                value, control[name], LARGER_IS_STORMY[name])
        v2_rows.append(record)
v2 = pd.DataFrame(v2_rows)


# --- V-1: recovery shape ---------------------------------------------------
v1_rows = []
for _, row in events.iterrows():
    onset, end = row["onset_utc"], row["end_utc"]
    pre_lo = onset - pd.Timedelta(minutes=60)
    pre = grid.loc[pre_lo:onset]
    post = grid.loc[end:end + pd.Timedelta(minutes=180)]
    if pre["temp_c"].dropna().empty or post["temp_c"].dropna().empty:
        v1_rows.append({"event": int(row["event"]), "status": "insufficient data"})
        continue

    baseline_t = float(pre["temp_c"].median())
    trough_t = float(grid.loc[onset:end, "temp_c"].min())
    depth = baseline_t - trough_t

    # Minutes after the window ends until temperature returns to 63% of the way back
    # (one time-constant, the usual convention for an exponential relaxation).
    target = trough_t + 0.63 * depth if depth > 0 else None
    recovery_min = None
    if target is not None:
        reached = post.index[post["temp_c"] >= target]
        if len(reached):
            recovery_min = round((reached[0] - end).total_seconds() / 60, 1)

    baseline_lux = float(pre["lux_base60"].median()) if pre["lux_base60"].notna().any() else None
    lux_recovery_min = None
    if baseline_lux and baseline_lux > 500:  # only meaningful in daylight
        target_lux = 0.63 * baseline_lux
        reached = post.index[post["lux"] >= target_lux]
        if len(reached):
            lux_recovery_min = round((reached[0] - end).total_seconds() / 60, 1)

    v1_rows.append({
        "event": int(row["event"]), "status": "ok", "hour_ict": int(row["hour_ict"]),
        "duration_min": float(row["duration_min"]),
        "baseline_temp_c": round(baseline_t, 2), "trough_temp_c": round(trough_t, 2),
        "depth_c": round(depth, 2),
        "temp_recovery_min_to_63pct": recovery_min,
        "daylight": bool(baseline_lux and baseline_lux > 500),
        "lux_recovery_min_to_63pct": lux_recovery_min,
    })
v1 = pd.DataFrame(v1_rows)

OUT.mkdir(parents=True, exist_ok=True)
v2.to_csv(OUT / "v2_preonset.csv", index=False)
v1.to_csv(OUT / "v1_recovery.csv", index=False)


# --- summary ---------------------------------------------------------------
print("\n" + "=" * 78)
print("V-2  STRICTLY PRE-ONSET, scored against a control of ordinary minutes")
print("=" * 78)
print("Percentile = how extreme the value is in the storm-like direction.")
print("50 = indistinguishable from an ordinary minute. 95+ = genuinely unusual.\n")
ok = v2[v2["status"] == "ok"]
summary = {}
for horizon in HORIZONS_MIN:
    sub = ok[ok["horizon_min"] == horizon]
    print(f"--- issue time = onset - {horizon} min   (n={len(sub)}) ---")
    for name in INDICATORS:
        col = sub[f"{name}_pctile_vs_control"].dropna()
        if col.empty:
            print(f"  {name:16s} no data")
            continue
        above95 = int((col >= 95).sum())
        print(f"  {name:16s} median pctile {col.median():6.1f}   "
              f"n>=95th: {above95}/{len(col)}")
        summary[f"{name}@{horizon}"] = {"median_pctile": float(col.median()),
                                        "n_at_or_above_95th": above95, "n": int(len(col))}
    print()

print("=" * 78)
print("V-1  RECOVERY SHAPE")
print("=" * 78)
good = v1[v1["status"] == "ok"]
rec = good["temp_recovery_min_to_63pct"].dropna()
print(f"occurrences analysed          : {len(good)}/{len(events)}")
print(f"temperature recovered to 63%  : {len(rec)}/{len(good)}")
if len(rec):
    print(f"  recovery minutes  min {rec.min():.0f}  p25 {rec.quantile(.25):.0f}  "
          f"median {rec.median():.0f}  p75 {rec.quantile(.75):.0f}  max {rec.max():.0f}")
lux_rec = good.loc[good["daylight"], "lux_recovery_min_to_63pct"].dropna()
print(f"daylight occurrences          : {int(good['daylight'].sum())}")
print(f"illuminance recovered to 63%  : {len(lux_rec)}")
if len(lux_rec):
    print(f"  recovery minutes  min {lux_rec.min():.0f}  median {lux_rec.median():.0f}  "
          f"max {lux_rec.max():.0f}")
print(f"depth (C)  median {good['depth_c'].median():.2f}  "
      f"min {good['depth_c'].min():.2f}  max {good['depth_c'].max():.2f}")

result = {
    "generated_at_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    "method": {
        "grid": "1-minute mean, interpolate(limit=10), same as the detector",
        "control": f"minutes at least {GUARD_MIN} min from any occurrence, all indicators present",
        "control_minutes": int(len(control_index)),
        "horizons_min": HORIZONS_MIN,
        "percentile_meaning": "fraction of control minutes less extreme in the storm-like direction",
        "recovery_definition": "minutes after the occurrence window until temperature regains 63% of its drop",
    },
    "caveats": [
        "the 28 occurrences are the output of one arbitrary threshold rule, not ground truth",
        "no independent label exists; nothing here identifies a mechanism",
        "wind is invalid before 2026-07-21T15:42Z, so gust is missing for the two earliest",
        "illuminance is meaningless at night, so the two night occurrences are excluded from it",
    ],
    "v2_summary": summary,
    "v1_summary": {
        "analysed": int(len(good)),
        "temp_recovery_available": int(len(rec)),
        "temp_recovery_min": (None if rec.empty else
                              {"min": float(rec.min()), "p25": float(rec.quantile(.25)),
                               "median": float(rec.median()), "p75": float(rec.quantile(.75)),
                               "max": float(rec.max())}),
        "daylight_occurrences": int(good["daylight"].sum()),
        "lux_recovery_available": int(len(lux_rec)),
        "lux_recovery_min": (None if lux_rec.empty else
                             {"min": float(lux_rec.min()), "median": float(lux_rec.median()),
                              "max": float(lux_rec.max())}),
    },
}
(OUT / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(f"\nartifacts: {OUT}")
