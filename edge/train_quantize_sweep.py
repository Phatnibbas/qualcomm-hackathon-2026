#!/usr/bin/env python3
"""Train + INT8-quantize a family sweep of time-series regressors, export
board-ready JSON (stdlib-only inference), and report the quantization cost.

Same frozen cohort as the deployed run: 125 station features, +30 min horizon,
chronological 60/20/20 with a 90-minute embargo. Residual target (AT_now + delta),
identical to the deployed challenger.
"""
import json
import math
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, HuberRegressor, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

SEED = 20260816
OUT = Path("sweep")
OUT.mkdir(exist_ok=True)

CSV = "../VFCD_3rd_landscape_20260816-0336_74700rows.csv"
A = {"timestamp": "timestamp_utc_iso", "wind_speed": "Wind speed (m/s)",
     "wind_direction": "Wind direction (°)", "temperature": "Temperature (°C)",
     "pressure": "Pressure (kPa)", "light": "Light (lux)", "humidity": "Humidity (%RH)",
     "noise": "Sound level (dB)", "pm25": "PM2.5 (µg/m³)"}
raw = pd.read_csv(CSV, low_memory=False)
w = pd.DataFrame({"timestamp": pd.to_datetime(raw[A["timestamp"]], utc=True, errors="coerce")})
for n, col in A.items():
    if n != "timestamp":
        w[n] = pd.to_numeric(raw[col], errors="coerce")
w = w.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp")
v = (w.temperature.between(.01, 60.) & w.humidity.between(1.01, 100.) & w.wind_speed.between(0., 40.)
     & w.pressure.between(80., 120.) & w.wind_direction.between(0., 360., inclusive="left")
     & w[["light", "noise", "pm25"]].notna().all(axis=1))
w = w.loc[v].set_index("timestamp")
num = ["wind_speed", "wind_direction", "temperature", "pressure", "light", "humidity", "noise", "pm25"]
b = w[num].resample("5min", label="right", closed="right").median()
b["raw_count"] = w.temperature.resample("5min", label="right", closed="right").count()
b = b.loc[b.raw_count >= 6].dropna()
b["at"] = (b.temperature
           + .33 * (b.humidity / 100.) * 6.105 * np.exp(17.27 * b.temperature / (237.7 + b.temperature))
           - .70 * b.wind_speed - 4.)
r = np.deg2rad(b.wind_direction.to_numpy())
b["wind_direction_sin"] = np.sin(r)
b["wind_direction_cos"] = np.cos(r)
b["light_log1p"] = np.log1p(np.maximum(b.light, 0.))
b["pm25_log1p"] = np.log1p(np.maximum(b.pm25, 0.))
PS = ["temperature", "humidity", "wind_speed", "wind_direction_sin", "wind_direction_cos",
      "light_log1p", "pressure", "pm25_log1p", "noise", "at"]
NAMES = (["t-%02dm_%s" % (55 - 5 * i, n) for i in range(12) for n in PS]
         + ["at_now", "local_time_sin", "local_time_cos", "day_of_year_sin", "day_of_year_cos"])

H = 6
rows = []
for i in range(11, len(b) - H):
    if b.index[i + H] - b.index[i - 11] != pd.Timedelta(minutes=5 * (11 + H)):
        continue
    blk = b.iloc[i - 11:i + 1]
    if not np.all(np.diff(blk.index.asi8) == pd.Timedelta(minutes=5).value):
        continue
    it = b.index[i]
    loc = it.tz_convert("Asia/Ho_Chi_Minh")
    mnt = loc.hour * 60 + loc.minute
    ex = [b.iloc[i]["at"], math.sin(2 * math.pi * mnt / 1440), math.cos(2 * math.pi * mnt / 1440),
          math.sin(2 * math.pi * loc.dayofyear / 366), math.cos(2 * math.pi * loc.dayofyear / 366)]
    rows.append((it, float(b.iloc[i + H]["at"]), float(b.iloc[i]["at"]),
                 np.concatenate([blk[PS].to_numpy(np.float64).reshape(-1), ex])))

iss = np.array([x[0] for x in rows])
y = np.array([x[1] for x in rows])
at = np.array([x[2] for x in rows])
X = np.vstack([x[3] for x in rows])
P0 = dict(tl=pd.Timestamp("2026-08-05T04:35:00Z"), vf=pd.Timestamp("2026-08-05T07:40:00Z"),
          vl=pd.Timestamp("2026-08-09T22:05:00Z"), tf=pd.Timestamp("2026-08-10T01:10:00Z"))
tr = iss <= P0["tl"]
va = (iss >= P0["vf"]) & (iss <= P0["vl"])
te = iss >= P0["tf"]
xfit = np.vstack([X[tr], X[va]])
yfit = np.concatenate([y[tr], y[va]])
atfit = np.concatenate([at[tr], at[va]])
xte, yte, atte = X[te], y[te], at[te]
sc = StandardScaler().fit(xfit)
Xs = sc.transform(xfit)
Xt = sc.transform(xte)
rfit = yfit - atfit

PERS = mean_absolute_error(yte, atte)
print("cohort: fit=%d test=%d features=%d" % (len(xfit), len(xte), X.shape[1]), flush=True)
print("baseline persistence: MAE=%.4f RMSE=%.4f" % (PERS, mean_squared_error(yte, atte) ** .5), flush=True)
print(flush=True)


def mae(p):
    return mean_absolute_error(yte, atte + p)


def q_int8(arr):
    a = np.asarray(arr, np.float64)
    s = max(float(np.max(np.abs(a))) / 127.0, 1e-12)
    return np.round(a / s).astype(np.int8), s


FAMILIES = [
    ("ridge", "linear", lambda: Ridge(alpha=1.0)),
    ("elasticnet", "linear", lambda: ElasticNet(alpha=.01, l1_ratio=.5, max_iter=5000)),
    ("lasso", "linear", lambda: Lasso(alpha=.001, max_iter=5000)),
    ("huber", "linear", lambda: HuberRegressor(alpha=.001, max_iter=300)),
    ("mlp_64_32", "mlp", lambda: MLPRegressor((64, 32), max_iter=400, early_stopping=True, random_state=SEED)),
    ("extra_trees_60", "tree", lambda: ExtraTreesRegressor(60, max_depth=10, min_samples_leaf=4, n_jobs=-1, random_state=SEED)),
    ("random_forest_60", "tree", lambda: RandomForestRegressor(60, max_depth=10, min_samples_leaf=4, n_jobs=-1, random_state=SEED)),
    ("grad_boost_100", "tree", lambda: GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=.05, random_state=SEED)),
]

print("%-18s %8s %8s %7s %9s %8s %6s" % ("family", "fp32MAE", "int8MAE", "delta", "params", "KB", "fit_s"), flush=True)
print("-" * 74, flush=True)
report = []
for name, kind, mk in FAMILIES:
    t0 = time.time()
    m = mk()
    m.fit(Xs, rfit)
    fit_s = time.time() - t0
    p32 = m.predict(Xt)
    mae32 = mae(p32)
    entry = {"family": name, "kind": kind, "fit_seconds": round(fit_s, 2),
             "test_mae_fp32": mae32, "test_rmse_fp32": mean_squared_error(yte, atte + p32) ** .5,
             "beats_persistence_fp32": bool(mae32 < PERS)}
    payload = {"format": "halo-sweep-v1", "run_id": "sweep-20260816",
               "n_features": len(NAMES), "feature_names": NAMES, "at_now_index": NAMES.index("at_now"),
               "feature_mean": sc.mean_.tolist(), "feature_scale": sc.scale_.tolist(),
               "output": "delta apparent temperature; final = AT_now + delta"}
    if kind == "linear":
        q, s = q_int8(m.coef_)
        pq = (Xt @ (q.astype(np.float64) * s)) + float(m.intercept_)
        entry["test_mae_int8"] = mae(pq)
        entry["params"] = int(np.count_nonzero(m.coef_))
        payload.update({"model_type": "linear_int8", "weight_scale": s,
                        "qweights": [int(x) for x in q], "bias": float(m.intercept_)})
    elif kind == "mlp":
        layers, h = [], Xt.copy()
        for i, (W, bb) in enumerate(zip(m.coefs_, m.intercepts_)):
            q, s = q_int8(W)
            h = h @ (q.astype(np.float64) * s) + bb
            last = i == len(m.coefs_) - 1
            if not last:
                h = np.maximum(h, 0.)
            layers.append({"relu": not last, "weight_scale": s,
                           "outputs": [{"bias": float(bb[j]),
                                        "indices": [int(k) for k in np.nonzero(q[:, j])[0]],
                                        "qweights": [int(x) for x in q[np.nonzero(q[:, j])[0], j]]}
                                       for j in range(W.shape[1])]})
        entry["test_mae_int8"] = mae(h.ravel())
        entry["params"] = int(sum(len(o["indices"]) for L in layers for o in L["outputs"]))
        payload.update({"model_type": "mlp_sparse_int8", "layers": layers})
    else:
        trees = list(m.estimators_.ravel()) if name.startswith("grad") else list(m.estimators_)
        off, feat, thr, lef, rig, val = [0], [], [], [], [], []
        for t in trees:
            s_ = t.tree_
            feat += s_.feature.astype(int).tolist()
            thr += s_.threshold.astype(float).tolist()
            lef += s_.children_left.astype(int).tolist()
            rig += s_.children_right.astype(int).tolist()
            val += s_.value[:, 0, 0].astype(float).tolist()
            off.append(len(feat))
        if name.startswith("grad"):
            base, agg, lr = float(m.init_.predict(Xt[:1])[0]), "sum", float(m.learning_rate)
        else:
            base, agg, lr = 0., "mean", 1.
        entry["test_mae_int8"] = None
        entry["int8_note"] = "tree thresholds are data values, not weights; per-tensor INT8 is not applicable"
        entry["params"] = len(feat)
        entry["trees"] = len(trees)
        payload.update({"model_type": "tree_ensemble", "aggregation": agg, "base_score": base,
                        "learning_rate": lr, "tree_offsets": off, "tree_feature": feat,
                        "tree_threshold": thr, "tree_left": lef, "tree_right": rig, "tree_value": val})
    p = OUT / ("%s.json" % name)
    p.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    entry["json_bytes"] = p.stat().st_size
    d = entry.get("test_mae_int8")
    print("%-18s %8.4f %8s %7s %9d %8.1f %6.1f" % (
        name, mae32,
        ("%.4f" % d) if d is not None else "n/a",
        ("%+.4f" % (d - mae32)) if d is not None else "n/a",
        entry["params"], entry["json_bytes"] / 1024, fit_s), flush=True)
    report.append(entry)

Path("sweep_report.json").write_text(json.dumps(
    {"cohort": {"fit": len(xfit), "test": len(xte), "features": int(X.shape[1]), "horizon_min": 30},
     "persistence_test_mae": PERS, "families": report,
     "quantization": "symmetric per-tensor INT8 on weights only; tree thresholds are data values and are not quantized",
     "claim_boundary": "retrospective chronological holdout; not prospective certification"},
    indent=2) + "\n", encoding="utf-8")
print(flush=True)
print("persistence MAE=%.4f  -- anything below this beats the baseline" % PERS, flush=True)
print("beat baseline: %s" % ([e["family"] for e in report if e["beats_persistence_fp32"]] or "NONE"), flush=True)
print("SWEEP_DONE", flush=True)
