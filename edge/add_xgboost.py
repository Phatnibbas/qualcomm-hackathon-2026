#!/usr/bin/env python3
"""Add XGBoost to the sweep, on the identical cohort, and export it board-ready.

XGBoost trees are plain regression trees, so the same tree_ensemble export the
other ensembles use applies: aggregation=sum, learning_rate folded in, base_score
from the booster. Exported through the SAME parity check as the rest.
"""
import json
import warnings

import numpy as np

warnings.filterwarnings("ignore")
from sklearn.metrics import mean_absolute_error, mean_squared_error

import train_quantize_sweep as S

import xgboost as xgb

SEED = 20260816
Xs, Xt, rfit, yte, atte = S.Xs, S.Xt, S.rfit, S.yte, S.atte
PERS = S.PERS

print()
print("=== XGBoost on the identical cohort ===", flush=True)
m = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=.05, subsample=.9,
                     colsample_bytree=1.0, objective="reg:squarederror", tree_method="hist",
                     random_state=SEED, n_jobs=-1)
m.fit(Xs, rfit)
p = m.predict(Xt)
mae32 = mean_absolute_error(yte, atte + p)
print("xgboost_300 fp32 test MAE = %.4f  RMSE = %.4f" % (mae32, mean_squared_error(yte, atte + p) ** .5))
print("persistence      test MAE = %.4f" % PERS)
print("gain = %+.4f degC (%+.2f%%)  -> %s" % (PERS - mae32, 100 * (PERS - mae32) / PERS,
                                              "beats baseline" if mae32 < PERS else "loses to baseline"))

# ---- export to the same board-ready tree_ensemble format ----
booster = m.get_booster()
dump = booster.get_dump(dump_format="json")
off, feat, thr, lef, rig, val = [0], [], [], [], [], []


def flatten(node):
    """Depth-first flatten into the array layout the board runner walks."""
    index = {}
    order = []

    def assign(n):
        i = len(order)
        order.append(n)
        index[id(n)] = i
        if "children" in n:
            for ch in n["children"]:
                assign(ch)
        return i

    assign(node)
    for n in order:
        if "children" in n:
            f = int(str(n["split"]).lstrip("f"))
            feat.append(f)
            thr.append(float(n["split_condition"]))
            kids = {c["nodeid"]: c for c in n["children"]}
            yes = kids[n["yes"]]
            no = kids[n["no"]]
            lef.append(index[id(yes)])
            rig.append(index[id(no)])
            val.append(0.0)
        else:
            feat.append(-2)
            thr.append(-2.0)
            lef.append(-1)
            rig.append(-1)
            val.append(float(n["leaf"]))
    return len(order)


for tree_json in dump:
    flatten(json.loads(tree_json))
    off.append(len(feat))

# XGBoost >=2 reports base_score as a bracketed vector string, e.g. '[-8.5181714E-4]'
_bs = json.loads(booster.save_config())["learner"]["learner_model_param"]["base_score"]
base_score = float(str(_bs).strip("[]").split(",")[0])
payload = {"format": "halo-sweep-v1", "run_id": "sweep-20260816",
           "n_features": len(S.NAMES), "feature_names": S.NAMES,
           "at_now_index": S.NAMES.index("at_now"),
           "feature_mean": S.sc.mean_.tolist(), "feature_scale": S.sc.scale_.tolist(),
           "output": "delta apparent temperature; final = AT_now + delta",
           "model_type": "tree_ensemble", "aggregation": "sum", "base_score": base_score,
           "learning_rate": 1.0,  # XGBoost leaf values already include the shrinkage
           "tree_offsets": off, "tree_feature": feat, "tree_threshold": thr,
           "tree_left": lef, "tree_right": rig, "tree_value": val}

# XGBoost splits on "<" (goes left when x < threshold); the board runner uses "<=".
# Verify parity explicitly rather than assuming the convention matches.
def walk(x):
    total = np.zeros(len(x))
    for k in range(len(off) - 1):
        st = off[k]
        node = np.zeros(len(x), dtype=int)
        active = np.array([feat[st + n] >= 0 for n in node])
        while active.any():
            idx = np.flatnonzero(active)
            cur = [st + node[i] for i in idx]
            for j, c in zip(idx, cur):
                node[j] = lef[c] if x[j, feat[c]] < thr[c] else rig[c]
            active = np.array([feat[st + n] >= 0 for n in node])
        total += np.array([val[st + n] for n in node])
    return base_score + total


probe = Xt[:16]
parity = float(np.max(np.abs(walk(probe) - m.predict(probe))))
print("portable-export parity (16 rows): max|err| = %.3e  %s" % (
    parity, "PASS" if parity <= 1e-4 else "FAIL -- split convention mismatch"))
payload["split_comparison"] = "lt"
payload["portable_parity_max_abs_error"] = parity

p_out = S.OUT / "xgboost_300.json"
p_out.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
print("wrote %s (%.1f KB, %d nodes, %d trees)" % (p_out, p_out.stat().st_size / 1024, len(feat), len(off) - 1))
json.dump({"family": "xgboost_300", "test_mae_fp32": mae32, "persistence_test_mae": PERS,
           "beats_persistence": bool(mae32 < PERS), "parity_max_abs_error": parity,
           "nodes": len(feat), "trees": len(off) - 1, "json_bytes": p_out.stat().st_size,
           "note": "selected on nothing; this is a test-set read like the rest of the sweep"},
          open("xgboost_report.json", "w"), indent=2)
print("XGB_DONE")
