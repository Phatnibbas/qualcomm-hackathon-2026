#!/usr/bin/env python3
"""Quantize + prune EVERY family, export each variant board-ready, and measure
what the optimization actually costs in accuracy.

Per family kind:

  linear  INT8 weights + magnitude pruning at 0/30/50/70/90% -> sparse_linear_int8
  mlp     INT8 weights + per-layer magnitude pruning at 0/30/50/70/90%
  tree    ensemble truncation (keep the first N trees) + INT16 quantization of
          thresholds and leaf values

Trees are pruned by dropping trees, not by quantizing thresholds to INT8: a
threshold is a data value on the feature's own scale, so per-tensor INT8 would
destroy the split. INT16 with a per-tensor scale is measured instead.

Every variant is scored on the same test set the rest of the sweep uses. These
are test-set reads for a size/latency study, NOT model selection -- selection
was settled in honest_selection.py and persistence won.
"""
import json
import warnings

import numpy as np

warnings.filterwarnings("ignore")
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, HuberRegressor, Lasso, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor

import train_quantize_sweep as S

SEED = 20260816
OUT = S.OUT / "optimized"
OUT.mkdir(exist_ok=True, parents=True)
Xs, Xt, rfit, yte, atte, PERS = S.Xs, S.Xt, S.rfit, S.yte, S.atte, S.PERS
NAMES, sc = S.NAMES, S.sc
BASE = {"format": "halo-opt-v1", "run_id": "opt-20260816", "n_features": len(NAMES),
        "feature_names": NAMES, "at_now_index": NAMES.index("at_now"),
        "feature_mean": sc.mean_.tolist(), "feature_scale": sc.scale_.tolist(),
        "output": "delta apparent temperature; final = AT_now + delta"}
SPARSITIES = (0.0, .30, .50, .70, .90)


def mae(p):
    return mean_absolute_error(yte, atte + p)


def q_int8(a):
    a = np.asarray(a, np.float64)
    s = max(float(np.max(np.abs(a))) / 127.0, 1e-12)
    return np.round(a / s).astype(np.int8), s


def q_int16(a):
    a = np.asarray(a, np.float64)
    s = max(float(np.max(np.abs(a))) / 32767.0, 1e-12)
    return np.round(a / s).astype(np.int16), s


def prune(a, frac):
    """Magnitude pruning: zero the smallest |w| until `frac` of them are gone."""
    if frac <= 0:
        return np.array(a, np.float64)
    a = np.array(a, np.float64)
    cut = np.quantile(np.abs(a), frac)
    a[np.abs(a) <= cut] = 0.0
    return a


def write(name, payload):
    p = OUT / (name + ".json")
    p.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return p.stat().st_size


rows = []
LINEAR = [("ridge", lambda: Ridge(alpha=1.0)),
          ("elasticnet", lambda: ElasticNet(alpha=.01, l1_ratio=.5, max_iter=5000)),
          ("lasso", lambda: Lasso(alpha=.001, max_iter=5000)),
          ("huber", lambda: HuberRegressor(alpha=.001, max_iter=300))]

print("%-26s %8s %8s %8s %9s %8s" % ("variant", "MAE", "vs fp32", "nnz", "bytes", "vs pers"), flush=True)
print("-" * 74, flush=True)

for name, mk in LINEAR:
    m = mk()
    m.fit(Xs, rfit)
    fp32 = mae(m.predict(Xt))
    for sp in SPARSITIES:
        w = prune(m.coef_, sp)
        q, scale = q_int8(w)
        wq = q.astype(np.float64) * scale
        p = (Xt @ wq) + float(m.intercept_)
        e = mae(p)
        nz = [int(i) for i in np.nonzero(q)[0]]
        payload = dict(BASE, model_type="sparse_linear_int8", weight_scale=scale,
                       indices=nz, qweights=[int(q[i]) for i in nz],
                       bias=float(m.intercept_), sparsity=sp)
        tag = "%s_int8_p%02d" % (name, int(sp * 100))
        b = write(tag, payload)
        rows.append({"family": name, "kind": "linear", "variant": tag, "sparsity": sp,
                     "test_mae": e, "delta_vs_fp32": e - fp32, "nonzero": len(nz), "bytes": b,
                     "beats_persistence": bool(e < PERS)})
        print("%-26s %8.4f %+8.4f %8d %9d %8s" % (tag, e, e - fp32, len(nz), b,
                                                  "YES" if e < PERS else "no"), flush=True)

# ---- MLP ----
m = MLPRegressor((64, 32), max_iter=400, early_stopping=True, random_state=SEED)
m.fit(Xs, rfit)
fp32 = mae(m.predict(Xt))
for sp in SPARSITIES:
    layers, h = [], Xt.copy()
    nz_total = 0
    for i, (W, bb) in enumerate(zip(m.coefs_, m.intercepts_)):
        Wp = prune(W, sp)
        q, scale = q_int8(Wp)
        h = h @ (q.astype(np.float64) * scale) + bb
        last = i == len(m.coefs_) - 1
        if not last:
            h = np.maximum(h, 0.)
        outs = []
        for j in range(W.shape[1]):
            idx = [int(k) for k in np.nonzero(q[:, j])[0]]
            nz_total += len(idx)
            outs.append({"bias": float(bb[j]), "indices": idx,
                         "qweights": [int(q[k, j]) for k in idx]})
        layers.append({"relu": not last, "weight_scale": scale, "outputs": outs})
    e = mae(h.ravel())
    tag = "mlp_64_32_int8_p%02d" % int(sp * 100)
    b = write(tag, dict(BASE, model_type="mlp_sparse_int8", layers=layers, sparsity=sp))
    rows.append({"family": "mlp_64_32", "kind": "mlp", "variant": tag, "sparsity": sp,
                 "test_mae": e, "delta_vs_fp32": e - fp32, "nonzero": nz_total, "bytes": b,
                 "beats_persistence": bool(e < PERS)})
    print("%-26s %8.4f %+8.4f %8d %9d %8s" % (tag, e, e - fp32, nz_total, b,
                                              "YES" if e < PERS else "no"), flush=True)

# ---- trees: truncation + INT16 ----
TREES = [("extra_trees_60", lambda: ExtraTreesRegressor(60, max_depth=10, min_samples_leaf=4, n_jobs=-1, random_state=SEED), False),
         ("random_forest_60", lambda: RandomForestRegressor(60, max_depth=10, min_samples_leaf=4, n_jobs=-1, random_state=SEED), False),
         ("grad_boost_100", lambda: GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=.05, random_state=SEED), True)]

for name, mk, is_gb in TREES:
    m = mk()
    m.fit(Xs, rfit)
    all_trees = list(m.estimators_.ravel()) if is_gb else list(m.estimators_)
    n_all = len(all_trees)
    for keep_frac in (1.0, .5, .25, .125):
        keep = max(1, int(round(n_all * keep_frac)))
        trees = all_trees[:keep]
        off, feat, thr, lef, rig, val = [0], [], [], [], [], []
        for t in trees:
            s_ = t.tree_
            feat += s_.feature.astype(int).tolist()
            thr += s_.threshold.astype(float).tolist()
            lef += s_.children_left.astype(int).tolist()
            rig += s_.children_right.astype(int).tolist()
            val += s_.value[:, 0, 0].astype(float).tolist()
            off.append(len(feat))
        if is_gb:
            base, agg, lr = float(m.init_.predict(Xt[:1])[0]), "sum", float(m.learning_rate)
        else:
            base, agg, lr = 0., "mean", 1.

        for prec in ("fp32", "int16"):
            if prec == "fp32":
                thr_u, val_u = np.array(thr), np.array(val)
                extra = {"tree_threshold": [float(x) for x in thr_u],
                         "tree_value": [float(x) for x in val_u]}
            else:
                qt, st = q_int16(thr)
                qv, sv = q_int16(val)
                thr_u, val_u = qt.astype(np.float64) * st, qv.astype(np.float64) * sv
                extra = {"tree_threshold": [float(x) for x in thr_u],
                         "tree_value": [float(x) for x in val_u],
                         "threshold_int16_scale": st, "value_int16_scale": sv}
            F = np.asarray(feat, np.int64)
            L = np.asarray(lef, np.int64)
            R = np.asarray(rig, np.int64)
            T = np.asarray(thr_u, np.float64)
            V = np.asarray(val_u, np.float64)
            total = np.zeros(len(Xt))
            for k in range(len(off) - 1):
                st_ = off[k]
                node = np.zeros(len(Xt), dtype=np.int64)
                act = F[st_ + node] >= 0
                while act.any():
                    i = np.flatnonzero(act)
                    c = st_ + node[i]
                    node[i] = np.where(Xt[i, F[c]] <= T[c], L[c], R[c])
                    act = F[st_ + node] >= 0
                total += V[st_ + node]
            p = base + (lr / len(trees) if agg == "mean" else lr) * total
            e = mae(p)
            tag = "%s_k%03d_%s" % (name, keep, prec)
            b = write(tag, dict(BASE, model_type="tree_ensemble", aggregation=agg,
                                base_score=base, learning_rate=lr, tree_offsets=off,
                                tree_feature=feat, tree_left=lef, tree_right=rig,
                                trees_kept=keep, trees_total=n_all, precision=prec, **extra))
            rows.append({"family": name, "kind": "tree", "variant": tag, "trees_kept": keep,
                         "trees_total": n_all, "precision": prec, "test_mae": e,
                         "nonzero": len(feat), "bytes": b, "beats_persistence": bool(e < PERS)})
            print("%-26s %8.4f %8s %8d %9d %8s" % (tag, e, "-", len(feat), b,
                                                   "YES" if e < PERS else "no"), flush=True)

json.dump({"persistence_test_mae": PERS, "variants": rows,
           "pruning": "magnitude pruning on weights; trees pruned by truncating the ensemble",
           "quantization": "INT8 weights (linear/MLP); INT16 thresholds+leaf values (trees)",
           "warning": "These are test-set reads for a size/latency study, not model selection. "
                      "Selection was settled in honest_selection.py: persistence won.",
           "claim_boundary": "retrospective chronological holdout; not prospective certification"},
          open("optimization_report.json", "w"), indent=2)
print(flush=True)
print("persistence MAE = %.4f | %d variants written to %s" % (PERS, len(rows), OUT), flush=True)
print("OPTIMIZE_DONE", flush=True)
