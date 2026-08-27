#!/usr/bin/env python3
"""Was RandomForest picked on the test set, or does it survive honest selection?

The sweep fitted on train+validation and read the test set for every family.
That is selection on the test set. This script redoes it the legitimate way:

  1. fit on TRAIN only
  2. rank families on VALIDATION only, and pick the winner there
  3. refit the winner on train+validation
  4. read the test set exactly once, for the already-committed winner

If the validation winner is also the family that beats persistence on test, the
result stands. If a different family wins validation, then the test-set number
was selection noise and must not be quoted as a gain.
"""
import json
import warnings

import numpy as np

warnings.filterwarnings("ignore")
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, HuberRegressor, Lasso, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

import train_quantize_sweep as S  # reuses the identical cohort construction

SEED = 20260816
X, y, at, iss, P0 = S.X, S.y, S.at, S.iss, S.P0
tr = iss <= P0["tl"]
va = (iss >= P0["vf"]) & (iss <= P0["vl"])
te = iss >= P0["tf"]

xtr, ytr, attr = X[tr], y[tr], at[tr]
xva, yva, atva = X[va], y[va], at[va]
xte, yte, atte = X[te], y[te], at[te]

sc_tr = StandardScaler().fit(xtr)
Xtr, Xva = sc_tr.transform(xtr), sc_tr.transform(xva)

FAMILIES = [
    ("ridge", lambda: Ridge(alpha=1.0)),
    ("elasticnet", lambda: ElasticNet(alpha=.01, l1_ratio=.5, max_iter=5000)),
    ("lasso", lambda: Lasso(alpha=.001, max_iter=5000)),
    ("huber", lambda: HuberRegressor(alpha=.001, max_iter=300)),
    ("mlp_64_32", lambda: MLPRegressor((64, 32), max_iter=400, early_stopping=True, random_state=SEED)),
    ("extra_trees_60", lambda: ExtraTreesRegressor(60, max_depth=10, min_samples_leaf=4, n_jobs=-1, random_state=SEED)),
    ("random_forest_60", lambda: RandomForestRegressor(60, max_depth=10, min_samples_leaf=4, n_jobs=-1, random_state=SEED)),
    ("grad_boost_100", lambda: GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=.05, random_state=SEED)),
]

pers_va = mean_absolute_error(yva, atva)
pers_te = mean_absolute_error(yte, atte)
print("STEP 1-2: fit on TRAIN only, rank on VALIDATION only (test set untouched)")
print("  persistence validation MAE = %.4f" % pers_va)
print()
print("  %-18s %10s %10s" % ("family", "val MAE", "vs pers"))
print("  " + "-" * 40)
scores = {}
for name, mk in FAMILIES:
    m = mk()
    m.fit(Xtr, ytr - attr)
    v = mean_absolute_error(yva, atva + m.predict(Xva))
    scores[name] = v
    print("  %-18s %10.4f %+10.4f" % (name, v, pers_va - v))

winner = min(scores, key=scores.get)
print()
print("  VALIDATION WINNER = %s (%.4f)" % (winner, scores[winner]))
beats_va = scores[winner] < pers_va
print("  beats persistence on validation: %s" % beats_va)
print()

print("STEP 3-4: refit the committed winner on train+validation, read test ONCE")
xfit = np.vstack([xtr, xva])
rfit = np.concatenate([ytr - attr, yva - atva])
sc = StandardScaler().fit(xfit)
m = dict(FAMILIES)[winner]()
m.fit(sc.transform(xfit), rfit)
te_mae = mean_absolute_error(yte, atte + m.predict(sc.transform(xte)))
gain = pers_te - te_mae
print("  persistence test MAE = %.4f" % pers_te)
print("  %s test MAE = %.4f" % (winner, te_mae))
print("  gain = %+.4f degC (%+.2f%%)" % (gain, 100 * gain / pers_te))
print()
verdict = ("HONEST GAIN: the validation winner also beats persistence on test"
           if beats_va and gain > 0 else
           "NOT AN HONEST GAIN: do not quote a learned win")
print("VERDICT: " + verdict)
json.dump({"validation_mae": scores, "persistence_validation_mae": pers_va,
           "validation_winner": winner, "winner_beats_persistence_on_validation": bool(beats_va),
           "persistence_test_mae": pers_te, "winner_test_mae": te_mae,
           "test_gain_degc": gain, "test_gain_fraction": gain / pers_te,
           "verdict": verdict,
           "protocol": "fit on train, select on validation, refit on train+validation, read test once"},
          open("honest_selection.json", "w"), indent=2)
print("HONEST_DONE")
