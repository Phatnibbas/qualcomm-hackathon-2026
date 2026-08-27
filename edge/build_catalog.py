#!/usr/bin/env python3
"""Build model_catalog.json: one row per deployable model, joining the offline
accuracy numbers to the artifact the board actually loads.

Accuracy comes from the sweep/optimization reports (measured on the held-out
test set on a laptop). Latency is filled in live by the board. Keeping them in
one table is the point: a model is only interesting if BOTH are acceptable.

XGBoost is listed but marked blocked -- its portable export failed the parity
gate (0.108 degC vs a 1e-4 tolerance), so it must not run on the board.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SWEEP = HERE / "sweep"
OPT = SWEEP / "optimized"

sweep_report = json.loads((HERE / "sweep_report.json").read_text(encoding="utf-8"))
opt_report = json.loads((HERE / "optimization_report.json").read_text(encoding="utf-8"))
PERS = float(sweep_report["persistence_test_mae"])

try:
    xgb_report = json.loads((HERE / "xgboost_report.json").read_text(encoding="utf-8"))
except FileNotFoundError:
    xgb_report = None

rows = []


def add(path, family, group, variant, mae, params, note="", status="deployable", extra=None):
    p = Path(path)
    if not p.is_file():
        return
    row = {"file": str(p.relative_to(HERE)).replace("\\", "/"),
           "family": family, "group": group, "variant": variant,
           "test_mae_degc": mae, "params": params, "bytes": p.stat().st_size,
           "status": status, "note": note}
    if mae is not None:
        row["vs_persistence_degc"] = round(mae - PERS, 4)
    if extra:
        row.update(extra)
    rows.append(row)


# --- the two models already deployed and demonstrated on the board ---
add(HERE / "edge_model.json", "persistence", "Baseline", "shipped",
    PERS, 0, "Operational model. Assume the temperature holds.")
add(HERE / "challenger_edge_model.json", "mlp_sparse_int8", "Deployed challenger",
    "P0 sparse-INT8", 0.9240, 3025,
    "The challenger from the frozen P0 Colab run. Loses to persistence.")

# --- base sweep, one per family ---
FAMILY_GROUP = {"ridge": "Linear", "elasticnet": "Linear", "lasso": "Linear", "huber": "Linear",
                "mlp_64_32": "Neural net", "extra_trees_60": "Tree ensemble",
                "random_forest_60": "Tree ensemble", "grad_boost_100": "Tree ensemble"}
for fam in sweep_report["families"]:
    name = fam["family"]
    add(SWEEP / (name + ".json"), name, FAMILY_GROUP.get(name, "Other"), "fp32/INT8 base",
        fam["test_mae_fp32"], fam["params"])

if xgb_report:
    add(SWEEP / "xgboost_300.json", "xgboost_300", "Tree ensemble", "fp32 base",
        xgb_report["test_mae_fp32"], xgb_report["nodes"],
        "BLOCKED: portable export parity %.3e exceeds the 1e-4 gate." % xgb_report["parity_max_abs_error"],
        status="blocked",
        extra={"parity_max_abs_error": xgb_report["parity_max_abs_error"]})

# --- quantized / pruned variants ---
for v in opt_report["variants"]:
    fam = v["family"]
    group = FAMILY_GROUP.get(fam, "Other")
    if v["kind"] == "tree":
        variant = "keep %d/%d trees, %s" % (v["trees_kept"], v["trees_total"], v["precision"])
    else:
        variant = "INT8, prune %d%%" % int(round(v["sparsity"] * 100))
    add(OPT / (v["variant"] + ".json"), fam, group, variant,
        v["test_mae"], v["nonzero"])

rows.sort(key=lambda r: (r["group"], r["family"], r["test_mae_degc"] if r["test_mae_degc"] is not None else 9e9))

catalog = {
    "persistence_test_mae_degc": PERS,
    "horizon_minutes": 30,
    "accuracy_source": "held-out chronological test set (1,308 windows), measured off-board",
    "latency_source": "measured live on this Arduino UNO Q",
    "selection_warning": (
        "Variants scoring below persistence here were NOT selected on validation. "
        "Honest selection (fit train, rank on validation, read test once) picked "
        "extra_trees_60, which LOSES on test. Persistence remains the operational model."),
    "claim_boundary": (
        "Station-derived shade apparent-temperature estimate at t+30 min. Not WBGT, "
        "not a medical or legal safety limit, not direct-sun exposure. "
        "No NPU/QNN/Hexagon/GPU acceleration is used or claimed."),
    "models": rows,
}
out = HERE / "model_catalog.json"
out.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")

dep = [r for r in rows if r["status"] == "deployable"]
blocked = [r for r in rows if r["status"] != "deployable"]
total_mb = sum(r["bytes"] for r in dep) / 1e6
print("catalog: %d deployable + %d blocked = %d models, %.2f MB" % (
    len(dep), len(blocked), len(rows), total_mb))
for g in sorted({r["group"] for r in rows}):
    n = len([r for r in rows if r["group"] == g])
    best = min((r["test_mae_degc"] for r in rows if r["group"] == g and r["test_mae_degc"]), default=None)
    print("  %-20s %2d models   best test MAE %s" % (g, n, ("%.4f" % best) if best else "n/a"))
print("wrote", out)
