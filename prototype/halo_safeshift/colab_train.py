"""Colab entry point: blocked validation, final refit, parity gate, fallback.

Run this in Colab against the frozen packet produced by ``prepare_colab.py``.
It contains no pipeline logic of its own: QC, resampling, windowing, folds,
baselines, export and the NumPy runtime all come from this package, so the
Colab run and the local preparation run share one implementation.

    python -m prototype.halo_safeshift.verify_packet --packet colab-input
    python -m prototype.halo_safeshift.colab_train \
        --packet colab-input --out artifacts/halo-safeshift/<run-id>

The shape of the protocol, and why each part is the way it is:

* **Bounded search.** The trials come from ``models.trials`` as an explicit
  list. Nothing expands at runtime, so the fitting budget is what the file says
  it is.
* **Selection scope.** Ranking uses blocked-validation MAE only. The prospective
  quarantine is never scored — not as a check, not as a tiebreak.
* **Final refit.** The exported estimator is refitted on *all* eligible
  historical-development windows after the trial id is frozen. Exporting the
  last fold's estimator would ship weights fitted on a prefix of the record.
* **Parity is a gate, not a report.** A candidate whose neutral-runtime error
  exceeds tolerance is ineligible and is not exported at all.
* **Operational fallback.** If the best learned challenger does not clear the
  observed engineering thresholds against persistence, the operational bundle
  *is* persistence. The challenger is still exported separately when its parity
  passes, clearly labelled as a challenger.

What it must never do: score the prospective quarantine, re-run selection after
seeing a held-out outcome, turn regime analysis into a selection gate, or claim
a certified untouched test.

**This script has not been executed. No model has been trained, selected or
evaluated for HALO SafeShift.**
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import load_config, make_run_id, sha256_file, write_json
from .baselines import (
    LocalTimeClimatology,
    PersistenceBaseline,
    from_residual,
    mean_absolute_error,
    regime_report,
    root_mean_squared_error,
    to_residual,
)
from .data import (
    apply_row_qc,
    load_channel_metadata,
    load_raw_csv,
    resample_five_minute,
    resolve_field_mapping,
)
from .export import (
    Preprocessing,
    export_estimator,
    export_persistence,
)
from .features import build_windows
from .inference import SafeShiftPredictor
from .models import (
    FAMILY_TO_EXPORT_TYPE,
    build_estimator,
    curated_trials,
    trial_budget_report,
    xgboost_policy_status,
)
from .splits import assert_fold_chronology, build_folds, fold_report
from .verify_packet import verify_packet

__all__ = ["ParityGateError", "fit_fold", "final_refit", "main"]


class ParityGateError(RuntimeError):
    """Raised when no candidate survives the neutral-runtime parity gate."""


def _tolerance_for(export_type: str, family: str, config: dict[str, Any]) -> float:
    tolerances = config["tolerances"]
    if family == "xgboost_optional":
        return float(tolerances["xgboost_tree_ensemble_fp64_max_abs_error"])
    if export_type == "tree_ensemble":
        return float(tolerances["tree_ensemble_fp64_max_abs_error"])
    if export_type == "mlp_relu":
        return float(tolerances["mlp_fp32_max_abs_error"])
    return float(tolerances["linear_fp64_max_abs_error"])


def _standardizer(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale <= 0] = 1.0
    return mean, scale


def fit_fold(
    trial: dict[str, Any],
    windows: Any,
    fold: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Fit inside one fold's train block and score its validation block."""
    x_train, y_train = windows.x[fold.train_mask], windows.y[fold.train_mask]
    x_valid, y_valid = windows.x[fold.validation_mask], windows.y[fold.validation_mask]
    at_train = windows.at_now[fold.train_mask]
    at_valid = windows.at_now[fold.validation_mask]

    mean, scale = _standardizer(x_train)
    residual = trial["parameterization"] == "residual"

    learn_target = to_residual(y_train, at_train) if residual else y_train
    estimator = build_estimator(trial["family"], trial["params"], config)
    estimator.fit((x_train - mean) / scale, learn_target)
    raw = estimator.predict((x_valid - mean) / scale)
    predicted = from_residual(raw, at_valid) if residual else raw

    persistence = PersistenceBaseline.from_schema(windows.schema).predict(x_valid)

    return {
        "trial_id": trial["trial_id"],
        "fold_index": fold.index,
        "n_train": int(x_train.shape[0]),
        "n_validation": int(x_valid.shape[0]),
        "validation_mae": mean_absolute_error(y_valid, predicted),
        "validation_rmse": root_mean_squared_error(y_valid, predicted),
        # Persistence on the IDENTICAL validation rows, so the per-fold
        # difference below is paired rather than two independent averages.
        "persistence_mae": mean_absolute_error(y_valid, persistence),
        "validation_regime": regime_report(y_valid, predicted, at_valid),
        "preprocessing_fit_scope": f"fold {fold.index} train block only",
    }


def final_refit(
    trial: dict[str, Any],
    windows: Any,
    config: dict[str, Any],
) -> tuple[Any, Preprocessing]:
    """Refit preprocessing and estimator on ALL eligible historical windows.

    Called only after the trial id is frozen. The fold estimators exist to rank
    trials; none of them is the thing that ships, because each saw only a prefix
    of the record.
    """
    mean, scale = _standardizer(windows.x)
    residual = trial["parameterization"] == "residual"
    learn_target = to_residual(windows.y, windows.at_now) if residual else windows.y
    estimator = build_estimator(trial["family"], trial["params"], config)
    estimator.fit((windows.x - mean) / scale, learn_target)
    preprocessing = Preprocessing(
        kind="standardize",
        mean=mean,
        scale=scale,
        fit_scope="all eligible historical-development windows (final refit)",
    )
    return estimator, preprocessing


def golden_vectors(windows: Any, config: dict[str, Any]) -> np.ndarray:
    """Deterministic probe rows drawn from historical development by position.

    Evenly spaced indices, not a random sample: the same packet must yield the
    same probe set on any machine, or a parity number cannot be reproduced.
    """
    count = min(int(config["tolerances"]["parity_probe_vectors"]), int(windows.x.shape[0]))
    if count <= 0:
        return np.zeros((0, windows.x.shape[1]), dtype=np.float64)
    positions = np.linspace(0, windows.x.shape[0] - 1, count).astype(np.int64)
    return windows.x[positions]


def parity_error(
    bundle_dir: Path,
    estimator: Any,
    preprocessing: Preprocessing,
    probe: np.ndarray,
    residual: bool,
    at_now: np.ndarray,
) -> float:
    """Max absolute difference between the training library and the neutral runtime."""
    predictor = SafeShiftPredictor.load(bundle_dir)
    if preprocessing.mean is None or preprocessing.scale is None:
        scaled = probe
    else:
        scaled = (probe - preprocessing.mean) / preprocessing.scale
    reference = np.asarray(estimator.predict(scaled), dtype=np.float64).ravel()
    if residual:
        reference = from_residual(reference, at_now)
    produced = predictor.predict_features(probe)
    return float(np.max(np.abs(produced - reference)))


def assert_parity_within(error: float, tolerance: float, label: str) -> float:
    """Hard gate. A candidate that does not reproduce is not a candidate.

    Parity is not a quality metric to report alongside the others — it is the
    condition under which the exported artifact means anything at all. A bundle
    that predicts something different from what was fitted is wrong regardless
    of how good the fitted model was.
    """
    if not float(error) <= float(tolerance):
        raise ParityGateError(
            f"{label}: neutral-runtime parity error {error} exceeds tolerance "
            f"{tolerance}. The candidate is ineligible and will not be exported."
        )
    return float(error)


def engineering_gate(
    per_fold: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Descriptive comparison of the challenger against persistence.

    Applies the observed engineering thresholds to a retrospective comparison.
    It is not a significance test, and the paired per-fold spread is reported
    beside the mean rather than folded into a single invented gate.
    """
    settings = config["selection"]["engineering_gate"]
    learned = np.asarray([row["validation_mae"] for row in per_fold], dtype=np.float64)
    baseline = np.asarray([row["persistence_mae"] for row in per_fold], dtype=np.float64)
    paired = baseline - learned

    mean_learned = float(np.mean(learned))
    mean_baseline = float(np.mean(baseline))
    absolute_gain = mean_baseline - mean_learned
    relative_gain = absolute_gain / mean_baseline if mean_baseline > 0 else 0.0

    clears_absolute = absolute_gain >= float(settings["absolute_mean_gain_min_degc"])
    clears_relative = relative_gain >= float(settings["relative_mean_gain_min_fraction"])
    consistent = bool(np.all(paired > 0))

    if clears_absolute and clears_relative and consistent:
        verdict = "pass"
    elif clears_absolute and clears_relative:
        verdict = "inconclusive"
    else:
        verdict = "fail"

    return {
        "status": settings["status"],
        "verdict": verdict,
        "mean_learned_mae_degc": mean_learned,
        "mean_persistence_mae_degc": mean_baseline,
        "absolute_mean_gain_degc": absolute_gain,
        "relative_mean_gain_fraction": relative_gain,
        "absolute_threshold_degc": settings["absolute_mean_gain_min_degc"],
        "relative_threshold_fraction": settings["relative_mean_gain_min_fraction"],
        "clears_absolute_threshold": bool(clears_absolute),
        "clears_relative_threshold": bool(clears_relative),
        "paired_per_fold_gain_degc": [float(v) for v in paired],
        "paired_gain_all_same_direction": consistent,
        "paired_gain_min_degc": float(np.min(paired)),
        "paired_gain_max_degc": float(np.max(paired)),
        "paired_gain_std_degc": float(np.std(paired)),
        "uncertainty_reporting": settings["uncertainty_reporting"],
        "boundary": settings["boundary"],
    }


def main(argv: list[str] | None = None) -> int:
    config = load_config()
    parser = argparse.ArgumentParser(description="HALO SafeShift Colab training entry point")
    parser.add_argument("--packet", type=Path, required=True, help="frozen colab-input directory")
    parser.add_argument("--out", type=Path, required=True, help="artifact output directory")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument(
        "--enable-xgboost",
        action="store_true",
        help=(
            "opt in to the four XGBoost trials. Without this flag they are excluded "
            "and the policy status stays skipped_by_policy. The flag alone does not "
            "make XGBoost eligible; the version check and the parity probe still gate it."
        ),
    )
    parser.add_argument(
        "--skip-packet-verification",
        action="store_true",
        help="NOT for normal use; retained only for offline debugging of a known-bad packet",
    )
    args = parser.parse_args(argv)

    run_id = args.run_id or make_run_id()
    args.out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Verify before reading any data
    # ------------------------------------------------------------------ #
    verification: dict[str, Any]
    if args.skip_packet_verification:
        verification = {"verified": False, "reason": "explicitly skipped by the operator"}
    else:
        verification = verify_packet(args.packet, config)

    mapping = resolve_field_mapping(
        load_channel_metadata(args.packet / "channel_metadata.json"), config
    )
    raw = load_raw_csv(args.packet / "station_raw.csv")
    admitted, row_counts = apply_row_qc(raw, mapping, config)
    binned = resample_five_minute(admitted, mapping, config)

    partition = json.loads((args.packet / "recovery_partition.json").read_text(encoding="utf-8"))
    recovery_start = partition.get("raw_feed_recovery_boundary") or partition.get(
        "recovery_start_utc"
    )
    if recovery_start:
        boundary = pd.Timestamp(recovery_start)
        historical = binned[pd.to_datetime(binned["bin_end_utc"], utc=True) < boundary].copy()
    else:
        historical = binned.copy()

    windows = build_windows(historical, mapping, config)
    folds = build_folds(windows, config)
    if not folds:
        raise SystemExit("no fold could be formed from the frozen packet")
    assert_fold_chronology(folds, windows)

    # ------------------------------------------------------------------ #
    # Baselines, per fold
    # ------------------------------------------------------------------ #
    baseline_rows: list[dict[str, Any]] = []
    for fold in folds:
        y_valid = windows.y[fold.validation_mask]
        at_valid = windows.at_now[fold.validation_mask]
        persistence = PersistenceBaseline.from_schema(windows.schema).predict(
            windows.x[fold.validation_mask]
        )
        climatology = LocalTimeClimatology.from_config(config).fit(
            windows.issue_times[fold.train_mask], windows.y[fold.train_mask]
        )
        baseline_rows.append(
            {
                "fold_index": fold.index,
                "n_validation": int(fold.n_validation),
                "persistence_mae": mean_absolute_error(y_valid, persistence),
                "climatology_mae": mean_absolute_error(
                    y_valid, climatology.predict(windows.issue_times[fold.validation_mask])
                ),
                "persistence_regime": regime_report(y_valid, persistence, at_valid),
            }
        )

    # ------------------------------------------------------------------ #
    # Bounded, explicit trial list
    # ------------------------------------------------------------------ #
    budget = trial_budget_report(config, enable_xgboost=args.enable_xgboost)
    xgboost_status = xgboost_policy_status(config, enable_xgboost=args.enable_xgboost)
    trials = curated_trials(config, enable_xgboost=args.enable_xgboost)

    results: list[dict[str, Any]] = []
    for trial in trials:
        per_fold = [fit_fold(trial, windows, fold, config) for fold in folds]
        results.append(
            {
                "trial_id": trial["trial_id"],
                "family": trial["family"],
                "parameterization": trial["parameterization"],
                "params": trial["params"],
                "parameterization_provenance": config["models"][
                    "parameterization_provenance"
                ].get(trial["parameterization"]),
                "mean_validation_mae": float(
                    np.mean([row["validation_mae"] for row in per_fold])
                ),
                "per_fold": per_fold,
            }
        )

    ranked = sorted(results, key=lambda row: row["mean_validation_mae"])
    best = ranked[0]
    best_trial = next(t for t in trials if t["trial_id"] == best["trial_id"])
    gate = engineering_gate(best["per_fold"], config)

    # ------------------------------------------------------------------ #
    # Final refit on ALL eligible historical-development windows
    # ------------------------------------------------------------------ #
    estimator, preprocessing = final_refit(best_trial, windows, config)
    export_type = FAMILY_TO_EXPORT_TYPE[best_trial["family"]]
    tolerance = _tolerance_for(export_type, best_trial["family"], config)

    probe = golden_vectors(windows, config)
    probe_at_now = probe[:, windows.schema["index"]["apparent_temperature_now"]]

    challenger_dir = args.out / "best_learned_challenger"
    export_estimator(
        challenger_dir,
        estimator,
        windows.schema,
        preprocessing,
        parameterization=best_trial["parameterization"],
        run_id=run_id,
        config=config,
        provenance={
            "selected_trial_id": best_trial["trial_id"],
            "selection_scope": config["selection"]["scope"],
            "final_refit_scope": config["selection"]["final_refit"]["rule"],
        },
    )
    challenger_parity = parity_error(
        challenger_dir,
        estimator,
        preprocessing,
        probe,
        best_trial["parameterization"] == "residual",
        probe_at_now,
    )
    challenger_eligible = challenger_parity <= tolerance

    # ------------------------------------------------------------------ #
    # Parity is a hard gate; persistence is the fallback
    # ------------------------------------------------------------------ #
    persistence_dir = args.out / "persistence_reference"
    export_persistence(persistence_dir, windows.schema, run_id=run_id, config=config)
    persistence_parity = parity_error(
        persistence_dir,
        PersistenceBaseline.from_schema(windows.schema),
        Preprocessing(kind="identity", fit_scope="not applicable"),
        probe,
        False,
        probe_at_now,
    )

    if not challenger_eligible:
        shutil.rmtree(challenger_dir)

    if challenger_eligible and gate["verdict"] == "pass":
        operational_model = "learned_challenger"
        operational_source = challenger_dir
    else:
        operational_model = "persistence"
        operational_source = persistence_dir

    if operational_model == "persistence":
        # If the fallback itself cannot reproduce, nothing eligible remains and
        # the run must fail rather than ship an unverified bundle.
        assert_parity_within(
            persistence_parity,
            float(config["tolerances"]["linear_fp64_max_abs_error"]),
            "persistence fallback (no eligible operational bundle would remain)",
        )

    operational_dir = args.out / "operational_bundle"
    if operational_dir.exists():
        shutil.rmtree(operational_dir)
    operational_dir.mkdir(parents=True)
    for item in sorted(operational_source.iterdir()):
        operational_dir.joinpath(item.name).write_bytes(item.read_bytes())

    report = {
        "artifact": "colab_training_report.json",
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "packet_verification": verification,
        "packet_source_manifest_sha256": sha256_file(args.packet / "source_manifest.json"),
        "row_counts": row_counts,
        "folds": fold_report(folds, windows, config),
        "baselines_per_fold": baseline_rows,
        "trial_budget": budget,
        "xgboost_policy": xgboost_status,
        "trials": ranked,
        "selection_scope": "retrospective blocked validation",
        "final_refit_scope": "all eligible historical-development windows",
        "prospective_quarantine_scored": False,
        "selected": {
            "trial_id": best_trial["trial_id"],
            "family": best_trial["family"],
            "params": best_trial["params"],
            "parameterization": best_trial["parameterization"],
            "export_type": export_type,
            "selection_rule": config["selection"]["rule"],
            "n_refit_windows": int(windows.x.shape[0]),
        },
        "parity_gate": {
            "status": config["selection"]["parity_gate"]["status"],
            "tolerance": tolerance,
            "n_probe_vectors": int(probe.shape[0]),
            "best_learned_challenger_max_abs_error": challenger_parity,
            "best_learned_challenger_eligible": bool(challenger_eligible),
            "persistence_max_abs_error": persistence_parity,
        },
        "engineering_gate_observed": gate,
        "evidence_status": "retrospective_only",
        "certified_untouched_test": False,
        "best_learned_challenger": (
            best_trial["trial_id"] if challenger_eligible else None
        ),
        "operational_model": operational_model,
        "claim_boundary": "not prospective generalization evidence",
        "boundary": [
            "Selection used validation blocks only.",
            "The prospective quarantine was not scored and did not inform any choice.",
            "Regime analysis is descriptive error analysis; it gated nothing.",
            "This report is not an UNO Q measurement.",
        ],
    }
    write_json(args.out / "colab_training_report.json", report)
    print(
        json.dumps(
            {
                "selected": report["selected"],
                "operational_model": operational_model,
                "engineering_gate_observed": gate["verdict"],
                "parity": report["parity_gate"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ParityGateError as exc:
        print(f"PARITY GATE FAILED: {exc}", file=sys.stderr)
        raise SystemExit(3) from exc
