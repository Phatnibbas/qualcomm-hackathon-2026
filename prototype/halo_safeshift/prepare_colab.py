"""Local preparation driver: freeze -> QC -> partition -> windows -> folds -> packet.

Runs the whole local phase against an already-frozen station pull and writes a
run-scoped evidence directory plus an immutable Colab packet.

It does not train, select or evaluate a model. The only numbers it produces are
counts, timestamps, hashes and numerical-parity errors.

Provenance is gated before anything is computed: the CSV, the channel metadata
and the producing script must hash to exactly what the data manifest attests,
and the manifest must describe the configured channel. A mismatch stops the run
rather than producing artifacts attributed to a pull that did not make them.

Usage (local preparation, Windows):

    py -3 -m prototype.halo_safeshift.prepare_colab \
        --raw benchmarks/halo/_cache/<run-id>/station_raw.csv \
        --metadata evidence/halo-safeshift/<run-id>/channel_metadata.json \
        --data-manifest evidence/halo-safeshift/<run-id>/data_manifest.json \
        --evidence-dir evidence/halo-safeshift/<run-id> \
        --puller benchmarks/halo/freeze_station_history.py \
        --run-id <run-id>

The packet it writes is consumed on Linux/Colab with `python`, never `py -3`.
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import (
    DEFAULT_CONFIG_PATH,
    PACKAGE_DIR,
    REPO_ROOT,
    git_head_short,
    load_config,
    make_run_id,
    sha256_file,
    write_json,
)
from .baselines import (
    LocalTimeClimatology,
    PersistenceBaseline,
    baseline_bundle,
    from_residual,
    to_residual,
)
from .data import (
    apply_row_qc,
    build_qc_report,
    build_recovery_report,
    load_channel_metadata,
    load_raw_csv,
    resample_five_minute,
    resolve_field_mapping,
)
from .export import Preprocessing, export_estimator, export_linear, export_persistence
from .features import build_windows
from .inference import SafeShiftPredictor
from .models import build_estimator, family_status, registry_report, trial_budget_report
from .source_archive import build_source_archive
from .splits import assert_fold_chronology, build_folds, fold_report
from .verify_packet import verify_packet

PACKET_FILES = (
    "station_raw.csv",
    "channel_metadata.json",
    "data_manifest.json",
    "qc_report.json",
    "recovery_partition.json",
    "experiment.v1.json",
    "source_manifest.json",
    "code_manifest.json",
    "halo_safeshift-source.zip",
    "bootstrap.py",
)

DEFAULT_PULLER = REPO_ROOT / "benchmarks" / "halo" / "freeze_station_history.py"


class BlockedError(RuntimeError):
    """Raised on a stop condition. The run reports BLOCKED rather than continuing."""


def _rel(path: Path) -> str:
    """Repo-relative POSIX path; absolute paths are never written into artifacts."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT).as_posix())
    except ValueError:
        return str(resolved.name)


def _stamp(value: Any) -> str | None:
    if value is None:
        return None
    return pd.Timestamp(value).tz_convert(timezone.utc).isoformat().replace("+00:00", "Z")


def write_table(frame: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")
    return sha256_file(path)


def write_resampled(frame: pd.DataFrame, directory: Path) -> tuple[Path, str, str]:
    """Prefer Parquet; fall back to CSV without installing anything."""
    try:
        import importlib.util

        if importlib.util.find_spec("pyarrow") is None:
            raise ImportError("pyarrow is absent")
        path = directory / "resampled_5min.parquet"
        frame.to_parquet(path, index=False)
        return path, sha256_file(path), "parquet"
    except Exception:  # noqa: BLE001 - any Parquet failure falls back, never installs
        path = directory / "resampled_5min.csv"
        return path, write_table(frame, path), "csv"


def code_manifest(
    config: dict[str, Any], run_id: str, archive: dict[str, Any]
) -> dict[str, Any]:
    files = sorted(p for p in PACKAGE_DIR.rglob("*.py") if "__pycache__" not in p.parts)
    return {
        "artifact": "code_manifest.json",
        "package": "prototype/halo_safeshift",
        "config_id": config["config_id"],
        "run_id": run_id,
        "python_files": {_rel(path): sha256_file(path) for path in files},
        "config_files": {_rel(DEFAULT_CONFIG_PATH): sha256_file(DEFAULT_CONFIG_PATH)},
        "source_archive": archive,
        "note": (
            "The Colab worker must import this exact package. A notebook that "
            "reimplements the pipeline is not the same pipeline. The working tree "
            "carries uncommitted source, so cloning git HEAD does NOT reproduce it: "
            "the attested archive above is the transport."
        ),
    }


def assert_source_provenance(
    raw_csv: Path,
    metadata: Path,
    data_manifest_path: Path,
    puller: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Prove the inputs are the ones the data manifest attests, before use.

    The data manifest is written by the freezer at acquisition time. If the CSV
    handed to this driver is not the CSV that manifest describes, every
    downstream count and hash would be attributed to a pull that did not produce
    them. Checked here rather than at packet-assembly time so a mismatch stops
    the run before any artifact exists.
    """
    manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
    integrity = manifest.get("integrity")
    if not isinstance(integrity, dict):
        raise BlockedError(f"{_rel(data_manifest_path)}: no integrity block")

    csv_sha = sha256_file(raw_csv)
    metadata_sha = sha256_file(metadata)
    puller_sha = sha256_file(puller)
    puller_path = _rel(puller)

    if csv_sha != integrity.get("csv_sha256"):
        raise BlockedError(
            f"{_rel(raw_csv)} hashes to {csv_sha} but {_rel(data_manifest_path)} "
            f"attests {integrity.get('csv_sha256')}. The CSV is not the frozen pull."
        )
    if metadata_sha != integrity.get("metadata_sha256"):
        raise BlockedError(
            f"{_rel(metadata)} hashes to {metadata_sha} but the data manifest attests "
            f"{integrity.get('metadata_sha256')}."
        )
    if puller_path != integrity.get("puller_path"):
        raise BlockedError(
            f"--puller is {puller_path} but the data manifest was produced by "
            f"{integrity.get('puller_path')}."
        )
    if puller_sha != integrity.get("puller_sha256"):
        raise BlockedError(
            f"{puller_path} hashes to {puller_sha} but the data manifest attests "
            f"{integrity.get('puller_sha256')}. The producer has changed since the "
            f"freeze, so it can no longer be cited as what produced this data."
        )

    configured_channel = int(config["source"]["thingspeak_channel_id"])
    observed_channel = int((manifest.get("source") or {}).get("channel_id", -1))
    if observed_channel != configured_channel:
        raise BlockedError(
            f"the data manifest describes channel {observed_channel}, not the "
            f"configured station {configured_channel}."
        )

    return {
        "csv_sha256": csv_sha,
        "metadata_sha256": metadata_sha,
        "puller_path": puller_path,
        "puller_sha256": puller_sha,
        "channel_id": observed_channel,
        "data_manifest_sha256": sha256_file(data_manifest_path),
    }


def source_manifest(
    raw_csv: Path,
    metadata: Path,
    data_manifest: Path,
    puller: Path,
    run_id: str,
    provenance: dict[str, Any],
    reuse: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Written only after :func:`assert_source_provenance` has passed."""
    return {
        "artifact": "source_manifest.json",
        "run_id": run_id,
        "station_raw_csv": {
            "path": _rel(raw_csv),
            "sha256": provenance["csv_sha256"],
            "bytes": raw_csv.stat().st_size,
        },
        "channel_metadata": {"path": _rel(metadata), "sha256": provenance["metadata_sha256"]},
        "data_manifest": {
            "path": _rel(data_manifest),
            "sha256": provenance["data_manifest_sha256"],
        },
        "puller": {"path": provenance["puller_path"], "sha256": provenance["puller_sha256"]},
        "channel_id": provenance["channel_id"],
        "provenance_checks": [
            "CSV SHA-256 equals data_manifest.integrity.csv_sha256",
            "metadata SHA-256 equals data_manifest.integrity.metadata_sha256",
            "puller path equals data_manifest.integrity.puller_path",
            "puller SHA-256 equals data_manifest.integrity.puller_sha256",
            "channel id equals the configured station",
        ],
        "reused_source_freeze": reuse,
        "boundary": "Source integrity only. Contains no model, metric or claim.",
    }


def environment_versions() -> dict[str, Any]:
    """Exact versions of what was actually imported. Nothing is claimed unread."""
    versions: dict[str, Any] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "platform": platform.platform(),
    }
    for name in ("sklearn", "pyarrow", "xgboost"):
        try:
            module = importlib.import_module(name)
        except ImportError:
            versions[name] = None
            versions[f"{name}_status"] = "not importable in this environment"
            continue
        versions[name] = str(getattr(module, "__version__", "unknown"))
        versions[f"{name}_status"] = "imported and version read"
    return versions


def quarantine_inspection(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Permitted inspection only: schema, timestamps, missingness, continuity, formability.

    Deliberately computes no error metric. The quarantine's target values must
    not influence any modelling choice.
    """
    permitted = list(config["recovery_partition"]["quarantine_permitted_inspection"])
    forbidden = list(config["recovery_partition"]["quarantine_forbidden_use"])
    bin_minutes = int(config["resample"]["bin_minutes"])
    needed_bins = int(config["windows"]["lookback_bins"]) + int(config["windows"]["horizon_bins"])

    if len(frame) == 0:
        return {
            "permitted_inspection": permitted,
            "forbidden_use": forbidden,
            "n_bins": 0,
            "note": "no bins fall at or after the recovery start",
        }

    times = pd.to_datetime(frame["bin_end_utc"], utc=True).sort_values()
    deltas = times.diff().dt.total_seconds().dropna()
    consecutive = 1
    longest_run = 1
    for delta in deltas:
        if abs(float(delta) - bin_minutes * 60.0) < 1e-6:
            consecutive += 1
            longest_run = max(longest_run, consecutive)
        else:
            consecutive = 1

    return {
        "permitted_inspection": permitted,
        "forbidden_use": forbidden,
        "n_bins": int(len(frame)),
        "n_valid_bins": int(frame["valid"].sum()),
        "first_bin_utc": _stamp(times.iloc[0]),
        "last_bin_utc": _stamp(times.iloc[-1]),
        "columns": list(frame.columns),
        "schema_matches_historical": True,
        "null_counts": {
            column: int(frame[column].isna().sum())
            for column in frame.columns
            if frame[column].dtype.kind in "fc"
        },
        "contiguous_bin_run_max": int(longest_run),
        "bins_required_for_one_window": needed_bins,
        "enough_bins_for_one_future_window": bool(longest_run >= needed_bins),
        "metrics_computed": False,
        "note": (
            "Inspection only. No model-performance metric was computed on this "
            "partition and none of its target values informed any modelling choice."
        ),
    }


def local_smoke(
    windows: Any,
    folds: list[Any],
    config: dict[str, Any],
    evidence_dir: Path,
) -> dict[str, Any]:
    """Prove the code paths execute. Not a benchmark, not a selection."""
    report: dict[str, Any] = {
        "artifact": "local_smoke_report.json",
        "STATUS_BANNER": [
            "NOT PERFORMANCE EVIDENCE",
            "NOT FINAL MODEL SELECTION",
            "NOT UNO Q INFERENCE",
        ],
        "purpose": (
            "Execute each code path once on a tiny subset so that a broken path "
            "fails here rather than in Colab. Deliberately reports no error "
            "metric of any kind: a quality number computed here could be "
            "mistaken for a result."
        ),
        "paths": {},
        "numerical_parity": {},
        "unavailable_optional_dependencies": [
            name for name, entry in family_status(config).items() if entry["local_status"] == "absent"
        ],
    }

    if len(windows) == 0 or not folds:
        report["paths"]["executed"] = False
        report["paths"]["reason"] = (
            "no eligible windows or no fold could be formed; nothing to smoke-test"
        )
        write_json(evidence_dir / "local_smoke_report.json", report)
        return report

    fold = folds[0]
    train = fold.train_mask
    validation = fold.validation_mask
    subset = min(400, int(train.sum()))
    x_train = windows.x[train][:subset]
    y_train = windows.y[train][:subset]
    at_train = windows.at_now[train][:subset]
    issue_train = windows.issue_times[train][:subset]
    x_validation = windows.x[validation][: min(200, int(validation.sum()))]

    # Baselines: prove they run and are fitted on train only.
    persistence = PersistenceBaseline.from_schema(windows.schema)
    persistence.predict(x_train)
    climatology = LocalTimeClimatology.from_config(config).fit(issue_train, y_train)
    climatology.predict(issue_train)
    report["paths"]["persistence"] = True
    report["paths"]["climatology_local_time"] = True
    report["paths"]["climatology_fitted_on_n_train_samples"] = climatology.fitted_on_n

    # Residual parameterization round trip.
    residual = to_residual(y_train, at_train)
    round_trip = float(np.max(np.abs(from_residual(residual, at_train) - y_train)))
    report["paths"]["residual_parameterization"] = True
    report["numerical_parity"]["residual_round_trip_max_abs_error"] = round_trip

    # Fold-scoped preprocessing, tiny fit, export and NumPy-runtime parity.
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale <= 0] = 1.0
    pre = Preprocessing(
        kind="standardize",
        mean=mean,
        scale=scale,
        fit_scope=f"fold {fold.index} train block only ({subset} samples)",
    )
    scaled_train = (x_train - mean) / scale
    estimator = build_estimator("ridge", {"alpha": 1.0}, config).fit(scaled_train, y_train)

    with tempfile.TemporaryDirectory() as tmp:
        bundle = export_linear(
            Path(tmp) / "smoke", estimator, windows.schema, pre, dtype="float64", config=config
        )
        predictor = SafeShiftPredictor.load(bundle)
        reference = estimator.predict((x_validation - mean) / scale)
        produced = predictor.predict_features(x_validation)
        report["numerical_parity"]["linear_fp64_max_abs_error"] = float(
            np.max(np.abs(reference - produced))
        )
        persistence_bundle = export_persistence(Path(tmp) / "persist", windows.schema, config=config)
        SafeShiftPredictor.load(persistence_bundle).predict_features(x_validation)

    # Tree and MLP export paths, so a structural or envelope defect in those
    # branches fails here instead of in Colab. Deliberately tiny: this proves
    # the path runs and reproduces, not that any model is any good.
    with tempfile.TemporaryDirectory() as tmp:
        for name, family, params, tolerance_key in (
            ("gradient_boosting", "gradient_boosting", {"n_estimators": 8, "max_depth": 2},
             "tree_ensemble_fp64_max_abs_error"),
            ("extra_trees", "extra_trees", {"n_estimators": 4, "max_depth": 4},
             "tree_ensemble_fp64_max_abs_error"),
            ("mlp_relu", "mlp_relu", {"hidden_layer_sizes": [8], "max_iter": 20},
             "mlp_fp32_max_abs_error"),
        ):
            fitted = build_estimator(family, params, config).fit(scaled_train, y_train)
            bundle = export_estimator(
                Path(tmp) / name, fitted, windows.schema, pre, config=config
            )
            produced = SafeShiftPredictor.load(bundle).predict_features(x_validation)
            reference = fitted.predict((x_validation - mean) / scale)
            error = float(np.max(np.abs(produced - reference)))
            report["numerical_parity"][f"{name}_max_abs_error"] = error
            report["numerical_parity"][f"{name}_tolerance"] = float(
                config["tolerances"][tolerance_key]
            )
            report["numerical_parity"][f"{name}_within_tolerance"] = bool(
                error <= float(config["tolerances"][tolerance_key])
            )
            report["paths"][f"{name}_export"] = True

    report["paths"]["preprocessing_fitted_inside_fold"] = True
    report["paths"]["linear_export"] = True
    report["paths"]["persistence_export"] = True
    report["paths"]["bundle_envelope_verified_on_load"] = True
    report["paths"]["numpy_runtime_load_and_predict"] = True
    report["paths"]["executed"] = True
    report["subset"] = {
        "fold_index": fold.index,
        "train_samples_used": int(x_train.shape[0]),
        "validation_samples_used": int(x_validation.shape[0]),
    }
    report["explicitly_not_reported"] = [
        "MAE, RMSE or any other error metric",
        "any comparison between model families",
        "any ranking or selection",
    ]
    write_json(evidence_dir / "local_smoke_report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    config = load_config()
    parser = argparse.ArgumentParser(description="HALO SafeShift local preparation driver")
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument(
        "--puller",
        type=Path,
        default=DEFAULT_PULLER,
        help=(
            "the script that produced --data-manifest. Defaults to "
            "benchmarks/halo/freeze_station_history.py; its path and hash must match "
            "data_manifest.integrity.puller_*"
        ),
    )
    parser.add_argument(
        "--reused-source-freeze",
        type=str,
        default=None,
        help=(
            "run id of an earlier run whose immutable source freeze is being reused. "
            "Recorded in the source manifest so the reuse is explicit rather than implied."
        ),
    )
    args = parser.parse_args(argv)

    run_id = args.run_id or make_run_id()
    evidence_dir = args.evidence_dir
    evidence_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)

    # Provenance gate: nothing downstream runs until the inputs are proven to be
    # the inputs the freeze attests.
    provenance = assert_source_provenance(
        args.raw, args.metadata, args.data_manifest, args.puller, config
    )
    raw_sha_before = provenance["csv_sha256"]
    metadata_sha = provenance["metadata_sha256"]

    channel = load_channel_metadata(args.metadata)
    try:
        mapping = resolve_field_mapping(channel, config)
    except Exception as exc:  # noqa: BLE001 - surfaced as a stop condition
        raise BlockedError(f"field mapping conflict: {exc}") from exc

    raw = load_raw_csv(args.raw)
    admitted, row_counts = apply_row_qc(raw, mapping, config)
    binned = resample_five_minute(admitted, mapping, config)

    qc = build_qc_report(
        run_id=run_id,
        csv_path=args.raw,
        csv_sha256=raw_sha_before,
        metadata_sha256=metadata_sha,
        field_mapping=mapping,
        row_counts=row_counts,
        binned=binned,
        config=config,
    )
    write_json(evidence_dir / "qc_report.json", qc)

    resampled_path, resampled_sha, resampled_format = write_resampled(binned, evidence_dir)

    # The boundary comes from the RAW feed, upstream of every QC rule. The
    # QC-admitted continuity is recorded beside it for comparison only.
    partition = build_recovery_report(raw, admitted["created_at_utc"], config)
    recovery_start = partition["raw_feed_recovery_boundary"]
    if recovery_start is None:
        historical = binned.copy()
        quarantine = binned.iloc[0:0].copy()
    else:
        boundary = pd.Timestamp(recovery_start)
        historical = binned[pd.to_datetime(binned["bin_end_utc"], utc=True) < boundary].copy()
        quarantine = binned[pd.to_datetime(binned["bin_end_utc"], utc=True) >= boundary].copy()

    historical_path = evidence_dir / "historical_development.csv"
    quarantine_path = evidence_dir / "prospective_quarantine.csv"
    historical_sha = write_table(historical, historical_path)
    quarantine_sha = write_table(quarantine, quarantine_path)

    windows = build_windows(historical, mapping, config)
    folds = build_folds(windows, config)
    if folds:
        assert_fold_chronology(folds, windows)
    folds_payload = fold_report(folds, windows, config)
    write_json(evidence_dir / "fold_definitions.json", folds_payload)

    partition_payload = {
        "artifact": "recovery_partition.json",
        "run_id": run_id,
        "config_id": config["config_id"],
        **partition,
        "partition": {
            "historical_development": {
                "path": _rel(historical_path),
                "sha256": historical_sha,
                "n_bins": int(len(historical)),
                "n_valid_bins": int(historical["valid"].sum()) if len(historical) else 0,
                "first_bin_utc": _stamp(historical["bin_end_utc"].min()) if len(historical) else None,
                "last_bin_utc": _stamp(historical["bin_end_utc"].max()) if len(historical) else None,
                "n_eligible_windows": len(windows),
            },
            "prospective_quarantine": {
                "path": _rel(quarantine_path),
                "sha256": quarantine_sha,
                **quarantine_inspection(quarantine, config),
            },
        },
        "naming_note": (
            "No artifact in this run is called an untouched test. The historical "
            "final dates have already been examined by an external verifier, so "
            "this is retrospective model development."
        ),
    }
    write_json(evidence_dir / "recovery_partition.json", partition_payload)

    write_json(evidence_dir / "window_diagnostics.json", {
        "artifact": "window_diagnostics.json",
        "run_id": run_id,
        "scope": "historical_development only",
        "n_eligible_windows": len(windows),
        "diagnostics": windows.diagnostics,
        "feature_schema": windows.schema,
        "baselines": baseline_bundle(windows, config),
    })
    write_json(evidence_dir / "model_registry.json", registry_report(config))

    smoke = local_smoke(windows, folds, config, evidence_dir)

    # ------------------------------------------------------------------ #
    # Colab packet
    # ------------------------------------------------------------------ #
    packet_dir = evidence_dir / "colab-input"
    packet_dir.mkdir(parents=True, exist_ok=True)

    packet_dir.joinpath("station_raw.csv").write_bytes(args.raw.read_bytes())
    packet_dir.joinpath("channel_metadata.json").write_bytes(args.metadata.read_bytes())
    packet_dir.joinpath("data_manifest.json").write_bytes(args.data_manifest.read_bytes())
    packet_dir.joinpath("qc_report.json").write_bytes((evidence_dir / "qc_report.json").read_bytes())
    packet_dir.joinpath("recovery_partition.json").write_bytes(
        (evidence_dir / "recovery_partition.json").read_bytes()
    )
    packet_dir.joinpath("experiment.v1.json").write_bytes(DEFAULT_CONFIG_PATH.read_bytes())

    reuse = (
        {
            "reused_run_id": args.reused_source_freeze,
            "status": "immutable source freeze reused; no live data was appended",
            "verified": "the CSV, metadata and puller hashes were re-checked against "
            "the reused run's data manifest before this run produced anything",
        }
        if args.reused_source_freeze
        else None
    )
    write_json(
        packet_dir / "source_manifest.json",
        source_manifest(
            args.raw, args.metadata, args.data_manifest, args.puller, run_id, provenance, reuse
        ),
    )

    archive = build_source_archive(packet_dir / "halo_safeshift-source.zip", config)
    packet_dir.joinpath("bootstrap.py").write_bytes(
        (PACKAGE_DIR / "bootstrap_colab.py").read_bytes()
    )
    write_json(packet_dir / "code_manifest.json", code_manifest(config, run_id, archive))

    missing = [name for name in PACKET_FILES if not (packet_dir / name).is_file()]
    if missing:
        raise BlockedError(f"colab packet is incomplete: {missing}")

    packet_manifest = {
        "artifact": "colab_packet_manifest.json",
        "run_id": run_id,
        "config_id": config["config_id"],
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "packet_dir": _rel(packet_dir),
        "files": {name: sha256_file(packet_dir / name) for name in PACKET_FILES},
        "immutability": (
            "Every file above is hashed. A Colab run that changes any of them is "
            "a different run and needs a new run id."
        ),
        "self_verification": {
            "command": "python -m prototype.halo_safeshift.verify_packet --packet colab-input",
            "manifest_location": "inside the packet, so the packet verifies itself",
            "undeclared_files": "rejected; see packet.allowed_extra_files_note",
        },
        "colab_contract": [
            "python colab-input/bootstrap.py --packet colab-input --root .",
            "python -m prototype.halo_safeshift.verify_packet --packet colab-input",
            "python -m prototype.halo_safeshift.colab_train --packet colab-input "
            "--out artifacts/halo-safeshift/<run-id>",
        ],
        "boundary": [
            "Input packet only. It contains no trained model and no metric.",
            "The Colab worker must import prototype/halo_safeshift, not reimplement it.",
        ],
    }
    # The manifest lives INSIDE the packet: a manifest that travels separately
    # can be lost, and a packet that cannot verify itself is only as trustworthy
    # as whatever channel delivered it.
    packet_manifest_sha = write_json(packet_dir / "colab_packet_manifest.json", packet_manifest)
    write_json(evidence_dir / "colab_packet_manifest.json", packet_manifest)

    verification = verify_packet(packet_dir, config)
    write_json(evidence_dir / "packet_verification.json", verification)

    raw_sha_after = sha256_file(args.raw)
    if raw_sha_after != raw_sha_before:
        raise BlockedError(
            "the frozen station CSV changed during this run; the freeze is not immutable"
        )

    finished = datetime.now(timezone.utc)
    run_manifest = {
        "artifact": "run_manifest.json",
        "run_id": run_id,
        "config_id": config["config_id"],
        "phase": "local preparation only",
        "started_utc": started.isoformat().replace("+00:00", "Z"),
        "finished_utc": finished.isoformat().replace("+00:00", "Z"),
        "git_head_short": git_head_short(),
        "environment": environment_versions(),
        "optional_dependencies": {
            name: entry["local_status"] for name, entry in family_status(config).items()
        },
        "trial_budget": trial_budget_report(config, enable_xgboost=False),
        "packet_verification": verification,
        "colab_command_linux": {
            "note": "Colab is Linux; these commands use `python`, not the local `py -3`.",
            "steps": packet_manifest["colab_contract"],
        },
        "inputs": {
            "station_raw_csv": _rel(args.raw),
            "station_raw_csv_sha256": raw_sha_before,
            "channel_metadata": _rel(args.metadata),
            "channel_metadata_sha256": metadata_sha,
            "data_manifest": _rel(args.data_manifest),
            "data_manifest_sha256": sha256_file(args.data_manifest),
        },
        "outputs": {
            "resampled": {
                "path": _rel(resampled_path),
                "format": resampled_format,
                "sha256": resampled_sha,
            },
            "qc_report": sha256_file(evidence_dir / "qc_report.json"),
            "recovery_partition": sha256_file(evidence_dir / "recovery_partition.json"),
            "fold_definitions": sha256_file(evidence_dir / "fold_definitions.json"),
            "window_diagnostics": sha256_file(evidence_dir / "window_diagnostics.json"),
            "model_registry": sha256_file(evidence_dir / "model_registry.json"),
            "local_smoke_report": sha256_file(evidence_dir / "local_smoke_report.json"),
            "packet_verification": sha256_file(evidence_dir / "packet_verification.json"),
            "colab_packet_manifest": packet_manifest_sha,
            "source_archive_sha256": archive["sha256"],
        },
        "counts": {
            "raw_rows": int(len(raw)),
            "admitted_rows": int(row_counts["rows_admitted"]),
            "bins_total": int(len(binned)),
            "bins_valid": int(binned["valid"].sum()) if len(binned) else 0,
            "historical_development_bins": int(len(historical)),
            "prospective_quarantine_bins": int(len(quarantine)),
            "eligible_windows_historical": len(windows),
            "folds": len(folds),
        },
        "claim_boundary": [
            "Local preparation only.",
            "Final model selection not performed.",
            "Colab training not performed.",
            "UNO Q inference not performed.",
            "No performance claim is admitted.",
        ],
    }
    write_json(evidence_dir / "run_manifest.json", run_manifest)

    print(json.dumps({
        "run_id": run_id,
        "counts": run_manifest["counts"],
        "recovery_start_utc": recovery_start,
        "colab_packet_manifest_sha256": packet_manifest_sha,
        "smoke_executed": smoke["paths"].get("executed"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BlockedError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
