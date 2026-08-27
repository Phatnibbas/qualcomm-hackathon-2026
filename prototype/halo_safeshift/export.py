"""Frozen, dependency-light export bundles.

A bundle directory contains exactly:

    manifest.json         model type, sibling SHA-256, parameters, boundaries
    model.npz             every numeric array, stored without pickle
    feature_schema.json   the frozen feature order
    preprocessing.json    descriptive preprocessing contract
    target_contract.json  what the target is, and what it is not

Nothing is pickled. The runtime loads ``model.npz`` with
``allow_pickle=False``, so an artifact that needs pickle to load is rejected
rather than executed.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import DEFAULT_CONFIG_PATH, REPO_ROOT, load_config, sha256_file, write_json
from .contracts import validate_preprocessing
from .target import target_contract
from .tree_export import (
    TreeExportError,
    export_sklearn_extra_trees,
    export_sklearn_gradient_boosting,
    export_xgboost_regressor,
    flatten_tree_ensemble,
    validate_flat_tree_arrays,
)

__all__ = [
    "ExportError",
    "SIBLING_FILES",
    "ENVELOPE_FILENAME",
    "Preprocessing",
    "write_bundle_envelope",
    "export_persistence",
    "export_linear",
    "export_mlp_relu",
    "export_tree_ensemble",
    "export_estimator",
]

SIBLING_FILES = ("model.npz", "feature_schema.json", "preprocessing.json", "target_contract.json")
ENVELOPE_FILENAME = "bundle_envelope.json"
BUNDLE_VERSION = "halo-safeshift-bundle-v1"
ENVELOPE_VERSION = "halo-safeshift-bundle-envelope-v1"


class ExportError(ValueError):
    """Raised when a bundle cannot be written in the neutral format."""


class Preprocessing:
    """Fold-fitted feature preprocessing, frozen into the bundle.

    ``mean``/``scale`` must be fitted on a fold's train block only. This class
    stores them; it does not fit them, so it cannot accidentally see a
    validation row.
    """

    def __init__(
        self,
        kind: str = "identity",
        mean: np.ndarray | None = None,
        scale: np.ndarray | None = None,
        clip_min: np.ndarray | None = None,
        clip_max: np.ndarray | None = None,
        fit_scope: str = "not fitted",
    ) -> None:
        if kind not in {"identity", "standardize"}:
            raise ExportError(f"unsupported preprocessing kind {kind!r}")
        if kind == "standardize" and (mean is None or scale is None):
            raise ExportError("standardize preprocessing requires mean and scale")
        self.kind = kind
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float64)
        self.scale = None if scale is None else np.asarray(scale, dtype=np.float64)
        self.clip_min = None if clip_min is None else np.asarray(clip_min, dtype=np.float64)
        self.clip_max = None if clip_max is None else np.asarray(clip_max, dtype=np.float64)
        self.fit_scope = fit_scope
        if (self.clip_min is None) != (self.clip_max is None):
            raise ExportError(
                "clipping bounds must be supplied together or not at all; one bound "
                "alone is a different transform from the one preprocessing.json will "
                "describe"
            )
        # Validate as early as the width can be inferred, using the same shared
        # contract the export and the runtime apply. Failing in the constructor
        # means a bad scaler never reaches a bundle directory at all.
        inferred = self._inferred_width()
        if inferred is not None:
            self.validate(inferred)

    def _inferred_width(self) -> int | None:
        for array in (self.mean, self.scale, self.clip_min, self.clip_max):
            if array is not None:
                return int(np.asarray(array).shape[0])
        return None

    def validate(self, n_features: int) -> dict[str, Any]:
        """Run the shared contract before the bundle is written.

        Uses exactly the checks the deployment runtime will re-run, so an
        artifact cannot pass export and then be rejected at load.
        """
        return validate_preprocessing(
            self.describe(), self.arrays(), n_features, error=ExportError
        )

    def arrays(self) -> dict[str, np.ndarray]:
        payload: dict[str, np.ndarray] = {}
        if self.mean is not None:
            payload["preproc_mean"] = self.mean
        if self.scale is not None:
            payload["preproc_scale"] = self.scale
        if self.clip_min is not None:
            payload["preproc_clip_min"] = self.clip_min
        if self.clip_max is not None:
            payload["preproc_clip_max"] = self.clip_max
        return payload

    def describe(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "arrays_in_model_npz": sorted(self.arrays().keys()),
            "clipping_applied": self.clip_min is not None or self.clip_max is not None,
            "fit_scope": self.fit_scope,
            "boundary": (
                "Preprocessing parameters must be fitted inside a single fold's "
                "train block. No validation or quarantine row may contribute."
            ),
        }


def _output_transform(parameterization: str, schema: dict[str, Any]) -> dict[str, Any]:
    if parameterization == "direct":
        return {"kind": "identity"}
    if parameterization == "residual":
        return {
            "kind": "residual_plus_feature",
            "feature_index": int(schema["feature_names"].index("apparent_temperature_now")),
            "feature_name": "apparent_temperature_now",
        }
    raise ExportError(f"unsupported parameterization {parameterization!r}")


def _write_bundle(
    bundle_dir: Path | str,
    *,
    model_type: str,
    arrays: dict[str, np.ndarray],
    schema: dict[str, Any],
    preprocessing: Preprocessing,
    model_params: dict[str, Any],
    parameterization: str,
    run_id: str | None,
    config: dict[str, Any],
    provenance: dict[str, Any] | None = None,
) -> Path:
    allowed = set(config["export"]["allowed_model_types"])
    if model_type not in allowed:
        raise ExportError(f"model_type {model_type!r} is not in allowed_model_types {sorted(allowed)}")

    target = Path(bundle_dir)
    target.mkdir(parents=True, exist_ok=True)

    n_features = int(schema["n_features"])
    preprocessing.validate(n_features)
    if model_type == "tree_ensemble":
        validate_flat_tree_arrays(
            arrays,
            n_features,
            aggregation=model_params.get("aggregation"),
            decision_rule=model_params.get("decision_rule"),
            comparison_dtype=model_params.get("split_comparison_dtype"),
            manifest_n_trees=model_params.get("n_trees"),
            manifest_n_nodes=model_params.get("n_nodes"),
            error=ExportError,
        )

    payload = {**arrays, **preprocessing.arrays()}
    for name, array in payload.items():
        if np.asarray(array).dtype == object:
            raise ExportError(f"array {name!r} has object dtype and would require pickle")
    npz_path = target / "model.npz"
    with npz_path.open("wb") as handle:
        np.savez(handle, allow_pickle=False, **payload)

    write_json(target / "feature_schema.json", schema)
    write_json(target / "preprocessing.json", preprocessing.describe())
    write_json(target / "target_contract.json", target_contract(config))

    schema_bytes = json.dumps(schema, indent=2, ensure_ascii=False).encode("utf-8")
    manifest = {
        "bundle_version": BUNDLE_VERSION,
        "config_id": config["config_id"],
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "model_type": model_type,
        "parameterization": parameterization,
        "parameterization_provenance": config["models"]["parameterization_provenance"].get(
            parameterization
        ),
        "n_features": int(schema["n_features"]),
        "feature_schema_sha256": sha256_file(target / "feature_schema.json"),
        "feature_schema_inline_sha256": hashlib.sha256(schema_bytes).hexdigest(),
        "model_params": model_params,
        "output_transform": _output_transform(parameterization, schema),
        "runtime_requirements": {
            "allowed": list(config["export"]["runtime_dependencies"]),
            "forbidden": list(config["export"]["forbidden_runtime_dependencies"]),
            "npz_allow_pickle": config["export"]["npz_allow_pickle"],
        },
        "provenance": provenance or {},
        "files": {name: sha256_file(target / name) for name in SIBLING_FILES},
        "claim_boundary": [
            "The bundle contains a predictor and its preprocessing. It contains no performance claim.",
            "Prediction is a station-derived shade apparent-temperature estimate, not a safety classification.",
        ],
    }
    write_json(target / "manifest.json", manifest)
    write_bundle_envelope(target, config=config)
    return target


def write_bundle_envelope(
    bundle_dir: Path | str,
    config: dict[str, Any] | None = None,
) -> Path:
    """Write the outer attestation over ``manifest.json`` and everything it names.

    ``manifest.json`` carries prediction-relevant metadata — the model type, the
    feature width, the residual transform's feature index, the tree and node
    counts. A bundle that verifies only against its own manifest cannot notice a
    manifest that was edited, because the manifest is both the claim and the
    authority for the claim. The envelope moves the attestation one level out:
    it hashes the manifest itself, every sibling, the runtime source files that
    will execute the prediction, and the experiment configuration, and it keeps
    an independent copy of the prediction-relevant fields so a partial edit
    shows up as a disagreement rather than as a new truth.
    """
    resolved = config or load_config()
    settings = resolved["envelope"]
    directory = Path(bundle_dir)
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise ExportError(f"{directory}: cannot envelope a bundle with no manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    params = manifest.get("model_params") or {}

    runtime_sources: dict[str, dict[str, Any]] = {}
    for relative in settings["runtime_source_files"]:
        source = REPO_ROOT / relative
        runtime_sources[relative] = (
            {"sha256": sha256_file(source), "present_at_export": True}
            if source.is_file()
            else {
                "sha256": None,
                "present_at_export": False,
                "note": "not present when the envelope was written; recorded as unverifiable",
            }
        )

    envelope = {
        "artifact": ENVELOPE_FILENAME,
        "envelope_version": ENVELOPE_VERSION,
        "purpose": settings["purpose"],
        "config_id": resolved["config_id"],
        "run_id": manifest.get("run_id"),
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "files": {
            "manifest.json": sha256_file(manifest_path),
            **{name: sha256_file(directory / name) for name in SIBLING_FILES},
        },
        "runtime_source_files": runtime_sources,
        "config_sha256": sha256_file(DEFAULT_CONFIG_PATH),
        "cross_checked_fields": list(settings["cross_checked_fields"]),
        "attested": {
            "model_type": manifest.get("model_type"),
            "n_features": manifest.get("n_features"),
            "feature_schema_sha256": manifest.get("feature_schema_sha256"),
            "output_transform": manifest.get("output_transform"),
            "model_params.n_trees": params.get("n_trees"),
            "model_params.n_nodes": params.get("n_nodes"),
            "preprocessing_kind": json.loads(
                (directory / "preprocessing.json").read_text(encoding="utf-8")
            ).get("kind"),
            "run_id": manifest.get("run_id"),
            "config_id": manifest.get("config_id"),
        },
        "boundary": [
            "Integrity attestation only. It contains no metric and no claim about model quality.",
            "It proves which bytes were exported, not that the model is any good.",
        ],
    }
    write_json(directory / ENVELOPE_FILENAME, envelope)
    return directory / ENVELOPE_FILENAME


# --------------------------------------------------------------------------- #
# Exporters
# --------------------------------------------------------------------------- #

def export_persistence(
    bundle_dir: Path | str,
    schema: dict[str, Any],
    *,
    run_id: str | None = None,
    config: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
) -> Path:
    """Persistence needs no weights, only the index of the AT-now feature."""
    resolved = config or load_config()
    index = int(schema["feature_names"].index("apparent_temperature_now"))
    return _write_bundle(
        bundle_dir,
        model_type="persistence",
        arrays={"at_now_index": np.asarray([index], dtype=np.int32)},
        schema=schema,
        preprocessing=Preprocessing(kind="identity", fit_scope="not applicable"),
        model_params={"at_now_feature_index": index, "definition": "AT(t + horizon) = AT(t)"},
        parameterization="direct",
        run_id=run_id,
        config=resolved,
        provenance=provenance,
    )


def export_linear(
    bundle_dir: Path | str,
    estimator: Any,
    schema: dict[str, Any],
    preprocessing: Preprocessing,
    *,
    parameterization: str = "direct",
    dtype: str = "float64",
    run_id: str | None = None,
    config: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
) -> Path:
    """Export a fitted linear regressor (Ridge, ElasticNet, ...)."""
    resolved = config or load_config()
    coef = np.asarray(getattr(estimator, "coef_"), dtype=np.float64).ravel()
    intercept = float(np.asarray(getattr(estimator, "intercept_")).ravel()[0])
    if coef.shape[0] != int(schema["n_features"]):
        raise ExportError(
            f"coefficient count {coef.shape[0]} does not match schema {schema['n_features']}"
        )
    numpy_dtype = np.float32 if dtype == "float32" else np.float64
    return _write_bundle(
        bundle_dir,
        model_type="linear",
        arrays={
            "linear_coef": coef.astype(numpy_dtype),
            "linear_intercept": np.asarray([intercept], dtype=numpy_dtype),
        },
        schema=schema,
        preprocessing=preprocessing,
        model_params={
            "source_estimator": type(estimator).__name__,
            "weight_dtype": dtype,
        },
        parameterization=parameterization,
        run_id=run_id,
        config=resolved,
        provenance=provenance,
    )


def export_mlp_relu(
    bundle_dir: Path | str,
    estimator: Any,
    schema: dict[str, Any],
    preprocessing: Preprocessing,
    *,
    parameterization: str = "direct",
    dtype: str = "float64",
    run_id: str | None = None,
    config: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
) -> Path:
    """Export a fitted ``MLPRegressor`` with ReLU hidden activations."""
    resolved = config or load_config()
    if getattr(estimator, "activation", None) != "relu":
        raise ExportError(
            f"only relu activations are exportable, got {getattr(estimator, 'activation', None)!r}"
        )
    if getattr(estimator, "out_activation_", "identity") != "identity":
        raise ExportError(
            f"only identity output activation is supported, got {estimator.out_activation_!r}"
        )
    coefs = list(getattr(estimator, "coefs_"))
    intercepts = list(getattr(estimator, "intercepts_"))
    if len(coefs) != len(intercepts):
        raise ExportError("weight/bias count mismatch")
    if np.asarray(coefs[0]).shape[0] != int(schema["n_features"]):
        raise ExportError("first layer input width does not match the schema")

    numpy_dtype = np.float32 if dtype == "float32" else np.float64
    arrays: dict[str, np.ndarray] = {"mlp_n_layers": np.asarray([len(coefs)], dtype=np.int32)}
    for index, (weight, bias) in enumerate(zip(coefs, intercepts)):
        arrays[f"mlp_W{index}"] = np.asarray(weight, dtype=numpy_dtype)
        arrays[f"mlp_b{index}"] = np.asarray(bias, dtype=numpy_dtype).ravel()

    return _write_bundle(
        bundle_dir,
        model_type="mlp_relu",
        arrays=arrays,
        schema=schema,
        preprocessing=preprocessing,
        model_params={
            "source_estimator": type(estimator).__name__,
            "n_layers": len(coefs),
            "hidden_layer_sizes": [int(np.asarray(w).shape[1]) for w in coefs[:-1]],
            "hidden_activation": "relu",
            "output_activation": "identity",
            "weight_dtype": dtype,
        },
        parameterization=parameterization,
        run_id=run_id,
        config=resolved,
        provenance=provenance,
    )


def export_tree_ensemble(
    bundle_dir: Path | str,
    neutral: dict[str, Any],
    schema: dict[str, Any],
    preprocessing: Preprocessing,
    *,
    parameterization: str = "direct",
    run_id: str | None = None,
    config: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
) -> Path:
    """Export an already-converted neutral tree ensemble."""
    resolved = config or load_config()
    if neutral.get("model_type") != "tree_ensemble":
        raise ExportError("neutral bundle is not a tree_ensemble")
    if neutral.get("decision_rule") not in set(resolved["export"]["tree_decision_rules"]):
        raise ExportError(f"unsupported decision_rule {neutral.get('decision_rule')!r}")

    arrays = flatten_tree_ensemble(neutral)
    return _write_bundle(
        bundle_dir,
        model_type="tree_ensemble",
        arrays=arrays,
        schema=schema,
        preprocessing=preprocessing,
        model_params={
            "aggregation": neutral["aggregation"],
            "base_score": float(neutral["base_score"]),
            "learning_rate": float(neutral["learning_rate"]),
            "decision_rule": neutral["decision_rule"],
            "split_comparison_dtype": neutral.get(
                "split_comparison_dtype",
                resolved["export"]["tree_split_comparison_dtype"],
            ),
            "n_trees": len(neutral["trees"]),
            "n_nodes": int(arrays["tree_offsets"][-1]),
            "source_estimator": neutral.get("source_estimator", "unknown"),
        },
        parameterization=parameterization,
        run_id=run_id,
        config=resolved,
        provenance=provenance,
    )


def export_estimator(
    bundle_dir: Path | str,
    estimator: Any,
    schema: dict[str, Any],
    preprocessing: Preprocessing,
    *,
    parameterization: str = "direct",
    dtype: str = "float64",
    run_id: str | None = None,
    config: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
) -> Path:
    """Dispatch a fitted estimator to the matching neutral exporter."""
    name = type(estimator).__name__
    common = {
        "parameterization": parameterization,
        "run_id": run_id,
        "config": config,
        "provenance": provenance,
    }
    if name in {"Ridge", "ElasticNet", "Lasso", "LinearRegression"}:
        return export_linear(bundle_dir, estimator, schema, preprocessing, dtype=dtype, **common)
    if name == "MLPRegressor":
        return export_mlp_relu(bundle_dir, estimator, schema, preprocessing, dtype=dtype, **common)
    if name == "GradientBoostingRegressor":
        neutral = export_sklearn_gradient_boosting(estimator)
    elif name == "ExtraTreesRegressor":
        neutral = export_sklearn_extra_trees(estimator)
    elif name == "XGBRegressor":
        neutral = export_xgboost_regressor(estimator)
    else:
        raise TreeExportError(f"no neutral exporter is registered for {name!r}")
    return export_tree_ensemble(bundle_dir, neutral, schema, preprocessing, **common)
