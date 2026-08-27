"""Pure-NumPy inference runtime.

Imports only the Python standard library and NumPy, because that is all the
UNO Q deployment runtime is permitted to provide. There is no scikit-learn, no
XGBoost, no pandas and no pickle on this path.

The predictor fails closed. Every rejection below raises rather than returning
a degraded number:

* a missing or unverifiable ``bundle_envelope.json``;
* an envelope that disagrees with ``manifest.json`` about anything that changes
  a prediction;
* missing manifest or missing sibling file;
* sibling SHA-256 mismatch;
* feature-schema mismatch;
* unsupported model type;
* malformed preprocessing (wrong shape, non-positive scale, half-declared
  clipping, JSON that disagrees with the arrays);
* a structurally invalid tree ensemble (unequal node arrays, bad offsets, a
  child pointing outside its own tree, an out-of-range split feature);
* wrong feature count;
* non-finite input;
* corrupted or object-dtype arrays;
* an artifact that would need ``allow_pickle=True`` to load.

``manifest.json`` is not self-authorising. It carries prediction-relevant
metadata, so it is verified *from outside* by the envelope before any field is
read out of it.

Missing-value routing in tree bundles is inert here by policy: non-finite input
is rejected before traversal, so the branch cannot be reached and the runtime
does not claim to support it.

It returns a prediction and runtime provenance. It returns no medical or
safety classification of any kind.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import validate_preprocessing
from .tree_export import (
    DEFAULT_SPLIT_COMPARISON_DTYPE,
    split_values,
    validate_flat_tree_arrays,
)

__all__ = ["InferenceError", "SafeShiftPredictor"]

SUPPORTED_MODEL_TYPES = ("persistence", "linear", "mlp_relu", "tree_ensemble")
SIBLING_FILES = ("model.npz", "feature_schema.json", "preprocessing.json", "target_contract.json")
ENVELOPE_FILENAME = "bundle_envelope.json"

LEAF = -1
MISSING_LEFT = 0


class InferenceError(RuntimeError):
    """Raised whenever the runtime refuses to serve a prediction."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise InferenceError(f"{path}: {label} is missing")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InferenceError(f"{path}: {label} is not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise InferenceError(f"{path}: {label} root must be an object")
    return payload


def _verify_envelope(directory: Path) -> dict[str, Any]:
    """Verify the outer attestation before ``manifest.json`` is trusted at all.

    Order matters. The manifest is the thing being checked, so nothing may be
    read out of it — not the model type, not the feature width, not the sibling
    hash list — until its own bytes have been attested from outside.
    """
    envelope = _read_json(directory / ENVELOPE_FILENAME, "bundle envelope")

    recorded = envelope.get("files")
    if not isinstance(recorded, dict) or "manifest.json" not in recorded:
        raise InferenceError(
            f"{directory}/{ENVELOPE_FILENAME}: envelope must attest manifest.json"
        )
    for name in ("manifest.json", *SIBLING_FILES):
        expected = recorded.get(name)
        if not isinstance(expected, str) or len(expected) != 64:
            raise InferenceError(
                f"{directory}/{ENVELOPE_FILENAME}: no valid SHA-256 attested for {name}"
            )
        path = directory / name
        if not path.is_file():
            raise InferenceError(f"{directory}: envelope attests {name}, which is missing")
        actual = _sha256(path)
        if actual != expected:
            raise InferenceError(
                f"{directory}: envelope SHA-256 mismatch for {name}\n"
                f"  attested {expected}\n  actual   {actual}"
            )
    return envelope


def _cross_check_envelope(
    directory: Path,
    envelope: dict[str, Any],
    manifest: dict[str, Any],
    preprocessing: dict[str, Any],
) -> None:
    """Independent copies of the prediction-relevant fields must agree.

    The hash check above catches a manifest edited without rebuilding the
    envelope. This catches the narrower case where both files exist but
    disagree about what the model actually is.
    """
    attested = envelope.get("attested")
    if not isinstance(attested, dict):
        raise InferenceError(
            f"{directory}/{ENVELOPE_FILENAME}: envelope carries no attested field block"
        )
    params = manifest.get("model_params") or {}
    observed = {
        "model_type": manifest.get("model_type"),
        "n_features": manifest.get("n_features"),
        "feature_schema_sha256": manifest.get("feature_schema_sha256"),
        "output_transform": manifest.get("output_transform"),
        "model_params.n_trees": params.get("n_trees"),
        "model_params.n_nodes": params.get("n_nodes"),
        "preprocessing_kind": preprocessing.get("kind"),
        "run_id": manifest.get("run_id"),
        "config_id": manifest.get("config_id"),
    }
    for field in envelope.get("cross_checked_fields", list(observed)):
        if field not in observed:
            continue
        if attested.get(field) != observed[field]:
            raise InferenceError(
                f"{directory}: envelope and manifest disagree on {field!r}\n"
                f"  envelope {attested.get(field)!r}\n  manifest {observed[field]!r}"
            )


class SafeShiftPredictor:
    """Frozen-bundle predictor for every neutral model type."""

    def __init__(
        self,
        bundle_dir: Path,
        manifest: dict[str, Any],
        schema: dict[str, Any],
        arrays: dict[str, np.ndarray],
    ) -> None:
        self.bundle_dir = bundle_dir
        self.manifest = manifest
        self.schema = schema
        self.arrays = arrays
        self.model_type = str(manifest["model_type"])
        self.n_features = int(manifest["n_features"])
        self.output_transform = dict(manifest.get("output_transform") or {"kind": "identity"})
        self.envelope: dict[str, Any] = {}
        self.preprocessing = self._load_preprocessing()

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #

    @classmethod
    def load(cls, bundle_dir: Path | str) -> "SafeShiftPredictor":
        directory = Path(bundle_dir)

        # The envelope is verified first and unconditionally. Reading anything
        # out of manifest.json before that would mean trusting the file whose
        # integrity has not been established yet.
        envelope = _verify_envelope(directory)

        manifest_path = directory / "manifest.json"
        manifest = _read_json(manifest_path, "manifest")

        model_type = manifest.get("model_type")
        if model_type not in SUPPORTED_MODEL_TYPES:
            raise InferenceError(
                f"unsupported model type {model_type!r}; supported: {list(SUPPORTED_MODEL_TYPES)}"
            )

        recorded = manifest.get("files")
        if not isinstance(recorded, dict):
            raise InferenceError(f"{manifest_path}: manifest.files must be an object")
        for name in SIBLING_FILES:
            sibling = directory / name
            if not sibling.is_file():
                raise InferenceError(f"{directory}: required sibling {name} is missing")
            expected = recorded.get(name)
            if not isinstance(expected, str) or len(expected) != 64:
                raise InferenceError(f"{manifest_path}: no valid SHA-256 recorded for {name}")
            actual = _sha256(sibling)
            if actual != expected:
                raise InferenceError(
                    f"{directory}: SHA-256 mismatch for {name}\n  expected {expected}\n  actual   {actual}"
                )

        schema_path = directory / "feature_schema.json"
        try:
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise InferenceError(f"{schema_path}: schema is not valid JSON: {exc.msg}") from exc
        if not isinstance(schema, dict) or "feature_names" not in schema:
            raise InferenceError(f"{schema_path}: schema must be an object with feature_names")
        if int(schema.get("n_features", -1)) != len(schema["feature_names"]):
            raise InferenceError(f"{schema_path}: n_features disagrees with feature_names length")
        if int(manifest.get("n_features", -1)) != int(schema["n_features"]):
            raise InferenceError(
                f"{directory}: manifest n_features {manifest.get('n_features')} disagrees with "
                f"schema n_features {schema.get('n_features')}"
            )

        try:
            with np.load(directory / "model.npz", allow_pickle=False) as handle:
                arrays = {key: np.asarray(handle[key]) for key in handle.files}
        except ValueError as exc:
            raise InferenceError(
                f"{directory}/model.npz: refused to load; the artifact appears to require "
                f"pickle or is corrupted ({exc})"
            ) from exc
        except OSError as exc:
            raise InferenceError(f"{directory}/model.npz: cannot be read ({exc})") from exc

        for key, array in arrays.items():
            if array.dtype == object:
                raise InferenceError(f"{directory}/model.npz: array {key!r} has object dtype")
            if np.issubdtype(array.dtype, np.floating) and not np.all(np.isfinite(array)):
                raise InferenceError(f"{directory}/model.npz: array {key!r} contains non-finite values")

        predictor = cls(directory, manifest, schema, arrays)
        _cross_check_envelope(directory, envelope, manifest, predictor.preprocessing)
        predictor.envelope = envelope
        predictor._validate_model_arrays()
        return predictor

    def _load_preprocessing(self) -> dict[str, Any]:
        path = self.bundle_dir / "preprocessing.json"
        payload = _read_json(path, "preprocessing")
        validate_preprocessing(payload, self.arrays, self.n_features, error=InferenceError)
        return payload

    def _validate_model_arrays(self) -> None:
        params = self.manifest.get("model_params") or {}
        if self.model_type == "persistence":
            if "at_now_index" not in self.arrays:
                raise InferenceError("persistence bundle is missing at_now_index")
            index = int(self.arrays["at_now_index"].ravel()[0])
            if not 0 <= index < self.n_features:
                raise InferenceError(f"persistence at_now_index {index} is out of range")
        elif self.model_type == "linear":
            for key in ("linear_coef", "linear_intercept"):
                if key not in self.arrays:
                    raise InferenceError(f"linear bundle is missing {key}")
            if self.arrays["linear_coef"].shape[0] != self.n_features:
                raise InferenceError("linear_coef width does not match n_features")
        elif self.model_type == "mlp_relu":
            if "mlp_n_layers" not in self.arrays:
                raise InferenceError("mlp bundle is missing mlp_n_layers")
            n_layers = int(self.arrays["mlp_n_layers"].ravel()[0])
            if n_layers < 1:
                raise InferenceError(f"mlp bundle declares {n_layers} layers")
            for layer in range(n_layers):
                for key in (f"mlp_W{layer}", f"mlp_b{layer}"):
                    if key not in self.arrays:
                        raise InferenceError(f"mlp bundle is missing {key}")
            if self.arrays["mlp_W0"].shape[0] != self.n_features:
                raise InferenceError("mlp_W0 input width does not match n_features")
        elif self.model_type == "tree_ensemble":
            # Every structural property is proven before a single row is
            # routed. Without this a corrupt child index surfaces as an
            # IndexError from inside NumPy, which reads like a library fault
            # instead of what it is: a rejected artifact.
            validate_flat_tree_arrays(
                self.arrays,
                self.n_features,
                aggregation=params.get("aggregation"),
                decision_rule=params.get("decision_rule"),
                comparison_dtype=params.get("split_comparison_dtype"),
                manifest_n_trees=params.get("n_trees"),
                manifest_n_nodes=params.get("n_nodes"),
                error=InferenceError,
            )

    # ------------------------------------------------------------------ #
    # Schema / input checks
    # ------------------------------------------------------------------ #

    def assert_schema(self, schema: dict[str, Any]) -> None:
        """Fail closed when a caller's schema disagrees with the bundle's."""
        if not isinstance(schema, dict):
            raise InferenceError("supplied schema must be an object")
        if schema.get("feature_names") != self.schema["feature_names"]:
            raise InferenceError("feature schema mismatch: feature_names differ from the bundle")
        if int(schema.get("n_features", -1)) != self.n_features:
            raise InferenceError("feature schema mismatch: n_features differs from the bundle")

    def _prepare(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        matrix = np.asarray(x, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2:
            raise InferenceError(f"expected a 2-D feature matrix, got shape {matrix.shape}")
        if matrix.shape[1] != self.n_features:
            raise InferenceError(
                f"expected {self.n_features} features, got {matrix.shape[1]}"
            )
        if not np.all(np.isfinite(matrix)):
            raise InferenceError("input contains non-finite values; refusing to predict")

        transformed = matrix
        if "preproc_clip_min" in self.arrays or "preproc_clip_max" in self.arrays:
            low = self.arrays.get("preproc_clip_min")
            high = self.arrays.get("preproc_clip_max")
            transformed = np.clip(
                transformed,
                None if low is None else low.astype(np.float64),
                None if high is None else high.astype(np.float64),
            )
        if self.preprocessing["kind"] == "standardize":
            mean = self.arrays["preproc_mean"].astype(np.float64)
            scale = self.arrays["preproc_scale"].astype(np.float64)
            transformed = (transformed - mean) / scale
        return matrix, transformed

    # ------------------------------------------------------------------ #
    # Forward passes
    # ------------------------------------------------------------------ #

    def _forward(self, prepared: np.ndarray, raw: np.ndarray) -> np.ndarray:
        if self.model_type == "persistence":
            index = int(self.arrays["at_now_index"].ravel()[0])
            return raw[:, index].copy()
        if self.model_type == "linear":
            coef = self.arrays["linear_coef"].astype(np.float64)
            intercept = float(self.arrays["linear_intercept"].ravel()[0])
            return prepared @ coef + intercept
        if self.model_type == "mlp_relu":
            n_layers = int(self.arrays["mlp_n_layers"].ravel()[0])
            hidden = prepared
            for layer in range(n_layers):
                weight = self.arrays[f"mlp_W{layer}"].astype(np.float64)
                bias = self.arrays[f"mlp_b{layer}"].astype(np.float64)
                hidden = hidden @ weight + bias
                if layer < n_layers - 1:
                    hidden = np.maximum(hidden, 0.0)
            return hidden.ravel() if hidden.shape[1] == 1 else hidden[:, 0]
        if self.model_type == "tree_ensemble":
            return self._forward_trees(prepared)
        raise InferenceError(f"unsupported model type {self.model_type!r}")

    def _forward_trees(self, prepared: np.ndarray) -> np.ndarray:
        params = self.manifest["model_params"]
        rule = params["decision_rule"]
        aggregation = params["aggregation"]
        # The source library compares at float32; reproducing that width is a
        # correctness requirement, not a rounding preference. See
        # export.tree_split_comparison_dtype_reason in the configuration.
        comparison_dtype = params.get(
            "split_comparison_dtype", DEFAULT_SPLIT_COMPARISON_DTYPE
        )
        offsets = self.arrays["tree_offsets"].astype(np.int64)
        feature_all = self.arrays["tree_node_feature"].astype(np.int64)
        threshold_all = self.arrays["tree_node_threshold"].astype(np.float64)
        left_all = self.arrays["tree_node_left"].astype(np.int64)
        right_all = self.arrays["tree_node_right"].astype(np.int64)
        value_all = self.arrays["tree_node_value"].astype(np.float64)

        # `tree_node_missing_go_to` is deliberately NOT read here. `_prepare`
        # rejects any non-finite input before this point, so a missing-value
        # branch is unreachable under the P0 finite-input policy. The field is
        # still exported, because an exporter must record what the source
        # library would have done, but honouring it here would advertise a
        # runtime capability that no input can ever reach.
        n_trees = offsets.shape[0] - 1
        total = np.zeros(prepared.shape[0], dtype=np.float64)
        for tree_index in range(n_trees):
            start, stop = int(offsets[tree_index]), int(offsets[tree_index + 1])
            feature = feature_all[start:stop]
            threshold = threshold_all[start:stop]
            left = left_all[start:stop]
            right = right_all[start:stop]
            value = value_all[start:stop]

            node = np.zeros(prepared.shape[0], dtype=np.int64)
            active = feature[node] != LEAF
            guard = feature.shape[0] + 1
            steps = 0
            while active.any():
                steps += 1
                if steps > guard:
                    raise InferenceError("tree traversal exceeded its node count; malformed tree")
                idx = np.nonzero(active)[0]
                current = node[idx]
                values = split_values(prepared[idx, feature[current]], comparison_dtype)
                if rule == "left_if_le":
                    go_left = values <= threshold[current]
                else:
                    go_left = values < threshold[current]
                node[idx] = np.where(go_left, left[current], right[current])
                active = feature[node] != LEAF
            total += value[node]

        if aggregation == "mean" and n_trees:
            total /= float(n_trees)
        return float(params["base_score"]) + float(params["learning_rate"]) * total

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def predict_features(self, x: np.ndarray) -> np.ndarray:
        """Predict the station-derived shade apparent-temperature estimate.

        Returns a 1-D float64 array. It carries no safety or medical
        classification.
        """
        raw, prepared = self._prepare(x)
        prediction = np.asarray(self._forward(prepared, raw), dtype=np.float64).ravel()

        transform = self.output_transform
        kind = transform.get("kind", "identity")
        if kind == "residual_plus_feature":
            index = int(transform["feature_index"])
            if not 0 <= index < self.n_features:
                raise InferenceError(f"output transform feature index {index} is out of range")
            prediction = prediction + raw[:, index]
        elif kind != "identity":
            raise InferenceError(f"unsupported output transform {kind!r}")

        if not np.all(np.isfinite(prediction)):
            raise InferenceError("model produced a non-finite prediction; refusing to return it")
        return prediction

    def runtime_provenance(self) -> dict[str, Any]:
        """Everything a viewer needs to tell which artifact produced a number."""
        return {
            "bundle_dir": str(self.bundle_dir.as_posix()),
            "bundle_version": self.manifest.get("bundle_version"),
            "model_type": self.model_type,
            "parameterization": self.manifest.get("parameterization"),
            "n_features": self.n_features,
            "manifest_sha256": _sha256(self.bundle_dir / "manifest.json"),
            "bundle_envelope_sha256": _sha256(self.bundle_dir / ENVELOPE_FILENAME),
            "envelope_version": self.envelope.get("envelope_version"),
            "missing_value_routing": (
                "inert: the runtime rejects non-finite input before traversal, so "
                "tree missing-value branches are unreachable and are not honoured"
            ),
            "model_npz_sha256": self.manifest["files"]["model.npz"],
            "feature_schema_sha256": self.manifest["files"]["feature_schema.json"],
            "preprocessing_kind": self.preprocessing["kind"],
            "output_transform": self.output_transform,
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "claim_boundary": [
                "Prediction only. No medical or safety classification is returned.",
                "Value is a station-derived shade apparent-temperature estimate in degC.",
            ],
        }
