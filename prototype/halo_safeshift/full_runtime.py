"""Dependency-light runtime for a full Colab portable linear/MLP bundle.

This is separate from the existing neutral P0 bundle runtime so the current P0
emergency service remains untouched until a P1 output passes all gates.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


class FullRuntimeError(RuntimeError):
    """A full-pipeline bundle is malformed, tampered, or unsupported."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FullRuntimeError(f"cannot read {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise FullRuntimeError(f"{path.name} must contain an object")
    return value


class PortablePredictor:
    """Run a Colab-exported linear or ReLU MLP bundle using NumPy only."""

    def __init__(self, bundle: Path, model: dict[str, Any], schema: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
        self.bundle = bundle
        self.model = model
        self.schema = schema
        self.arrays = arrays
        self.n_features = int(model["n_features"])
        self.model_type = str(model["model_type"])
        self.feature_names = list(schema["feature_names"])
        preprocessing = model.get("preprocessing") or {}
        self.mean = np.asarray(preprocessing.get("mean", arrays.get("mean")), dtype=np.float64)
        self.scale = np.asarray(preprocessing.get("scale", arrays.get("scale")), dtype=np.float64)
        if self.mean.shape != (self.n_features,) or self.scale.shape != (self.n_features,):
            raise FullRuntimeError("preprocessing mean/scale do not match feature width")
        if not np.all(np.isfinite(self.mean)) or not np.all(np.isfinite(self.scale)) or np.any(self.scale <= 0):
            raise FullRuntimeError("preprocessing has invalid mean/scale")
        if self.model_type not in {"persistence", "linear", "mlp_relu", "tree_ensemble"}:
            raise FullRuntimeError(
                f"unsupported portable model type {self.model_type!r}; a tree export must clear its own parity gate"
            )

    @classmethod
    def load(cls, bundle_dir: str | Path, *, require_fused: bool = False) -> "PortablePredictor":
        bundle = Path(bundle_dir)
        manifest = _read_json(bundle / "manifest.json")
        expected = manifest.get("files")
        if not isinstance(expected, dict):
            raise FullRuntimeError("manifest files must be an object")
        for name in ("model.json", "feature_schema.json", "model.npz"):
            path = bundle / name
            if not path.is_file() or expected.get(name) != _sha256(path):
                raise FullRuntimeError(f"hash verification failed for {name}")
        model = _read_json(bundle / "model.json")
        schema = _read_json(bundle / "feature_schema.json")
        names = schema.get("feature_names")
        if not isinstance(names, list) or int(schema.get("n_features", -1)) != len(names):
            raise FullRuntimeError("invalid feature schema")
        if int(model.get("n_features", -1)) != len(names):
            raise FullRuntimeError("model and feature schema widths disagree")
        runtime_names = model.get("runtime_input_names")
        if runtime_names != names:
            raise FullRuntimeError("model runtime inputs differ from frozen feature schema")
        satellite = [name for name in names if str(name).startswith("satellite_")]
        if require_fused and not satellite:
            raise FullRuntimeError("fused mode rejected: bundle contains no satellite features")
        try:
            with np.load(bundle / "model.npz", allow_pickle=False) as archive:
                arrays = {key: np.asarray(archive[key]) for key in archive.files}
        except (OSError, ValueError) as exc:
            raise FullRuntimeError(f"model.npz cannot be loaded safely: {exc}") from exc
        if any(value.dtype == object for value in arrays.values()):
            raise FullRuntimeError("object arrays are forbidden")
        return cls(bundle, model, schema, arrays)

    def predict(self, features: np.ndarray) -> np.ndarray:
        raw = np.asarray(features, dtype=np.float64)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        if raw.ndim != 2 or raw.shape[1] != self.n_features:
            raise FullRuntimeError(f"expected N×{self.n_features} feature matrix")
        if not np.all(np.isfinite(raw)):
            raise FullRuntimeError("non-finite runtime input")
        x = (raw - self.mean) / self.scale
        if self.model_type == "persistence":
            at_index = int(self.model.get("at_now_index", -1))
            if not 0 <= at_index < self.n_features:
                raise FullRuntimeError("persistence model lacks a valid station AT-now index")
            prediction = raw[:, at_index]
        elif self.model_type == "linear":
            prediction = x @ np.asarray(self.arrays["coef"], dtype=np.float64) + float(np.asarray(self.arrays["intercept"]).ravel()[0])
        elif self.model_type == "mlp_relu":
            hidden = x
            count = int(self.model["layers"])
            for index in range(count):
                hidden = hidden @ np.asarray(self.arrays[f"W{index}"], dtype=np.float64) + np.asarray(self.arrays[f"b{index}"], dtype=np.float64)
                if index < count - 1:
                    hidden = np.maximum(hidden, 0.0)
            prediction = hidden.ravel()
        else:
            offsets = np.asarray(self.arrays["tree_offsets"], dtype=np.int64)
            feature = np.asarray(self.arrays["tree_feature"], dtype=np.int64)
            threshold = np.asarray(self.arrays["tree_threshold"], dtype=np.float64)
            left = np.asarray(self.arrays["tree_left"], dtype=np.int64)
            right = np.asarray(self.arrays["tree_right"], dtype=np.int64)
            value = np.asarray(self.arrays["tree_value"], dtype=np.float64)
            total = np.zeros(raw.shape[0], dtype=np.float64)
            for tree in range(len(offsets) - 1):
                start, end = int(offsets[tree]), int(offsets[tree + 1])
                node = np.zeros(raw.shape[0], dtype=np.int64)
                active = feature[start + node] >= 0
                while active.any():
                    idx = np.flatnonzero(active); global_node = start + node[idx]
                    node[idx] = np.where(x[idx, feature[global_node]] <= threshold[global_node], left[global_node], right[global_node])
                    active = feature[start + node] >= 0
                total += value[start + node]
            if self.model.get("aggregation") == "mean": total /= len(offsets) - 1
            prediction = float(self.model.get("base_score", 0.0)) + float(self.model.get("learning_rate", 1.0)) * total
        if self.model.get("target") == "residual":
            at_index = int(self.model.get("at_now_index", -1))
            if not 0 <= at_index < self.n_features:
                raise FullRuntimeError("residual model lacks a valid station AT-now index")
            prediction = prediction + raw[:, at_index]
        if not np.all(np.isfinite(prediction)):
            raise FullRuntimeError("model emitted non-finite prediction")
        return np.asarray(prediction, dtype=np.float64)

    def provenance(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "model_sha256": _sha256(self.bundle / "model.npz"),
            "schema_sha256": _sha256(self.bundle / "feature_schema.json"),
            "n_features": self.n_features,
            "satellite_features": [name for name in self.feature_names if str(name).startswith("satellite_")],
        }
