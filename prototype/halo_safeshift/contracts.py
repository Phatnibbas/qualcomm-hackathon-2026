"""Structural contracts shared by the exporter and the deployment runtime.

Both sides must agree on what a well-formed artifact is. If the exporter
validated one set of conditions and the runtime validated another, an artifact
could pass export and fail at prediction time, or worse, pass both while being
wrong in a way neither side looked at. The checks therefore live here once and
each side supplies the exception type it wants raised.

Imports only the Python standard library and NumPy, because the deployment
runtime imports this module.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "PREPROCESSING_KINDS",
    "CLIP_KEYS",
    "validate_preprocessing",
]

PREPROCESSING_KINDS = ("identity", "standardize")
CLIP_KEYS = ("preproc_clip_min", "preproc_clip_max")


def _vector(name: str, array: Any, n_features: int, error: type[Exception]) -> np.ndarray:
    values = np.asarray(array)
    if values.dtype == object:
        raise error(f"{name} has object dtype and would require pickle")
    if values.ndim != 1 or int(values.shape[0]) != int(n_features):
        raise error(
            f"{name} must have shape ({int(n_features)},), got {tuple(values.shape)}"
        )
    values = values.astype(np.float64)
    if not np.all(np.isfinite(values)):
        raise error(f"{name} contains a non-finite value")
    return values


def validate_preprocessing(
    payload: dict[str, Any],
    arrays: dict[str, Any] | Any,
    n_features: int,
    *,
    error: type[Exception] = ValueError,
) -> dict[str, Any]:
    """Fail closed on any malformed preprocessing payload.

    Checked here rather than at first use because a silently wrong scale does
    not crash — it returns a plausible number. The clipping arrays are checked
    as a pair: one present without the other means the artifact declares a
    bound in one direction only, which is a different transform from the one
    the exporter intended and from the one the JSON describes.

    ``payload`` is ``preprocessing.json``; ``arrays`` is the loaded ``model.npz``
    mapping. The two must agree, so a JSON that claims ``identity`` while the
    npz carries a scaler is rejected rather than quietly ignored.
    """
    if not isinstance(payload, dict):
        raise error("preprocessing.json must contain an object")
    kind = payload.get("kind")
    if kind not in PREPROCESSING_KINDS:
        raise error(f"unsupported preprocessing kind {kind!r}")

    present = {key for key in ("preproc_mean", "preproc_scale", *CLIP_KEYS) if key in arrays}

    if kind == "standardize":
        for key in ("preproc_mean", "preproc_scale"):
            if key not in arrays:
                raise error(f"standardize preprocessing requires {key} in model.npz")
        _vector("preproc_mean", arrays["preproc_mean"], n_features, error)
        scale = _vector("preproc_scale", arrays["preproc_scale"], n_features, error)
        if not bool(np.all(scale > 0)):
            raise error("preproc_scale must be strictly positive")
    else:
        stray = present & {"preproc_mean", "preproc_scale"}
        if stray:
            raise error(
                f"preprocessing.json declares kind={kind!r} but model.npz carries "
                f"{sorted(stray)}; the JSON and the arrays describe different transforms"
            )

    have_clip = [key for key in CLIP_KEYS if key in arrays]
    if have_clip and len(have_clip) != len(CLIP_KEYS):
        missing = [key for key in CLIP_KEYS if key not in arrays]
        raise error(
            f"clipping arrays must be present together or not at all; "
            f"found {have_clip}, missing {missing}"
        )
    if have_clip:
        low = _vector("preproc_clip_min", arrays["preproc_clip_min"], n_features, error)
        high = _vector("preproc_clip_max", arrays["preproc_clip_max"], n_features, error)
        if not bool(np.all(low <= high)):
            bad = int(np.argmax(low > high))
            raise error(
                f"preproc_clip_min exceeds preproc_clip_max at feature index {bad} "
                f"({float(low[bad])} > {float(high[bad])}); the clip would empty the range"
            )

    declared = payload.get("arrays_in_model_npz")
    if declared is not None:
        declared_set = set(declared)
        actual = {
            key
            for key in ("preproc_mean", "preproc_scale", *CLIP_KEYS)
            if key in arrays
        }
        if declared_set != actual:
            raise error(
                f"preprocessing.json declares arrays {sorted(declared_set)} but "
                f"model.npz carries {sorted(actual)}"
            )

    declared_clipping = payload.get("clipping_applied")
    if declared_clipping is not None and bool(declared_clipping) != bool(have_clip):
        raise error(
            f"preprocessing.json says clipping_applied={declared_clipping} but "
            f"model.npz {'carries' if have_clip else 'carries no'} clipping arrays"
        )

    return {
        "kind": kind,
        "n_features": int(n_features),
        "standardize_arrays_present": kind == "standardize",
        "clipping_applied": bool(have_clip),
    }
