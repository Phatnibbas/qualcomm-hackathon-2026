"""Neutral tree-ensemble representation, independent of sklearn and XGBoost.

The deployment runtime may import only the Python standard library and NumPy,
so a trained ensemble is converted into flat arrays:

    {
      "model_type": "tree_ensemble",
      "aggregation": "mean" | "sum",
      "base_score": 0.0,
      "learning_rate": 1.0,
      "decision_rule": "left_if_le" | "left_if_lt",
      "trees": [
        {"feature": [], "threshold": [], "left": [], "right": [],
         "value": [], "missing_go_to": []}
      ]
    }

``decision_rule`` is an addition to the bare schema and it is load-bearing:
scikit-learn splits on ``x <= threshold`` while XGBoost splits on
``x < threshold``. Leaving it implicit would make a later XGBoost conversion
silently wrong on ties, so the exporter states it and the evaluator honours it.

Leaf nodes carry ``feature = -1`` and ``left = right = -1``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "TreeExportError",
    "LEAF",
    "MISSING_LEFT",
    "MISSING_RIGHT",
    "evaluate_tree_ensemble",
    "export_sklearn_gradient_boosting",
    "export_sklearn_extra_trees",
    "export_xgboost_regressor",
    "flatten_tree_ensemble",
    "unflatten_tree_ensemble",
]

LEAF = -1
MISSING_LEFT = 0
MISSING_RIGHT = 1


class TreeExportError(ValueError):
    """Raised when an estimator cannot be converted to the neutral format."""


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def evaluate_tree_ensemble(bundle: dict[str, Any], x: np.ndarray) -> np.ndarray:
    """Pure-NumPy evaluation of the neutral representation."""
    matrix = np.asarray(x, dtype=np.float64)
    if matrix.ndim != 2:
        raise TreeExportError(f"expected a 2-D feature matrix, got shape {matrix.shape}")

    rule = bundle.get("decision_rule", "left_if_le")
    if rule not in {"left_if_le", "left_if_lt"}:
        raise TreeExportError(f"unsupported decision_rule {rule!r}")
    aggregation = bundle.get("aggregation", "sum")
    if aggregation not in {"mean", "sum"}:
        raise TreeExportError(f"unsupported aggregation {aggregation!r}")

    comparison_dtype = bundle.get("split_comparison_dtype", DEFAULT_SPLIT_COMPARISON_DTYPE)
    trees = bundle["trees"]
    total = np.zeros(matrix.shape[0], dtype=np.float64)
    for tree in trees:
        total += _evaluate_tree(tree, matrix, rule, comparison_dtype)

    learning_rate = float(bundle.get("learning_rate", 1.0))
    base_score = float(bundle.get("base_score", 0.0))
    if aggregation == "mean":
        total /= float(len(trees)) if trees else 1.0
    return base_score + learning_rate * total


SPLIT_COMPARISON_DTYPES = {"float32": np.float32, "float64": np.float64}
DEFAULT_SPLIT_COMPARISON_DTYPE = "float32"


def split_values(values: np.ndarray, comparison_dtype: str) -> np.ndarray:
    """Round feature values to the width the source library compares at.

    scikit-learn casts X to float32 before traversal and XGBoost is float32
    internally, so a float64 comparison is not "more precise" — it is a
    different function. A sample within one float32 step of a threshold routes
    down the other branch and returns a plausible wrong number instead of
    raising, which is why this is a correctness issue and not a tolerance one.
    """
    try:
        target = SPLIT_COMPARISON_DTYPES[comparison_dtype]
    except KeyError as exc:
        raise TreeExportError(
            f"unsupported split comparison dtype {comparison_dtype!r}; "
            f"supported: {sorted(SPLIT_COMPARISON_DTYPES)}"
        ) from exc
    return values.astype(target).astype(np.float64)


def _evaluate_tree(
    tree: dict[str, Any],
    matrix: np.ndarray,
    rule: str,
    comparison_dtype: str = DEFAULT_SPLIT_COMPARISON_DTYPE,
) -> np.ndarray:
    feature = np.asarray(tree["feature"], dtype=np.int64)
    threshold = np.asarray(tree["threshold"], dtype=np.float64)
    left = np.asarray(tree["left"], dtype=np.int64)
    right = np.asarray(tree["right"], dtype=np.int64)
    value = np.asarray(tree["value"], dtype=np.float64)
    missing_go_to = np.asarray(tree["missing_go_to"], dtype=np.int64)

    n_rows = matrix.shape[0]
    node = np.zeros(n_rows, dtype=np.int64)
    active = feature[node] != LEAF
    # A binary tree of N nodes has depth < N; the loop terminates.
    guard = int(feature.shape[0]) + 1
    steps = 0
    while active.any():
        steps += 1
        if steps > guard:  # pragma: no cover - structural guard
            raise TreeExportError("tree traversal exceeded its node count; malformed tree")
        idx = np.nonzero(active)[0]
        current = node[idx]
        values = split_values(matrix[idx, feature[current]], comparison_dtype)
        if rule == "left_if_le":
            go_left = values <= threshold[current]
        else:
            go_left = values < threshold[current]
        missing = ~np.isfinite(values)
        if missing.any():
            go_left = np.where(missing, missing_go_to[current] == MISSING_LEFT, go_left)
        node[idx] = np.where(go_left, left[current], right[current])
        active = feature[node] != LEAF
    return value[node]


# --------------------------------------------------------------------------- #
# sklearn exporters
# --------------------------------------------------------------------------- #

def _export_sklearn_tree(sklearn_tree: Any) -> dict[str, list[Any]]:
    internal = sklearn_tree.tree_
    n_nodes = int(internal.node_count)
    children_left = internal.children_left
    children_right = internal.children_right
    feature = internal.feature
    threshold = internal.threshold
    values = internal.value

    missing_left = getattr(internal, "missing_go_to_left", None)

    out_feature: list[int] = []
    out_threshold: list[float] = []
    out_left: list[int] = []
    out_right: list[int] = []
    out_value: list[float] = []
    out_missing: list[int] = []

    for node in range(n_nodes):
        is_leaf = int(children_left[node]) == LEAF
        out_feature.append(LEAF if is_leaf else int(feature[node]))
        out_threshold.append(0.0 if is_leaf else float(threshold[node]))
        out_left.append(int(children_left[node]))
        out_right.append(int(children_right[node]))
        out_value.append(float(np.asarray(values[node]).ravel()[0]))
        if is_leaf or missing_left is None:
            out_missing.append(MISSING_LEFT)
        else:
            out_missing.append(MISSING_LEFT if int(missing_left[node]) else MISSING_RIGHT)

    return {
        "feature": out_feature,
        "threshold": out_threshold,
        "left": out_left,
        "right": out_right,
        "value": out_value,
        "missing_go_to": out_missing,
    }


def export_sklearn_gradient_boosting(estimator: Any) -> dict[str, Any]:
    """Convert a fitted ``GradientBoostingRegressor`` to the neutral format."""
    estimators = getattr(estimator, "estimators_", None)
    if estimators is None:
        raise TreeExportError("estimator is not fitted (no estimators_)")
    if getattr(estimator, "n_outputs_", 1) != 1:
        raise TreeExportError("only single-output regression is supported")
    if np.asarray(estimators).shape[1] != 1:
        raise TreeExportError("only single-output boosting stages are supported")

    init = getattr(estimator, "init_", None)
    if init is None or getattr(estimator, "init", None) == "zero":
        base_score = 0.0
    else:
        probe = np.zeros((1, int(estimator.n_features_in_)), dtype=np.float64)
        base_score = float(np.asarray(init.predict(probe)).ravel()[0])

    trees = [_export_sklearn_tree(stage[0]) for stage in estimators]
    return {
        "model_type": "tree_ensemble",
        "aggregation": "sum",
        "base_score": base_score,
        "learning_rate": float(estimator.learning_rate),
        "decision_rule": "left_if_le",
        "split_comparison_dtype": "float32",
        "source_estimator": "sklearn.ensemble.GradientBoostingRegressor",
        "n_features": int(estimator.n_features_in_),
        "trees": trees,
    }


def export_sklearn_extra_trees(estimator: Any) -> dict[str, Any]:
    """Convert a fitted ``ExtraTreesRegressor`` to the neutral format."""
    estimators = getattr(estimator, "estimators_", None)
    if estimators is None:
        raise TreeExportError("estimator is not fitted (no estimators_)")
    if getattr(estimator, "n_outputs_", 1) != 1:
        raise TreeExportError("only single-output regression is supported")

    trees = [_export_sklearn_tree(tree) for tree in estimators]
    return {
        "model_type": "tree_ensemble",
        "aggregation": "mean",
        "base_score": 0.0,
        "learning_rate": 1.0,
        "decision_rule": "left_if_le",
        "split_comparison_dtype": "float32",
        "source_estimator": "sklearn.ensemble.ExtraTreesRegressor",
        "n_features": int(estimator.n_features_in_),
        "trees": trees,
    }


# --------------------------------------------------------------------------- #
# Optional XGBoost exporter interface
# --------------------------------------------------------------------------- #

def export_xgboost_regressor(estimator: Any) -> dict[str, Any]:
    """Convert a fitted ``xgboost.XGBRegressor`` to the neutral format.

    XGBoost is not installed on the local preparation machine, so this path is
    exercised only in the Colab phase, which must prove prediction parity
    before an XGBoost model may be selected for the UNO Q.

    Implementation note: XGBoost splits on ``x < threshold``, hence
    ``decision_rule = "left_if_lt"``.
    """
    booster = getattr(estimator, "get_booster", None)
    if booster is None:
        raise TreeExportError("estimator does not expose get_booster(); not an XGBRegressor")
    frame = booster().trees_to_dataframe()

    base_score = float(getattr(estimator, "base_score", None) or 0.0)
    trees: list[dict[str, list[Any]]] = []
    for tree_id in sorted(frame["Tree"].unique()):
        block = frame[frame["Tree"] == tree_id]
        node_ids = list(block["ID"])
        index_of = {node_id: position for position, node_id in enumerate(node_ids)}

        feature: list[int] = []
        threshold: list[float] = []
        left: list[int] = []
        right: list[int] = []
        value: list[float] = []
        missing_go_to: list[int] = []
        for _, row in block.iterrows():
            if str(row["Feature"]) == "Leaf":
                feature.append(LEAF)
                threshold.append(0.0)
                left.append(LEAF)
                right.append(LEAF)
                value.append(float(row["Gain"]))
                missing_go_to.append(MISSING_LEFT)
                continue
            feature.append(int(str(row["Feature"]).lstrip("f")))
            threshold.append(float(row["Split"]))
            left.append(index_of[row["Yes"]])
            right.append(index_of[row["No"]])
            value.append(0.0)
            missing_go_to.append(
                MISSING_LEFT if index_of[row["Missing"]] == index_of[row["Yes"]] else MISSING_RIGHT
            )
        trees.append(
            {
                "feature": feature,
                "threshold": threshold,
                "left": left,
                "right": right,
                "value": value,
                "missing_go_to": missing_go_to,
            }
        )

    return {
        "model_type": "tree_ensemble",
        "aggregation": "sum",
        "base_score": base_score,
        "learning_rate": 1.0,
        "decision_rule": "left_if_lt",
        "split_comparison_dtype": "float32",
        "source_estimator": "xgboost.XGBRegressor",
        "n_features": int(getattr(estimator, "n_features_in_", 0)),
        "trees": trees,
    }


# --------------------------------------------------------------------------- #
# Flat array form for pickle-free npz storage
# --------------------------------------------------------------------------- #

def flatten_tree_ensemble(bundle: dict[str, Any]) -> dict[str, np.ndarray]:
    """Concatenate every tree into flat arrays plus an offset index.

    ``numpy.savez`` cannot store a ragged list of trees without pickling, and
    the runtime must load with ``allow_pickle=False``.
    """
    features: list[int] = []
    thresholds: list[float] = []
    lefts: list[int] = []
    rights: list[int] = []
    values: list[float] = []
    missing: list[int] = []
    offsets: list[int] = [0]

    for tree in bundle["trees"]:
        features.extend(int(v) for v in tree["feature"])
        thresholds.extend(float(v) for v in tree["threshold"])
        lefts.extend(int(v) for v in tree["left"])
        rights.extend(int(v) for v in tree["right"])
        values.extend(float(v) for v in tree["value"])
        missing.extend(int(v) for v in tree["missing_go_to"])
        offsets.append(len(features))

    return {
        "tree_node_feature": np.asarray(features, dtype=np.int32),
        "tree_node_threshold": np.asarray(thresholds, dtype=np.float64),
        "tree_node_left": np.asarray(lefts, dtype=np.int32),
        "tree_node_right": np.asarray(rights, dtype=np.int32),
        "tree_node_value": np.asarray(values, dtype=np.float64),
        "tree_node_missing_go_to": np.asarray(missing, dtype=np.int8),
        "tree_offsets": np.asarray(offsets, dtype=np.int32),
    }


TREE_ARRAY_KEYS = (
    "tree_node_feature",
    "tree_node_threshold",
    "tree_node_left",
    "tree_node_right",
    "tree_node_value",
    "tree_node_missing_go_to",
)

SUPPORTED_AGGREGATIONS = ("mean", "sum")
SUPPORTED_DECISION_RULES = ("left_if_le", "left_if_lt")


def validate_flat_tree_arrays(
    arrays: Any,
    n_features: int,
    *,
    aggregation: str | None = None,
    decision_rule: str | None = None,
    comparison_dtype: str | None = None,
    manifest_n_trees: int | None = None,
    manifest_n_nodes: int | None = None,
    error: type[Exception] = TreeExportError,
) -> dict[str, int]:
    """Prove a flat ensemble is structurally traversable *before* traversing it.

    Every check below exists because the failure it catches would otherwise
    surface as an ``IndexError`` or an infinite loop deep inside the prediction
    path, where it looks like a NumPy bug rather than a corrupted artifact.
    Child indices are tree-local, so a child that points outside its own tree
    would silently read a different tree's node.

    ``error`` lets the runtime raise ``InferenceError`` while the exporter
    raises ``TreeExportError``; the checks themselves are identical, so the two
    sides cannot drift.
    """
    missing = [key for key in (*TREE_ARRAY_KEYS, "tree_offsets") if key not in arrays]
    if missing:
        raise error(f"tree_ensemble bundle is missing {missing}")

    lengths = {key: int(np.asarray(arrays[key]).shape[0]) for key in TREE_ARRAY_KEYS}
    if len(set(lengths.values())) != 1:
        raise error(f"tree node arrays have unequal lengths: {lengths}")
    n_nodes = next(iter(lengths.values()))

    offsets = np.asarray(arrays["tree_offsets"], dtype=np.int64)
    if offsets.ndim != 1 or offsets.shape[0] < 2:
        raise error("tree_offsets must be a 1-D array with at least two entries")
    if int(offsets[0]) != 0:
        raise error(f"tree_offsets must start at zero, got {int(offsets[0])}")
    if bool(np.any(np.diff(offsets) <= 0)):
        raise error(
            "tree_offsets must be strictly increasing; a zero-length tree has no "
            "root to start traversal from"
        )
    if int(offsets[-1]) != n_nodes:
        raise error(
            f"final tree offset {int(offsets[-1])} does not equal the node count {n_nodes}"
        )

    n_trees = int(offsets.shape[0] - 1)
    if manifest_n_trees is not None and int(manifest_n_trees) != n_trees:
        raise error(
            f"manifest declares {manifest_n_trees} trees but the arrays contain {n_trees}"
        )
    if manifest_n_nodes is not None and int(manifest_n_nodes) != n_nodes:
        raise error(
            f"manifest declares {manifest_n_nodes} nodes but the arrays contain {n_nodes}"
        )
    if aggregation is not None and aggregation not in SUPPORTED_AGGREGATIONS:
        raise error(f"unsupported aggregation {aggregation!r}")
    if decision_rule is not None and decision_rule not in SUPPORTED_DECISION_RULES:
        raise error(f"unsupported decision_rule {decision_rule!r}")
    if comparison_dtype is not None and comparison_dtype not in SPLIT_COMPARISON_DTYPES:
        raise error(
            f"unsupported split comparison dtype {comparison_dtype!r}; "
            f"supported: {sorted(SPLIT_COMPARISON_DTYPES)}"
        )

    feature_all = np.asarray(arrays["tree_node_feature"], dtype=np.int64)
    threshold_all = np.asarray(arrays["tree_node_threshold"], dtype=np.float64)
    left_all = np.asarray(arrays["tree_node_left"], dtype=np.int64)
    right_all = np.asarray(arrays["tree_node_right"], dtype=np.int64)
    missing_all = np.asarray(arrays["tree_node_missing_go_to"], dtype=np.int64)
    value_all = np.asarray(arrays["tree_node_value"], dtype=np.float64)

    if not np.all(np.isin(missing_all, (MISSING_LEFT, MISSING_RIGHT))):
        raise error("tree_node_missing_go_to must contain only 0 (left) or 1 (right)")
    if not np.all(np.isfinite(value_all)):
        raise error("tree_node_value contains a non-finite leaf value")

    for tree_index in range(n_trees):
        start, stop = int(offsets[tree_index]), int(offsets[tree_index + 1])
        size = stop - start
        feature = feature_all[start:stop]
        left = left_all[start:stop]
        right = right_all[start:stop]
        threshold = threshold_all[start:stop]

        internal = feature != LEAF
        leaf = ~internal

        if bool(np.any(leaf & ((left != LEAF) | (right != LEAF)))):
            raise error(f"tree {tree_index}: a leaf node declares a child")
        if bool(np.any(internal & ((left < 0) | (left >= size)))):
            raise error(
                f"tree {tree_index}: a left child index falls outside its own tree "
                f"(valid range 0..{size - 1})"
            )
        if bool(np.any(internal & ((right < 0) | (right >= size)))):
            raise error(
                f"tree {tree_index}: a right child index falls outside its own tree "
                f"(valid range 0..{size - 1})"
            )
        if bool(np.any(internal & ((feature < 0) | (feature >= int(n_features))))):
            raise error(
                f"tree {tree_index}: an internal node splits on a feature index outside "
                f"0..{int(n_features) - 1}"
            )
        if bool(np.any(internal & ~np.isfinite(threshold))):
            raise error(f"tree {tree_index}: an internal node has a non-finite threshold")

    return {"n_trees": n_trees, "n_nodes": n_nodes}


def unflatten_tree_ensemble(
    arrays: dict[str, np.ndarray] | Any,
    aggregation: str,
    base_score: float,
    learning_rate: float,
    decision_rule: str,
    split_comparison_dtype: str = DEFAULT_SPLIT_COMPARISON_DTYPE,
) -> dict[str, Any]:
    """Rebuild the neutral bundle from flat arrays."""
    offsets = np.asarray(arrays["tree_offsets"], dtype=np.int64)
    trees: list[dict[str, Any]] = []
    for position in range(offsets.shape[0] - 1):
        start, stop = int(offsets[position]), int(offsets[position + 1])
        trees.append(
            {
                "feature": np.asarray(arrays["tree_node_feature"][start:stop], dtype=np.int64),
                "threshold": np.asarray(arrays["tree_node_threshold"][start:stop], dtype=np.float64),
                "left": np.asarray(arrays["tree_node_left"][start:stop], dtype=np.int64),
                "right": np.asarray(arrays["tree_node_right"][start:stop], dtype=np.int64),
                "value": np.asarray(arrays["tree_node_value"][start:stop], dtype=np.float64),
                "missing_go_to": np.asarray(
                    arrays["tree_node_missing_go_to"][start:stop], dtype=np.int64
                ),
            }
        )
    return {
        "model_type": "tree_ensemble",
        "aggregation": aggregation,
        "base_score": float(base_score),
        "learning_rate": float(learning_rate),
        "decision_rule": decision_rule,
        "split_comparison_dtype": split_comparison_dtype,
        "trees": trees,
    }
