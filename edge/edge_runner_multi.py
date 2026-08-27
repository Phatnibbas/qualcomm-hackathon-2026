#!/usr/bin/env python3
"""Stdlib-only UNO Q inference for every model family in the sweep.

The board has no numpy (verified: ModuleNotFoundError), so every path here uses
only the standard library. Supported model_type values:

  persistence          return AT_now
  linear_int8          INT8 weights + float scale, single dot product
  sparse_linear_int8   magnitude-pruned INT8 linear, gathered dot product
  mlp_sparse_int8      pruned INT8 MLP, ReLU hidden layers
  tree_ensemble        ExtraTrees / RandomForest (mean) or GradientBoosting (sum)

Every family uses the same optimization strategy as edge_runner_fast.py: all
casts, divisions and scale folds happen once in compile_model(), never per call.

No NPU, QNN, Hexagon or GPU is used or claimed. CPU Python only.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import time
from operator import itemgetter, mul
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _single_gather(k):
    return lambda h: (h[k],)


def _empty_gather(_h):
    return ()


def compile_model(model):
    """Hoist every cast, division and scale multiply out of the hot path."""
    kind = model["model_type"]
    c = {"kind": kind,
         "at_index": int(model["at_now_index"]),
         "n_features": int(model["n_features"])}
    if kind == "persistence":
        return c

    # fold (v - mean) / scale -> v * a + b   (division becomes multiplication)
    c["a"] = tuple(1.0 / float(s) for s in model["feature_scale"])
    c["b"] = tuple(-float(m) / float(s) for m, s in zip(model["feature_mean"], model["feature_scale"]))

    if kind == "linear_int8":
        scale = float(model["weight_scale"])
        c["w"] = tuple(scale * int(q) for q in model["qweights"])   # fold INT8 scale in
        c["bias"] = float(model["bias"])
        return c

    if kind == "sparse_linear_int8":
        # magnitude-pruned linear: only the surviving weights are stored
        scale = float(model["weight_scale"])
        idx = tuple(int(i) for i in model["indices"])
        c["w"] = tuple(scale * int(q) for q in model["qweights"])
        if len(idx) > 1:
            c["gather"] = itemgetter(*idx)
        elif len(idx) == 1:
            c["gather"] = _single_gather(idx[0])
        else:
            c["gather"] = _empty_gather
        c["bias"] = float(model["bias"])
        return c

    if kind == "mlp_sparse_int8":
        layers = []
        for layer in model["layers"]:
            scale = float(layer["weight_scale"])
            neurons = []
            for neuron in layer["outputs"]:
                idx = tuple(int(i) for i in neuron["indices"])
                weights = tuple(scale * int(q) for q in neuron["qweights"])
                if len(idx) > 1:
                    gather = itemgetter(*idx)
                elif len(idx) == 1:
                    gather = _single_gather(idx[0])
                else:
                    gather = _empty_gather
                neurons.append((float(neuron["bias"]), gather, weights))
            layers.append((bool(layer["relu"]), tuple(neurons)))
        c["layers"] = tuple(layers)
        return c

    if kind == "tree_ensemble":
        c["offsets"] = tuple(int(o) for o in model["tree_offsets"])
        c["feature"] = tuple(int(f) for f in model["tree_feature"])
        c["threshold"] = tuple(float(t) for t in model["tree_threshold"])
        c["left"] = tuple(int(x) for x in model["tree_left"])
        c["right"] = tuple(int(x) for x in model["tree_right"])
        c["value"] = tuple(float(x) for x in model["tree_value"])
        c["n_trees"] = len(c["offsets"]) - 1
        c["base"] = float(model.get("base_score", 0.0))
        # fold the mean aggregation and the learning rate into one scalar
        lr = float(model.get("learning_rate", 1.0))
        c["scale"] = lr / c["n_trees"] if model.get("aggregation") == "mean" else lr
        return c

    raise ValueError("unsupported model_type: %r" % kind)


def predict(c, features):
    n = c["n_features"]
    if len(features) != n:
        raise ValueError(f"expected {n} features, got {len(features)}")
    at_now = features[c["at_index"]]
    kind = c["kind"]
    if kind == "persistence":
        if not math.isfinite(at_now):
            raise ValueError("non-finite input")
        return float(at_now)

    h = list(map(lambda v, a, b: v * a + b, features, c["a"], c["b"]))
    if not all(map(math.isfinite, h)):
        raise ValueError("non-finite input")

    if kind == "linear_int8":
        return at_now + c["bias"] + sum(map(mul, c["w"], h))

    if kind == "sparse_linear_int8":
        return at_now + c["bias"] + sum(map(mul, c["w"], c["gather"](h)))

    if kind == "mlp_sparse_int8":
        for relu, neurons in c["layers"]:
            if relu:
                h = [t if (t := bias + sum(map(mul, w, g(h)))) > 0.0 else 0.0
                     for bias, g, w in neurons]
            else:
                h = [bias + sum(map(mul, w, g(h))) for bias, g, w in neurons]
        return at_now + h[0]

    # tree_ensemble
    offsets, feature, threshold = c["offsets"], c["feature"], c["threshold"]
    left, right, value = c["left"], c["right"], c["value"]
    total = 0.0
    for k in range(c["n_trees"]):
        node = offsets[k]
        f = feature[node]
        while f >= 0:
            node = offsets[k] + (left[node] if h[f] <= threshold[node] else right[node])
            f = feature[node]
        total += value[node]
    return at_now + c["base"] + c["scale"] * total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--samples", default="replay_samples.json")
    p.add_argument("--repeat", type=int, default=200)
    p.add_argument("--label", default="")
    a = p.parse_args()
    model, samples = load(a.model), load(a.samples)
    rows = samples["samples"] if isinstance(samples, dict) else samples
    c = compile_model(model)

    t0 = time.perf_counter()
    produced = [predict(c, row["features"]) for row in rows]
    compile_and_first_pass_s = time.perf_counter() - t0

    times = []
    for i in range(a.repeat):
        row = rows[i % len(rows)]["features"]
        t0 = time.perf_counter_ns()
        predict(c, row)
        times.append((time.perf_counter_ns() - t0) / 1e6)
    ordered = sorted(times)
    print(json.dumps({
        "label": a.label or Path(a.model).stem,
        "runtime": "Arduino UNO Q" if platform.machine().lower() in {"aarch64", "arm64"} else platform.machine(),
        "hostname": platform.node(), "kernel": platform.release(), "python": platform.python_version(),
        "model_type": model["model_type"], "model_file": Path(a.model).name,
        "model_bytes": Path(a.model).stat().st_size,
        "samples": len(rows), "repeat": a.repeat,
        "latency_ms": {"median": statistics.median(times),
                       "p95": ordered[max(0, math.ceil(.95 * len(ordered)) - 1)],
                       "max": max(times)},
        "first_pass_seconds": compile_and_first_pass_s,
        "first_predictions": produced[:3],
        "acceleration_claim": "none; CPU Python only. No NPU/QNN/Hexagon/GPU.",
    }))


if __name__ == "__main__":
    main()
