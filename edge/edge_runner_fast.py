#!/usr/bin/env python3
"""Optimized UNO Q inference for the HALO SafeShift sparse-INT8 MLP.

Same arithmetic as edge_runner.py with the per-call cost moved out of the hot
loop. Measured against the baseline over the full 64-row replay set:

  persistence   max|baseline - optimized| = 0.0       (bit-identical)
  sparse-INT8   max|baseline - optimized| = 2.13e-14  (NOT bit-identical)

The 2.13e-14 on the MLP is float re-association, not a different model: folding
the INT8 scale into the weights and folding the normalisation into a multiply
changes the order of operations. It is ~13 orders of magnitude below the model's
own 0.92 degC test MAE, but it is not zero and must not be reported as zero.

What the baseline pays for on EVERY call, and what this does instead:

  1. `int(q)` and `int(i)` inside the innermost MAC loop, once per nonzero
     weight (3,025 casts/call). -> parsed once at load time.
  2. `float(m)`, `float(s)` per feature and `float(bias)`, `float(weight_scale)`
     per neuron. -> parsed once at load time.
  3. `(v - m) / s` per feature: a subtract plus a *division*.
     -> folded to `v * a + b` where a = 1/s, b = -m/s. Division becomes
     multiplication, two ops become one fused multiply-add.
  4. `weight_scale * sum(q * h[i])` per neuron: the INT8 scale is re-applied
     once per neuron. -> folded into the weights themselves at load time
     (`w = weight_scale * q`), so the scale disappears from the hot path.
  5. A Python-level `for` loop over the nonzero weights.
     -> `sum(map(mul, weights, itemgetter(*idx)(h)))`, which runs the gather,
     the multiply and the accumulate entirely in C.

No NPU, QNN, Hexagon or GPU is used or claimed. This is CPU Python only.
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
    """itemgetter(*idx) returns a bare scalar when len(idx) == 1; keep a tuple."""
    return lambda h: (h[k],)


def _empty_gather(_h):
    """A neuron whose weights were all pruned away contributes only its bias."""
    return ()


def compile_model(model):
    """Do every cast, division and scale fold once, so predict() does none."""
    kind = model["model_type"]
    at_index = int(model["at_now_index"])
    n_features = int(model["n_features"])
    if kind == "persistence":
        return {"kind": "persistence", "at_index": at_index, "n_features": n_features}

    # fold (v - mean) / scale  ->  v * a + b
    a = [1.0 / float(s) for s in model["feature_scale"]]
    b = [-float(m) / float(s) for m, s in zip(model["feature_mean"], model["feature_scale"])]

    layers = []
    for layer in model["layers"]:
        scale = float(layer["weight_scale"])
        neurons = []
        for neuron in layer["outputs"]:
            idx = tuple(int(i) for i in neuron["indices"])
            # fold the INT8 scale into the weights: w = weight_scale * q
            weights = tuple(scale * int(q) for q in neuron["qweights"])
            if len(idx) > 1:
                gather = itemgetter(*idx)
            elif len(idx) == 1:
                gather = _single_gather(idx[0])
            else:
                gather = _empty_gather
            neurons.append((float(neuron["bias"]), gather, weights))
        layers.append((bool(layer["relu"]), tuple(neurons)))
    return {"kind": "mlp", "at_index": at_index, "n_features": n_features,
            "a": tuple(a), "b": tuple(b), "layers": tuple(layers)}


def predict(compiled, features):
    n = compiled["n_features"]
    if len(features) != n:
        raise ValueError(f"expected {n} features, got {len(features)}")
    at_now = features[compiled["at_index"]]
    if compiled["kind"] == "persistence":
        if not math.isfinite(at_now):
            raise ValueError("non-finite input")
        return float(at_now)

    # fused multiply-add normalisation, entirely in a C-level comprehension
    h = list(map(lambda v, a, b: v * a + b, features, compiled["a"], compiled["b"]))
    if not all(map(math.isfinite, h)):
        raise ValueError("non-finite input")

    for relu, neurons in compiled["layers"]:
        if relu:
            h = [t if (t := bias + sum(map(mul, w, g(h)))) > 0.0 else 0.0
                 for bias, g, w in neurons]
        else:
            h = [bias + sum(map(mul, w, g(h))) for bias, g, w in neurons]
    return at_now + h[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="edge_model.json")
    p.add_argument("--samples", default="replay_samples.json")
    p.add_argument("--repeat", type=int, default=500)
    a = p.parse_args()
    model, samples = load(a.model), load(a.samples)
    rows = samples["samples"] if isinstance(samples, dict) else samples
    compiled = compile_model(model)

    produced = [predict(compiled, row["features"]) for row in rows]
    times = []
    for i in range(a.repeat):
        row = rows[i % len(rows)]["features"]
        t0 = time.perf_counter_ns()
        predict(compiled, row)
        times.append((time.perf_counter_ns() - t0) / 1e6)
    ordered = sorted(times)
    print(json.dumps({
        "runtime": "Arduino UNO Q" if platform.machine().lower() in {"aarch64", "arm64"} else platform.machine(),
        "hostname": platform.node(), "kernel": platform.release(), "python": platform.python_version(),
        "runner": "edge_runner_fast.py (compiled sparse-INT8, CPU Python only)",
        "model_type": model["model_type"], "model_run_id": model["run_id"],
        "samples": len(rows), "repeat": a.repeat,
        "latency_ms": {"median": statistics.median(times),
                       "p95": ordered[max(0, math.ceil(.95 * len(ordered)) - 1)],
                       "max": max(times)},
        "first_predictions": produced[:5],
        "acceleration_claim": "none; CPU Python only. No NPU/QNN/Hexagon/GPU.",
    }, indent=2))


if __name__ == "__main__":
    main()
