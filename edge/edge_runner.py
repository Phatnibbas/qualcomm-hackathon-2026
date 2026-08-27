#!/usr/bin/env python3
import argparse, json, math, platform, statistics, time
from pathlib import Path

def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def predict(model, features):
    if len(features) != model["n_features"]:
        raise ValueError(f"expected {model['n_features']} features, got {len(features)}")
    if not all(math.isfinite(float(v)) for v in features):
        raise ValueError("non-finite input")
    at_now = float(features[model["at_now_index"]])
    if model["model_type"] == "persistence":
        return at_now
    h = [(float(v) - float(m)) / float(s) for v, m, s in zip(features, model["feature_mean"], model["feature_scale"])]
    for layer in model["layers"]:
        nxt = []
        ws = float(layer["weight_scale"])
        for neuron in layer["outputs"]:
            total = float(neuron["bias"])
            total += ws * sum(int(q) * h[int(i)] for i, q in zip(neuron["indices"], neuron["qweights"]))
            nxt.append(max(0.0, total) if layer["relu"] else total)
        h = nxt
    return at_now + h[0]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="edge_model.json")
    p.add_argument("--samples", default="replay_samples.json")
    p.add_argument("--repeat", type=int, default=200)
    a = p.parse_args()
    model, samples = load(a.model), load(a.samples)
    rows = samples["samples"]
    produced = [predict(model, row["features"]) for row in rows]
    times = []
    for i in range(a.repeat):
        row = rows[i % len(rows)]["features"]
        t0 = time.perf_counter_ns(); predict(model, row); times.append((time.perf_counter_ns() - t0) / 1e6)
    ordered = sorted(times)
    result = {
        "runtime": "Arduino UNO Q" if platform.machine().lower() in {"aarch64", "arm64"} else platform.machine(),
        "hostname": platform.node(), "kernel": platform.release(), "python": platform.python_version(),
        "model_type": model["model_type"], "model_run_id": model["run_id"],
        "samples": len(rows), "repeat": a.repeat,
        "latency_ms": {"median": statistics.median(times), "p95": ordered[max(0, math.ceil(.95 * len(ordered)) - 1)], "max": max(times)},
        "first_predictions": produced[:5],
    }
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
