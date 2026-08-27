"""Build the self-contained emergency HALO SafeShift Colab notebook.

This generator never trains a model.  It only writes notebook JSON so the
actual compute remains in Colab, as required by the emergency workflow.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "HALO_SafeShift_Training.ipynb"


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


INTRO = r'''# HALO SafeShift — Emergency Colab Pipeline

**One manual action:** choose `VFCD_3rd_landscape_20260816-0336_74700rows.csv`
when the upload dialog appears. Then use **Runtime → Run all**.

This notebook performs the heavy work in Colab:

1. validates and resamples the 74,700-row station CSV;
2. builds leakage-safe 60-minute windows for a +30-minute target;
3. trains a compact MLP with GPU support when Colab provides one;
4. compares it with persistence on a chronological holdout;
5. measures 0/30/50/70% magnitude pruning;
6. exports FP32, dynamic-range and full-INT8 TFLite models;
7. exports a dependency-free sparse INT8 runner for Arduino UNO Q;
8. downloads one ZIP containing the model, metrics, figures, replay samples,
   hashes and board runner.

## Claim boundary

The target is a **station-derived shade apparent-temperature estimate** using
temperature, humidity and observed station wind. It is **not WBGT**, a medical
prediction, a legal safety limit, or a direct-sun measurement. The evaluation
is retrospective and chronological. UNO Q latency is measured only after the
downloaded artifact is actually run on the board.
'''


SETUP = r'''# 1) Environment + upload (the only manual action)
import os, sys, json, math, time, hashlib, zipfile, platform, subprocess
from pathlib import Path
from datetime import datetime, timezone

EXPECTED_NAME = "VFCD_3rd_landscape_20260816-0336_74700rows.csv"
SEED = 20260816
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-colab"
OUT = Path("halo_safeshift_artifacts") / RUN_ID
OUT.mkdir(parents=True, exist_ok=True)

def ensure_packages():
    required = {"numpy": "numpy", "pandas": "pandas", "matplotlib": "matplotlib", "tensorflow": "tensorflow"}
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])

ensure_packages()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def locate_or_upload_csv():
    exact = list(Path(".").glob(f"**/{EXPECTED_NAME}"))
    if exact:
        return exact[0]
    candidates = list(Path(".").glob("**/*74700rows.csv"))
    if len(candidates) == 1:
        return candidates[0]
    if "google.colab" not in sys.modules:
        raise FileNotFoundError(f"Place {EXPECTED_NAME} beside this notebook")
    from google.colab import files
    print(f"UPLOAD NOW: choose {EXPECTED_NAME}")
    uploaded = files.upload()
    if EXPECTED_NAME in uploaded:
        return Path(EXPECTED_NAME)
    csvs = [Path(name) for name in uploaded if name.lower().endswith(".csv")]
    if len(csvs) != 1:
        raise RuntimeError("Upload exactly one CSV")
    return csvs[0]

CSV_PATH = locate_or_upload_csv()
CSV_SHA256 = sha256_file(CSV_PATH)
GPU = tf.config.list_physical_devices("GPU")
ENV = {
    "run_id": RUN_ID,
    "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "csv_name": CSV_PATH.name,
    "csv_sha256": CSV_SHA256,
    "csv_bytes": CSV_PATH.stat().st_size,
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "tensorflow": tf.__version__,
    "gpu_visible": [d.name for d in GPU],
    "training_device": "GPU when TensorFlow places supported ops there; otherwise CPU",
    "edge_acceleration_claim": "none; UNO Q runtime is measured separately",
}
print(json.dumps(ENV, indent=2))
'''


PIPELINE = r'''# 2) Data preparation — deterministic, chronological, no interpolation
ALIASES = {
    "timestamp": ["timestamp_utc_iso", "created_at", "timestamp"],
    "wind_speed": ["Wind speed (m/s)", "WindSpeed", "wind_speed"],
    "wind_direction": ["Wind direction (°)", "WindDirection", "wind_direction"],
    "temperature": ["Temperature (°C)", "Temperature", "temperature"],
    "pressure": ["Pressure (kPa)", "Pressure", "pressure"],
    "light": ["Light (lux)", "Light", "light"],
    "humidity": ["Humidity (%RH)", "Humidity", "humidity"],
    "noise": ["Sound level (dB)", "Noise", "noise"],
    "pm25": ["PM2.5 (µg/m³)", "PM2.5", "pm25"],
}

def resolve_columns(frame):
    resolved = {}
    for semantic, names in ALIASES.items():
        hits = [name for name in names if name in frame.columns]
        if not hits:
            raise ValueError(f"Missing {semantic}; tried {names}; available={list(frame.columns)}")
        resolved[semantic] = hits[0]
    return resolved

raw = pd.read_csv(CSV_PATH, low_memory=False)
columns = resolve_columns(raw)
if len(raw) != 74700:
    raise ValueError(f"Expected 74,700 rows from filename, read {len(raw):,}")

work = pd.DataFrame({"timestamp": pd.to_datetime(raw[columns["timestamp"]], utc=True, errors="coerce")})
for semantic in ["wind_speed", "wind_direction", "temperature", "pressure", "light", "humidity", "noise", "pm25"]:
    work[semantic] = pd.to_numeric(raw[columns[semantic]], errors="coerce")

qc = {"raw_rows": int(len(work))}
work = work.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp")
qc["timestamp_valid_unique_rows"] = int(len(work))
valid = (
    work["temperature"].between(0.01, 60.0)
    & work["humidity"].between(1.01, 100.0)
    & work["wind_speed"].between(0.0, 40.0)
    & work["pressure"].between(80.0, 120.0)
    & work["wind_direction"].between(0.0, 360.0, inclusive="left")
    & work[["light", "noise", "pm25"]].notna().all(axis=1)
)
qc["rows_rejected_physical_or_missing"] = int((~valid).sum())
work = work.loc[valid].set_index("timestamp")

numeric_cols = ["wind_speed", "wind_direction", "temperature", "pressure", "light", "humidity", "noise", "pm25"]
agg = work[numeric_cols].resample("5min", label="right", closed="right").median()
counts = work["temperature"].resample("5min", label="right", closed="right").count().rename("raw_count")
bins = agg.join(counts)
bins = bins[bins["raw_count"] >= 6].dropna()
qc["valid_5min_bins"] = int(len(bins))
qc["first_valid_bin_utc"] = bins.index.min().isoformat()
qc["last_valid_bin_utc"] = bins.index.max().isoformat()

def apparent_temperature(temp_c, rh_percent, wind_mps):
    e = (rh_percent / 100.0) * 6.105 * np.exp(17.27 * temp_c / (237.7 + temp_c))
    return temp_c + 0.33 * e - 0.70 * wind_mps - 4.0

bins["apparent_temperature"] = apparent_temperature(
    bins["temperature"].to_numpy(), bins["humidity"].to_numpy(), bins["wind_speed"].to_numpy()
)

PER_STEP = [
    "temperature", "humidity", "wind_speed", "wind_direction_sin", "wind_direction_cos",
    "light_log1p", "pressure", "pm25_log1p", "noise", "apparent_temperature",
]
feature_names = [f"t-{55 - 5*i:02d}m_{name}" for i in range(12) for name in PER_STEP]
feature_names += ["apparent_temperature_now", "local_time_sin", "local_time_cos", "day_of_year_sin", "day_of_year_cos"]
AT_NOW_INDEX = len(feature_names) - 5

values = bins.copy()
rad = np.deg2rad(values["wind_direction"].to_numpy())
values["wind_direction_sin"] = np.sin(rad)
values["wind_direction_cos"] = np.cos(rad)
values["light_log1p"] = np.log1p(np.maximum(values["light"], 0.0))
values["pm25_log1p"] = np.log1p(np.maximum(values["pm25"], 0.0))

X, y_delta, y_abs, at_now, issue_times, target_times = [], [], [], [], [], []
times = values.index
expected_span = pd.Timedelta(minutes=85)  # t-55 through t+30
for i in range(11, len(values) - 6):
    if times[i + 6] - times[i - 11] != expected_span:
        continue
    block = values.iloc[i - 11:i + 1]
    if not np.all(np.diff(block.index.asi8) == pd.Timedelta(minutes=5).value):
        continue
    sequence = block[PER_STEP].to_numpy(dtype=np.float32).reshape(-1)
    current_at = float(values.iloc[i]["apparent_temperature"])
    future_at = float(values.iloc[i + 6]["apparent_temperature"])
    local = times[i].tz_convert("Asia/Ho_Chi_Minh")
    minute = local.hour * 60 + local.minute
    doy = local.dayofyear
    extras = np.array([
        current_at,
        math.sin(2 * math.pi * minute / 1440), math.cos(2 * math.pi * minute / 1440),
        math.sin(2 * math.pi * doy / 366), math.cos(2 * math.pi * doy / 366),
    ], dtype=np.float32)
    X.append(np.concatenate([sequence, extras]))
    y_delta.append(future_at - current_at)
    y_abs.append(future_at)
    at_now.append(current_at)
    issue_times.append(times[i])
    target_times.append(times[i + 6])

X = np.asarray(X, dtype=np.float32)
y_delta = np.asarray(y_delta, dtype=np.float32)
y_abs = np.asarray(y_abs, dtype=np.float32)
at_now = np.asarray(at_now, dtype=np.float32)
issue_times = pd.DatetimeIndex(issue_times)
target_times = pd.DatetimeIndex(target_times)
if len(X) < 1000:
    raise RuntimeError(f"Only {len(X)} windows; expected at least 1,000")

# Chronological 60/20/20 with a 90-minute embargo around boundaries.
n = len(X)
b1, b2, embargo = int(n * 0.60), int(n * 0.80), 18
idx_train = np.arange(0, b1 - embargo)
idx_val = np.arange(b1 + embargo, b2 - embargo)
idx_test = np.arange(b2 + embargo, n)
if min(len(idx_train), len(idx_val), len(idx_test)) == 0:
    raise RuntimeError("Chronological split is empty")

mean = X[idx_train].mean(axis=0).astype(np.float32)
scale = X[idx_train].std(axis=0).astype(np.float32)
scale[scale < 1e-6] = 1.0
Xs = ((X - mean) / scale).astype(np.float32)

split = {
    "policy": "chronological 60/20/20 with 90-minute embargo",
    "train": {"n": len(idx_train), "first": issue_times[idx_train[0]].isoformat(), "last": issue_times[idx_train[-1]].isoformat()},
    "validation": {"n": len(idx_val), "first": issue_times[idx_val[0]].isoformat(), "last": issue_times[idx_val[-1]].isoformat()},
    "test": {"n": len(idx_test), "first": issue_times[idx_test[0]].isoformat(), "last": issue_times[idx_test[-1]].isoformat()},
}
qc["eligible_windows"] = int(n)
qc["n_features"] = int(X.shape[1])
print(json.dumps({"qc": qc, "split": split}, indent=2))
'''


TRAIN = r'''# 3) Train compact MLP in Colab and compare with persistence
def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))

def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1],), name="features"),
    tf.keras.layers.Dense(64, activation="relu", name="dense_64"),
    tf.keras.layers.Dense(32, activation="relu", name="dense_32"),
    tf.keras.layers.Dense(1, name="delta_at"),
], name="halo_safeshift_mlp")
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mae", metrics=["mae"])

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, min_delta=1e-4, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.5, min_lr=1e-5),
]
t0 = time.perf_counter()
history = model.fit(
    Xs[idx_train], y_delta[idx_train],
    validation_data=(Xs[idx_val], y_delta[idx_val]),
    epochs=120, batch_size=128, callbacks=callbacks, verbose=2,
)
training_seconds = time.perf_counter() - t0

def predict_absolute(keras_model, indices):
    delta = keras_model.predict(Xs[indices], batch_size=512, verbose=0).reshape(-1)
    return at_now[indices] + delta

val_pred = predict_absolute(model, idx_val)
test_pred = predict_absolute(model, idx_test)
val_persistence = at_now[idx_val]
test_persistence = at_now[idx_test]

base_metrics = {
    "validation": {
        "mlp_mae_degc": mae(y_abs[idx_val], val_pred),
        "persistence_mae_degc": mae(y_abs[idx_val], val_persistence),
    },
    "test": {
        "mlp_mae_degc": mae(y_abs[idx_test], test_pred),
        "mlp_rmse_degc": rmse(y_abs[idx_test], test_pred),
        "persistence_mae_degc": mae(y_abs[idx_test], test_persistence),
        "persistence_rmse_degc": rmse(y_abs[idx_test], test_persistence),
    },
    "training_seconds": training_seconds,
    "epochs_ran": len(history.history["loss"]),
}
gain = base_metrics["test"]["persistence_mae_degc"] - base_metrics["test"]["mlp_mae_degc"]
relative_gain = gain / base_metrics["test"]["persistence_mae_degc"]
base_metrics["test"]["absolute_mae_gain_degc"] = gain
base_metrics["test"]["relative_mae_gain_fraction"] = relative_gain
base_metrics["test"]["learned_model_gate"] = bool(gain >= 0.05 and relative_gain >= 0.05)
print(json.dumps(base_metrics, indent=2))
'''


OPTIMIZE_EXPORT = r'''# 4) Magnitude pruning, INT8 weight quantization, TFLite export, edge bundle
original_weights = [np.asarray(w, dtype=np.float32).copy() for w in model.get_weights()]

def forward_numpy(x_scaled, weights):
    h = np.asarray(x_scaled, dtype=np.float32)
    n_layers = len(weights) // 2
    for layer in range(n_layers):
        h = h @ weights[2 * layer] + weights[2 * layer + 1]
        if layer < n_layers - 1:
            h = np.maximum(h, 0.0)
    return h.reshape(-1)

def prune_weights(weights, sparsity):
    result = [w.copy() for w in weights]
    if sparsity <= 0:
        return result
    for i in range(0, len(result), 2):
        kernel = result[i]
        flat = np.abs(kernel).reshape(-1)
        k = int(math.floor(sparsity * flat.size))
        if k > 0:
            threshold = np.partition(flat, k - 1)[k - 1]
            kernel[np.abs(kernel) <= threshold] = 0.0
    return result

pruning_rows = []
candidate_weights = {}
base_val = base_metrics["validation"]["mlp_mae_degc"]
allowed_val_regression = max(0.03, base_val * 0.02)
for sparsity in (0.0, 0.30, 0.50, 0.70):
    weights = prune_weights(original_weights, sparsity)
    candidate_weights[sparsity] = weights
    val_abs = at_now[idx_val] + forward_numpy(Xs[idx_val], weights)
    test_abs = at_now[idx_test] + forward_numpy(Xs[idx_test], weights)
    kernel_total = sum(weights[i].size for i in range(0, len(weights), 2))
    kernel_zero = sum(np.count_nonzero(weights[i] == 0) for i in range(0, len(weights), 2))
    pruning_rows.append({
        "requested_sparsity": sparsity,
        "actual_sparsity": kernel_zero / kernel_total,
        "validation_mae_degc": mae(y_abs[idx_val], val_abs),
        "test_mae_degc": mae(y_abs[idx_test], test_abs),
        "eligible_by_validation": mae(y_abs[idx_val], val_abs) <= base_val + allowed_val_regression,
    })

eligible = [row for row in pruning_rows if row["eligible_by_validation"]]
chosen_pruning = max(eligible, key=lambda row: row["requested_sparsity"])
chosen_weights = candidate_weights[chosen_pruning["requested_sparsity"]]

def quantize_kernels(weights):
    quantized, dequantized, layer_meta = [], [], []
    for layer in range(len(weights) // 2):
        kernel, bias = weights[2 * layer], weights[2 * layer + 1]
        max_abs = float(np.max(np.abs(kernel)))
        qscale = max_abs / 127.0 if max_abs > 0 else 1.0
        q = np.clip(np.rint(kernel / qscale), -127, 127).astype(np.int8)
        quantized.append((q, bias.astype(np.float32), qscale))
        dequantized.extend([(q.astype(np.float32) * qscale), bias.astype(np.float32)])
        layer_meta.append({"layer": layer, "scale": qscale, "zero_weights": int(np.count_nonzero(q == 0)), "weights": int(q.size)})
    return quantized, dequantized, layer_meta

quantized_layers, dequantized_weights, quant_meta = quantize_kernels(chosen_weights)
quant_val = at_now[idx_val] + forward_numpy(Xs[idx_val], dequantized_weights)
quant_test = at_now[idx_test] + forward_numpy(Xs[idx_test], dequantized_weights)
quant_val_mae = mae(y_abs[idx_val], quant_val)
quant_test_mae = mae(y_abs[idx_test], quant_test)
quant_eligible = quant_val_mae <= chosen_pruning["validation_mae_degc"] + 0.03
quant_test_gain = base_metrics["test"]["persistence_mae_degc"] - quant_test_mae
quant_test_relative_gain = quant_test_gain / base_metrics["test"]["persistence_mae_degc"]

# Build a Keras clone carrying the selected pruned weights for TFLite conversion.
pruned_model = tf.keras.models.clone_model(model)
pruned_model.set_weights(chosen_weights)
sample = np.zeros((1, X.shape[1]), dtype=np.float32)
_ = pruned_model(sample)

def write_tflite(path, mode):
    converter = tf.lite.TFLiteConverter.from_keras_model(pruned_model)
    if mode in {"dynamic", "int8"}:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if mode == "int8":
        def representative():
            limit = min(512, len(idx_train))
            for row in Xs[idx_train[:limit]]:
                yield [row.reshape(1, -1).astype(np.float32)]
        converter.representative_dataset = representative
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    blob = converter.convert()
    Path(path).write_bytes(blob)
    return len(blob)

tflite_sizes = {}
for mode, name in [("fp32", "model_fp32.tflite"), ("dynamic", "model_dynamic_int8.tflite"), ("int8", "model_full_int8.tflite")]:
    try:
        tflite_sizes[mode] = write_tflite(OUT / name, mode)
    except Exception as exc:
        tflite_sizes[mode] = {"error": str(exc)}

def sparse_int8_payload(layers):
    payload = []
    for layer_no, (q, bias, qscale) in enumerate(layers):
        outputs = []
        for out_idx in range(q.shape[1]):
            nonzero = np.flatnonzero(q[:, out_idx])
            outputs.append({
                "indices": nonzero.astype(int).tolist(),
                "qweights": q[nonzero, out_idx].astype(int).tolist(),
                "bias": float(bias[out_idx]),
            })
        payload.append({
            "input_width": int(q.shape[0]), "output_width": int(q.shape[1]),
            "weight_scale": float(qscale), "relu": layer_no < len(layers) - 1,
            "outputs": outputs,
        })
    return payload

learned_edge = {
    "format": "halo-safeshift-sparse-int8-v1",
    "model_type": "mlp_sparse_int8",
    "run_id": RUN_ID,
    "n_features": int(X.shape[1]),
    "feature_names": feature_names,
    "feature_mean": mean.astype(float).tolist(),
    "feature_scale": scale.astype(float).tolist(),
    "at_now_index": AT_NOW_INDEX,
    "output": "delta apparent temperature; final = AT_now + delta",
    "layers": sparse_int8_payload(quantized_layers),
    "claim_boundary": "station-derived shade apparent-temperature estimate; not WBGT or a medical/legal limit",
}

gate = bool(
    quant_eligible
    and quant_test_gain >= 0.05
    and quant_test_relative_gain >= 0.05
)
operational_edge = learned_edge if gate else {
    "format": "halo-safeshift-persistence-v1", "model_type": "persistence",
    "run_id": RUN_ID, "n_features": int(X.shape[1]), "feature_names": feature_names,
    "at_now_index": AT_NOW_INDEX,
    "claim_boundary": "learned challenger did not clear the frozen gate; honest persistence fallback",
}

(OUT / "challenger_edge_model.json").write_text(json.dumps(learned_edge, separators=(",", ":")), encoding="utf-8")
(OUT / "edge_model.json").write_text(json.dumps(operational_edge, separators=(",", ":")), encoding="utf-8")
(OUT / "feature_schema.json").write_text(json.dumps({"n_features": len(feature_names), "feature_names": feature_names, "at_now_index": AT_NOW_INDEX}, indent=2), encoding="utf-8")

EDGE_RUNNER = r"""#!/usr/bin/env python3
import argparse, hashlib, json, math, platform, statistics, time
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
    expected_key = "challenger_at_degc" if model["model_type"] == "mlp_sparse_int8" else "persistence_at_degc"
    expected = [float(row[expected_key]) for row in rows]
    max_abs_error = max(abs(a - b) for a, b in zip(produced, expected))
    if max_abs_error > 1e-4:
        raise RuntimeError(f"runtime/reference mismatch: max abs error {max_abs_error}")
    times = []
    for i in range(a.repeat):
        row = rows[i % len(rows)]["features"]
        t0 = time.perf_counter_ns(); predict(model, row); times.append((time.perf_counter_ns() - t0) / 1e6)
    ordered = sorted(times)
    result = {
        "runtime": "Arduino UNO Q" if platform.machine().lower() in {"aarch64", "arm64"} else platform.machine(),
        "hostname": platform.node(), "kernel": platform.release(), "python": platform.python_version(),
        "model_type": model["model_type"], "model_run_id": model["run_id"],
        "model_sha256": hashlib.sha256(Path(a.model).read_bytes()).hexdigest(),
        "samples": len(rows), "repeat": a.repeat,
        "reference_max_abs_error_degc": max_abs_error,
        "latency_ms": {"median": statistics.median(times), "p95": ordered[max(0, math.ceil(.95 * len(ordered)) - 1)], "max": max(times)},
        "first_predictions": produced[:5],
    }
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
"""
(OUT / "edge_runner.py").write_text(EDGE_RUNNER, encoding="utf-8")

replay_rows = []
for j in idx_test[:64]:
    replay_rows.append({
        "issue_time_utc": issue_times[j].isoformat(), "target_time_utc": target_times[j].isoformat(),
        "features": X[j].astype(float).tolist(), "observed_at_degc": float(y_abs[j]),
        "persistence_at_degc": float(at_now[j]), "challenger_at_degc": float(quant_test[np.where(idx_test == j)[0][0]]),
    })
(OUT / "replay_samples.json").write_text(json.dumps({"samples": replay_rows}, separators=(",", ":")), encoding="utf-8")

optimization = {
    "pruning_candidates": pruning_rows,
    "selected_pruning_by_validation": chosen_pruning,
    "int8_weight_storage": {
        "validation_mae_degc": quant_val_mae,
        "test_mae_degc": quant_test_mae,
        "test_absolute_gain_vs_persistence_degc": quant_test_gain,
        "test_relative_gain_vs_persistence_fraction": quant_test_relative_gain,
        "eligible_by_validation": quant_eligible,
        "layers": quant_meta,
    },
    "tflite_bytes": tflite_sizes,
    "operational_model": "learned_sparse_int8" if gate else "persistence",
    "learned_gate_pass": gate,
    "boundary": "Pruning and quantization are retained by validation quality; speed is not claimed until measured on UNO Q.",
}
print(json.dumps(optimization, indent=2))
'''


REPORT = r'''# 5) Metrics, figures, hashes and automatic download
metrics = {
    "run_id": RUN_ID,
    "target": "station-derived shade apparent-temperature estimate at t+30 minutes",
    "evaluation_status": "retrospective chronological holdout; not prospective certification",
    "environment": ENV,
    "qc": qc,
    "split": split,
    "base_model": base_metrics,
    "optimization": optimization,
    "forbidden_claims": ["WBGT", "medical prediction", "legal safety limit", "UNO Q NPU/QNN/Hexagon acceleration"],
}
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
(OUT / "qc_report.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
(OUT / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")

plt.figure(figsize=(8, 4.2))
plt.plot(history.history["loss"], label="train MAE")
plt.plot(history.history["val_loss"], label="validation MAE")
plt.xlabel("Epoch"); plt.ylabel("MAE of ΔAT (°C)"); plt.title("HALO SafeShift training history")
plt.grid(alpha=.25); plt.legend(); plt.tight_layout(); plt.savefig(OUT / "training_history.png", dpi=180); plt.close()

show = min(288, len(idx_test))
plt.figure(figsize=(11, 4.5))
plt.plot(issue_times[idx_test[:show]], y_abs[idx_test[:show]], label="Observed +30 min", linewidth=2)
plt.plot(issue_times[idx_test[:show]], quant_test[:show], label="HALO quantized", linewidth=1.5)
plt.plot(issue_times[idx_test[:show]], at_now[idx_test[:show]], label="Persistence", alpha=.75)
plt.ylabel("Apparent temperature estimate (°C)"); plt.xlabel("Issue time UTC")
plt.title("Chronological held-out replay"); plt.grid(alpha=.2); plt.legend(); plt.tight_layout()
plt.savefig(OUT / "heldout_replay.png", dpi=180); plt.close()

labels = [f"{int(100*r['requested_sparsity'])}%" for r in pruning_rows]
vals = [r["validation_mae_degc"] for r in pruning_rows]
plt.figure(figsize=(7, 4.2)); plt.bar(labels, vals, color="#ef6c35")
plt.xlabel("Magnitude pruning"); plt.ylabel("Validation MAE (°C)"); plt.title("Pruning quality trade-off")
plt.grid(axis="y", alpha=.2); plt.tight_layout(); plt.savefig(OUT / "pruning_tradeoff.png", dpi=180); plt.close()

manifest = {}
for path in sorted(OUT.iterdir()):
    if path.is_file() and path.name != "manifest.json":
        manifest[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
(OUT / "manifest.json").write_text(json.dumps({"run_id": RUN_ID, "files": manifest, "csv_sha256": CSV_SHA256}, indent=2), encoding="utf-8")

zip_path = Path(f"HALO_SafeShift_{RUN_ID}.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for path in sorted(OUT.iterdir()):
        if path.is_file():
            zf.write(path, path.name)

summary = {
    "RESULT": "TRAINING_AND_EXPORT_FINISHED",
    "operational_model": optimization["operational_model"],
    "test_mlp_mae_degc": base_metrics["test"]["mlp_mae_degc"],
    "test_persistence_mae_degc": base_metrics["test"]["persistence_mae_degc"],
    "selected_pruning": chosen_pruning["actual_sparsity"],
    "quantized_test_mae_degc": optimization["int8_weight_storage"]["test_mae_degc"],
    "zip": str(zip_path), "zip_bytes": zip_path.stat().st_size, "zip_sha256": sha256_file(zip_path),
    "NEXT": "Download ZIP, then deploy edge_model.json + edge_runner.py + replay_samples.json to UNO Q.",
}
print("\n" + "=" * 72)
print(json.dumps(summary, indent=2))
print("=" * 72)

if "google.colab" in sys.modules:
    from google.colab import files
    files.download(str(zip_path))
'''


nb = {
    "cells": [
        markdown(INTRO),
        code(SETUP),
        code(PIPELINE),
        code(TRAIN),
        code(OPTIMIZE_EXPORT),
        code(REPORT),
    ],
    "metadata": {
        "accelerator": "GPU",
        "colab": {"name": "HALO_SafeShift_Training.ipynb", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Wrote {OUT}")
