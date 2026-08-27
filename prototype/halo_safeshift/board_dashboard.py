#!/usr/bin/env python3
"""Offline HALO SafeShift replay dashboard for the Arduino UNO Q.

The service intentionally shows the learned challenger beside persistence and
the later observation.  It never hides the frozen model gate outcome.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

def load(path: Path | str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def predict(model: dict, features: list[float]) -> float:
    if len(features) != model["n_features"]:
        raise ValueError(f"expected {model['n_features']} features, got {len(features)}")
    if not all(math.isfinite(float(value)) for value in features):
        raise ValueError("non-finite input")
    at_now = float(features[model["at_now_index"]])
    if model["model_type"] == "persistence":
        return at_now
    hidden = [
        (float(value) - float(mean)) / float(scale)
        for value, mean, scale in zip(
            features, model["feature_mean"], model["feature_scale"], strict=True
        )
    ]
    for layer in model["layers"]:
        next_hidden = []
        weight_scale = float(layer["weight_scale"])
        for neuron in layer["outputs"]:
            total = float(neuron["bias"])
            total += weight_scale * sum(
                int(weight) * hidden[int(index)]
                for index, weight in zip(
                    neuron["indices"], neuron["qweights"], strict=True
                )
            )
            next_hidden.append(max(0.0, total) if layer["relu"] else total)
        hidden = next_hidden
    return at_now + hidden[0]


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>HALO SafeShift — UNO Q</title>
<style>
:root{--ink:#12233b;--muted:#607089;--paper:#f4f8fb;--card:#fff;--orange:#ef6c35;--blue:#2448d8;--green:#0b8a62;--red:#b63a3a}
*{box-sizing:border-box} body{margin:0;background:linear-gradient(135deg,#eef8ff,#f7f2ec);color:var(--ink);font-family:Segoe UI,Arial,sans-serif}
header{padding:18px 28px;background:#1829b8;color:white;display:flex;align-items:center;justify-content:space-between}
h1{font-size:28px;margin:0;letter-spacing:.3px}.tag{font-weight:700;background:#ffbf33;color:#18233a;padding:7px 12px;border-radius:5px}
main{max-width:1320px;margin:auto;padding:20px}.sub{color:var(--muted);font-size:14px;margin:0 0 15px}
.grid{display:grid;grid-template-columns:1.25fr .75fr;gap:16px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
.card,.panel{background:var(--card);border:1px solid #d9e3ec;border-radius:10px;box-shadow:0 5px 18px #1b355512;padding:16px}
.label{font-size:12px;text-transform:uppercase;letter-spacing:.8px;color:var(--muted);font-weight:700}.value{font-size:31px;font-weight:800;margin-top:7px}.unit{font-size:15px;color:var(--muted)}
.ai{border-top:6px solid var(--orange)}.base{border-top:6px solid var(--blue)}.obs{border-top:6px solid var(--green)}.now{border-top:6px solid #6a55c7}
.panel{margin-top:15px}.row{display:flex;justify-content:space-between;gap:12px;border-bottom:1px solid #e7edf2;padding:9px 0}.row:last-child{border:0}
.ok{color:var(--green);font-weight:800}.warn{color:var(--red);font-weight:800}.proof{font-family:Consolas,monospace;font-size:13px}
button{background:var(--orange);color:#fff;border:0;border-radius:7px;padding:12px 20px;font-weight:800;cursor:pointer}button:hover{filter:brightness(.94)}
.gate{padding:12px;border-radius:7px;background:#fff3e8;border-left:5px solid var(--orange);font-size:14px;margin-top:12px}
.legend{display:flex;gap:18px;font-size:13px;color:var(--muted);margin-top:12px}.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px}
canvas{width:100%;height:255px;background:#fbfdff;border-radius:6px;margin-top:10px}
@media(max-width:900px){.grid{grid-template-columns:1fr}.cards{grid-template-columns:repeat(2,1fr)}}
</style></head>
<body><header><div><h1>HALO SafeShift</h1><div>30-minute site apparent-temperature forecast at the edge</div></div><div class="tag">RUNNING ON ARDUINO UNO Q</div></header>
<main><p class="sub">Shade estimate from site temperature, humidity and observed wind. Not WBGT, a medical prediction, or a legal safety limit.</p>
<section class="grid"><div><div class="cards">
<div class="card now"><div class="label">Current estimate</div><div class="value" id="now">—</div><span class="unit">°C</span></div>
<div class="card ai"><div class="label">AI challenger +30 min</div><div class="value" id="ai">—</div><span class="unit">°C</span></div>
<div class="card base"><div class="label">Persistence +30 min</div><div class="value" id="base">—</div><span class="unit">°C</span></div>
<div class="card obs"><div class="label">Later observed</div><div class="value" id="obs">—</div><span class="unit">°C</span></div>
</div>
<div class="panel"><div style="display:flex;justify-content:space-between;align-items:center"><div><b>Held-out chronological replay</b><div class="sub" id="time">—</div></div><button onclick="nextRow()">NEXT REPLAY STEP</button></div>
<canvas id="chart" width="900" height="255"></canvas><div class="legend"><span><i class="dot" style="background:#ef6c35"></i>AI challenger</span><span><i class="dot" style="background:#2448d8"></i>Persistence</span><span><i class="dot" style="background:#0b8a62"></i>Observed</span></div></div>
</div><aside>
<div class="panel" style="margin-top:0"><div class="label">Measured model quality</div>
<div class="row"><span>AI challenger test MAE</span><b id="aiMae">—</b></div><div class="row"><span>Persistence test MAE</span><b id="baseMae">—</b></div>
<div class="row"><span>Pruning</span><b id="prune">—</b></div><div class="row"><span>Full INT8 TFLite</span><b id="size">—</b></div>
<div class="gate"><b>Transparent model gate:</b> the AI challenger is deployed and measured, but persistence remains the operational fallback because it achieved lower chronological test error.</div></div>
<div class="panel proof"><div class="label">Runtime proof</div><div class="row"><span>Host</span><b id="host">—</b></div><div class="row"><span>Kernel</span><b id="kernel">—</b></div><div class="row"><span>Python</span><b id="python">—</b></div><div class="row"><span>AI latency</span><b id="latency">—</b></div><div class="row"><span>Model hash</span><b id="hash">—</b></div><div class="row"><span>Network</span><b>local Wi-Fi / offline-ready</b></div></div>
</aside></section></main>
<script>
let history=[]; const colors={ai:'#ef6c35',base:'#2448d8',obs:'#0b8a62'};
function fmt(x,n=2){return Number(x).toFixed(n)}
function draw(){let c=document.getElementById('chart'),x=c.getContext('2d');x.clearRect(0,0,c.width,c.height);if(history.length<2)return;let vals=history.flatMap(r=>[r.ai,r.base,r.obs]),mn=Math.min(...vals)-.3,mx=Math.max(...vals)+.3;function px(i){return 35+i*(c.width-55)/(history.length-1)}function py(v){return 15+(mx-v)*(c.height-35)/(mx-mn)};x.strokeStyle='#d5e0e8';x.beginPath();for(let j=0;j<5;j++){let y=15+j*(c.height-35)/4;x.moveTo(30,y);x.lineTo(c.width-10,y)}x.stroke();for(let key of ['ai','base','obs']){x.strokeStyle=colors[key];x.lineWidth=3;x.beginPath();history.forEach((r,i)=>{i?x.lineTo(px(i),py(r[key])):x.moveTo(px(i),py(r[key]))});x.stroke()}}
async function nextRow(){let r=await fetch('/api/next').then(x=>x.json());document.getElementById('now').textContent=fmt(r.current);document.getElementById('ai').textContent=fmt(r.ai);document.getElementById('base').textContent=fmt(r.persistence);document.getElementById('obs').textContent=fmt(r.observed);document.getElementById('time').textContent=`Issue ${r.issue_time_utc} → target ${r.target_time_utc}`;document.getElementById('latency').textContent=`${fmt(r.ai_latency_ms,3)} ms`;history.push({ai:r.ai,base:r.persistence,obs:r.observed});if(history.length>50)history.shift();draw()}
async function boot(){let s=await fetch('/api/status').then(x=>x.json());document.getElementById('aiMae').textContent=`${fmt(s.metrics.ai_test_mae_degc,3)} °C`;document.getElementById('baseMae').textContent=`${fmt(s.metrics.persistence_test_mae_degc,3)} °C`;document.getElementById('prune').textContent=`${fmt(100*s.metrics.pruning,0)}%`;document.getElementById('size').textContent=`${(s.metrics.int8_bytes/1024).toFixed(1)} KiB`;document.getElementById('host').textContent=s.runtime.hostname;document.getElementById('kernel').textContent=s.runtime.kernel;document.getElementById('python').textContent=s.runtime.python;document.getElementById('hash').textContent=s.runtime.challenger_sha256.slice(0,16);for(let i=0;i<12;i++)await nextRow()};boot();
</script></body></html>"""


class State:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.challenger_path = root / "challenger_edge_model.json"
        self.challenger = load(self.challenger_path)
        self.persistence = load(root / "edge_model.json")
        self.samples = load(root / "replay_samples.json")["samples"]
        self.metrics = load(root / "metrics.json")
        self.index = 0
        self.lock = threading.Lock()

    @staticmethod
    def _sha256(path: Path) -> str:
        import hashlib

        return hashlib.sha256(path.read_bytes()).hexdigest()

    def status(self) -> dict:
        optimization = self.metrics["optimization"]
        return {
            "runtime": {
                "board": "Arduino UNO Q",
                "hostname": platform.node(),
                "kernel": platform.release(),
                "machine": platform.machine(),
                "python": platform.python_version(),
                "challenger_sha256": self._sha256(self.challenger_path),
                "run_id": self.challenger["run_id"],
            },
            "metrics": {
                "ai_test_mae_degc": optimization["int8_weight_storage"]["test_mae_degc"],
                "persistence_test_mae_degc": self.metrics["base_model"]["test"]["persistence_mae_degc"],
                "pruning": optimization["selected_pruning_by_validation"]["actual_sparsity"],
                "int8_bytes": optimization["tflite_bytes"]["int8"],
                "learned_gate_pass": optimization["learned_gate_pass"],
                "evaluation": self.metrics["evaluation_status"],
            },
            "claim_boundary": self.metrics["forbidden_claims"],
        }

    def next(self) -> dict:
        with self.lock:
            row = self.samples[self.index % len(self.samples)]
            self.index += 1
        features = row["features"]
        t0 = time.perf_counter_ns()
        ai = predict(self.challenger, features)
        latency_ms = (time.perf_counter_ns() - t0) / 1e6
        baseline = predict(self.persistence, features)
        if not all(math.isfinite(v) for v in (ai, baseline)):
            raise RuntimeError("non-finite dashboard prediction")
        return {
            "sequence": self.index,
            "issue_time_utc": row["issue_time_utc"],
            "target_time_utc": row["target_time_utc"],
            "current": row["persistence_at_degc"],
            "ai": ai,
            "persistence": baseline,
            "observed": row["observed_at_degc"],
            "ai_latency_ms": latency_ms,
        }


def handler_for(state: State):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, content_type: str, payload: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:  # noqa: N802 - stdlib API
            path = urlparse(self.path).path
            if path in {"/", "/index.html"}:
                self._send(200, "text/html; charset=utf-8", HTML.encode())
            elif path == "/api/status":
                self._send(200, "application/json", json.dumps(state.status()).encode())
            elif path == "/api/next":
                self._send(200, "application/json", json.dumps(state.next()).encode())
            elif path == "/healthz":
                self._send(200, "application/json", b'{"ok":true}')
            else:
                self._send(404, "application/json", b'{"error":"not found"}')

        def log_message(self, format: str, *args) -> None:
            print(f"{self.address_string()} - {format % args}", flush=True)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    state = State(args.root)
    server = ThreadingHTTPServer((args.host, args.port), handler_for(state))
    print(json.dumps({"status": "ready", "url": f"http://{args.host}:{args.port}", **state.status()["runtime"]}), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
