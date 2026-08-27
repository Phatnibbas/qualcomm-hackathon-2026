#!/usr/bin/env python3
"""Board-only Wi-Fi replay dashboard for a validated full Colab run.

It intentionally refuses to label a model fused unless the runtime bundle has
satellite inputs.  This process contains no training and no laptop inference.
"""

from __future__ import annotations

import argparse
import json
import platform
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

try:  # package execution in tests / deployed standalone scripts on the UNO Q
    from .full_runtime import FullRuntimeError, PortablePredictor
except ImportError:  # pragma: no cover - only reached by the board copy
    from full_runtime import FullRuntimeError, PortablePredictor


CLAIM_BOUNDARY = "Shade estimate from site temperature, humidity and wind; not WBGT or a medical/legal safety limit. Retrospective replay, not prospective certification."


HTML = """<!doctype html><html lang=\"en\"><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>HALO SafeShift</title>
<style>body{margin:0;background:#f2f5f4;color:#17231f;font:16px Georgia,serif}header{background:#133b32;color:#fff;padding:24px}main{max-width:1200px;margin:auto;padding:20px}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.card,section{background:#fff;border:1px solid #d6e0da;padding:16px;margin-bottom:14px}.label{font:700 12px ui-monospace,monospace;text-transform:uppercase;color:#51685e}.value{font-size:31px;font-weight:700}button{background:#d6632d;color:#fff;border:0;padding:12px 16px;font-weight:bold;cursor:pointer}.boundary{border-left:5px solid #d6632d;padding:12px;background:#fff7ee}.proof{font:13px ui-monospace,monospace;white-space:pre-wrap}@media(max-width:800px){.grid{grid-template-columns:repeat(2,1fr)}}</style>
<header><h1>HALO SafeShift</h1><div>Arduino UNO Q on-device replay</div></header><main>
<p class=\"boundary\">Shade estimate from site temperature, humidity and wind; not WBGT or a medical/legal safety limit. Retrospective replay, not prospective certification.</p>
<div class=\"grid\"><div class=\"card\"><div class=label>Current AT</div><div id=current class=value>—</div></div><div class=\"card\"><div class=label>+30 prediction</div><div id=prediction class=value>—</div></div><div class=\"card\"><div class=label>Persistence</div><div id=persistence class=value>—</div></div><div class=\"card\"><div class=label>Later observed</div><div id=observed class=value>Hidden</div></div></div>
<section><button id=next>Advance replay +5 min</button> <span id=mode></span><p id=time></p></section><section><div class=label>Runtime / data proof</div><pre id=proof class=proof>—</pre></section></main>
<script>async function status(){let s=await fetch('/api/status').then(r=>r.json());document.querySelector('#mode').textContent='mode: '+s.mode;document.querySelector('#proof').textContent=JSON.stringify(s,null,2)}async function next(){let r=await fetch('/api/next').then(r=>r.json());for(let k of ['current','prediction','persistence'])document.querySelector('#'+k).textContent=Number(r[k]).toFixed(2)+' °C';document.querySelector('#observed').textContent=r.observed===null?'Hidden until target time':Number(r.observed).toFixed(2)+' °C';document.querySelector('#time').textContent='Issue '+r.issue_time_utc+' → target '+r.target_time_utc+' | '+(r.target_revealed?'target revealed':'target withheld');await status()}document.querySelector('#next').onclick=next;status()</script></html>"""


class DashboardState:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.predictor = PortablePredictor.load(root / "operational_bundle")
        self.replay = json.loads((root / "replay_samples.json").read_text(encoding="utf-8"))
        if not self.replay:
            raise FullRuntimeError("replay_samples.json is empty")
        self.metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
        self.manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        self.index = 0
        self.virtual_time: str | None = None
        self.lock = threading.Lock()

    @property
    def mode(self) -> str:
        return "fused" if self.predictor.provenance()["satellite_features"] else "station-only"

    def status(self) -> dict:
        provenance = self.predictor.provenance()
        return {
            "board": {"hostname": platform.node(), "kernel": platform.release(), "machine": platform.machine(), "python": platform.python_version()},
            "mode": self.mode,
            "model_hash": provenance["model_sha256"],
            "schema_hash": provenance["schema_sha256"],
            "satellite_feature_count": len(provenance["satellite_features"]),
            "satellite_gate": self.manifest.get("satellite_gate", {}).get("status"),
            "fusion_gate": self.manifest.get("fusion_gate", {}).get("pass"),
            "learned_gate": self.manifest.get("learned_gate", {}).get("pass"),
            "measured_board_latency": self.manifest.get("board_latency", "not measured in this bundle"),
            "claim_boundary": CLAIM_BOUNDARY,
        }

    def next(self) -> dict:
        with self.lock:
            row = self.replay[self.index % len(self.replay)]
            position = self.index
            self.virtual_time = row["issue_time_utc"]
            self.index += 1
        # Portable feature vectors are optional in replay; persistence fallback
        # remains deterministic without them. A learned bundle must ship them.
        current = float(row["current_at_degc"])
        persistence = float(row["persistence_degc"])
        if "features" in row:
            started = time.perf_counter_ns()
            prediction = float(self.predictor.predict(np.asarray(row["features"], dtype=float))[0])
            latency_ms = (time.perf_counter_ns() - started) / 1e6
        else:
            prediction, latency_ms = persistence, 0.0
        observed_row = self.replay[position - 6] if position >= 6 else None
        revealed = observed_row is not None
        return {"issue_time_utc": row["issue_time_utc"], "target_time_utc": row["target_time_utc"], "current": current, "prediction": prediction, "persistence": persistence, "observed": float(observed_row["observed_at_degc"]) if revealed else None, "observed_target_time_utc": observed_row["target_time_utc"] if revealed else None, "target_revealed": revealed, "inference_latency_ms": latency_ms, "station_missingness": row.get("station_missingness"), "satellite_frame_age_minutes": row.get("satellite_frame_age_minutes"), "satellite_lag_assumption_minutes": row.get("satellite_lag_assumption_minutes")}


def handler_for(state: DashboardState):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status); self.send_header("Content-Type", content_type); self.send_header("Content-Length", str(len(body))); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            try:
                if path in {"/", "/index.html"}: self.reply(200, "text/html; charset=utf-8", HTML.encode())
                elif path == "/healthz": self.reply(200, "application/json", b'{"ok":true}')
                elif path == "/api/status": self.reply(200, "application/json", json.dumps(state.status()).encode())
                elif path == "/api/next": self.reply(200, "application/json", json.dumps(state.next()).encode())
                else: self.reply(404, "application/json", b'{"error":"not found"}')
            except (FullRuntimeError, ValueError, KeyError) as exc:
                self.reply(503, "application/json", json.dumps({"error": str(exc)}).encode())

        def log_message(self, format: str, *args) -> None: pass
    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", required=True, type=Path); parser.add_argument("--host", default="0.0.0.0"); parser.add_argument("--port", type=int, default=8765); args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), handler_for(DashboardState(args.root)))
    print(json.dumps({"ready": True, "url": f"http://{args.host}:{args.port}"}), flush=True); server.serve_forever()


if __name__ == "__main__":
    main()
