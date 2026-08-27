#!/usr/bin/env python3
"""HALO SafeShift live service for the Arduino UNO Q.

Standalone: plug in power, join the Wi-Fi, open the page. No ADB, no laptop, no
cloud inference. The board pulls live station telemetry from the public ThingSpeak
channel, rebuilds the exact 125-feature vector the models were trained on, and runs
every deployable model on this board.

Stdlib only -- the UNO Q has no numpy, pandas or requests.

Feature contract (must match training exactly, or the prediction is meaningless):
  12 consecutive 5-minute bins, t-55m .. t-0m, each bin the MEDIAN of its raw
  samples and requiring at least 6 raw samples, each contributing 10 values in
  this order:
      temperature, humidity, wind_speed, wind_direction_sin, wind_direction_cos,
      light_log1p, pressure, pm25_log1p, noise, at
  then 5 extras: at_now, local_time_sin, local_time_cos, day_of_year_sin,
  day_of_year_cos.
Bins use label="right", closed="right", i.e. bin (t-5m, t] is labelled t.

If fewer than 12 consecutive complete bins are available, the service reports
NOT READY and refuses to predict. It never extrapolates or back-fills.

Claim boundary: station-derived shade apparent-temperature estimate at t+30 min.
Not WBGT, not a medical or legal safety limit, not direct-sun exposure.
Retrospective models applied live; this is not a certified forecast.
No NPU/QNN/Hexagon/GPU acceleration is used or claimed.
"""
from __future__ import annotations

import argparse
import json
import math
import threading
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import edge_runner_multi as RT

CHANNEL = 3428136
FIELD_MAP = {"field1": "wind_speed", "field2": "wind_direction", "field3": "temperature",
             "field4": "pressure", "field5": "light", "field6": "humidity",
             "field7": "noise", "field8": "pm25"}
PER_STEP = ["temperature", "humidity", "wind_speed", "wind_direction_sin", "wind_direction_cos",
            "light_log1p", "pressure", "pm25_log1p", "noise", "at"]
ICT = timezone(timedelta(hours=7))
BIN_S = 300
N_BINS = 12
MIN_RAW = 6
HORIZON_MIN = 30
CLAIM = ("Shade apparent-temperature estimate from site temperature, humidity and wind. "
         "Not WBGT, not a medical or legal safety limit, not direct-sun exposure. "
         "Retrospective models applied live; not a certified forecast. "
         "CPU Python only -- no NPU, QNN, Hexagon or GPU.")


def qc_ok(r):
    return (0.01 <= r["temperature"] <= 60.0 and 1.01 <= r["humidity"] <= 100.0
            and 0.0 <= r["wind_speed"] <= 40.0 and 80.0 <= r["pressure"] <= 120.0
            and 0.0 <= r["wind_direction"] < 360.0)


def apparent_temperature(t, rh, wind):
    e = (rh / 100.0) * 6.105 * math.exp(17.27 * t / (237.7 + t))
    return t + 0.33 * e - 0.70 * wind - 4.0


def _at_equation_block(ta, rh, ws, at_now):
    """Describe the apparent-temperature equation so the UI can render it.

    Everything the UI needs to colour-code lives here rather than in JavaScript:
    which symbols are measured this minute, which are fixed by the published
    equation, what each one means, and its unit. A judge should be able to redo
    the arithmetic from the screen alone.
    """
    e = (rh / 100.0) * 6.105 * math.exp(17.27 * ta / (237.7 + ta))
    return {
        "source": "Australian Bureau of Meteorology, non-radiation apparent temperature",
        "source_url": "https://www.bom.gov.au/info/thermal_stress/",
        "equations": {
            "vapour_pressure": "e = RH/100 * 6.105 * exp(17.27*Ta/(237.7+Ta))",
            "apparent_temperature": "AT = Ta + 0.33*e - 0.70*ws - 4.00",
        },
        "equation_status": "fixed; published equation, not fitted to our data",
        "symbols": [
            {"symbol": "Ta", "meaning": "Air temperature (dry bulb)", "unit": "degC",
             "kind": "measured", "source": "SEN0658 station", "value": round(ta, 2)},
            {"symbol": "RH", "meaning": "Relative humidity", "unit": "% RH",
             "kind": "measured", "source": "SEN0658 station", "value": round(rh, 2)},
            {"symbol": "ws", "meaning": "Wind speed", "unit": "m/s",
             "kind": "measured", "source": "SEN0658 station", "value": round(ws, 2)},
            {"symbol": "e", "meaning": "Water-vapour pressure", "unit": "hPa",
             "kind": "computed", "source": "from Ta and RH", "value": round(e, 3)},
            {"symbol": "AT", "meaning": "Apparent temperature (shade estimate)", "unit": "degC",
             "kind": "computed", "source": "the number this product reports", "value": round(at_now, 2)},
        ],
        # Units matter here: 0.33 is degC per hPa and 0.70 is degC per m/s, which
        # is why the equation only balances with RH in percent and wind in m/s.
        "constants": [
            {"symbol": "6.105", "meaning": "Magnus reference vapour pressure", "unit": "hPa"},
            {"symbol": "17.27", "meaning": "Magnus coefficient", "unit": "dimensionless"},
            {"symbol": "237.7", "meaning": "Magnus coefficient", "unit": "degC"},
            {"symbol": "0.33", "meaning": "Vapour-pressure weight", "unit": "degC per hPa"},
            {"symbol": "0.70", "meaning": "Wind-cooling weight", "unit": "degC per m/s"},
            {"symbol": "4.00", "meaning": "Offset", "unit": "degC"},
        ],
        "wind_height_note": ("Station wind is used as observed (~15 m AGL); the BoM equation "
                             "references 10 m. No height correction is applied, because no "
                             "site-validated roughness length exists."),
    }


def median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def fetch(results=260, timeout=20):
    url = "https://api.thingspeak.com/channels/%d/feeds.json?results=%d" % (CHANNEL, results)
    with urllib.request.urlopen(url, timeout=timeout) as fh:
        return json.loads(fh.read().decode("utf-8"))


def to_bins(feeds, now_epoch=None):
    """Group raw samples into right-labelled 5-minute bins of medians.

    Only CLOSED bins are returned. A bin labelled t covers (t-5m, t], so it is
    not complete until the wall clock passes t. Using the still-filling bin would
    feed the models a median over a partial window, which is not what training
    saw -- and it makes the newest bin label sit in the future.
    """
    if now_epoch is None:
        now_epoch = int(datetime.now(timezone.utc).timestamp())
    buckets = {}
    newest_raw = [0]
    for f in feeds:
        try:
            ts = datetime.strptime(f["created_at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            row = {}
            for fld, name in FIELD_MAP.items():
                v = f.get(fld)
                if v is None or v == "":
                    raise ValueError("missing " + fld)
                row[name] = float(v)
        except (ValueError, TypeError, KeyError):
            continue
        if not qc_ok(row):
            continue
        epoch = int(ts.timestamp())
        label = ((epoch + BIN_S - 1) // BIN_S) * BIN_S      # closed="right", label="right"
        buckets.setdefault(label, []).append(row)
        newest_raw[0] = max(newest_raw[0], epoch)

    bins = {}
    for label, rows in buckets.items():
        if len(rows) < MIN_RAW or label > now_epoch:        # skip the still-filling bin
            continue
        b = {name: median([r[name] for r in rows]) for name in FIELD_MAP.values()}
        b["raw_count"] = len(rows)
        b["at"] = apparent_temperature(b["temperature"], b["humidity"], b["wind_speed"])
        rad = math.radians(b["wind_direction"])
        b["wind_direction_sin"] = math.sin(rad)
        b["wind_direction_cos"] = math.cos(rad)
        b["light_log1p"] = math.log1p(max(b["light"], 0.0))
        b["pm25_log1p"] = math.log1p(max(b["pm25"], 0.0))
        bins[label] = b
    return bins, newest_raw[0]


def build_features(bins):
    """Return (features, issue_epoch, diagnostics) or (None, None, diagnostics)."""
    if not bins:
        return None, None, {"reason": "no complete 5-minute bin yet", "complete_bins": 0}
    newest = max(bins)
    want = [newest - BIN_S * i for i in range(N_BINS - 1, -1, -1)]   # t-55m .. t-0m
    missing = [w for w in want if w not in bins]
    if missing:
        return None, None, {
            "reason": "need %d consecutive complete bins ending at the newest one" % N_BINS,
            "complete_bins": len(bins), "missing_bins": len(missing),
            "missing_utc": [datetime.fromtimestamp(w, timezone.utc).isoformat() for w in missing[:6]]}
    feats = []
    for w in want:
        b = bins[w]
        feats.extend(b[k] for k in PER_STEP)
    loc = datetime.fromtimestamp(newest, ICT)
    minute = loc.hour * 60 + loc.minute
    feats.extend([bins[newest]["at"],
                  math.sin(2 * math.pi * minute / 1440), math.cos(2 * math.pi * minute / 1440),
                  math.sin(2 * math.pi * loc.timetuple().tm_yday / 366),
                  math.cos(2 * math.pi * loc.timetuple().tm_yday / 366)])
    diag = {"complete_bins": len(bins),
            "window_start_utc": datetime.fromtimestamp(want[0], timezone.utc).isoformat(),
            "raw_counts": [bins[w]["raw_count"] for w in want]}
    return feats, newest, diag


class State:
    def __init__(self, catalog, models, refresh):
        self.catalog = catalog
        self.models = models          # [(meta, compiled)]
        self.refresh = refresh
        self.lock = threading.Lock()
        self.snapshot = {"status": "starting", "claim_boundary": CLAIM}
        self.stop = threading.Event()

    def poll_once(self):
        started = time.time()
        try:
            data = fetch()
        except Exception as exc:                              # network is the expected failure
            with self.lock:
                self.snapshot = {"status": "upstream_error", "error": repr(exc),
                                 "fetched_utc": datetime.now(timezone.utc).isoformat(),
                                 "claim_boundary": CLAIM}
            return
        feeds = data.get("feeds", [])
        now = datetime.now(timezone.utc)
        bins, newest_raw = to_bins(feeds, int(now.timestamp()))
        feats, issue, diag = build_features(bins)
        # Age is measured from the newest RAW sample, not from the bin label: the
        # label is the right edge of a closed window, so it trails the sample.
        raw_age = None if not newest_raw else round(now.timestamp() - newest_raw, 1)
        snap = {"fetched_utc": now.isoformat(), "channel": CHANNEL,
                "channel_name": data.get("channel", {}).get("name"),
                "raw_samples": len(feeds), "diagnostics": diag,
                "data_age_seconds": raw_age,
                "fetch_seconds": round(time.time() - started, 3),
                "claim_boundary": CLAIM,
                "persistence_test_mae_degc": self.catalog.get("persistence_test_mae_degc"),
                "selection_warning": self.catalog.get("selection_warning"),
                "horizon_minutes": HORIZON_MIN,
                "acceleration_claim": "none; CPU Python only. No NPU/QNN/Hexagon/GPU."}
        if feats is None:
            snap["status"] = "not_ready"
            with self.lock:
                self.snapshot = snap
            return

        issue_dt = datetime.fromtimestamp(issue, timezone.utc)
        at_now = feats[-5]      # extras are at_now, time sin/cos, day-of-year sin/cos
        # The newest closed bin is step 11 of 12, so its block starts at 11*len(PER_STEP).
        # Exposed so the UI can show the BoM equation with this run's own numbers
        # rather than a static picture of a formula.
        newest_block = (N_BINS - 1) * len(PER_STEP)
        ta = feats[newest_block + PER_STEP.index("temperature")]
        rh = feats[newest_block + PER_STEP.index("humidity")]
        ws = feats[newest_block + PER_STEP.index("wind_speed")]
        preds = []
        t_all = time.perf_counter()
        for meta, compiled in self.models:
            t0 = time.perf_counter_ns()
            try:
                value = RT.predict(compiled, feats)
                err = None
            except Exception as exc:
                value, err = None, repr(exc)
            latency = (time.perf_counter_ns() - t0) / 1e6
            preds.append({**meta,
                          "predicted_at_plus30_degc": None if value is None else round(value, 3),
                          "delta_vs_now_degc": None if value is None else round(value - at_now, 3),
                          "inference_latency_ms": round(latency, 4),
                          "error": err})
        snap.update({
            "status": "ok",
            "issue_time_utc": issue_dt.isoformat(),
            "issue_time_ict": datetime.fromtimestamp(issue, ICT).strftime("%Y-%m-%d %H:%M:%S"),
            "target_time_ict": datetime.fromtimestamp(issue + HORIZON_MIN * 60, ICT).strftime("%Y-%m-%d %H:%M:%S"),
            "bin_close_lag_seconds": round((now - issue_dt).total_seconds(), 1),
            "current_at_degc": round(at_now, 3),
            "at_equation": _at_equation_block(ta, rh, ws, at_now),
            "models_run": len(preds),
            "all_models_latency_ms": round((time.perf_counter() - t_all) * 1000, 2),
            "blocked_models": self.catalog.get("blocked", []),
            "predictions": preds})
        with self.lock:
            self.snapshot = snap

    def run(self):
        while not self.stop.is_set():
            self.poll_once()
            self.stop.wait(self.refresh)

    def get(self):
        with self.lock:
            return dict(self.snapshot)


HTML = r"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>HALO SafeShift - live on Arduino UNO Q</title>
<style>
*{box-sizing:border-box}
body{margin:0;background:#eef2f1;color:#16211d;font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif}
header{background:linear-gradient(135deg,#0f3830,#1b5c4c);color:#fff;padding:18px 22px}
header h1{margin:0;font-size:22px;letter-spacing:.2px}
header .sub{opacity:.85;font-size:13px;margin-top:3px}
header .live{display:inline-block;width:9px;height:9px;border-radius:50%;background:#4ade80;margin-right:6px;animation:p 1.6s infinite}
@keyframes p{0%,100%{opacity:1}50%{opacity:.25}}
main{max-width:1180px;margin:auto;padding:18px}
.note{border-left:4px solid #d6632d;background:#fff6ec;padding:10px 13px;margin-bottom:14px;font-size:13px;border-radius:0 4px 4px 0}
.warn{border-left-color:#b4472e;background:#fdefec}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin-bottom:16px}
.card{background:#fff;border:1px solid #dbe5e1;border-radius:6px;padding:12px 14px}
.k{font:600 10.5px ui-monospace,monospace;text-transform:uppercase;color:#5c7268;letter-spacing:.07em}
.v{font-size:26px;font-weight:700;margin-top:4px;line-height:1.15}
.v small{font-size:14px;font-weight:400;color:#5c7268}
.ok{color:#11795a}.bad{color:#a8322d}
.bar{display:flex;gap:8px;align-items:center;margin:14px 0 8px}
.bar h2{margin:0;font-size:15px}
.bar input{flex:1;padding:7px 10px;border:1px solid #cfdad5;border-radius:5px;font-size:13px;min-width:120px}
.tablewrap{overflow-x:auto;background:#fff;border:1px solid #dbe5e1;border-radius:6px}
table{width:100%;border-collapse:collapse;font-size:13px;min-width:820px}
th{background:#f4f8f6;font:600 10.5px ui-monospace,monospace;text-transform:uppercase;color:#5c7268;
   padding:9px 11px;text-align:left;border-bottom:1px solid #dbe5e1;white-space:nowrap;cursor:pointer;user-select:none}
th.n,td.n{text-align:right}
td{padding:8px 11px;border-bottom:1px solid #f0f4f2;white-space:nowrap}
tr.grp td{background:#f7faf9;font:600 10.5px ui-monospace,monospace;text-transform:uppercase;
          color:#5c7268;letter-spacing:.07em}
tr.op{background:#eefaf4}
tr.op td:first-child::after{content:"OPERATIONAL";margin-left:8px;font:600 9px ui-monospace,monospace;
  background:#11795a;color:#fff;padding:2px 5px;border-radius:3px;vertical-align:middle}
tr.blk{opacity:.55}
tr.blk td:first-child::after{content:"BLOCKED";margin-left:8px;font:600 9px ui-monospace,monospace;
  background:#a8322d;color:#fff;padding:2px 5px;border-radius:3px;vertical-align:middle}
td.mono{font-family:ui-monospace,monospace}
.var{color:#5c7268;font-size:12px}
.foot{color:#5c7268;font-size:12.5px;margin-top:12px}
.eq{background:#fff;border:1px solid #dbe5e1;border-left:4px solid #1b5c4c;border-radius:0 6px 6px 0;
    padding:11px 14px;margin-bottom:14px}
.eqline{font-family:ui-monospace,monospace;font-size:15px;margin-top:7px;letter-spacing:.01em}
.eqwhy{font-size:12.5px;color:#3d5249;margin-top:5px}
.eq .op{color:#8a9c94}
.mea{color:#0b6b50;font-weight:700;background:#e4f7ef;padding:1px 4px;border-radius:3px}
.con{color:#9a5312;font-weight:700;background:#fdf0e2;padding:1px 4px;border-radius:3px}
.cmp{color:#2d4f9e;font-weight:700;background:#e8eefb;padding:1px 4px;border-radius:3px}
.legend{margin-top:10px;display:flex;gap:8px;flex-wrap:wrap}
.chip{font:600 10px ui-monospace,monospace;text-transform:uppercase;letter-spacing:.06em;
      padding:3px 7px;border-radius:3px}
.eqtabwrap{overflow-x:auto;margin-top:10px}
table.eqtab{width:100%;border-collapse:collapse;font-size:12.5px;min-width:560px}
table.eqtab th{background:#f4f8f6;font:600 10px ui-monospace,monospace;text-transform:uppercase;
   color:#5c7268;letter-spacing:.06em;padding:6px 9px;text-align:left;border-bottom:1px solid #dbe5e1}
table.eqtab td{padding:5px 9px;border-bottom:1px solid #f0f4f2}
table.eqtab td.s{font-family:ui-monospace,monospace;font-weight:700}
table.eqtab td.n{text-align:right;font-family:ui-monospace,monospace}
table.eqtab td.u{font-family:ui-monospace,monospace;color:#5c7268}
.eqsub{font-family:ui-monospace,monospace;font-size:12.5px;margin-top:9px;color:#16211d}
.eqsub.dim{color:#5c7268;font-size:11.5px;font-family:inherit;margin-top:5px}
details{margin-top:14px}summary{cursor:pointer;font:600 11px ui-monospace,monospace;
  text-transform:uppercase;color:#5c7268;letter-spacing:.07em;padding:6px 0}
pre{background:#fff;border:1px solid #dbe5e1;border-radius:6px;padding:11px;overflow-x:auto;font-size:11.5px;max-height:340px}
</style>
<header>
  <h1><span class="live"></span>HALO SafeShift</h1>
  <div class="sub">Live inference on Arduino UNO Q &middot; station telemetry from ThingSpeak &middot; no laptop in the loop</div>
</header>
<main>
<p class="note" id="claim">&nbsp;</p>
<div class="eq" id="eqbox">
  <div class="k">Apparent temperature &mdash; exactly how this number is computed</div>
  <div class="eqwhy">The equation below is <b>fixed</b>: it is the published Bureau of Meteorology
    non-radiation apparent temperature, not something fitted to our data. Only the
    <span class="mea">measured</span> symbols change.</div>
  <div class="eqline"><span class="cmp">e</span> <span class="op">=</span> <span class="mea">RH</span><span class="op">/</span><span class="con">100</span>
    <span class="op">&times;</span> <span class="con">6.105</span> <span class="op">&times; exp(</span><span class="con">17.27</span><span class="op">&middot;</span><span class="mea">Ta</span>
    <span class="op">/ (</span><span class="con">237.7</span> <span class="op">+</span> <span class="mea">Ta</span><span class="op">))</span></div>
  <div class="eqline"><span class="cmp">AT</span> <span class="op">=</span> <span class="mea">Ta</span> <span class="op">+</span> <span class="con">0.33</span><span class="op">&middot;</span><span class="cmp">e</span>
    <span class="op">&minus;</span> <span class="con">0.70</span><span class="op">&middot;</span><span class="mea">ws</span> <span class="op">&minus;</span> <span class="con">4.00</span></div>
  <div class="legend">
    <span class="chip mea">measured now</span>
    <span class="chip con">fixed constant</span>
    <span class="chip cmp">computed</span>
  </div>
  <div class="eqtabwrap"><table class="eqtab">
    <thead><tr><th>Symbol</th><th>What it is</th><th>Unit</th><th>Kind</th><th class="n">Value now</th></tr></thead>
    <tbody id="eqrows"><tr><td colspan="5">loading...</td></tr></tbody>
  </table></div>
  <div class="eqsub" id="eqnow">&nbsp;</div>
  <div class="eqsub dim" id="eqnote">&nbsp;</div>
</div>
<div class="grid">
  <div class="card"><div class="k">Status</div><div class="v" id="st">-</div></div>
  <div class="card"><div class="k">Current AT</div><div class="v"><span id="cur">-</span><small> &deg;C</small></div></div>
  <div class="card"><div class="k">Data age</div><div class="v"><span id="age">-</span><small> s</small></div></div>
  <div class="card"><div class="k">Models on board</div><div class="v" id="nm">-</div></div>
  <div class="card"><div class="k">All models</div><div class="v"><span id="tot">-</span><small> ms</small></div></div>
</div>
<p class="note warn" id="warn">&nbsp;</p>
<div class="bar">
  <h2>Models running on this board</h2>
  <input id="q" placeholder="filter: huber, tree, prune 90, int16 ...">
</div>
<div class="tablewrap"><table>
<thead><tr>
  <th data-s="family">Model</th><th data-s="variant">Variant</th>
  <th class="n" data-s="test_mae_degc">Test MAE &deg;C</th>
  <th class="n" data-s="vs_persistence_degc">vs persist</th>
  <th class="n" data-s="predicted_at_plus30_degc">+30 min &deg;C</th>
  <th class="n" data-s="delta_vs_now_degc">&Delta; now</th>
  <th class="n" data-s="inference_latency_ms">Latency ms</th>
  <th class="n" data-s="bytes">Size KB</th>
</tr></thead><tbody id="rows"><tr><td colspan="8">loading...</td></tr></tbody></table></div>
<p class="foot" id="times">&nbsp;</p>
<details><summary>Raw service state (JSON)</summary><pre id="raw">loading...</pre></details>
</main>
<script>
let SNAP=null, SORT="group", DIR=1;
const f=(v,d)=>v==null?"-":Number(v).toFixed(d);
function render(){
 if(!SNAP)return;
 const s=SNAP;
 document.getElementById('claim').textContent=s.claim_boundary||'';
 const eq=s.at_equation, U=t=>String(t).replace(/degC/g,'°C');
 if(eq&&eq.symbols){
   const kindClass={measured:'mea',computed:'cmp',constant:'con'};
   let rows='';
   for(const y of eq.symbols)
     rows+='<tr><td class="s"><span class="'+(kindClass[y.kind]||'')+'">'+y.symbol+'</span></td><td>'+y.meaning
          +'</td><td class="u">'+U(y.unit)+'</td><td>'+y.kind+' &middot; '+y.source
          +'</td><td class="n">'+y.value+'</td></tr>';
   for(const c of eq.constants||[])
     rows+='<tr><td class="s"><span class="con">'+c.symbol+'</span></td><td>'+c.meaning
          +'</td><td class="u">'+U(c.unit)+'</td><td>constant &middot; fixed by BoM</td><td class="n">'+c.symbol+'</td></tr>';
   document.getElementById('eqrows').innerHTML=rows;
   const g=k=>{const y=(eq.symbols||[]).find(z=>z.symbol===k);return y?y.value:null};
   document.getElementById('eqnow').innerHTML=
     'substituted now: &nbsp;<span class="cmp">e</span>='+g('e')+' hPa &nbsp;&rarr;&nbsp; <span class="mea">'+g('Ta')
     +'</span> + <span class="con">0.33</span>&middot;<span class="cmp">'+g('e')+'</span> &minus; <span class="con">0.70</span>&middot;<span class="mea">'
     +g('ws')+'</span> &minus; <span class="con">4.00</span> = <b><span class="cmp">'+g('AT')+' &deg;C</span></b>';
   document.getElementById('eqnote').textContent=eq.wind_height_note||'';
 }else{
   document.getElementById('eqrows').innerHTML='<tr><td colspan="5">waiting for a complete station window</td></tr>';
   document.getElementById('eqnow').textContent='';
   document.getElementById('eqnote').textContent='';
 }
 document.getElementById('warn').textContent=s.selection_warning||'';
 const st=document.getElementById('st');
 st.textContent=s.status; st.className='v '+(s.status==='ok'?'ok':'bad');
 document.getElementById('cur').textContent=f(s.current_at_degc,2);
 document.getElementById('age').textContent=s.data_age_seconds??'-';
 document.getElementById('nm').textContent=s.models_run??'-';
 document.getElementById('tot').textContent=f(s.all_models_latency_ms,1);
 document.getElementById('times').textContent = s.status==='ok'
   ? ('Issue '+s.issue_time_ict+' ICT  →  target '+s.target_time_ict+' ICT   ('+s.horizon_minutes+' min horizon, '+s.raw_samples+' raw samples)')
   : ((s.diagnostics&&s.diagnostics.reason)||s.error||('fetched '+(s.fetched_utc||'-')));
 document.getElementById('raw').textContent=JSON.stringify(s,null,2);

 const tb=document.getElementById('rows');
 let p=(s.predictions||[]).slice();
 if(!p.length){tb.innerHTML='<tr><td colspan=8>'+((s.diagnostics&&s.diagnostics.reason)||s.error||'waiting for data')+'</td></tr>';return;}
 const q=document.getElementById('q').value.trim().toLowerCase();
 if(q)p=p.filter(r=>(r.family+' '+r.variant+' '+r.group).toLowerCase().includes(q));
 if(SORT==='group'){p.sort((a,b)=>a.group.localeCompare(b.group)||(a.test_mae_degc??9)-(b.test_mae_degc??9));}
 else{p.sort((a,b)=>{const x=a[SORT],y=b[SORT];
   if(x==null)return 1; if(y==null)return -1;
   return (typeof x==='string'?x.localeCompare(y):x-y)*DIR;});}
 let html='',last=null;
 for(const r of p){
  if(SORT==='group'&&r.group!==last){last=r.group;html+='<tr class=grp><td colspan=8>'+r.group+'</td></tr>';}
  const cls=r.family==='persistence'?'op':(r.status!=='deployable'?'blk':'');
  html+='<tr class="'+cls+'"><td>'+r.family+'</td><td class=var>'+r.variant+'</td>'
   +'<td class="n mono">'+f(r.test_mae_degc,4)+'</td>'
   +'<td class="n mono '+((r.vs_persistence_degc??0)<0?'ok':'')+'">'+(r.vs_persistence_degc==null?'-':(r.vs_persistence_degc>0?'+':'')+f(r.vs_persistence_degc,4))+'</td>'
   +'<td class="n mono">'+(r.error?'ERR':f(r.predicted_at_plus30_degc,2))+'</td>'
   +'<td class="n mono">'+(r.delta_vs_now_degc==null?'-':(r.delta_vs_now_degc>0?'+':'')+f(r.delta_vs_now_degc,2))+'</td>'
   +'<td class="n mono">'+f(r.inference_latency_ms,3)+'</td>'
   +'<td class="n mono">'+f(r.bytes/1024,1)+'</td></tr>';
 }
 tb.innerHTML=html;
}
document.querySelectorAll('th[data-s]').forEach(th=>th.onclick=()=>{
 const k=th.dataset.s; DIR=(SORT===k)?-DIR:1; SORT=k; render();});
document.getElementById('q').oninput=render;
async function tick(){
 try{SNAP=await (await fetch('/api/state',{cache:'no-store'})).json();render();}
 catch(e){document.getElementById('st').textContent='page error';}
}
tick(); setInterval(tick,5000);
</script></html>"""


def handler_for(state):
    class H(BaseHTTPRequestHandler):
        def reply(self, code, ctype, body):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                self.reply(200, "text/html; charset=utf-8", HTML.encode())
            elif path == "/api/state":
                self.reply(200, "application/json", json.dumps(state.get()).encode())
            elif path == "/api/catalog":
                self.reply(200, "application/json", json.dumps(state.catalog).encode())
            elif path == "/healthz":
                ok = state.get().get("status") in ("ok", "not_ready")
                self.reply(200 if ok else 503, "application/json", json.dumps({"ok": ok}).encode())
            else:
                self.reply(404, "application/json", b'{"error":"not found"}')

        def log_message(self, *args):
            pass
    return H


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", default="model_catalog.json")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--refresh", type=int, default=60)
    a = p.parse_args()

    cat = json.loads(Path(a.catalog).read_text(encoding="utf-8"))
    models, blocked, failed = [], [], []
    for row in cat["models"]:
        if row.get("status") != "deployable":
            blocked.append({k: row[k] for k in ("family", "variant", "note", "status")})
            continue
        try:
            m = json.loads(Path(row["file"]).read_text(encoding="utf-8"))
            compiled = RT.compile_model(m)
        except Exception as exc:
            failed.append({"file": row["file"], "error": repr(exc)})
            continue
        meta = {k: row.get(k) for k in
                ("family", "group", "variant", "test_mae_degc", "vs_persistence_degc",
                 "bytes", "params", "status", "note")}
        meta["model_type"] = m["model_type"]
        meta["file"] = row["file"]
        models.append((meta, compiled))
    cat["blocked"] = blocked
    print("loaded %d models, %d blocked, %d failed to load" % (len(models), len(blocked), len(failed)), flush=True)
    for x in failed:
        print("  LOAD FAILED %s: %s" % (x["file"], x["error"]), flush=True)

    state = State(cat, models, a.refresh)
    threading.Thread(target=state.run, daemon=True).start()
    srv = ThreadingHTTPServer((a.host, a.port), handler_for(state))
    print(json.dumps({"ready": True, "port": a.port, "refresh_seconds": a.refresh,
                      "models": len(models), "blocked": len(blocked)}), flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
