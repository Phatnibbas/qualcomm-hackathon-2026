#!/usr/bin/env python3
"""Baseline vs optimized inference, same model, same inputs, same machine."""
import json, math, statistics, time, platform, sys
from pathlib import Path
import edge_runner as base
import edge_runner_fast as fast

def bench(fn, rows, repeat):
    for i in range(min(50, len(rows))): fn(rows[i % len(rows)]["features"])  # warm
    t=[]
    for i in range(repeat):
        f=rows[i%len(rows)]["features"]
        t0=time.perf_counter_ns(); fn(f); t.append((time.perf_counter_ns()-t0)/1e6)
    o=sorted(t)
    return {"median":statistics.median(t),"p95":o[max(0,math.ceil(.95*len(o))-1)],"max":max(t)}

samples=json.loads(Path("replay_samples.json").read_text(encoding="utf-8"))
rows=samples["samples"] if isinstance(samples,dict) else samples
REPEAT=int(sys.argv[1]) if len(sys.argv)>1 else 3000
print("machine=%s python=%s rows=%d repeat=%d"%(platform.machine(),platform.python_version(),len(rows),REPEAT))
print()
out={"machine":platform.machine(),"hostname":platform.node(),"python":platform.python_version(),
     "kernel":platform.release(),"repeat":REPEAT,"samples":len(rows),"models":{}}
for name,path in (("persistence (shipped)","edge_model.json"),("challenger sparse-INT8 MLP","challenger_edge_model.json")):
    if not Path(path).exists(): print("SKIP %s (missing)"%path); continue
    m=json.loads(Path(path).read_text(encoding="utf-8"))
    c=fast.compile_model(m)
    pb=[base.predict(m,r["features"]) for r in rows]
    pf=[fast.predict(c,r["features"]) for r in rows]
    diff=max(abs(x-y) for x,y in zip(pb,pf))
    rb=bench(lambda f,m=m: base.predict(m,f), rows, REPEAT)
    rf=bench(lambda f,c=c: fast.predict(c,f), rows, REPEAT)
    sp=rb["median"]/rf["median"]
    nz=sum(len(o["indices"]) for L in m.get("layers",[]) for o in L["outputs"])
    print("### %s   (%s, %d nonzero weights)"%(name,m["model_type"],nz))
    print("   numerical identity  max|baseline - optimized| = %.3e  %s"%(diff,"IDENTICAL" if diff==0.0 else "DIFFERS"))
    print("   baseline   edge_runner.py       median %8.4f ms   p95 %8.4f ms"%(rb["median"],rb["p95"]))
    print("   optimized  edge_runner_fast.py  median %8.4f ms   p95 %8.4f ms"%(rf["median"],rf["p95"]))
    print("   speedup                         %.2fx   (%.1f%% latency removed)"%(sp,(1-rf["median"]/rb["median"])*100))
    print("   throughput  %.0f -> %.0f inferences/sec"%(1000/rb["median"],1000/rf["median"]))
    print()
    out["models"][name]={"model_type":m["model_type"],"nonzero_weights":nz,"max_abs_diff":diff,
                         "baseline_ms":rb,"optimized_ms":rf,"speedup":sp}
Path("inference_optimization.json").write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8")
print("wrote inference_optimization.json")
