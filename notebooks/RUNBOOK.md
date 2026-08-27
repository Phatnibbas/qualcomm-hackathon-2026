# HALO SafeShift — Colab run book

How to execute `HALO_SafeShift_Training.ipynb`, what it will produce, and what
to hand back.

---

## Before you start

**Upload archive** (already built, 11 members, 1,675,636 bytes):

```
<scratchpad>/colab-input-20260815T190622Z-99afc2c.zip
sha256 825c9733f2ea453c7f47e97668ebc28266de6191f73c51c15c44e6198845080a
```

Regenerate it any time from the admitted packet:

```powershell
python -c "import zipfile;from pathlib import Path;src=Path('evidence/halo-safeshift/20260815T190622Z-99afc2c/colab-input');z=zipfile.ZipFile('colab-input.zip','w',zipfile.ZIP_DEFLATED);[z.write(p,'colab-input/'+p.name) for p in sorted(src.iterdir()) if p.is_file()];z.close()"
```

## Steps

1. Open <https://colab.research.google.com> → **Upload** →
   `notebooks/HALO_SafeShift_Training.ipynb`.
2. **Runtime → Change runtime type → CPU.** A GPU is not needed and is not used
   (see *Deviation* below). If one is attached anyway it is recorded as an
   environment fact only.
3. **Runtime → Run all.**
4. At section 2 the file picker opens — select the upload ZIP. This is the
   **only** manual action.
5. Wait. Section 11 is the long cell: **≈ 1.5–2 hours** on a Colab CPU runtime.
   Keep the tab open; Colab reclaims idle runtimes.
6. When it finishes, **File → Save**, then re-run the final three cells so the
   executed notebook and its HTML are captured into the archive.
7. Download `HALO-SafeShift-Colab-<colab-run-id>.zip` (auto-download is
   attempted; otherwise use the file browser).

## Hand back

* the ZIP and its printed SHA-256;
* the final banner block (run id, baselines, gate verdict, operational model,
  parity, duration);
* any cell that stopped, verbatim.

Then place these where the deliverable list expects them:

```
notebooks/HALO_SafeShift_Training.executed.ipynb
notebooks/HALO_SafeShift_Training.executed.html
```

Both are inside the ZIP under `<colab-run-id>/notebook/`.

---

## Deviation from the brief: XGBoost runs on CPU, Extra Trees on one core

The scope override asks for `device="cuda"` XGBoost and all-cores Extra Trees.
**Neither is possible without modifying the admitted packet**, and packet
immutability wins:

* XGBoost hyper-parameters come from `models.trials` in the frozen
  `experiment.v1.json`. The four declared XGBoost trials set only
  `n_estimators`, `max_depth`, `learning_rate` and `subsample` — there is no
  `device` key. Adding one changes `halo_safeshift-source.zip`, which breaks
  `7c4ad03e…` and fails `verify_packet` before training starts.
* `build_estimator` applies `kwargs.setdefault("n_jobs", 1)` to Extra Trees, and
  the declared ET trials do not override it.
* `tree_method="hist"` and `objective="reg:squarederror"` are already the
  XGBoost defaults, so those two parts of the override are satisfied as-is.
  `random_state=20260815` is injected from `config.seed`, which matches.

The notebook therefore runs XGBoost on **CPU** and Extra Trees on **one core**,
and records `xgboost_device: cpu` as a fact rather than silently implying GPU
training happened.

If you want the GPU/multicore behaviour, it needs a **re-issued packet** with
those keys in the trial params — new run id, new hashes, and re-admission by the
orchestrator. That is a decision for you, not something to patch at runtime.

---

## What the notebook will not do

* score the prospective quarantine (12 bins, held);
* certify an untouched test (`certified_untouched_test = false`);
* touch ADB, Docker, firmware or the station;
* claim UNO Q inference, latency, RSS, offline operation, or any
  NPU/QNN/Hexagon/GPU acceleration.

## Stop conditions

The notebook halts rather than degrading a test. If you hit one, send the cell
output — do **not** loosen a tolerance, change a threshold, or add trials.

| Stop | Meaning |
|---|---|
| packet identity / verification failure | wrong or altered packet |
| pinned dependency not importable | install failed — a restart will not fix it |
| version mismatch after install | restart runtime, re-run **from section 3** |
| leakage sentinel failure | window geometry is not what the schema declares |
| trial count ≠ 26 / 30 | budget violated |
| empty fold | fold protocol changed |
| parity gate failure (exit 3) | no candidate reproduces in the neutral runtime |
| final refit ≠ all eligible windows | refit scope violated |
| ZIP fails its manifest | output is not trustworthy |

If XGBoost cannot be installed, set `ENABLE_XGBOOST = False` and re-run from
section 1. The 26-trial run is fully valid; XGBoost is opt-in and its absence is
recorded as `skipped_by_policy`.

---

## Expected shape of the result

From the local dry run (real data, real pipeline), persistence is a strong
baseline: **mean MAE 0.8415 °C** across the five folds, against climatology's
**1.7237 °C**. Single-fold probes had Gradient Boosting at 0.897 and Extra Trees
at 0.803 against persistence's 0.800 on the largest fold.

So `operational_model = persistence` is a realistic outcome, not a failure. The
fallback is designed: a learned model that cannot consistently beat *"assume it
stays the same"* has not earned a place on the board. If that happens, the best
learned challenger is still exported and labelled as a challenger, and the board
package still contains optimization candidates built from it.

## Pre-flight already done locally

* packet verified — `verified: true`, 10 declared files, 29 archive members;
* both admitted hashes re-computed and matched;
* target equation checked against an independent restatement (max diff `0.0`),
  RH-unit guard confirmed to reject a 0–1 fraction;
* sections 0–10 executed against the real packet;
* whole notebook executed with a reduced 2-trial substitution — figures,
  optimization candidates, `board_deploy/`, manifests and ZIP verification all
  ran and passed;
* tree-pruning path exercised separately on Extra Trees and Gradient Boosting
  bundles;
* all 59 code cells parse, and no cell uses a name an earlier cell has not
  defined.

None of the above is training evidence. It is proof the code runs.
