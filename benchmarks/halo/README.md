# HALO feasibility probe — tooling

Scripts behind `docs/HALO_PHASE2_PROBE.md`. Read that document for what the
numbers mean and, more importantly, for what they do **not** support.

Nothing here trains a model or decides an architecture. These are exploratory
measurement tools whose historical outputs are reference-only. They require the V-gate
checks before any result can carry a design decision.

## Run order

```powershell
py -3 benchmarks/halo/pull_station_history.py        # ~2 min, 11 pages of 8000 rows
py -3 benchmarks/halo/describe_station.py            # coverage, ranges, firmware epochs
py -3 benchmarks/halo/detect_threshold_rule_occurrences.py  # rule occurrences + hour histogram
py -3 benchmarks/halo/himawari_geometry.py           # which segment holds the station
py -3 benchmarks/halo/fetch_satellite_day.py 20260809 2 14 30
py -3 benchmarks/halo/fetch_satellite_day.py 20260810 2 14 30
py -3 benchmarks/halo/export_result.py               # events.csv, result.json
py -3 benchmarks/halo/make_figures.py                # the three figures
```

`fetch_satellite_day.py` takes `YYYYMMDD start_hour_utc end_hour_utc step_minutes`
and is **resumable** — it skips timelines already in the output file, so an
interrupted run costs nothing. Fetch/decode failures are persisted separately as
`sat_<YYYYMMDD>_failures.json`; a missing statistic must not be inferred from stdout.

## Where things go

| Path | Contents | Tracked? |
|---|---|---|
| `benchmarks/halo/_cache/` | `saigon_full.csv` (~11 MB), pickled frames | **no** — gitignored |
| `evidence/halo-probe-2026-08-11/` | ring statistics, event table, figures, `result.json` | yes |

## Dependencies

`numpy`, `pandas`, `matplotlib` — all already present. **`satpy` is deliberately
not used**: `hsd.py` reads Himawari Standard Data directly, which avoids a heavy
dependency tree on Windows and keeps the calibration path auditable.

## Reading `hsd.py`

The reader takes constants from the file header, but that does **not** make it
self-validating. Plausible summary temperatures can coexist with byte-order,
calibration, navigation or pixel-level errors. Its `__main__` output is an internal
sanity check only. V-3 must compare the same file pixel-by-pixel with a reference reader.

## Known rough edges

- Pixel geometry is approximated as 2 km square. At 34° from the sub-satellite
  point the true footprint is larger and anisotropic, so ring radii in kilometres
  are approximate. Fix this before quoting a distance as a result.
- `fetch_satellite_day.py` reads segment 04 only and exports a nominal 25 km statistic.
  Larger radii are forbidden until segment stitching and geometry are validated.
- The 220 K deep-convection threshold is hard-coded in both
  `fetch_satellite_day.py` and `hsd.py`'s callers. It is a starting choice, not a
  validated one.
