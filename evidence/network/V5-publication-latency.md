# V-5 — publication latency, measured for the first time

The campaign has called this *"the single number that decides whether any inference-time
satellite use is possible at all"* and recorded it as **UNKNOWN, never measured**. It now
has a value, with its conditions recorded.

Script: `benchmarks/halo/measure_publication_latency.py`.
Artifact: `publication-latency-20260814T121147Z.json`.

## What was run

Three consecutive nominal 10-minute slots were watched. For each, the exact object the
project's region of interest needs — Himawari-9 AHI, band **B13**, segment **04**,
resolution R20 — was polled with HTTP HEAD every 20 s from the slot's nominal time until
it answered 200. Deadline 40 min; a slot that never appeared would have been recorded as
a timeout, not dropped.

## Result

| nominal slot (UTC) | published after | polls |
|---|---|---|
| 11:40 | **11.79 min** | 35 |
| 11:50 | **11.96 min** | 30 |
| 12:00 | **11.77 min** | 29 |

**3/3 slots. Median 11.79 min. Slowest 11.96 min.** Total spread across the three:
**11 seconds.**

## What this changes

Combined with the board measurements taken the same day, the full chain from scan to
usable brightness temperature on the board is:

| stage | time |
|---|---|
| nominal scan start → object downloadable | **11.8 – 12.0 min** |
| fetch + bz2 decompress on board over 4G | 15.4 s (0.26 min) |
| numpy decode | 0.36 s |
| **one segment, end to end** | **≈ 12.1 min** |
| two segments (the ROI spills into segment 05) | **≈ 12.3 min** |

Publication — not the network, not the CPU — is 97 % of that. The 4G link and the
Cortex-A53 are almost irrelevant to freshness, and nothing done to the board can improve
it.

### The number to plan against is not 12 minutes. It is 17, and at worst 22.

12 minutes is the **best case**: the age of an observation at the instant its file
appears. But files arrive every 10 minutes while each takes ~11.8 minutes to appear, so
between publications the newest available observation keeps ageing.

At 12:00 sharp, the 11:50 slot is not out yet (due 12:02.0). The newest file available is
11:40 — an observation **20 minutes old**.

With cadence *C* = 10 min and latency *L* = 11.8 min, the age of the freshest available
observation sweeps a sawtooth:

| | age of newest available observation |
|---|---|
| best case (a file has just appeared) | **L = 11.8 min** |
| mean | **L + C/2 ≈ 16.8 min** |
| worst case (just before the next file) | **L + C ≈ 21.8 min** |

Adding the board's own fetch and decode (≈31 s for the two segments the ROI needs) barely
moves it.

**So the board is always looking at a sky between roughly 12 and 22 minutes in the past,
about 17 minutes on average. It never sees the present.** A design that assumes
"near-real-time" satellite imagery does not survive this measurement. For a product
claiming 30 minutes of warning, ~17 of those minutes are already spent before any
inference runs.

## What is NOT controlled — read this before quoting the number

Applying the project's own standard, this is *one measurement with its conditions
recorded*, not a property of the product:

- **One session**, roughly 32 minutes of wall clock, on **2026-08-14**.
- **One time of day**: 11:40–12:12 UTC (18:40–19:12 ICT). Publication load may vary by hour.
- **One segment (04) and one band (B13).** AHI scans progressively, so other segments do
  **not** necessarily publish at the same delay. This number is for our segment only.
- **One product** (Himawari-9 via the NOAA S3 mirror) and **one network**. Latency should
  be provider-side, but that was not verified by measuring from a second network.
- **Three samples.** It says nothing about outages, day-to-day variation, or reliability.

**One correction in the favourable direction, unmeasured:** the nominal slot time is the
**start of the full-disk scan**, not the moment our pixel was observed. Segment 04 is
scanned some minutes into the sweep, so the true observation-to-availability delay for
our pixel is **less** than 11.8 min by that offset. How much less has not been measured.
The conservative figure — **11.96 min** — is the one to plan against.

## What it does not answer

It does not establish that satellite data carries useful signal at any horizon for this
site. Latency and usefulness are different questions; this measures only the first. The
retracted lead-time table and the unvalidated decoder both still stand between this number
and any claim about what the satellite can predict.
