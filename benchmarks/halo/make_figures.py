"""Exploratory figures for the HALO feasibility probe.

Three figures, each with one job:
  fig1  the co-timing story - satellite cooling above, station response below,
        on one shared time axis (small multiples; never a dual axis)
  fig2  event day against control day, same measure, one scale
  fig3  when the threshold-rule occurrences happen (descriptive only)

Palette is the validated dataviz reference instance (blue/orange categorical
pair: ALL CHECKS PASS, light surface #fcfcfb).
"""

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.dates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent / "_cache"
EVIDENCE = Path(__file__).resolve().parents[2] / "evidence" / "halo-probe-2026-08-11"
ICT = "Asia/Ho_Chi_Minh"

# --- palette (dataviz reference instance) ---------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES1 = "#2a78d6"   # blue, categorical slot 1
SERIES2 = "#eb6834"   # orange, categorical slot 2
CRITICAL = "#d03b3b"  # status: the rule-occurrence marker
RAMP = {25: "#0d366b"}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
    "font.size": 9,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": BASELINE,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK2,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "grid.linestyle": "-",          # solid hairline, never dashed
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.frameon": False,
    "savefig.facecolor": SURFACE,
    "savefig.bbox": "tight",
})

station = pd.read_pickle(HERE / "saigon.pkl")
station = station.loc[station["temp_c"] > 1].copy()
station["ict"] = station["ts"].dt.tz_convert(ICT)
station["gust"] = (
    station["wind_ms"] - station["wind_ms"].rolling(60, min_periods=20).median()
)

events = pd.read_pickle(HERE / "events.pkl")
events["ict"] = events["peak"].dt.tz_convert(ICT)


def satellite(day):
    raw = json.loads((EVIDENCE / f"sat_{day}.json").read_text())
    rows = []
    for stamp, rings in sorted(raw.items()):
        utc = pd.Timestamp(f"{day[:4]}-{day[4:6]}-{day[6:8]}T{stamp[:2]}:{stamp[2:]}:00Z")
        row = {"ict": utc.tz_convert(ICT)}
        for radius in (25,):
            entry = rings.get(str(radius))
            row[f"deep{radius}"] = entry["deep_pct"] if entry else np.nan
            row[f"min{radius}"] = entry["min"] if entry else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def hourfmt(axis):
    axis.xaxis.set_major_formatter(mpl.dates.DateFormatter("%H:%M", tz=ICT))
    axis.xaxis.set_major_locator(mpl.dates.HourLocator(interval=2))


def tidy(axis, label):
    axis.set_ylabel(label, fontsize=8.5)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_axisbelow(True)


# =========================================================================
# FIGURE 1 - the co-timing story
# =========================================================================
DAY = "20260809"
window = (
    pd.Timestamp("2026-08-09 09:00", tz=ICT),
    pd.Timestamp("2026-08-09 21:00", tz=ICT),
)
sat = satellite(DAY)
local = station[(station["ict"] >= window[0]) & (station["ict"] <= window[1])]
marks = events[(events["ict"] >= window[0]) & (events["ict"] <= window[1])]["ict"]

figure, axes = plt.subplots(
    5, 1, figsize=(8.2, 9.4), sharex=True,
    gridspec_kw={"height_ratios": [1.5, 1, 1, 1, 1], "hspace": 0.16},
)

top = axes[0]
for radius in (25,):
    top.plot(sat["ict"], sat[f"deep{radius}"], color=RAMP[radius], linewidth=2,
             marker="o", markersize=4, markeredgecolor=SURFACE,
             markeredgewidth=0.8, label=f"nominal {radius} km, segment 04 only")
top.set_ylim(-3, 118)
tidy(top, "B13 pixels below 220 K\n(nominal 25 km, S04 only)")
top.set_yticks([0, 20, 40, 60, 80, 100])
# Legend sits in the headroom above the data so it can never collide with
# either the traces or the vertical event labels.
top.legend(loc="upper left", bbox_to_anchor=(0, 1.005), fontsize=8,
           labelcolor=INK2, ncol=3, columnspacing=1.4, handlelength=1.6)
top.set_title(
    "B13 threshold statistic and local threshold-rule occurrences on one day\n"
    "2026-08-09, Ho Chi Minh City  ·  satellite band 13 (10.4 µm) vs ground station 3428136",
    fontsize=10.5, color=INK, loc="left", pad=10,
)

for axis, column, label, scale in (
    (axes[1], "temp_c", "air temperature\n(°C)", 1),
    (axes[2], "rh_pct", "relative humidity\n(%RH)", 1),
    (axes[3], "wind_ms", "wind speed\n(m/s)", 1),
    (axes[4], "lux", "illuminance\n(thousand lux)", 1e-3),
):
    axis.plot(local["ict"], local[column] * scale, color=SERIES1, linewidth=1.6)
    tidy(axis, label)

for axis in axes:
    for index, mark in enumerate(marks):
        axis.axvline(mark, color=CRITICAL, linewidth=1.2, alpha=0.85, zorder=0)
        if axis is top:
            top.annotate(
                f"rule hit {mark:%H:%M}", xy=(mark, 2), xytext=(4, 0),
                textcoords="offset points", color=CRITICAL, fontsize=8,
                rotation=90, va="bottom", ha="left",
            )

hourfmt(axes[-1])
axes[-1].set_xlabel("time of day (ICT)", fontsize=8.5, color=INK2)
figure.text(
    0.008, 0.005,
    "Red lines mark occurrences of a station threshold rule "
    "(air temperature falling ≥2 °C while humidity rises ≥8 %RH within 20 min).\n"
    "Satellite statistic uses segment 04 only and a nominal 25 km mask centred on an assumed coordinate; "
    "the decoder and 220 K threshold are unvalidated.",
    fontsize=7.2, color=MUTED, va="bottom",
)
for suffix in ("png", "pdf"):
    figure.savefig(EVIDENCE / f"fig1_cotiming.{suffix}", dpi=170)
plt.close(figure)

# =========================================================================
# FIGURE 2 - event day vs control day
# =========================================================================
figure, axis = plt.subplots(figsize=(7.4, 3.5))
for day, color, label in (
    ("20260809", SERIES1, "2026-08-09  (3 threshold-rule occurrences)"),
    ("20260810", SERIES2, "2026-08-10  (no occurrence detected)"),
):
    frame = satellite(day)
    hours = frame["ict"].dt.hour + frame["ict"].dt.minute / 60
    axis.plot(hours, frame["deep25"], color=color, linewidth=2, marker="o",
              markersize=4.5, markeredgecolor=SURFACE, markeredgewidth=0.8,
              label=label)

axis.set_xlim(8.7, 21.3)
axis.set_ylim(-3, 100)
axis.set_xticks(range(9, 22, 2))
axis.set_xticklabels([f"{h:02d}:00" for h in range(9, 22, 2)])
tidy(axis, "% of B13 pixels < 220 K\nnominal 25 km, S04 only")
axis.set_xlabel("time of day (ICT)", fontsize=8.5, color=INK2)
axis.legend(loc="upper right", fontsize=8.5, labelcolor=INK2)
axis.set_title(
    "Two-day comparison of a satellite statistic around the assumed station location\n"
    "Descriptive only (n=2 days); physical cause and predictive value are unknown",
    fontsize=10.5, color=INK, loc="left", pad=8,
)
for suffix in ("png", "pdf"):
    figure.savefig(EVIDENCE / f"fig2_contrast.{suffix}", dpi=170)
plt.close(figure)

# =========================================================================
# FIGURE 3 - local-hour distribution (descriptive only)
# =========================================================================
counts = events["ict"].dt.hour.value_counts().reindex(range(24), fill_value=0)

figure, axis = plt.subplots(figsize=(7.4, 3.2))
bars = axis.bar(counts.index, counts.values, width=0.72, color=SERIES1)
axis.set_xticks(range(0, 24, 2))
axis.set_xticklabels([f"{h:02d}" for h in range(0, 24, 2)])
axis.set_xlim(-0.8, 23.8)
axis.set_ylim(0, max(counts.values) + 1.6)
tidy(axis, "rule occurrences")
axis.set_xlabel("hour of day (ICT)", fontsize=8.5, color=INK2)
axis.grid(axis="x", visible=False)
axis.set_title(
    f"Local-hour distribution of {int(counts.sum())} threshold-rule occurrences\n"
    "The detector never sees a clock; this pattern does not identify the physical cause",
    fontsize=10.5, color=INK, loc="left", pad=8,
)
# Direct-label selectively: the peak only.
peak = int(counts.idxmax())
axis.annotate(
    f"{int(counts[peak])} occurrences", xy=(peak, counts[peak]), xytext=(0, 5),
    textcoords="offset points", ha="center", fontsize=8.5, color=INK2,
)
axis.axvspan(-0.8, 6.5, color=GRID, alpha=0.55, zorder=0)
axis.annotate(
    "00:00–06:30  no occurrences", xy=(2.8, max(counts.values) + 0.5),
    ha="center", fontsize=8, color=INK2,
)
for suffix in ("png", "pdf"):
    figure.savefig(EVIDENCE / f"fig3_hours.{suffix}", dpi=170)
plt.close(figure)

print("wrote fig1_cotiming, fig2_contrast, fig3_hours  (.png and .pdf)")
print(f"events in window for fig1: {list(marks.dt.strftime('%H:%M'))}")
