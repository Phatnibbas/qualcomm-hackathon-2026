"""Cache a day of Himawari B13 statistics around the station.

Downloads one segment per timeline and keeps only a nominal 25 km statistic. Larger
radii are invalid with one segment because segment 04 ends inside the old 50 km mask.
The coordinate and 2 km/pixel distance are approximate, so this remains reference-only.
"""

import json
import sys
from pathlib import Path

import numpy as np

from hsd import fetch, read_bt

STATION_LINE, STATION_COL = 2178.6, 1059.7
SEGMENT, SEGMENT_FIRST_LINE = 4, 1651
KM_PER_PIXEL = 2.0
DEEP_K = 220.0
HERE = Path(__file__).parent / "_cache"
EVIDENCE = Path(__file__).resolve().parents[2] / "evidence" / "halo-probe-2026-08-11"

_distance_cache = {}


def distance_grid(shape):
    if shape not in _distance_cache:
        row = STATION_LINE - SEGMENT_FIRST_LINE
        yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
        _distance_cache[shape] = np.hypot(yy - row, xx - STATION_COL) * KM_PER_PIXEL
    return _distance_cache[shape]


def reduce_segment(bt):
    distance = distance_grid(bt.shape)
    valid = np.isfinite(bt) & (bt > 150)
    out = {}
    for radius in (25,):
        window = bt[(distance <= radius) & valid]
        if not window.size:
            out[str(radius)] = None
            continue
        out[str(radius)] = {
            "min": float(window.min()),
            "mean": float(window.mean()),
            "p10": float(np.percentile(window, 10)),
            "deep_pct": float((window < DEEP_K).mean() * 100),
        }
    return out


def main(day, start_utc, end_utc, step_minutes):
    target = EVIDENCE / f"sat_{day}.json"
    failure_target = EVIDENCE / f"sat_{day}_failures.json"
    cached = json.loads(target.read_text()) if target.exists() else {}
    failures = json.loads(failure_target.read_text()) if failure_target.exists() else {}
    # Older runs exported 50/100/150 km keys from a single segment even though those
    # masks cross the segment boundary. Preserve only the bounded 25 km statistic.
    cached = {
        stamp: {"25": rings.get("25")}
        for stamp, rings in cached.items()
        if isinstance(rings, dict) and rings.get("25") is not None
    }
    target.write_text(json.dumps(cached, indent=1), encoding="utf-8")

    minutes = range(start_utc * 60, end_utc * 60 + 1, step_minutes)
    for total in minutes:
        stamp = f"{total // 60:02d}{total % 60:02d}"
        if stamp in cached:
            continue
        try:
            bt, _ = read_bt(fetch("B13", SEGMENT, f"{day}_{stamp}"))
        except Exception as exc:  # noqa: BLE001
            failures[stamp] = {
                "error_type": type(exc).__name__,
                "message": str(exc),
                "boundary": "fetch/decode failure; no satellite statistic produced",
            }
            failure_target.write_text(json.dumps(failures, indent=1), encoding="utf-8")
            print(f"  {stamp} failed: {type(exc).__name__}")
            continue
        cached[stamp] = reduce_segment(bt)
        failures.pop(stamp, None)
        ring25 = cached[stamp].get("25")
        if ring25 is None:
            failures[stamp] = {
                "error_type": "NoValidPixels",
                "message": "nominal-25-km mask contains no valid brightness-temperature pixels",
                "boundary": "decode completed; no satellite statistic produced",
            }
            print(f"  {stamp} ok  nominal-25-km mask has no valid pixels")
        else:
            print(f"  {stamp} ok  r25 min={ring25['min']:.1f}K "
                  f"deep={ring25['deep_pct']:.0f}%")
        target.write_text(json.dumps(cached, indent=1), encoding="utf-8")
        failure_target.write_text(json.dumps(failures, indent=1), encoding="utf-8")

    print(f"{day}: {len(cached)} timelines -> {target}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
