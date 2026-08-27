"""Time an anonymous HSD fetch over whatever network is currently attached.

This measures TRANSPORT ONLY: how long one real Himawari segment takes to arrive and
decompress on this connection. It deliberately uses a FIXED historical timestamp so a
missing file can never be confused with a slow link.

IT IS NOT V-5. V-5 is publication latency -- the wall-clock delay between an
observation and its object becoming listable. That question is untouched here, and a
fast link does not answer it.

A result is only meaningful with its conditions, so --network is required and is
written into the artifact. Do not average across different --network labels.

    py -3 benchmarks/halo/measure_network.py --network "4g-hotspot-at-home"
"""

import argparse
import bz2
import json
import platform
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

BASE = "https://noaa-himawari9.s3.amazonaws.com"
# Fixed and historical on purpose: isolates transport from publication.
STAMP, BAND, SEGMENT = "20260809_0700", "B13", 4
OUT = Path(__file__).resolve().parents[2] / "evidence" / "network"


def key_for(stamp, band, segment, resolution="R20"):
    day, hhmm = stamp[:8], stamp[9:]
    return (f"AHI-L1b-FLDK/{day[:4]}/{day[4:6]}/{day[6:8]}/{hhmm}/"
            f"HS_H09_{day}_{hhmm}_{band}_FLDK_{resolution}_S{segment:02d}10.DAT.bz2")


def time_one_fetch(url):
    """Return timings in seconds plus the byte counts, or an error dict."""
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(url, timeout=180) as response:
            opened = time.perf_counter()
            payload = response.read()
            downloaded = time.perf_counter()
        raw = bz2.decompress(payload)
        decompressed = time.perf_counter()
    except Exception as exc:                      # noqa: BLE001 - the failure IS data
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                "seconds_to_failure": round(time.perf_counter() - started, 3)}

    return {
        "ok": True,
        "connect_and_first_response_s": round(opened - started, 3),
        "download_s": round(downloaded - opened, 3),
        "decompress_s": round(decompressed - downloaded, 3),
        "total_s": round(decompressed - started, 3),
        "compressed_bytes": len(payload),
        "decompressed_bytes": len(raw),
        "download_MiB_per_s": round(len(payload) / (1024 * 1024) / max(downloaded - opened, 1e-9), 3),
    }


def git_head():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:                             # noqa: BLE001
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", required=True,
                        help="Label for the connection under test, e.g. "
                             "'4g-hotspot-viettel-at-home'. Recorded verbatim.")
    parser.add_argument("--runs", type=int, default=3,
                        help="Repeat count. One sample is not a measurement.")
    parser.add_argument("--note", default="",
                        help="Free text: signal bars, location, who else is on the link.")
    args = parser.parse_args()

    url = f"{BASE}/{key_for(STAMP, BAND, SEGMENT)}"
    print(f"target : {url}")
    print(f"network: {args.network}")
    print(f"runs   : {args.runs}\n")

    runs = []
    for index in range(1, args.runs + 1):
        result = time_one_fetch(url)
        runs.append(result)
        if result["ok"]:
            print(f"run {index}: {result['total_s']}s total "
                  f"({result['connect_and_first_response_s']}s to first response, "
                  f"{result['download_s']}s download, "
                  f"{result['decompress_s']}s decompress) "
                  f"= {result['download_MiB_per_s']} MiB/s")
        else:
            print(f"run {index}: FAILED after {result['seconds_to_failure']}s "
                  f"-- {result['error']}")

    good = [r for r in runs if r["ok"]]
    artifact = {
        "what_this_measures": "transport time for one fixed historical HSD segment",
        "what_this_does_not_measure": (
            "publication latency (V-5), and conditions at the venue during the demo"
        ),
        "provenance": {
            "command": " ".join(sys.argv),
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_head": git_head(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "conditions": {
            "network_label": args.network,
            "note": args.note,
            "object": key_for(STAMP, BAND, SEGMENT),
        },
        "runs": runs,
        "successful_runs": len(good),
        "median_total_s": (sorted(r["total_s"] for r in good)[len(good) // 2]
                           if good else None),
        "slowest_total_s": max((r["total_s"] for r in good), default=None),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in args.network)
    path = OUT / f"network-{safe}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    print(f"\nsuccessful runs: {len(good)}/{args.runs}")
    if good:
        # The slowest run is the one that decides whether a demo survives, not the median.
        print(f"median total   : {artifact['median_total_s']}s")
        print(f"slowest total  : {artifact['slowest_total_s']}s  <- plan against this")
    print(f"artifact       : {path}")
    print("\nThis is an UPPER BOUND unless it was taken in Hoi truong I at demo time.")


if __name__ == "__main__":
    main()
