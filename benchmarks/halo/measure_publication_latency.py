"""V-5: measure Himawari publication latency — observation time to first downloadable.

This is the number the campaign has never had. Cadence is NOT latency: a 10-minute scan
schedule says nothing about when the object becomes listable. Everything about
inference-time satellite use depends on this and on nothing else measured so far.

Method: pick upcoming nominal 10-minute slots, poll S3 with HEAD until the object for
that slot answers 200, and record the wall-clock delay from the slot's nominal
observation time. A slot that never appears within the deadline is recorded as a
timeout, not as a missing measurement.

Segment matters: AHI scans progressively, so segments do not all publish together. This
polls the segment the project's region of interest actually needs.

    py -3 benchmarks/halo/measure_publication_latency.py --slots 3

Caveats that travel with every number this produces:
  - measured from THIS network, at THIS time of day, against THIS provider mirror;
  - nominal slot time is the START of the scan, not the observation of our pixel;
  - a fast result does not prove the path is reliable, only that it was fast now.
"""

import argparse
import datetime as dt
import json
import platform
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://noaa-himawari9.s3.amazonaws.com"
OUT = Path(__file__).resolve().parents[2] / "evidence" / "network"


def key_for(slot, band, segment, resolution="R20"):
    day = slot.strftime("%Y%m%d")
    hhmm = slot.strftime("%H%M")
    return (f"AHI-L1b-FLDK/{day[:4]}/{day[4:6]}/{day[6:8]}/{hhmm}/"
            f"HS_H09_{day}_{hhmm}_{band}_FLDK_{resolution}_S{segment:02d}10.DAT.bz2")


def head(url, timeout=20):
    """Return (status, content_length) or (None, error_string)."""
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.headers.get("Content-Length")
    except urllib.error.HTTPError as exc:
        return exc.code, None
    except Exception as exc:                       # noqa: BLE001 - failure is data
        return None, f"{type(exc).__name__}: {exc}"


def next_slots(count, lead_s=45):
    """Nominal 10-minute slots that have not yet started (plus a small lead)."""
    now = dt.datetime.now(dt.timezone.utc)
    first = now.replace(minute=(now.minute // 10) * 10, second=0, microsecond=0)
    while first <= now + dt.timedelta(seconds=lead_s):
        first += dt.timedelta(minutes=10)
    return [first + dt.timedelta(minutes=10 * i) for i in range(count)]


def wait_for(slot, band, segment, poll_s, deadline_s):
    url = f"{BASE}/{key_for(slot, band, segment)}"
    attempts = []
    while True:
        now = dt.datetime.now(dt.timezone.utc)
        elapsed = (now - slot).total_seconds()
        if elapsed < 0:                            # slot has not happened yet
            time.sleep(min(poll_s, -elapsed + 1))
            continue
        status, extra = head(url)
        attempts.append({"at_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                         "seconds_after_slot": round(elapsed, 1), "status": status})
        if status == 200:
            return {"ok": True, "slot_utc": slot.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "object": key_for(slot, band, segment),
                    "latency_s": round(elapsed, 1),
                    "latency_min": round(elapsed / 60, 2),
                    "content_length": extra, "poll_attempts": len(attempts),
                    "attempts": attempts}
        if elapsed > deadline_s:
            return {"ok": False, "slot_utc": slot.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "object": key_for(slot, band, segment),
                    "error": f"not published within {deadline_s}s",
                    "last_status": status, "poll_attempts": len(attempts),
                    "attempts": attempts}
        time.sleep(poll_s)


def git_head():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:                              # noqa: BLE001
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slots", type=int, default=3, help="How many 10-minute slots to watch.")
    parser.add_argument("--band", default="B13")
    parser.add_argument("--segment", type=int, default=4, help="Segment the ROI needs.")
    parser.add_argument("--poll-s", type=float, default=20.0)
    parser.add_argument("--deadline-s", type=float, default=2400.0, help="Give up after this long.")
    parser.add_argument("--network", default="unlabelled", help="Connection under test; recorded verbatim.")
    args = parser.parse_args()

    slots = next_slots(args.slots)
    print(f"watching {args.slots} slot(s) for {args.band} segment {args.segment:02d}")
    for slot in slots:
        print(f"  {slot.strftime('%Y-%m-%dT%H:%M:%SZ')}")
    print(f"polling every {args.poll_s}s, giving up at {args.deadline_s / 60:.0f} min\n", flush=True)

    results = []
    for slot in slots:
        result = wait_for(slot, args.band, args.segment, args.poll_s, args.deadline_s)
        results.append(result)
        if result["ok"]:
            print(f"{result['slot_utc']}  PUBLISHED after {result['latency_min']} min "
                  f"({result['latency_s']}s, {result['poll_attempts']} polls)", flush=True)
        else:
            print(f"{result['slot_utc']}  TIMEOUT -- {result['error']}", flush=True)

    good = [r for r in results if r["ok"]]
    artifact = {
        "what_this_measures": "V-5 publication latency: nominal slot time to first HTTP 200 on the object",
        "what_this_does_not_measure": (
            "download time, decode time, reliability over days, or latency of any other product"
        ),
        "caveats": [
            "nominal slot time is the scan START, not the observation time of our pixel",
            "measured from one network at one time of day against one mirror",
            "segments do not all publish together; this is the ROI segment only",
        ],
        "provenance": {
            "command": " ".join(sys.argv),
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_head": git_head(), "python": sys.version.split()[0],
            "platform": platform.platform(), "network_label": args.network,
        },
        "band": args.band, "segment": args.segment,
        "slots": results, "successful_slots": len(good),
        "median_latency_min": (sorted(r["latency_min"] for r in good)[len(good) // 2]
                               if good else None),
        "slowest_latency_min": max((r["latency_min"] for r in good), default=None),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"publication-latency-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    print(f"\nsuccessful slots: {len(good)}/{len(results)}")
    if good:
        print(f"median  : {artifact['median_latency_min']} min")
        print(f"slowest : {artifact['slowest_latency_min']} min  <- plan against this")
    print(f"artifact: {path}")
    print("\nThis is V-5 for ONE product, ONE segment, ONE session. It is not a reliability claim.")


if __name__ == "__main__":
    main()
