"""Pull the full Saigon ThingSpeak history (channel 3428136) into a local CSV.

Read-only against the public API. Writes only into the scratchpad.
ThingSpeak caps `results` at 8000, so we page backwards with `end`.
"""

import csv
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

CHANNEL = 3428136
OUT = Path(__file__).parent / "_cache" / "saigon_full.csv"
FIELDS = [f"field{i}" for i in range(1, 9)]


def fetch(end_iso=None, results=8000):
    url = (
        f"https://api.thingspeak.com/channels/{CHANNEL}/feeds.json"
        f"?results={results}&status=true"
    )
    if end_iso:
        url += "&end=" + urllib.parse.quote(end_iso)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return json.load(response)
        except Exception as exc:  # noqa: BLE001 - transient network
            if attempt == 3:
                raise
            print(f"  retry {attempt + 1} after {type(exc).__name__}")
            time.sleep(3)
    return None


def main():
    rows = {}
    end_iso = None
    page = 0

    while True:
        page += 1
        data = fetch(end_iso)
        feeds = data.get("feeds", [])
        if not feeds:
            print(f"page {page}: empty, stopping")
            break

        new = 0
        for feed in feeds:
            entry_id = feed.get("entry_id")
            if entry_id is not None and entry_id not in rows:
                rows[entry_id] = feed
                new += 1

        oldest = feeds[0]["created_at"]
        print(f"page {page}: {len(feeds)} rows, {new} new, oldest {oldest}, total {len(rows)}")

        if new == 0:
            break

        # Step `end` one second before the oldest row we just received.
        stamp = datetime.fromisoformat(oldest.replace("Z", "+00:00")) - timedelta(seconds=1)
        end_iso = stamp.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        if page > 20:
            print("page cap reached")
            break

    ordered = [rows[k] for k in sorted(rows)]
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["entry_id", "created_at", "status", *FIELDS])
        for feed in ordered:
            writer.writerow(
                [feed.get("entry_id"), feed.get("created_at"), feed.get("status") or ""]
                + [feed.get(name) or "" for name in FIELDS]
            )

    print(f"\nwrote {len(ordered)} rows -> {OUT}")
    if ordered:
        print(f"range: {ordered[0]['created_at']} .. {ordered[-1]['created_at']}")
        print(f"entry_id: {ordered[0]['entry_id']} .. {ordered[-1]['entry_id']}")


if __name__ == "__main__":
    main()
