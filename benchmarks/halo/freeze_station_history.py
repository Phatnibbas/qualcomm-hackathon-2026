"""Freeze the full Saigon ThingSpeak history (channel 3428136) with a manifest.

This is the upgraded, run-scoped successor to `pull_station_history.py`.

⚠ **Why it is a separate file.** The 2026-08-11 feasibility probe pins the
SHA-256 of every script that produced it, including `pull_station_history.py`,
in `evidence/halo-probe-2026-08-11/result.json`. `benchmarks/validate_agent_docs.py`
verifies those hashes, so editing that script in place silently invalidates a
frozen historical provenance chain — it fails the repository gate with
`producer_sha256 hash mismatch`, and the only way to "fix" that by editing the
evidence file would be to falsify what produced the probe. Adding the new
behaviour here leaves the probe's chain verifiable and this script free to
evolve. See the completion report for the reported contradiction.

`pull_station_history.py` is therefore unchanged and remains the probe's
producer of record. Use *this* script for HALO SafeShift freezes.

Read-only against the public API. Writes only where the caller points it.
ThingSpeak caps `results` at 8000, so we page backwards with `end`.

Beyond the raw feed this script also freezes:

* the channel metadata verbatim, including the fieldN -> label mapping, so no
  downstream stage has to guess the semantics from field position;
* a manifest with source URLs, pull window, entry/timestamp ranges, row count,
  and SHA-256 of the CSV, the metadata and this script;
* an ordering/duplicate/cadence/gap analysis, including the latest outage
  longer than the configured threshold and the first entry after it.

Configured values (channel id, page size, outage threshold, gap-report
thresholds, semantic field mapping) come from
`prototype/halo_safeshift/config/experiment.v1.json`. Nothing scientific is
hard-coded here, and no current entry id, timestamp or run id is baked in.

Usage:

    py -3 benchmarks/halo/freeze_station_history.py
    py -3 benchmarks/halo/freeze_station_history.py \
        --out benchmarks/halo/_cache/<run-id>/station_raw.csv \
        --metadata-out evidence/halo-safeshift/<run-id>/channel_metadata.json \
        --manifest evidence/halo-safeshift/<run-id>/data_manifest.json
"""

import argparse
import csv
import hashlib
import io
import json
import os
import statistics
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "prototype" / "halo_safeshift" / "config" / "experiment.v1.json"
DEFAULT_OUT = Path(__file__).parent / "_cache" / "safeshift_station.csv"
FIELDS = [f"field{i}" for i in range(1, 9)]


class FreezeError(RuntimeError):
    """Raised when a freeze cannot be completed honestly. The run stops."""


def load_config(path=CONFIG_PATH):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_relative(path):
    """Repository-relative POSIX path, or refuse.

    An absolute path baked into a manifest is unusable on any other machine and
    leaks the operator's directory layout. A path outside the repository is
    refused outright rather than silently rewritten.
    """
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError as exc:
        raise FreezeError(
            f"{resolved}: output paths must live inside the repository ({REPO_ROOT})"
        ) from exc


def require_new_path(path, label):
    """Refuse to overwrite. A freeze that clobbers an earlier freeze is not frozen."""
    target = Path(path)
    if target.exists():
        raise FreezeError(
            f"{repo_relative(target)}: {label} already exists. A freeze must not "
            f"overwrite a previous freeze — the earlier run's manifest still "
            f"attests these bytes. Use a new run-scoped directory."
        )
    return target


def atomic_write_bytes(path, payload):
    """Write via a temporary file and replace, so no reader sees a partial file.

    A crash midway through a direct write leaves a truncated CSV that still
    looks like a CSV, and its hash would then be recorded as if it were the
    complete pull.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".partial")
    with open(temporary, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    return target


def atomic_write_text(path, text):
    return atomic_write_bytes(path, text.encode("utf-8"))


def parse_iso(value):
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def iso(value):
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def fetch(url, timeout, attempts, sleep_seconds):
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return json.load(response)
        except Exception as exc:  # noqa: BLE001 - transient network
            if attempt == attempts - 1:
                raise
            print(f"  retry {attempt + 1} after {type(exc).__name__}")
            time.sleep(sleep_seconds)
    return None


def feeds_url(base, results, end_iso=None, status=True):
    url = f"{base}?results={results}"
    if status:
        url += "&status=true"
    if end_iso:
        url += "&end=" + urllib.parse.quote(end_iso)
    return url


def analyse(ordered, config):
    """Ordering, duplicates, cadence and gaps over the pulled feed."""
    thresholds = list(config["outage"]["gap_report_thresholds_seconds"])
    long_outage_seconds = float(config["outage"]["long_outage_hours"]) * 3600.0

    stamps = []
    unparsable = 0
    for row in ordered:
        try:
            stamps.append((row["entry_id"], parse_iso(row["created_at"])))
        except (TypeError, ValueError):
            unparsable += 1

    entry_ids = [entry for entry, _ in stamps]
    times = [when for _, when in stamps]

    duplicate_entry_ids = _duplicates(entry_ids)
    duplicate_timestamps = _duplicates([iso(t) for t in times])

    non_monotonic = sum(1 for i in range(1, len(times)) if times[i] < times[i - 1])

    deltas = [(times[i] - times[i - 1]).total_seconds() for i in range(1, len(times))]
    gaps = {}
    for threshold in thresholds:
        gaps[f"gaps_over_{int(threshold)}s"] = [
            {
                "after_entry_id": entry_ids[i - 1],
                "gap_start_utc": iso(times[i - 1]),
                "gap_end_utc": iso(times[i]),
                "gap_seconds": round(deltas[i - 1], 3),
            }
            for i in range(1, len(times))
            if deltas[i - 1] > threshold
        ]

    long_outages = [
        {
            "after_entry_id": entry_ids[i - 1],
            "gap_start_utc": iso(times[i - 1]),
            "gap_end_utc": iso(times[i]),
            "gap_hours": round(deltas[i - 1] / 3600.0, 4),
            "first_entry_id_after_gap": entry_ids[i],
        }
        for i in range(1, len(times))
        if deltas[i - 1] > long_outage_seconds
    ]

    largest = None
    if deltas:
        index = max(range(len(deltas)), key=lambda i: deltas[i])
        largest = {
            "after_entry_id": entry_ids[index],
            "gap_start_utc": iso(times[index]),
            "gap_end_utc": iso(times[index + 1]),
            "gap_seconds": round(deltas[index], 3),
            "gap_hours": round(deltas[index] / 3600.0, 4),
        }

    cadence = {}
    if deltas:
        ordered_deltas = sorted(deltas)
        cadence = {
            "n_intervals": len(deltas),
            "min_seconds": round(ordered_deltas[0], 3),
            "median_seconds": round(statistics.median(ordered_deltas), 3),
            "p95_seconds": round(ordered_deltas[max(0, int(0.95 * len(ordered_deltas)) - 1)], 3),
            "max_seconds": round(ordered_deltas[-1], 3),
        }

    return {
        "unparsable_timestamps": unparsable,
        "timestamp_order": "non-decreasing" if non_monotonic == 0 else f"{non_monotonic} out-of-order pairs",
        "non_monotonic_pairs": non_monotonic,
        "duplicate_entry_ids": duplicate_entry_ids,
        "duplicate_timestamps": duplicate_timestamps,
        "cadence": cadence,
        "gap_counts": {key: len(value) for key, value in gaps.items()},
        "gaps": gaps,
        "largest_gap": largest,
        "long_outage_threshold_hours": config["outage"]["long_outage_hours"],
        "long_outages": long_outages,
        "latest_long_outage": long_outages[-1] if long_outages else None,
        "first_entry_after_latest_long_outage": (
            {
                "entry_id": long_outages[-1]["first_entry_id_after_gap"],
                "created_at_utc": long_outages[-1]["gap_end_utc"],
            }
            if long_outages
            else None
        ),
    }


def _duplicates(values):
    seen = set()
    duplicates = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        else:
            seen.add(value)
    return sorted(duplicates, key=str)


def validate_channel(channel_meta, config, expected_channel_id):
    """Fail closed on a channel that is not the configured station.

    Two different failures hide behind a successful HTTP 200: the API can return
    a channel object for a different station, and it can return feeds with no
    channel block at all. Either would produce a manifest that claims to
    describe channel 3428136 while containing something else.
    """
    if not isinstance(channel_meta, dict) or not channel_meta:
        raise FreezeError(
            "the API returned no channel metadata block. Field semantics are "
            "resolved from published labels, so a freeze without metadata cannot "
            "be interpreted and is refused."
        )
    observed = channel_meta.get("id")
    if observed is None or int(observed) != int(expected_channel_id):
        raise FreezeError(
            f"channel metadata reports id {observed!r} but this freeze targets "
            f"{int(expected_channel_id)}. Refusing to write a manifest that would "
            f"attribute one station's data to another."
        )

    table = config["field_semantics"]["label_to_semantic"]
    required = list(config["field_semantics"]["feature_required_semantics"])
    resolved = {
        table.get(str(channel_meta.get(f"field{i}", "")).strip())
        for i in range(1, 9)
    }
    missing = [name for name in required if name not in resolved]
    if missing:
        raise FreezeError(
            f"channel metadata does not publish label(s) for required feature "
            f"semantics {missing}. The feature schema cannot be constructed from "
            f"this channel, so the freeze is refused at acquisition rather than "
            f"failing later during preparation."
        )
    return True


def pull_all(fetch_page, base, results, page_cap):
    """Page backwards until the history is exhausted, or refuse a truncated pull.

    Completion means a page came back empty or contributed no new entry id.
    Reaching the page cap first means the record is longer than this pull, and a
    partial history recorded as if it were complete is exactly the sort of
    silent truncation the manifest is supposed to make impossible.
    """
    rows = {}
    urls = []
    channel_meta = None
    end_iso = None
    page = 0
    complete = False

    while True:
        page += 1
        url = feeds_url(base, results, end_iso)
        urls.append(url)
        data = fetch_page(url) or {}
        if channel_meta is None:
            channel_meta = data.get("channel")
        feeds = data.get("feeds", [])
        if not feeds:
            print(f"page {page}: empty, stopping")
            complete = True
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
            complete = True
            break

        stamp = parse_iso(oldest) - timedelta(seconds=1)
        end_iso = stamp.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        if page >= page_cap:
            break

    if not complete:
        raise FreezeError(
            f"page cap {page_cap} was reached before the history was exhausted "
            f"({len(rows)} rows so far). This freeze would be a truncated prefix "
            f"recorded as a complete history. Raise source.page_cap and re-run."
        )
    if not rows:
        raise FreezeError(
            "the API returned zero rows. An empty freeze has nothing to attest and "
            "would produce a manifest describing no data."
        )
    return [rows[key] for key in sorted(rows)], urls, channel_meta, complete


def read_frozen_csv(path):
    """Read an already-frozen CSV back into feed-shaped rows. No network.

    Used by ``--from-frozen-csv`` so that an immutable freeze can be
    re-manifested after this script changes, without re-pulling and therefore
    without appending live data.
    """
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise FreezeError(f"{repo_relative(path)}: frozen CSV contains no rows")
    ordered = []
    for row in rows:
        entry = {
            "entry_id": int(row["entry_id"]),
            "created_at": row["created_at"],
            "status": row.get("status") or "",
        }
        for name in FIELDS:
            entry[name] = row.get(name) or None
        ordered.append(entry)
    ordered.sort(key=lambda item: item["entry_id"])
    return ordered


def semantic_mapping(channel, config):
    """Explicit metadata-derived fieldN -> label -> semantic mapping."""
    table = config["field_semantics"]["label_to_semantic"]
    mapping = {}
    for index in range(1, 9):
        key = f"field{index}"
        label = channel.get(key)
        if label is None or str(label).strip() == "":
            continue
        label = str(label).strip()
        mapping[key] = {"label": label, "semantic": table.get(label)}
    return mapping


def main(argv=None, fetch_page=None):
    config = load_config()
    source = config["source"]

    raw_argv = list(argv) if argv is not None else list(sys.argv[1:])
    parser = argparse.ArgumentParser(
        description="Read-only ThingSpeak station history pull, freeze and manifest."
    )
    parser.add_argument("--channel", type=int, default=int(source["thingspeak_channel_id"]))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--metadata-out", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--from-frozen-csv",
        type=Path,
        default=None,
        help=(
            "re-manifest an existing immutable freeze instead of pulling. No network "
            "call is made and no live data is appended; the CSV bytes are copied "
            "verbatim so their SHA-256 is unchanged."
        ),
    )
    parser.add_argument(
        "--from-frozen-metadata",
        type=Path,
        default=None,
        help="channel metadata belonging to the reused freeze (required with --from-frozen-csv)",
    )
    parser.add_argument(
        "--reused-source-manifest",
        type=Path,
        default=None,
        help="the previous run's data_manifest.json, referenced in the new manifest",
    )
    args = parser.parse_args(argv)

    configured = int(source["thingspeak_channel_id"])
    if args.channel == int(source["second_station_channel_id_excluded"]):
        parser.error(source["second_station_exclusion_reason"])
    if args.channel != configured:
        parser.error(
            f"--channel {args.channel} is not the configured station {configured}. "
            f"This freezer serves exactly one channel; pointing it elsewhere would "
            f"produce artifacts the rest of the pipeline would misread."
        )

    metadata_path = args.metadata_out or args.out.with_name(
        args.out.stem + "_channel_metadata.json"
    )
    manifest_path = args.manifest or args.out.with_name(args.out.stem + "_manifest.json")
    # Refuse before any network call: discovering a clobber after a six-minute
    # pull wastes the pull and tempts an overwrite.
    require_new_path(args.out, "raw CSV")
    require_new_path(metadata_path, "channel metadata")
    require_new_path(manifest_path, "manifest")

    base = source["feeds_url_template"].format(channel=args.channel)
    results = int(source["max_results_per_page"])
    timeout = int(source["request_timeout_seconds"])
    attempts = int(source["retry_attempts"])
    sleep_seconds = float(source["retry_sleep_seconds"])
    page_cap = int(source["page_cap"])

    def live_fetch(url):
        return fetch(url, timeout, attempts, sleep_seconds)

    reuse = None
    pull_start = datetime.now(timezone.utc)

    if args.from_frozen_csv is not None:
        if args.from_frozen_metadata is None:
            parser.error("--from-frozen-csv requires --from-frozen-metadata")
        source_csv = args.from_frozen_csv
        source_csv_sha = sha256_file(source_csv)
        ordered = read_frozen_csv(source_csv)
        reused_metadata = json.loads(args.from_frozen_metadata.read_text(encoding="utf-8"))
        channel_meta = reused_metadata.get("channel") or reused_metadata
        urls = [reused_metadata.get("source_url")] if reused_metadata.get("source_url") else []
        complete = True
        validate_channel(channel_meta, config, args.channel)

        # Byte-for-byte copy: the reused freeze keeps its identity, so the new
        # manifest's csv_sha256 is the same value the earlier run attested.
        atomic_write_bytes(args.out, source_csv.read_bytes())
        if sha256_file(args.out) != source_csv_sha:
            raise FreezeError("the reused CSV changed while being copied; refusing to continue")
        reuse = {
            "mode": "reused immutable source freeze",
            "reused_csv_path": repo_relative(source_csv),
            "reused_csv_sha256": source_csv_sha,
            "reused_metadata_path": repo_relative(args.from_frozen_metadata),
            "reused_metadata_sha256": sha256_file(args.from_frozen_metadata),
            "previous_source_manifest": (
                {
                    "path": repo_relative(args.reused_source_manifest),
                    "sha256": sha256_file(args.reused_source_manifest),
                }
                if args.reused_source_manifest is not None
                else None
            ),
            "network_calls": 0,
            "live_data_appended": False,
            "boundary": (
                "The bytes were acquired by the earlier run recorded in "
                "previous_source_manifest. This manifest was produced by the current "
                "version of this script reading those bytes offline. integrity.puller_* "
                "identifies what produced THIS manifest; acquisition identifies what "
                "pulled the data. Conflating the two would let a corrected script claim "
                "credit for a pull it never made."
            ),
        }
        print(f"\nre-manifested {len(ordered)} reused rows -> {repo_relative(args.out)}")
    else:
        fetcher = fetch_page if fetch_page is not None else live_fetch
        ordered, urls, channel_meta, complete = pull_all(fetcher, base, results, page_cap)
        validate_channel(channel_meta, config, args.channel)

        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(["entry_id", "created_at", "status", *FIELDS])
        for feed in ordered:
            writer.writerow(
                [feed.get("entry_id"), feed.get("created_at"), feed.get("status") or ""]
                + [feed.get(name) if feed.get(name) is not None else "" for name in FIELDS]
            )
        atomic_write_text(args.out, buffer.getvalue())
        print(f"\nwrote {len(ordered)} rows -> {repo_relative(args.out)}")

    pull_end = datetime.now(timezone.utc)

    metadata_payload = {
        "artifact": "channel_metadata.json",
        "fetched_utc": iso(pull_start),
        "source_url": urls[0] if urls else None,
        "channel": channel_meta,
        "reused_source_freeze": reuse,
        "field_mapping_from_metadata": semantic_mapping(channel_meta or {}, config),
        "boundary": (
            "Channel metadata is recorded verbatim. Field semantics are resolved "
            "from the published labels, never from field position."
        ),
    }
    atomic_write_text(
        metadata_path, json.dumps(metadata_payload, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"wrote channel metadata -> {repo_relative(metadata_path)}")

    analysis = analyse(ordered, config)
    manifest = {
        "artifact": "data_manifest.json",
        "config_id": config["config_id"],
        "source": {
            "channel_id": args.channel,
            "channel_name": (channel_meta or {}).get("name"),
            "source_urls": urls,
            "page_count": len(urls),
            "page_cap": page_cap,
            "results_per_page": results,
            "complete_history": bool(complete),
            "complete_history_definition": (
                "The pull ended because a page returned no rows or no new entry id, "
                "not because the page cap was reached. A cap-truncated pull raises "
                "instead of being recorded, so this field is never false in a "
                "manifest that exists."
            ),
        },
        "pull": {
            "start_utc": iso(pull_start),
            "end_utc": iso(pull_end),
            "duration_seconds": round((pull_end - pull_start).total_seconds(), 3),
            "puller_command": " ".join(
                ["py", "-3", repo_relative(Path(__file__)), *raw_argv]
            ),
            "network_pull": reuse is None,
        },
        "acquisition": reuse
        or {
            "mode": "live pull",
            "acquired_by": repo_relative(Path(__file__)),
            "acquired_by_sha256": sha256_file(Path(__file__)),
        },
        "rows": {
            "count": len(ordered),
            "first_entry_id": ordered[0].get("entry_id") if ordered else None,
            "last_entry_id": ordered[-1].get("entry_id") if ordered else None,
            "first_timestamp_utc": ordered[0].get("created_at") if ordered else None,
            "last_timestamp_utc": ordered[-1].get("created_at") if ordered else None,
            "channel_last_entry_id_at_pull": (channel_meta or {}).get("last_entry_id"),
        },
        "field_mapping_from_metadata": semantic_mapping(channel_meta or {}, config),
        "raw_csv_columns": ["entry_id", "created_at", "status", *FIELDS],
        "raw_column_policy": (
            "Raw source columns are written verbatim and are never renamed or "
            "reordered. Semantic names live in the mapping above."
        ),
        "integrity": {
            "csv_path": repo_relative(args.out),
            "csv_sha256": sha256_file(args.out),
            "metadata_path": repo_relative(metadata_path),
            "metadata_sha256": sha256_file(metadata_path),
            "puller_path": repo_relative(Path(__file__)),
            "puller_sha256": sha256_file(Path(__file__)),
            "puller_role": (
                "the script that produced THIS manifest. When `acquisition.mode` is a "
                "reused freeze, the bytes were pulled earlier by a different script "
                "version; see `acquisition` for what actually fetched them."
            ),
            "path_policy": "repository-relative POSIX paths only; absolute paths are refused",
        },
        "continuity": analysis,
        "recovery_context": {
            "cause": "user-confirmed manual station reset",
            "firmware_root_cause": "unresolved",
            "boundary": (
                "Resumption after a manual reset is not evidence that the firmware "
                "hang is fixed. Only the outage interval, the first entry after "
                "recovery and the state at pull time are recorded."
            ),
        },
        "boundary": [
            "Read-only acquisition and integrity record. No model, metric or claim is produced here.",
            "Feed continuity is not sensor completeness: a device that fails to post creates no entry id to be missing.",
        ],
    }
    atomic_write_text(manifest_path, json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote manifest -> {repo_relative(manifest_path)}")

    if ordered:
        print(f"range: {ordered[0]['created_at']} .. {ordered[-1]['created_at']}")
        print(f"entry_id: {ordered[0]['entry_id']} .. {ordered[-1]['entry_id']}")
        print(f"gap counts: {analysis['gap_counts']}")
        print(f"latest long outage: {analysis['latest_long_outage']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FreezeError as exc:
        print(f"FREEZE REFUSED: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
