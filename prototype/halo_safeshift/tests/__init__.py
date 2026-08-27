"""Test package for HALO SafeShift local preparation.

Shared synthetic builders live here so that individual test modules stay
focused. Everything below is synthetic: no test in this package depends on the
frozen station data, and no test reports a model metric.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def reseal(bundle_dir: Path) -> Path:
    """Rebuild ``bundle_envelope.json`` after a deliberate mutation.

    The envelope is checked before anything else, so a test that tampers with
    ``manifest.json`` and does not reseal only ever proves the envelope works.
    Resealing lets a test isolate the *inner* check it actually cares about —
    and the un-resealed case is covered by its own dedicated tests.
    """
    from ..export import write_bundle_envelope

    write_bundle_envelope(bundle_dir)
    return bundle_dir


def rewrite_manifest(bundle_dir: Path, **changes: Any) -> Path:
    """Mutate manifest fields and reseal, so the inner validators are reached."""
    path = Path(bundle_dir) / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(changes)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    return reseal(Path(bundle_dir))

CHANNEL_METADATA = {
    "id": 3428136,
    "name": "MakerLab Giadinh Station",
    "field1": "WindSpeed",
    "field2": "WindDirection",
    "field3": "Temperature",
    "field4": "Pressure",
    "field5": "Light",
    "field6": "Humidity",
    "field7": "Noise",
    "field8": "PM2.5",
}

FIELD_MAPPING = {
    "field1": "wind_speed",
    "field2": "wind_direction",
    "field3": "temperature",
    "field4": "pressure",
    "field5": "light",
    "field6": "humidity",
    "field7": "noise",
    "field8": "pm25",
}

EPOCH = datetime(2026, 7, 22, 0, 0, 0, tzinfo=timezone.utc)


def raw_row(
    entry_id: int,
    when: datetime,
    *,
    temperature: float = 30.0,
    humidity: float = 70.0,
    wind_speed: float = 1.5,
    wind_direction: float = 180.0,
    pressure: float = 100.5,
    light: float = 12000.0,
    noise: float = 60.0,
    pm25: float = 25.0,
    status: str = "ALL_OK",
) -> dict[str, str]:
    """One raw ThingSpeak-shaped row, with the real field positions."""
    return {
        "entry_id": str(entry_id),
        "created_at": when.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": status,
        "field1": f"{wind_speed}",
        "field2": f"{wind_direction}",
        "field3": f"{temperature}",
        "field4": f"{pressure}",
        "field5": f"{light}",
        "field6": f"{humidity}",
        "field7": f"{noise}",
        "field8": f"{pm25}",
    }


def raw_frame(rows: list[dict[str, str]]) -> pd.DataFrame:
    """Build the frame shape that ``data.load_raw_csv`` produces."""
    frame = pd.DataFrame(rows, dtype=str)
    frame["created_at_utc"] = pd.to_datetime(
        frame["created_at"], utc=True, format="ISO8601", errors="coerce"
    )
    frame["entry_id_int"] = pd.to_numeric(frame["entry_id"], errors="coerce").astype("Int64")
    return frame


def dense_rows(
    n_bins: int,
    *,
    start: datetime = EPOCH,
    per_bin: int = 8,
    seconds_between: int = 30,
    first_entry_id: int = 1,
    temperature_at=lambda index: 30.0 + 0.01 * index,
    skip_bins: tuple[int, ...] = (),
) -> list[dict[str, str]]:
    """Rows dense enough that every five-minute bin is valid.

    ``skip_bins`` omits whole bins, which is how a gap or outage is simulated.
    """
    rows: list[dict[str, str]] = []
    entry_id = first_entry_id
    for bin_index in range(n_bins):
        if bin_index in skip_bins:
            continue
        bin_start = start + timedelta(minutes=5 * bin_index)
        for slot in range(per_bin):
            when = bin_start + timedelta(seconds=1 + slot * seconds_between)
            rows.append(
                raw_row(
                    entry_id,
                    when,
                    temperature=round(temperature_at(bin_index), 4),
                    humidity=round(70.0 + math.sin(bin_index / 7.0), 4),
                    wind_speed=round(1.5 + 0.1 * math.cos(bin_index / 5.0), 4),
                )
            )
            entry_id += 1
    return rows
