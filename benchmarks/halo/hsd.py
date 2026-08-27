"""Minimal reader for Himawari Standard Data (HSD) infrared segments.

Written instead of pulling in satpy: we only need one band, one segment, and
brightness temperature. Every constant comes from the file's own header. Plausible
summary values are only an internal sanity check; this reader is unvalidated until a
same-file pixel-by-pixel comparison with a reference implementation passes.
"""

import bz2
import struct
import urllib.request

import numpy as np

BASE = "https://noaa-himawari9.s3.amazonaws.com"


def fetch(band, segment, stamp, resolution="R20"):
    """Download and decompress one HSD segment. `stamp` is 'YYYYmmdd_HHMM'."""
    day = stamp[:8]
    hhmm = stamp[9:]
    key = (f"AHI-L1b-FLDK/{day[:4]}/{day[4:6]}/{day[6:8]}/{hhmm}/"
           f"HS_H09_{day}_{hhmm}_{band}_FLDK_{resolution}_S{segment:02d}10.DAT.bz2")
    with urllib.request.urlopen(f"{BASE}/{key}", timeout=120) as response:
        return bz2.decompress(response.read())


def _blocks(raw):
    """Walk the 11 header blocks; each starts with number(u8) + length(u16)."""
    offsets, cursor = {}, 0
    for _ in range(11):
        number, length = struct.unpack_from("<BH", raw, cursor)
        offsets[number] = (cursor, length)
        cursor += length
    return offsets, cursor


def read_bt(raw):
    """Return (brightness_temperature_2d, metadata)."""
    offsets, data_start = _blocks(raw)

    # Block 2 - data information
    base, _ = offsets[2]
    bits, columns, lines = struct.unpack_from("<HHH", raw, base + 3)

    # Block 7 - segment information.  The HSD wire order is total segments
    # then segment number.  This corrects only the historical metadata label;
    # it does NOT validate calibration/navigation/pixels (Satpy remains the
    # required reference gate for any P1 satellite feature).
    base, _ = offsets[7]
    total_segments, segment_number, first_line = struct.unpack_from("<BBH", raw, base + 3)

    # Block 5 - calibration information
    base, _ = offsets[5]
    band_number, wavelength_um, valid_bits, error_count, outside_count = \
        struct.unpack_from("<HdHHH", raw, base + 3)
    gain, constant = struct.unpack_from("<dd", raw, base + 3 + 16)
    # IR-specific block: radiance -> effective temperature -> brightness temp
    c0, c1, c2, C0, C1, C2, light_speed, planck, boltzmann = \
        struct.unpack_from("<9d", raw, base + 3 + 32)

    counts = np.frombuffer(
        raw, dtype="<u2", count=columns * lines, offset=data_start
    ).reshape(lines, columns).astype(np.float64)

    invalid = (counts == error_count) | (counts == outside_count)
    radiance = gain * counts + constant          # W m-2 sr-1 um-1
    radiance = np.where(radiance <= 0, np.nan, radiance)

    # Planck inversion in SI: convert um -> m and per-um -> per-m.
    lam = wavelength_um * 1e-6
    spectral = radiance * 1e6
    effective = (planck * light_speed / boltzmann / lam) / np.log(
        1.0 + (2.0 * planck * light_speed ** 2) / (spectral * lam ** 5)
    )
    bt = c0 + c1 * effective + c2 * effective ** 2
    bt[invalid] = np.nan

    return bt, {
        "band": band_number, "wavelength_um": wavelength_um,
        "columns": columns, "lines": lines, "bits": bits,
        "segment": segment_number, "of": total_segments, "first_line": first_line,
    }


if __name__ == "__main__":
    raw = fetch("B13", 4, "20260809_0930")
    bt, meta = read_bt(raw)
    print("metadata:", meta)
    finite = bt[np.isfinite(bt)]
    print(f"brightness temperature: min {finite.min():.1f} K  "
          f"median {np.median(finite):.1f} K  max {finite.max():.1f} K")
    print("SANITY ONLY: plausible range does not validate the decoder; run V-3 reference comparison")
