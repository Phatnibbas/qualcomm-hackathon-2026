"""Which Himawari full-disk segment contains Ho Chi Minh City, and how big is it?

Uses the CGMS geostationary projection used by the AHI HSD format, then
verifies the answer against the real object listing on NOAA's public S3.
No credentials, no bulk download - HTTP range requests only.
"""

import math
import urllib.request

SUB_LON = 140.7          # Himawari-9 sub-satellite longitude
STATION = (10.80, 106.70)  # MakerLab, Binh Thanh, HCMC (approximate)

# CGMS constants for the AHI 2 km grid (bands 5-16, e.g. B13). 5500x5500.
COFF = LOFF = 2750.5
CFAC = LFAC = 20466275
GRID = 5500
SEGMENTS = 10


def latlon_to_pixel(lat, lon):
    """CGMS LRIT/HRIT normalized geostationary projection -> (line, column)."""
    lat_r, lon_r = math.radians(lat), math.radians(lon)
    # geodetic -> geocentric latitude on the WGS84-ish ellipsoid AHI uses
    c_lat = math.atan(0.993305616 * math.tan(lat_r))
    rl = 6356.7523 / math.sqrt(1 - 0.00669438444 * math.cos(c_lat) ** 2)

    r1 = 42164.0 - rl * math.cos(c_lat) * math.cos(lon_r - math.radians(SUB_LON))
    r2 = -rl * math.cos(c_lat) * math.sin(lon_r - math.radians(SUB_LON))
    r3 = rl * math.sin(c_lat)

    if r1 * (r1 - 42164.0) + r2 * r2 + r3 * r3 > 0:
        raise ValueError("point is not visible from the satellite")

    rn = math.sqrt(r1 * r1 + r2 * r2 + r3 * r3)
    x = math.degrees(math.atan2(-r2, r1))
    y = math.degrees(math.asin(-r3 / rn))

    column = COFF + x * (2 ** -16) * CFAC
    line = LOFF + y * (2 ** -16) * LFAC
    return line, column


line, column = latlon_to_pixel(*STATION)
rows_per_segment = GRID // SEGMENTS
segment = int(line // rows_per_segment) + 1

print("=" * 68)
print("GEOMETRY")
print("=" * 68)
print(f"station           {STATION[0]}N {STATION[1]}E")
print(f"pixel (2 km grid) line {line:.1f}, column {column:.1f}  of {GRID}")
print(f"rows per segment  {rows_per_segment}")
print(f"-> SEGMENT {segment:02d} of {SEGMENTS}")

print("\nbounding box for a +/-150 km region of interest:")
for name, (dlat, dlon) in {
    "N edge": (1.35, 0), "S edge": (-1.35, 0),
    "W edge": (0, -1.37), "E edge": (0, 1.37),
}.items():
    edge_line, edge_col = latlon_to_pixel(STATION[0] + dlat, STATION[1] + dlon)
    print(f"  {name}: line {edge_line:7.1f}  col {edge_col:7.1f}  "
          f"(segment {int(edge_line // rows_per_segment) + 1:02d})")

print()
print("=" * 68)
print("OBJECT SIZES ON NOAA S3  (2026-07-15 06:00 UTC = 13:00 ICT)")
print("=" * 68)

BASE = "https://noaa-himawari9.s3.amazonaws.com"
PREFIX = "AHI-L1b-FLDK/2026/07/15/0600"

measured = {}
for band in ("B08", "B13", "B03"):
    resolution = {"B08": "R20", "B13": "R20", "B03": "R05"}[band]
    key = f"{PREFIX}/HS_H09_20260715_0600_{band}_FLDK_{resolution}_S{segment:02d}10.DAT.bz2"
    request = urllib.request.Request(f"{BASE}/{key}", method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            size = int(response.headers["Content-Length"]) / 1024 / 1024
            measured[band] = size
            print(f"  {band} S{segment:02d}  {size:6.2f} MiB   OK")
    except Exception as exc:  # noqa: BLE001
        print(f"  {band} S{segment:02d}  FAILED: {type(exc).__name__} {exc}")

print("\n  (a full 16-band x 10-segment timeline is 448.6 MiB compressed)")

if not measured.get("B13"):
    raise SystemExit("\nCannot size the dataset without a measured B13 object.")

print()
print("=" * 68)
print("REFERENCE SIZE SCENARIOS — NO DATASET OR PRODUCT SELECTED")
print("=" * 68)
# Sizes come from the HEAD requests above, never from a guessed constant:
# an earlier version of this script hard-coded 9 MiB and overstated every
# figure below by ~3x.
b13, b08 = measured["B13"], measured.get("B08", 0.0)
print(f"  (using measured B13 {b13:.2f} MiB, B08 {b08:.2f} MiB per segment)")
for label, per_timeline, cadence_min in (
    ("B13, S04 only, 10 min", b13, 10),
    ("B13, S04+S05, 10 min", b13 * 2, 10),
    ("B13, S04+S05, 30 min", b13 * 2, 30),
    ("B13+B08, S04+S05, 30 min", (b13 + b08) * 2, 30),
):
    timelines = 27 * 24 * 60 / cadence_min
    print(f"  {label:<28} {timelines:6.0f} timelines  ~{timelines * per_timeline / 1024:5.1f} GiB / 27 days")

print()
print("  Reference scenario: only windows around threshold-rule occurrences plus a")
print("  matched sample of quiet timelines is far smaller than any row above.")
print("  e.g. 28 rule occurrences x 4 h at 10 min + 1000 comparison timelines, S04 only:")
print(f"       ~{1672 * b13 / 1024:.1f} GiB")
