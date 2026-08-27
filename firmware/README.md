# Station firmware — ESP32-WROOM + DFRobot SEN0658

MicroPython firmware for the weather station that produces every input HALO
SafeShift consumes. It polls a 9-in-1 RS485 sensor over Modbus-RTU, shows the
current reading on a 16×2 I²C LCD, and posts eight fields to a public
ThingSpeak channel over HTTPS.

## Files

| File | Purpose |
|---|---|
| `saigon_station.py` | The firmware. Upload to the board as `main.py`. |
| `station_secrets.example.py` | Template for credentials. Copy to `station_secrets.py`, fill in, upload alongside. |

`station_secrets.py` is gitignored and is **not** in this repository. The
firmware imports `WIFI_SSID`, `WIFI_PASSWORD` and `THINGSPEAK_WRITE_API_KEY`
from it; if the import fails the board prints a warning and refuses to connect
rather than falling back to a hard-coded value.

## Hardware

| Part | Detail |
|---|---|
| MCU | ESP32-WROOM (ESP32 classic), MicroPython |
| Sensor | DFRobot SEN0658, RS485 / Modbus-RTU, IP54, 10–30 V DC |
| RS485 adapter | UART2 — `RX_PIN = 16`, `TX_PIN = 17` |
| Display | 16×2 HD44780 over PCF8574 I²C — `SDA = 22`, `SCL = 21` |
| Mounting | Rooftop mast, approximately 15 m above local ground |

## Sensor parameters

The SEN0658 reports **nine** parameters. **None of them is rainfall** — some
retailer listings carry the word "Rain" in the product title, but the parameter
list on the DFRobot wiki, Mouser and DigiKey has no precipitation channel.

| Parameter | Range | Accuracy |
|---|---|---|
| Wind speed (ultrasonic) | 0–40 m/s | ±0.5 + 2 % FS, **0.5 m/s start threshold** |
| Wind direction | 0–359° | ±3° |
| Temperature | −40…80 °C | ±0.5 °C |
| Humidity | 0–99 % RH | ±3 % RH |
| Pressure | 0–120 kPa | ±0.15 kPa |
| Illuminance | 0–200 000 lux | ±7 % |
| Noise | 30–120 dB | ±0.5 dB |
| PM2.5 | 0–1000 µg/m³ | ±3 % FS |
| PM10 | 0–1000 µg/m³ | ±3 % FS |

Source: <https://wiki.dfrobot.com/sen0658/>

## ThingSpeak field map

A free ThingSpeak channel has eight field slots, so **PM10 is measured but not
published**. Historical PM10 is therefore absent from the dataset in `data/`,
not merely unexported.

| Field | Value |
|---|---|
| 1 | WindSpeed |
| 2 | WindDirection |
| 3 | Temperature |
| 4 | Pressure |
| 5 | Light |
| 6 | Humidity |
| 7 | Noise |
| 8 | PM2.5 |

Channel `3428136`, public: <https://thingspeak.com/channels/3428136>

## Timing constants that actually exist in the code

| Constant | Value | Meaning |
|---|---|---|
| `UPLOAD_INTERVAL_MS` | 20 000 | Minimum spacing between ThingSpeak posts |
| `SENSOR_INTERVAL_MS` | 5 000 | Modbus poll period |
| `MODBUS_GAP_MS` | 220 | Gap between consecutive Modbus frames |
| `WIFI_RETRIES` | 10 | Connection attempts at boot |
| `WIFI_TIMEOUT_MS` | 10 000 | Per-attempt Wi-Fi timeout |
| `CALM_MS` | 0.5 | Wind speed below this is reported as calm |

The observed posting cadence on the public channel is a **median of 27 s**,
longer than `UPLOAD_INTERVAL_MS` because the Modbus read and the HTTPS round
trip both add time.

## `status` field vocabulary

The firmware writes a short marker into the ThingSpeak `status` field:

| Marker | Meaning |
|---|---|
| `BOOT.` | Board has just started. One marker per reset. |
| `ALL_OK` | Every Modbus group read successfully this cycle. |
| `FAIL-ALL` | No group could be read. |
| `FAIL-<groups>` | Named groups failed, e.g. `FAIL-THN-PM`. |

Counting `BOOT.` markers in the channel history is how unplanned resets are
detected after the fact.

## Known limitations — read before relying on this firmware

These are measured properties of the code as published, not hypotheticals.

- **There is no self-recovery.** The firmware contains no watchdog timer, no
  `machine.reset()` call, and passes no timeout to `requests.post()`. The
  `timeout=1000` on the UART constructor is a Modbus read timeout and does not
  bound the network path. If the HTTPS request blocks, the board stays blocked
  until it is power-cycled.
- **An HTTPS/TLS hang is unresolved.** The board has been observed blocking
  inside a synchronous TLS handshake. Attempts to bound it with a socket
  timeout, with raw TLS, and with non-blocking TLS all failed on the physical
  board; the non-blocking attempt produced an mbedTLS MPI allocation error. The
  station currently runs the rolled-back firmware published here. Do not
  reintroduce those three fixes without board-level verification.
- **Wind data before 2026-07-21T15:42Z is unusable.** A scale divisor was
  corrected from `/100` to `/10` at that point. Readings before it are
  approximately 10× low. The dataset in `data/` starts after the fix.
- **Pressure resolves to 0.1 kPa = 1 hPa** over an observed range of only
  100.0–101.2 kPa. A gust-front pressure jump of 0.5–2 hPa is 0–2 quantisation
  steps, so pressure is not a sensitive predictor on this hardware.
- **PM response time is up to 90 s** per the DFRobot specification, against a
  ~27 s posting cadence. Consecutive PM records are therefore **not**
  independent measurements.
