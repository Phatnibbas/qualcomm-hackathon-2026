"""Template for firmware/station_secrets.py.

Copy this file to `station_secrets.py`, fill in the three values, and upload
BOTH `saigon_station.py` and `station_secrets.py` to the ESP32.

`station_secrets.py` is gitignored. Do not commit it, do not paste it into a
chat, and do not include it in an archive you share.
"""

WIFI_SSID = "your-wifi-ssid"
WIFI_PASSWORD = "your-wifi-password"

# ThingSpeak *Write* API key for your own channel.
# Get it from https://thingspeak.com -> your channel -> API Keys.
THINGSPEAK_WRITE_API_KEY = "YOUR_WRITE_API_KEY"
