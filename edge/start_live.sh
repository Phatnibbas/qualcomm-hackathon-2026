#!/bin/sh
# Start the HALO SafeShift live service detached, so it survives the adb session
# and keeps running on the board alone. No sudo required.
cd "$(dirname "$0")" || exit 1
pkill -f halo_live.py 2>/dev/null
sleep 1
setsid nohup python3 halo_live.py \
  --catalog model_catalog.json \
  --host 0.0.0.0 --port 8080 --refresh 60 \
  < /dev/null > halo_live.log 2>&1 &
echo "started pid $!"
