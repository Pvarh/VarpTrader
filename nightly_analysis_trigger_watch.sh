#!/bin/bash
# Polls for nightly-analysis triggers written by the container.

set -euo pipefail

WORKDIR='/root/Varptrader/autotrader'
TRIGGER_FILE="$WORKDIR/logs/nightly_analysis_trigger.json"
PROCESSING_FILE="$WORKDIR/logs/nightly_analysis_trigger.processing.json"

if [ ! -f "$TRIGGER_FILE" ]; then
  exit 0
fi

mv "$TRIGGER_FILE" "$PROCESSING_FILE"
trap 'rm -f "$PROCESSING_FILE"' EXIT

read_trigger_field() {
  local field="$1"
  python3 - "$PROCESSING_FILE" "$field" <<'PY'
import json
import sys

path = sys.argv[1]
field = sys.argv[2]
with open(path, encoding="utf-8") as fh:
    data = json.load(fh)
value = data.get(field)
print("" if value is None else value)
PY
}

MODEL="$(read_trigger_field model)"

echo "[$(date -u)] nightly_analysis_trigger_watch | model=${MODEL:-default}"
CMD=(python3 /root/Varptrader/nightly_analysis_host.py)
if [ -n "${MODEL:-}" ]; then
  CMD+=(--model "$MODEL")
fi
"${CMD[@]}"
