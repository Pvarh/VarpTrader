#!/bin/bash
# Polls for starvation-triggered overseer requests written by the container.

set -euo pipefail

WORKDIR='/root/Varptrader/autotrader'
TRIGGER_FILE="$WORKDIR/logs/overseer_trigger.json"
PROCESSING_FILE="$WORKDIR/logs/overseer_trigger.processing.json"

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
if isinstance(value, bool):
    print("true" if value else "false")
elif value is None:
    print("")
else:
    print(value)
PY
}

TRIGGER_REASON="$(read_trigger_field trigger_reason)"
DEEP="$(read_trigger_field deep)"
MODEL="$(read_trigger_field model)"

CMD=(/root/Varptrader/overseer_host.sh --trigger-reason "${TRIGGER_REASON:-signal_starvation}")
if [ "$DEEP" = "true" ]; then
  CMD+=(--deep)
fi
if [ -n "${MODEL:-}" ]; then
  CMD+=(--model "$MODEL")
fi

echo "[$(date -u)] overseer_trigger_watch | trigger=${TRIGGER_REASON:-signal_starvation} deep=${DEEP:-false} model=${MODEL:-default}"
"${CMD[@]}"
