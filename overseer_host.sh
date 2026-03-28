#!/bin/bash
set -euo pipefail

exec python3 /root/Varptrader/overseer_host.py "$@"
