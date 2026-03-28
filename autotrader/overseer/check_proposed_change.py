"""CLI wrapper for overseer config-change guard checks."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime

from overseer.change_memory import check_proposed_change


def _parse_json_value(raw: str):
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def main() -> int:
    parser = argparse.ArgumentParser(description="Check whether an overseer config change is allowed.")
    parser.add_argument("--parameter", required=True, help="Dot-path parameter name, e.g. strategies.rsi_momentum.rsi_overbought")
    parser.add_argument("--old-json", required=True, help="Current value encoded as JSON")
    parser.add_argument("--new-json", required=True, help="Proposed value encoded as JSON")
    parser.add_argument(
        "--change-log-path",
        default="overseer/change_log.json",
        help="Path to change_log.json",
    )
    parser.add_argument(
        "--as-of",
        default=None,
        help="Optional ISO timestamp for deterministic checks",
    )
    args = parser.parse_args()

    as_of = datetime.fromisoformat(args.as_of) if args.as_of else None
    result = check_proposed_change(
        parameter=args.parameter,
        old_value=_parse_json_value(args.old_json),
        new_value=_parse_json_value(args.new_json),
        change_log_path=args.change_log_path,
        as_of=as_of,
    )
    sys.stdout.write(json.dumps(result, indent=2, default=str))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
