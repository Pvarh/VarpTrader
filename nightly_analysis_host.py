#!/usr/bin/env python3
"""Host-side Claude launcher for nightly analysis."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from overseer_host import (
    TRADER_USER,
    WORKDIR,
    _append_log,
    _claude_env,
    _drop_privileges,
    _ensure_trader_home,
    _host_path,
    _load_env_file,
    _normalize_claude_error,
    _resolve_auth_mode,
    _resolve_trader_account,
)


def _prepare(model: str | None) -> dict[str, object]:
    cmd = [
        "docker",
        "exec",
        "autotrader",
        "python",
        "-m",
        "analysis.host_nightly",
        "--host-prepare",
    ]
    if model:
        cmd.extend(["--model", model])
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "prepare failed")
    return json.loads(result.stdout)


def _finalize(state_file: str, response_file: str) -> dict[str, object]:
    cmd = [
        "docker",
        "exec",
        "autotrader",
        "python",
        "-m",
        "analysis.host_nightly",
        "--host-finalize",
        "--state-file",
        state_file,
        "--response-file",
        response_file,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "finalize failed")
    return json.loads(result.stdout)


def _invoke_claude(prompt: str, model: str, env_file: dict[str, str]) -> str:
    account = _resolve_trader_account()
    _ensure_trader_home(account)
    auth_mode = _resolve_auth_mode(env_file)
    child_env = _claude_env(env_file, account, auth_mode)

    result = subprocess.run(
        [
            "claude",
            "--print",
            "--model",
            model,
            "-p",
            prompt,
        ],
        capture_output=True,
        text=True,
        timeout=900,
        env=child_env,
        preexec_fn=_drop_privileges(account),
    )
    stdout = (result.stdout or "").strip()
    if result.returncode != 0 and not stdout:
        message = result.stderr.strip() or f"Claude exited {result.returncode}"
        return _normalize_claude_error(f"[Nightly Analysis Error] {message}", auth_mode)
    return _normalize_claude_error(stdout, auth_mode)


def run(model: str | None = None) -> dict[str, object]:
    env_file = _load_env_file(WORKDIR / ".env")
    payload = _prepare(model)
    prompt = str(payload["prompt"])
    state_file = str(payload["state_file"])
    response_file = str(payload["response_file"])
    resolved_model = str(payload["model"])

    _append_log(
        f"nightly_analysis_host_started | model={resolved_model} user={TRADER_USER}"
    )
    raw_output = _invoke_claude(prompt, resolved_model, env_file)
    response_path = _host_path(response_file)
    response_path.parent.mkdir(parents=True, exist_ok=True)
    response_path.write_text(raw_output, encoding="utf-8")
    finalized = _finalize(state_file, response_file)
    _append_log(
        "nightly_analysis_host_complete | run_id={} approved={} rejected={}".format(
            finalized.get("run_id"),
            finalized.get("approved"),
            finalized.get("rejected"),
        )
    )
    return finalized


def main() -> None:
    parser = argparse.ArgumentParser(description="Host nightly analysis runner")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    try:
        result = run(model=args.model)
    except Exception as exc:
        _append_log(f"nightly_analysis_host_error | err={exc}")
        raise
    print(json.dumps(result))


if __name__ == "__main__":
    main()
