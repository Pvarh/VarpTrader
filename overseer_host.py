#!/usr/bin/env python3
"""Host-side overseer launcher with background job orchestration."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any

try:
    import pwd
except ImportError:  # pragma: no cover - Windows local tests
    pwd = None


ROOT_DIR = Path(os.environ.get("VARPTRADER_ROOT", "/root/Varptrader"))
WORKDIR = Path(os.environ.get("VARPTRADER_WORKDIR", str(ROOT_DIR / "autotrader")))
ENV_FILE = WORKDIR / ".env"
LOG_DIR = WORKDIR / "logs"
REPORTS_DIR = WORKDIR / "overseer" / "reports"
STATE_DIR = LOG_DIR / "overseer_state"
PID_FILE = STATE_DIR / "current.pid"
STATUS_FILE = STATE_DIR / "current_status.json"
PROMPT_FILE = STATE_DIR / "current_prompt.txt"
HOST_LOG_FILE = LOG_DIR / "overseer_host.log"
LIMITS_FILE = STATE_DIR / "limits.json"
TRADER_USER = os.environ.get("VARPTRADER_RUN_AS_USER", "trader")
_CLAUDE_AUTH_ENV_KEYS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
)


@dataclass
class HostConfig:
    model: str
    trigger_reason: str
    deep: bool
    timeout_sec: int
    effort: str
    test: bool = False


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp() -> str:
    return _utc_now().strftime("%Y-%m-%d %H:%M:%S UTC")


def _run_id() -> str:
    return _utc_now().strftime("%Y-%m-%d_%H%M%S")


def _iso_now() -> str:
    return _utc_now().isoformat()


def _load_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip()
    return env


def _append_log(message: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with HOST_LOG_FILE.open("a", encoding="utf-8") as fh:
        fh.write(f"[{_timestamp()}] {message}\n")


def _write_status(payload: dict[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _read_status() -> dict[str, Any]:
    if not STATUS_FILE.exists():
        return {}
    try:
        return json.loads(STATUS_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _read_limits() -> dict[str, Any]:
    if not LIMITS_FILE.exists():
        return {}
    try:
        return json.loads(LIMITS_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _write_limits(payload: dict[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    LIMITS_FILE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _pid_running(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _current_pid() -> int | None:
    if not PID_FILE.exists():
        return None
    try:
        return int(PID_FILE.read_text(encoding="utf-8").strip())
    except ValueError:
        return None


def _resolve_trader_account():
    if pwd is None:
        raise RuntimeError("pwd module unavailable on this platform")
    try:
        return pwd.getpwnam(TRADER_USER)
    except KeyError as exc:
        raise RuntimeError(f"user {TRADER_USER!r} not found") from exc


def _ensure_trader_home(account) -> None:
    home = Path(account.pw_dir)
    claude_dir = home / ".claude"
    debug_dir = claude_dir / "debug"

    for path in (home, claude_dir, debug_dir):
        path.mkdir(parents=True, exist_ok=True)
        os.chown(path, account.pw_uid, account.pw_gid)

    for path in home.iterdir():
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        if stat.st_uid != account.pw_uid or stat.st_gid != account.pw_gid:
            os.chown(path, account.pw_uid, account.pw_gid)


def _resolve_auth_mode(env: dict[str, str]) -> str:
    mode = (
        env.get("VARPTRADER_CLAUDE_AUTH_MODE")
        or os.environ.get("VARPTRADER_CLAUDE_AUTH_MODE")
        or "login"
    ).strip().lower()
    return mode if mode in {"login", "api", "auto"} else "login"


def _claude_env(base_env: dict[str, str], account, auth_mode: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(base_env)
    env.update({
        "HOME": account.pw_dir,
        "USER": account.pw_name,
        "LOGNAME": account.pw_name,
    })
    if auth_mode == "login":
        for key in _CLAUDE_AUTH_ENV_KEYS:
            env.pop(key, None)
    return env


def _drop_privileges(account):
    def _demote() -> None:
        os.setgid(account.pw_gid)
        os.setuid(account.pw_uid)

    return _demote


def _build_prompt(context: str, cfg: HostConfig) -> str:
    if not cfg.deep and cfg.trigger_reason == "signal_starvation":
        return (
            "You are VarpTrader Overseer handling a signal-starvation incident.\n\n"
            f"Trigger reason: {cfg.trigger_reason}\n\n"
            f"{context}\n\n"
            "=== YOUR TASK ===\n"
            "Use the provided context as ground truth. Do not reread files already summarized here unless you need to edit that exact file.\n"
            "Focus ONLY on the starvation problem.\n"
            "1. Explain why there were 0 trades/signals in the last 24h using exact evidence.\n"
            "2. Identify the single most likely blocking filter or missing setup.\n"
            "3. Apply at most one safe config change to reduce starvation. Do not ask for approval.\n"
            "4. Run `docker exec autotrader python -m pytest tests/ -q`.\n"
            "5. If config changed, run `cd /root/Varptrader/autotrader && docker compose restart autotrader`.\n"
            "6. Write a concise plain-text report with the root cause, the exact change, and test result.\n"
        )

    intro = (
        "You are VarpTrader Overseer performing a DEEP WEEKLY REVIEW with full autonomy."
        if cfg.deep
        else "You are VarpTrader Overseer performing a nightly check."
    )
    tasks = (
        """=== YOUR TASK ===
Work through ALL of the following thoroughly:
1. Full performance audit: For every strategy, compute win rate, avg PnL, max losing streak.
2. Root-cause analysis: WHY is the worst strategy losing? Wrong direction bias? Wrong thresholds?
3. Parameter optimisation: Suggest and apply tighter parameter values. Edit /root/Varptrader/autotrader/config.json.
4. Regime filter audit: Is ADX threshold too tight or loose? Adjust if needed.
5. Signal starvation: If fired:blocked < 1:5, loosen two most aggressive filters.
6. Risk sizing: If max_drawdown > 5%, reduce position_size_pct 20%. If < 1% with 10+ trades, increase 10%.
7. Strategy enable/disable: Disable Sharpe < 0 strategies over 20+ trades.
8. Run full tests: docker exec autotrader python -m pytest tests/ -v
9. Rebuild if code changed: cd /root/Varptrader/autotrader && docker compose restart autotrader
10. Write full markdown weekly report with before/after tables for every change."""
        if cfg.deep
        else """=== YOUR TASK ===
1. Diagnose the single biggest performance problem with evidence (win rate, PnL, counts).
2. Signal starvation: If 0 trades in 24h, identify which filter is blocking and recommend loosening.
3. Auto-disable: If any strategy has 0% win rate over 7+ trades in 7 days, set enabled:false in /root/Varptrader/autotrader/config.json.
4. Loosen filter: If blocked:fired > 10:1, loosen the most aggressive filter in config.json.
5. Run tests: docker exec autotrader python -m pytest tests/ -q
6. Restart if changed: cd /root/Varptrader/autotrader && docker compose restart autotrader
7. Write concise report with key numbers and list every change made."""
    )
    return (
        f"{intro}\n\n"
        f"Trigger reason: {cfg.trigger_reason}\n\n"
        f"{context}\n\n"
        f"{tasks}\n"
    )


def _build_preview(report: str, max_lines: int = 12) -> str:
    lines: list[str] = []
    for raw_line in report.splitlines():
        line = raw_line.rstrip()
        if not line:
            continue
        lines.append(line)
        if len(lines) >= max_lines:
            break
    return "\n".join(lines) if lines else "(no report output)"


def _send_telegram(message: str, env: dict[str, str]) -> None:
    token = env.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = env.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        return
    payload = json.dumps({"chat_id": chat_id, "text": message}).encode("utf-8")
    request = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=10):
        pass


def _report_filename(cfg: HostConfig) -> Path:
    suffix = "_deep" if cfg.deep else ""
    return REPORTS_DIR / f"{_utc_now().strftime('%Y-%m-%d_%H')}{suffix}.txt"


def _build_telegram_summary(cfg: HostConfig, report_file: Path, report: str) -> str:
    mode = "DEEP" if cfg.deep else "NIGHTLY"
    return (
        "OVERSEER UPDATE\n\n"
        f"Trigger: {cfg.trigger_reason}\n"
        f"Mode: {mode}\n"
        f"Report: {report_file}\n\n"
        f"{_build_preview(report)}"
    )


def _build_skip_summary(cfg: HostConfig, reason: str) -> str:
    mode = "DEEP" if cfg.deep else "NIGHTLY"
    return (
        "OVERSEER SKIPPED\n\n"
        f"Trigger: {cfg.trigger_reason}\n"
        f"Mode: {mode}\n"
        f"Reason: {reason}"
    )


def _normalize_claude_error(report: str, auth_mode: str) -> str:
    lowered = report.lower()
    if auth_mode == "login" and "not logged in" in lowered:
        return (
            "[Overseer Error] Claude login mode is enabled, but user "
            f"'{TRADER_USER}' is not logged in. Run `sudo -u {TRADER_USER} -H claude`, "
            "then complete `/login`."
        )
    return report


def _extract_report_from_stdout(stdout: str) -> str:
    """Extract the plain overseer report from CLI stdout."""
    text = stdout.strip()
    if not text:
        return ""

    marker = "OVERSEER REPORT"
    if marker not in text:
        return text

    lines = text.splitlines()
    for index, line in enumerate(lines):
        if marker in line:
            remainder = lines[index + 2 :]
            while remainder and not remainder[0].strip():
                remainder.pop(0)
            return "\n".join(remainder).strip()
    return text


def _resolve_report_file(cfg: HostConfig, started_at: datetime) -> Path | None:
    """Find the report file produced by the current run."""
    candidates: list[Path] = []
    for path in REPORTS_DIR.glob("*.txt"):
        if cfg.deep != path.name.endswith("_deep.txt"):
            continue
        try:
            modified_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        except OSError:
            continue
        if modified_at >= started_at - timedelta(seconds=5):
            candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.stat().st_mtime)


def _host_path(path_str: str) -> Path:
    """Translate a container /app path into the shared host workspace."""
    posix = PurePosixPath(path_str)
    if posix.is_absolute() and posix.parts[:2] == ("/", "app"):
        return WORKDIR / Path(*posix.parts[2:])
    return Path(path_str) if Path(path_str).is_absolute() else WORKDIR / path_str


def _docker_json(cmd: list[str], timeout: int) -> dict[str, Any]:
    """Run a docker exec command that returns a JSON payload."""
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "docker command failed")
    payload = (result.stdout or "").strip()
    if not payload:
        raise RuntimeError("docker command returned empty output")
    return json.loads(payload)


def _docker_copy_to_container(host_path: Path, container_path: str) -> None:
    """Copy a host file into the running autotrader container."""
    result = subprocess.run(
        ["docker", "cp", str(host_path), f"autotrader:{container_path}"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "docker cp failed")


def _prepare_host_run(cfg: HostConfig) -> dict[str, Any]:
    cmd = [
        "docker",
        "exec",
        "autotrader",
        "python",
        "-m",
        "overseer.run_overseer",
        "--host-prepare",
        "--trigger-reason",
        cfg.trigger_reason,
        "--model",
        cfg.model,
    ]
    if cfg.deep:
        cmd.append("--deep")
    return _docker_json(cmd, timeout=120)


def _finalize_host_run(state_file: str, report_file: str) -> dict[str, Any]:
    return _docker_json(
        [
            "docker",
            "exec",
            "autotrader",
            "python",
            "-m",
            "overseer.run_overseer",
            "--host-finalize",
            "--state-file",
            state_file,
            "--report-file",
            report_file,
        ],
        timeout=120,
    )


def _run_overseer_pipeline(cfg: HostConfig, env: dict[str, str]) -> tuple[str, str, Path | None]:
    """Execute the host Claude run with container-side prepare/finalize."""
    prepare_payload = _prepare_host_run(cfg)
    prompt = str(prepare_payload["prompt"])
    state_file = str(prepare_payload["state_file"])
    raw_report_rel = str(prepare_payload["raw_report_file"])
    raw_report_path = _host_path(raw_report_rel)
    raw_report_path.parent.mkdir(parents=True, exist_ok=True)
    PROMPT_FILE.write_text(prompt, encoding="utf-8")
    _append_log(
        f"overseer_context_built | chars={len(prompt)} run_id={prepare_payload['run_id']}"
    )

    state, report = _invoke_claude(prompt, cfg, env)
    raw_report_path.write_text(report + "\n", encoding="utf-8")
    _docker_copy_to_container(raw_report_path, raw_report_rel)

    finalize_payload = _finalize_host_run(state_file=state_file, report_file=raw_report_rel)
    finalized_report = str(finalize_payload.get("report", report))
    report_path = _host_path(str(finalize_payload.get("report_path", raw_report_rel)))
    generated_report_path = WORKDIR / "overseer" / "reports" / f"{prepare_payload['run_id']}_overseer_report.txt"
    if generated_report_path.exists():
        report_path = generated_report_path
        finalized_report = generated_report_path.read_text(encoding="utf-8").strip()
    final_state = "failed" if "[Overseer Error]" in finalized_report else state
    if final_state == "completed" and state != "completed":
        final_state = state
    return final_state, finalized_report, report_path


def _limits_for_today(limits: dict[str, Any]) -> tuple[str, dict[str, int]]:
    today = _utc_now().strftime("%Y-%m-%d")
    daily = limits.setdefault("daily_counts", {})
    bucket = daily.setdefault(today, {})
    bucket.setdefault("total", 0)
    bucket.setdefault("signal_starvation", 0)
    bucket.setdefault("nightly", 0)
    bucket.setdefault("deep", 0)
    for day in list(daily.keys()):
        if day != today:
            daily.pop(day, None)
    return today, bucket


def _admission_guard(cfg: HostConfig) -> tuple[bool, str]:
    limits = _read_limits()
    now = _utc_now()

    low_credit_until = _parse_dt(limits.get("low_credit_until"))
    if low_credit_until and low_credit_until > now:
        return False, f"low credit lockout until {low_credit_until.strftime('%Y-%m-%d %H:%M UTC')}"

    _, today_bucket = _limits_for_today(limits)
    total_cap = max(1, int(os.environ.get("OVERSEER_MAX_RUNS_PER_DAY", "4")))
    starvation_cap = max(1, int(os.environ.get("OVERSEER_MAX_STARVATION_RUNS_PER_DAY", "2")))
    starvation_cooldown_sec = max(0, int(os.environ.get("OVERSEER_STARVATION_COOLDOWN_SEC", str(6 * 3600))))

    if today_bucket["total"] >= total_cap:
        return False, f"daily run cap reached ({today_bucket['total']}/{total_cap})"

    if cfg.deep:
        if today_bucket["deep"] >= 1:
            return False, "deep run already started today"
        return True, ""

    if cfg.trigger_reason == "signal_starvation":
        if today_bucket["signal_starvation"] >= starvation_cap:
            return False, f"signal-starvation daily cap reached ({today_bucket['signal_starvation']}/{starvation_cap})"
        last_started = _parse_dt(limits.get("last_started", {}).get("signal_starvation"))
        if last_started and (now - last_started).total_seconds() < starvation_cooldown_sec:
            next_retry = last_started + timedelta(seconds=starvation_cooldown_sec)
            return False, f"signal-starvation cooldown until {next_retry.strftime('%Y-%m-%d %H:%M UTC')}"

    return True, ""


def _record_run_started(cfg: HostConfig) -> None:
    limits = _read_limits()
    _, today_bucket = _limits_for_today(limits)
    now_iso = _iso_now()
    last_started = limits.setdefault("last_started", {})
    today_bucket["total"] += 1
    if cfg.deep:
        today_bucket["deep"] += 1
        last_started["deep"] = now_iso
    else:
        today_bucket[cfg.trigger_reason] = int(today_bucket.get(cfg.trigger_reason, 0)) + 1
        last_started[cfg.trigger_reason] = now_iso
    _write_limits(limits)


def _record_run_result(cfg: HostConfig, state: str, report: str) -> None:
    if state != "failed":
        return
    if "credit balance is too low" not in report.lower():
        return
    limits = _read_limits()
    lockout_sec = max(3600, int(os.environ.get("OVERSEER_LOW_CREDIT_LOCKOUT_SEC", str(24 * 3600))))
    until = _utc_now() + timedelta(seconds=lockout_sec)
    limits["low_credit_until"] = until.isoformat()
    limits["low_credit_reason"] = "credit balance is too low"
    limits["low_credit_trigger_reason"] = cfg.trigger_reason
    _write_limits(limits)


def _invoke_claude(prompt: str, cfg: HostConfig, env: dict[str, str]) -> tuple[str, str]:
    account = _resolve_trader_account()
    _ensure_trader_home(account)
    auth_mode = _resolve_auth_mode(env)
    cmd = [
        "claude",
        "--print",
        "--output-format",
        "json",
        "--no-session-persistence",
        "--permission-mode",
        "bypassPermissions",
        "--tools",
        "Bash,Read,Edit,Write,Glob,Grep",
        "--effort",
        cfg.effort,
        "--model",
        cfg.model,
        "-p",
        prompt,
    ]
    child_env = _claude_env(
        {"ANTHROPIC_API_KEY": env.get("ANTHROPIC_API_KEY", "")},
        account,
        auth_mode,
    )
    _append_log(
        f"overseer_invoking_claude | model={cfg.model} trigger={cfg.trigger_reason} mode={'deep' if cfg.deep else 'nightly'} timeout={cfg.timeout_sec}s auth_mode={auth_mode}"
    )
    _write_status({
        "state": "running",
        "step": "claude_running",
        "trigger_reason": cfg.trigger_reason,
        "deep": cfg.deep,
        "model": cfg.model,
        "started_at": _timestamp(),
        "pid": os.getpid(),
    })
    try:
        result = subprocess.run(
            cmd,
            cwd=str(WORKDIR),
            capture_output=True,
            text=True,
            timeout=cfg.timeout_sec,
            env=child_env,
            preexec_fn=_drop_privileges(account),
        )
    except subprocess.TimeoutExpired:
        return "timed_out", f"[Overseer Error] Claude timed out after {cfg.timeout_sec}s."

    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    if result.returncode != 0:
        message = stdout or stderr or f"Claude exited {result.returncode}"
        return "failed", _normalize_claude_error(f"[Overseer Error] {message}", auth_mode)
    if not stdout:
        return "failed", "[Overseer Error] Claude returned empty output."
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return "completed", _normalize_claude_error(stdout, auth_mode)

    if payload.get("subtype") != "success" or payload.get("is_error"):
        message = payload.get("result") or stderr or "[Overseer Error] Claude returned an error."
        return "failed", _normalize_claude_error(str(message), auth_mode)

    report = str(payload.get("result") or "").strip()
    if not report:
        return "failed", "[Overseer Error] Claude returned empty result."
    return "completed", _normalize_claude_error(report, auth_mode)


def _build_context() -> str:
    result = subprocess.run(
        ["docker", "exec", "autotrader", "python", "-m", "overseer.run_overseer", "--context-only"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "context build failed")
    context = result.stdout.strip()
    if not context:
        raise RuntimeError("empty context from docker")
    return context


def _run_job(cfg: HostConfig) -> int:
    env = _load_env_file(ENV_FILE)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()), encoding="utf-8")

    _append_log(
        f"overseer_host_started | model={cfg.model} deep={cfg.deep} trigger={cfg.trigger_reason} pid={os.getpid()}"
    )
    _write_status({
        "state": "running",
        "step": "building_context",
        "trigger_reason": cfg.trigger_reason,
        "deep": cfg.deep,
        "model": cfg.model,
        "started_at": _timestamp(),
        "pid": os.getpid(),
    })

    try:
        if cfg.test:
            context = _build_context()
            _append_log(f"overseer_context_built | chars={len(context)}")
            prompt = _build_prompt(context, cfg)
            PROMPT_FILE.write_text(prompt, encoding="utf-8")
            preview = prompt[:3000]
            _append_log("overseer_test_complete")
            print(preview)
            return 0

        state, report, report_file = _run_overseer_pipeline(cfg, env)
        _record_run_result(cfg, state, report)
        if report_file is None:
            report_file = _report_filename(cfg)
            report_file.write_text(report + "\n", encoding="utf-8")
        _append_log(f"overseer_report_saved | path={report_file}")
        _write_status({
            "state": state,
            "step": "done",
            "trigger_reason": cfg.trigger_reason,
            "deep": cfg.deep,
            "model": cfg.model,
            "started_at": _read_status().get("started_at"),
            "finished_at": _timestamp(),
            "pid": os.getpid(),
            "report_file": str(report_file),
            "report_preview": _build_preview(report, max_lines=4),
        })
        return 0 if state == "completed" else 1
    except Exception as exc:
        _append_log(f"overseer_host_error | err={exc}")
        _write_status({
            "state": "failed",
            "step": "error",
            "trigger_reason": cfg.trigger_reason,
            "deep": cfg.deep,
            "model": cfg.model,
            "started_at": _read_status().get("started_at"),
            "finished_at": _timestamp(),
            "pid": os.getpid(),
            "error": str(exc),
        })
        return 1
    finally:
        if PID_FILE.exists():
            PID_FILE.unlink()


def _start_background(cfg: HostConfig) -> int:
    current_pid = _current_pid()
    if _pid_running(current_pid):
        status = _read_status()
        print(json.dumps({
            "state": "already_running",
            "pid": current_pid,
            "status": status,
        }))
        return 0
    if current_pid and not _pid_running(current_pid):
        PID_FILE.unlink(missing_ok=True)
        status = _read_status()
        if status.get("state") in {"queued", "running"}:
            status["state"] = "stale"
            status["finished_at"] = _timestamp()
            _write_status(status)

    allowed, reason = _admission_guard(cfg)
    if not allowed:
        env = _load_env_file(ENV_FILE)
        _append_log(
            f"overseer_start_skipped | trigger={cfg.trigger_reason} deep={cfg.deep} reason={reason}"
        )
        _write_status({
            "state": "skipped",
            "step": "admission_guard",
            "trigger_reason": cfg.trigger_reason,
            "deep": cfg.deep,
            "model": cfg.model,
            "finished_at": _timestamp(),
            "reason": reason,
        })
        try:
            _send_telegram(_build_skip_summary(cfg, reason), env)
        except Exception as exc:  # pragma: no cover - best effort on host
            _append_log(f"overseer_skip_telegram_error | err={exc}")
        print(json.dumps({
            "state": "skipped",
            "trigger_reason": cfg.trigger_reason,
            "deep": cfg.deep,
            "reason": reason,
        }))
        return 0

    _record_run_started(cfg)

    args = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--model",
        cfg.model,
        "--trigger-reason",
        cfg.trigger_reason,
        "--timeout-sec",
        str(cfg.timeout_sec),
        "--effort",
        cfg.effort,
    ]
    if cfg.deep:
        args.append("--deep")
    if cfg.test:
        args.append("--test")

    proc = subprocess.Popen(
        args,
        cwd=str(ROOT_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    _write_status({
        "state": "queued",
        "step": "spawned",
        "trigger_reason": cfg.trigger_reason,
        "deep": cfg.deep,
        "model": cfg.model,
        "spawned_at": _timestamp(),
        "pid": proc.pid,
    })
    print(json.dumps({
        "state": "queued",
        "pid": proc.pid,
        "trigger_reason": cfg.trigger_reason,
        "deep": cfg.deep,
        "model": cfg.model,
    }))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="VarpTrader host overseer runner")
    parser.add_argument("--deep", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--trigger-reason", default="nightly")
    parser.add_argument("--model", default=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"))
    env_timeout = os.environ.get("CLAUDE_TIMEOUT_SEC", "").strip()
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=int(env_timeout) if env_timeout else None,
    )
    parser.add_argument("--effort", choices=["low", "medium", "high", "max"])
    parser.add_argument("--foreground", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    if args.status:
        status = _read_status()
        pid = status.get("pid")
        if status.get("state") in {"queued", "running"} and not _pid_running(pid):
            status["state"] = "stale"
            status["finished_at"] = _timestamp()
            _write_status(status)
        print(json.dumps(status, indent=2))
        return 0

    cfg = HostConfig(
        model=args.model,
        trigger_reason=args.trigger_reason,
        deep=args.deep,
        timeout_sec=args.timeout_sec if args.timeout_sec is not None else (1800 if args.deep else 600),
        effort=args.effort or ("medium" if args.deep else "low"),
        test=args.test,
    )

    if args.worker or args.foreground:
        return _run_job(cfg)
    return _start_background(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
