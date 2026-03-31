"""Validate and safely apply configuration changes.

Primary write mode is atomic write-to-temp + rename. On bind-mounted
single-file volumes where ``os.replace`` can fail with ``EBUSY`` or
similar, the updater falls back to an in-place overwrite.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

from loguru import logger


class ConfigUpdater:
    """Validates proposed config changes against safety bounds and applies them.

    Bounds are read from the ``"bounds"`` key in the config JSON file at
    initialization time.  If the key is missing, sensible defaults are used.
    """

    # Default bounds used when the config file does not contain a "bounds" key.
    _DEFAULT_BOUNDS: dict[str, dict[str, float]] = {
        "stop_loss_pct": {"min": 0.005, "max": 0.05},
        "position_size_pct": {"min": 0.005, "max": 0.03},
        "rsi_oversold": {"min": 20, "max": 40},
        "rsi_overbought": {"min": 60, "max": 80},
    }

    # Maps parameter names to their nested location inside config.json.
    PARAM_PATHS: dict[str, list[str]] = {
        "stop_loss_pct": ["risk", "stop_loss_pct"],
        "position_size_pct": ["risk", "position_size_pct"],
        "rsi_oversold": ["strategies", "rsi_momentum", "rsi_oversold"],
        "rsi_overbought": ["strategies", "rsi_momentum", "rsi_overbought"],
    }

    def __init__(self, config_path: str) -> None:
        """Initialize with path to the config JSON file.

        Reads bounds from ``config["bounds"]`` if present, otherwise
        falls back to built-in defaults.

        Args:
            config_path: Filesystem path to ``config.json``.
        """
        self.config_path = Path(config_path)
        self._bounds: dict[str, dict[str, float]] = self._load_bounds()
        logger.info(
            "config_updater_initialized | config_path={config_path} bounds_keys={bounds_keys}",
            config_path=str(self.config_path),
            bounds_keys=list(self._bounds.keys()),
        )

    def _load_bounds(self) -> dict[str, dict[str, float]]:
        """Load bounds from the config file's ``"bounds"`` key.

        Returns:
            Bounds dictionary.  Falls back to ``_DEFAULT_BOUNDS`` on
            any error or if the key is missing.
        """
        try:
            with open(self.config_path, "r", encoding="utf-8") as fh:
                cfg: dict = json.load(fh)
            bounds = cfg.get("bounds")
            if isinstance(bounds, dict) and bounds:
                logger.info("bounds_loaded_from_config | count={count}", count=len(bounds))
                return bounds
            logger.info("bounds_key_missing_or_empty_using_defaults")
            return dict(self._DEFAULT_BOUNDS)
        except (OSError, json.JSONDecodeError):
            logger.warning("bounds_load_failed_using_defaults | config_path={config_path}", config_path=str(self.config_path))
            return dict(self._DEFAULT_BOUNDS)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_changes(
        self, changes: dict[str, Any]
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Validate each proposed change against bounds.

        Args:
            changes: Mapping of parameter name to proposed new value.

        Returns:
            A 2-tuple of ``(approved, rejected)`` where *approved* is a dict
            of accepted changes and *rejected* is a list of dicts each
            containing ``"param"``, ``"value"``, and ``"reason"`` keys.
        """
        approved: dict[str, Any] = {}
        rejected: list[dict[str, Any]] = []

        for param, value in changes.items():
            if param not in self._bounds:
                rejected.append({
                    "param": param,
                    "value": value,
                    "reason": f"unknown parameter: {param}",
                })
                logger.warning(
                    "config_change_rejected | param={param} value={value} reason=unknown parameter",
                    param=param,
                    value=value,
                )
                continue

            bounds = self._bounds[param]
            lo = bounds["min"]
            hi = bounds["max"]

            if value < lo:
                rejected.append({
                    "param": param,
                    "value": value,
                    "reason": f"value {value} below minimum {lo}",
                })
                logger.warning(
                    "config_change_rejected | param={param} value={value} reason=below minimum min={min_val}",
                    param=param,
                    value=value,
                    min_val=lo,
                )
            elif value > hi:
                rejected.append({
                    "param": param,
                    "value": value,
                    "reason": f"value {value} above maximum {hi}",
                })
                logger.warning(
                    "config_change_rejected | param={param} value={value} reason=above maximum max={max_val}",
                    param=param,
                    value=value,
                    max_val=hi,
                )
            else:
                approved[param] = value
                logger.info(
                    "config_change_approved | param={param} value={value}",
                    param=param,
                    value=value,
                )

        logger.info(
            "config_validation_complete | approved_count={approved_count} rejected_count={rejected_count}",
            approved_count=len(approved),
            rejected_count=len(rejected),
        )
        return approved, rejected

    # ------------------------------------------------------------------
    # Atomic write
    # ------------------------------------------------------------------

    @staticmethod
    def _set_nested(cfg: dict, path: list[str], value: Any) -> None:
        """Set a value in a nested dict following *path* keys.

        Args:
            cfg: The config dictionary to mutate in-place.
            path: List of successive keys leading to the target leaf.
            value: Value to set at the leaf.
        """
        node = cfg
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = value

    def apply_changes(self, approved: dict[str, Any]) -> None:
        """Write approved changes to config.json atomically.

        Reads the current config, updates the nested paths for each
        approved parameter, writes to a temporary file in the same
        directory, then renames it over the original to guarantee
        atomicity on the same filesystem.

        Does nothing when *approved* is empty.

        Uses a process-wide lock shared with ``load_config`` to prevent
        race conditions when the config hot-reloader runs concurrently.

        Args:
            approved: Mapping of parameter name to new value (already validated).
        """
        if not approved:
            logger.info("apply_changes_skipped | reason=no approved changes")
            return

        # Import the shared config lock from main to prevent concurrent reads
        try:
            from main import _config_lock
        except ImportError:
            _config_lock = threading.Lock()

        # Read current config
        try:
            with _config_lock, open(self.config_path, "r", encoding="utf-8") as fh:
                cfg: dict = json.load(fh)
        except (OSError, json.JSONDecodeError):
            logger.exception("config_read_failed | path={path}", path=str(self.config_path))
            raise

        # Apply each approved change to the nested structure
        for param, value in approved.items():
            path = self.PARAM_PATHS.get(param)
            if path is None:
                logger.warning(
                    "apply_changes_unknown_path | param={param} reason=no PARAM_PATHS entry",
                    param=param,
                )
                continue
            old_value = cfg
            for key in path:
                old_value = old_value.get(key, {}) if isinstance(old_value, dict) else None
            self._set_nested(cfg, path, value)
            logger.info(
                "config_param_updated | param={param} old_value={old_value} new_value={new_value}",
                param=param,
                old_value=old_value,
                new_value=value,
            )

        # Atomic write: write to temp file in same directory, then rename
        config_dir = self.config_path.parent
        tmp_path = ""
        with _config_lock:
            try:
                fd, tmp_path = tempfile.mkstemp(
                    dir=str(config_dir), suffix=".tmp", prefix=".config_"
                )
                with os.fdopen(fd, "w", encoding="utf-8") as tmp_fh:
                    json.dump(cfg, tmp_fh, indent=2)
                    tmp_fh.write("\n")
                os.replace(tmp_path, str(self.config_path))
            except OSError as exc:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

                if exc.errno in {16, 18, 1, 13}:
                    logger.warning(
                        "config_atomic_write_fallback_in_place | path={path} errno={errno}",
                        path=str(self.config_path),
                        errno=exc.errno,
                    )
                    with open(self.config_path, "w", encoding="utf-8") as fh:
                        json.dump(cfg, fh, indent=2)
                        fh.write("\n")
                else:
                    logger.exception(
                        "config_atomic_write_failed | path={path}",
                        path=str(self.config_path),
                    )
                    raise

        logger.info(
            "config_changes_applied | params={params} config_path={config_path}",
            params=list(approved.keys()),
            config_path=str(self.config_path),
        )
