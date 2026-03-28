"""Tests for config_updater bound validation and atomic writes."""

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, ".")

from analysis.config_updater import ConfigUpdater


@pytest.fixture
def sample_config() -> dict:
    """Provide a baseline config matching config.json structure."""
    return {
        "risk": {
            "stop_loss_pct": 0.015,
            "position_size_pct": 0.02,
        },
        "strategies": {
            "rsi_momentum": {
                "rsi_oversold": 32,
                "rsi_overbought": 68,
            },
        },
        "bounds": {
            "stop_loss_pct": {"min": 0.005, "max": 0.05},
            "position_size_pct": {"min": 0.005, "max": 0.03},
            "rsi_oversold": {"min": 20, "max": 40},
            "rsi_overbought": {"min": 60, "max": 80},
        },
    }


@pytest.fixture
def config_file(sample_config: dict, tmp_path: Path) -> Path:
    """Write sample config to a temp file and return the path."""
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(sample_config, indent=2))
    return config_path


@pytest.fixture
def updater(config_file: Path) -> ConfigUpdater:
    """Create a ConfigUpdater instance with a temp config file."""
    return ConfigUpdater(str(config_file))


# ===========================================================================
# Bound validation tests
# ===========================================================================

class TestBoundValidation:
    """Ensure config changes are validated against hard-coded bounds."""

    def test_accept_stop_loss_within_bounds(self, updater: ConfigUpdater) -> None:
        """stop_loss_pct=0.02 is within [0.005, 0.05] — should be accepted."""
        changes = {"stop_loss_pct": 0.02}
        approved, rejected = updater.validate_changes(changes)
        assert "stop_loss_pct" in approved
        assert len(rejected) == 0

    def test_reject_stop_loss_below_min(self, updater: ConfigUpdater) -> None:
        """stop_loss_pct=0.001 is below min 0.005 — should be rejected."""
        changes = {"stop_loss_pct": 0.001}
        approved, rejected = updater.validate_changes(changes)
        assert "stop_loss_pct" not in approved
        assert len(rejected) == 1

    def test_reject_stop_loss_above_max(self, updater: ConfigUpdater) -> None:
        """stop_loss_pct=0.10 is above max 0.05 — should be rejected."""
        changes = {"stop_loss_pct": 0.10}
        approved, rejected = updater.validate_changes(changes)
        assert "stop_loss_pct" not in approved
        assert len(rejected) == 1

    def test_accept_position_size_within_bounds(self, updater: ConfigUpdater) -> None:
        """position_size_pct=0.01 is within [0.005, 0.03]."""
        changes = {"position_size_pct": 0.01}
        approved, rejected = updater.validate_changes(changes)
        assert "position_size_pct" in approved

    def test_reject_position_size_too_large(self, updater: ConfigUpdater) -> None:
        """position_size_pct=0.05 exceeds max 0.03."""
        changes = {"position_size_pct": 0.05}
        approved, rejected = updater.validate_changes(changes)
        assert "position_size_pct" not in approved
        assert len(rejected) == 1

    def test_accept_rsi_oversold_within_bounds(self, updater: ConfigUpdater) -> None:
        """rsi_oversold=25 is within [20, 40]."""
        changes = {"rsi_oversold": 25}
        approved, rejected = updater.validate_changes(changes)
        assert "rsi_oversold" in approved

    def test_reject_rsi_oversold_below_min(self, updater: ConfigUpdater) -> None:
        """rsi_oversold=15 is below min 20."""
        changes = {"rsi_oversold": 15}
        approved, rejected = updater.validate_changes(changes)
        assert "rsi_oversold" not in approved

    def test_reject_rsi_overbought_above_max(self, updater: ConfigUpdater) -> None:
        """rsi_overbought=85 exceeds max 80."""
        changes = {"rsi_overbought": 85}
        approved, rejected = updater.validate_changes(changes)
        assert "rsi_overbought" not in approved

    def test_accept_rsi_overbought_at_boundary(self, updater: ConfigUpdater) -> None:
        """rsi_overbought=80 is exactly at max — should be accepted."""
        changes = {"rsi_overbought": 80}
        approved, rejected = updater.validate_changes(changes)
        assert "rsi_overbought" in approved

    def test_accept_rsi_oversold_at_boundary(self, updater: ConfigUpdater) -> None:
        """rsi_oversold=20 is exactly at min — should be accepted."""
        changes = {"rsi_oversold": 20}
        approved, rejected = updater.validate_changes(changes)
        assert "rsi_oversold" in approved

    def test_mixed_valid_and_invalid_changes(self, updater: ConfigUpdater) -> None:
        """Some changes valid, others not — only valid ones approved."""
        changes = {
            "stop_loss_pct": 0.02,      # valid
            "position_size_pct": 0.10,  # invalid
            "rsi_oversold": 30,         # valid
            "rsi_overbought": 90,       # invalid
        }
        approved, rejected = updater.validate_changes(changes)
        assert "stop_loss_pct" in approved
        assert "rsi_oversold" in approved
        assert "position_size_pct" not in approved
        assert "rsi_overbought" not in approved
        assert len(rejected) == 2

    def test_unknown_keys_are_rejected(self, updater: ConfigUpdater) -> None:
        """Keys not in bounds dict should be rejected."""
        changes = {"unknown_param": 42}
        approved, rejected = updater.validate_changes(changes)
        assert "unknown_param" not in approved
        assert len(rejected) == 1


# ===========================================================================
# Atomic write tests
# ===========================================================================

class TestAtomicConfigWrite:
    """Ensure config is written atomically (temp file → rename)."""

    def test_config_file_updated_after_apply(self, updater: ConfigUpdater, config_file: Path) -> None:
        """Config file on disk should reflect approved changes after apply."""
        changes = {"stop_loss_pct": 0.03}
        approved, _ = updater.validate_changes(changes)
        updater.apply_changes(approved)
        updated = json.loads(config_file.read_text())
        assert updated["risk"]["stop_loss_pct"] == 0.03

    def test_original_preserved_on_no_changes(self, updater: ConfigUpdater, config_file: Path) -> None:
        """Config file unchanged when no changes are approved."""
        original = config_file.read_text()
        updater.apply_changes({})
        assert config_file.read_text() == original

    def test_rsi_changes_written_to_strategies_section(self, updater: ConfigUpdater, config_file: Path) -> None:
        """RSI parameter changes should update the strategies section."""
        changes = {"rsi_oversold": 28, "rsi_overbought": 72}
        approved, _ = updater.validate_changes(changes)
        updater.apply_changes(approved)
        updated = json.loads(config_file.read_text())
        assert updated["strategies"]["rsi_momentum"]["rsi_oversold"] == 28
        assert updated["strategies"]["rsi_momentum"]["rsi_overbought"] == 72

    def test_concurrent_writes_dont_corrupt(self, updater: ConfigUpdater, config_file: Path) -> None:
        """Sequential rapid writes should not corrupt the config file."""
        for i in range(10):
            val = 0.005 + (i * 0.005)
            val = min(val, 0.05)
            approved = {"stop_loss_pct": val}
            updater.apply_changes(approved)
        result = json.loads(config_file.read_text())
        assert 0.005 <= result["risk"]["stop_loss_pct"] <= 0.05

    def test_bind_mount_fallback_overwrites_in_place(
        self, updater: ConfigUpdater, config_file: Path
    ) -> None:
        """EBUSY on rename should fall back to an in-place overwrite."""
        approved = {"stop_loss_pct": 0.03}

        with patch(
            "analysis.config_updater.os.replace",
            side_effect=OSError(16, "Device or resource busy"),
        ):
            updater.apply_changes(approved)

        updated = json.loads(config_file.read_text())
        assert updated["risk"]["stop_loss_pct"] == 0.03
