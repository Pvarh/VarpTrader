"""Pytest configuration for autotrader tests.

Pre-loads the ``overseer_host`` module from within the autotrader package so
that test_overseer_host.py can import it even when the parent directory
(/root/Varptrader) is not writable at test time.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

# Ensure autotrader root is on the path before any test module is collected.
_AUTOTRADER_ROOT = Path(__file__).resolve().parents[1]
if str(_AUTOTRADER_ROOT) not in sys.path:
    sys.path.insert(0, str(_AUTOTRADER_ROOT))

# Pre-register overseer_host so that test_overseer_host.py's top-level
# ``from overseer_host import …`` resolves to our local copy regardless of
# the sys.path manipulation the test file performs itself.
if "overseer_host" not in sys.modules:
    import overseer_host  # noqa: F401 — side-effect import to register module
