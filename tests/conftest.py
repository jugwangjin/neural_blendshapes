"""Ensure repo root is on sys.path (``config``, ``model``, …)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
root = str(ROOT)
if root not in sys.path:
    sys.path.insert(0, root)
