"""Repo-root import bootstrap (for scripts under project root)."""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
_PROCESSING_ROOT = _REPO_ROOT / "processing"
for _root in (_REPO_ROOT, _PROCESSING_ROOT):
    _p = str(_root)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from processing.paths import (
    ASSETS_DIR,
    FLAME_MODEL,
    FLAME_STATIC_EMBEDDING,
    FLAME_UV_MESH,
    ICT_CANONICAL,
    ICT_NPY,
    PROCESSING_ROOT,
    REPO_ROOT,
    setup_import_paths,
)

__all__ = [
    "REPO_ROOT",
    "PROCESSING_ROOT",
    "ASSETS_DIR",
    "FLAME_MODEL",
    "FLAME_STATIC_EMBEDDING",
    "FLAME_UV_MESH",
    "ICT_NPY",
    "ICT_CANONICAL",
    "setup_import_paths",
]
