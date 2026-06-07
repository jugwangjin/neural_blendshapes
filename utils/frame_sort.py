"""Numeric sort for image sequence filenames (``1`` … ``10`` vs ``00001`` …)."""

from __future__ import annotations

import re
from pathlib import Path

_NUMERIC_STEM = re.compile(r"^(\d+)$")
_SUFFIX_NUM = re.compile(r"^(.*?)(\d+)$")


def frame_sort_key(path: Path) -> tuple:
    """
    Sort key for frame files.

    - ``1.png``, ``00010.png`` → numeric stem (1 < 2 < 10)
    - ``f_12.png``, ``frame_3.png`` → prefix + trailing number
    - otherwise → lexicographic stem fallback
    """
    stem = Path(path).stem
    m = _NUMERIC_STEM.fullmatch(stem)
    if m is not None:
        return (0, int(m.group(1)), "")
    m = _SUFFIX_NUM.fullmatch(stem)
    if m is not None and m.group(1):
        return (1, m.group(1), int(m.group(2)))
    return (2, stem)


def sort_frame_paths(paths: list[Path]) -> list[Path]:
    return sorted(paths, key=frame_sort_key)
