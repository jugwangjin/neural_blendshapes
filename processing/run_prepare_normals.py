"""Legacy entry — use ``python -m processing.face_normals.run_prepare_normals``."""

import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
runpy.run_module("processing.face_normals.run_prepare_normals", run_name="__main__")
