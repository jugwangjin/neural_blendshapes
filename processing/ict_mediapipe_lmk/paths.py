"""Re-export shared bootstrap (use processing.paths)."""

from processing.paths import REPO_ROOT, PROCESSING_ROOT, setup_import_paths

__all__ = ["REPO_ROOT", "PROCESSING_ROOT", "setup_import_paths"]
