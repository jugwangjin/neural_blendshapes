"""Copy training code snapshot + config/stage state at run start."""

import json
import shutil
from collections import Counter
from dataclasses import asdict, fields, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Whole packages (recursive; primary source of truth after train/losses/model refactor).
SNAPSHOT_ROOTS = (
    "training",
    "model",
    "losses",
    "rendering",
    "utils",
    "dataset",
    "tests",
)

# Implementation notes tied to the training stack (not user-facing guides).
SNAPSHOT_DOC_DIRS = (
    "docs/implementation",
)

SNAPSHOT_TOPLEVEL_FILES = (
    "config.py",
    "train.py",
    "train_time_measure.py",
    "run_config.py",
    "run_trains.py",
    "sweep_params.py",
    "sweep_configs.py",
    "render_control_video.py",
    "bootstrap_paths.py",
)

SNAPSHOT_EXTS = {".py", ".json", ".yaml", ".yml", ".txt", ".md"}

SNAPSHOT_EXCLUDE_DIR_NAMES = {
    "__pycache__",
    ".git",
    ".cursor",
    "outputs",
    "checkpoints",
    "renders",
    "codes",
    "analysis",
}


def _is_excluded_path(rel: Path) -> bool:
    return any(part in SNAPSHOT_EXCLUDE_DIR_NAMES for part in rel.parts)


def _add_tree_files(root: Path, base: Path, rels: set[str]):
    if not base.is_dir():
        return
    for p in base.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in SNAPSHOT_EXTS:
            continue
        rel = p.relative_to(root)
        if _is_excluded_path(rel):
            continue
        rels.add(str(rel).replace("\\", "/"))


def _iter_snapshot_rel_paths(root: Path) -> list[str]:
    rels: set[str] = set()

    for rel in SNAPSHOT_TOPLEVEL_FILES:
        if (root / rel).is_file():
            rels.add(rel)

    for folder in SNAPSHOT_ROOTS:
        _add_tree_files(root, root / folder, rels)

    for folder in SNAPSHOT_DOC_DIRS:
        _add_tree_files(root, root / folder, rels)

    return sorted(rels)


def _json_default(obj: Any):
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serializable: {type(obj)}")


def _stage_spec_dict(spec) -> dict:
    if is_dataclass(spec):
        return {f.name: getattr(spec, f.name) for f in fields(spec)}
    return {"repr": repr(spec)}


def dump_training_code(root: Path, out_dir: Path, cfg, schedule) -> Path:
    """
    Mirror training-related sources under ``out_dir`` and write ``config.json``,
    ``STAGE_SCHEDULE.json``, ``dump_meta.json``.
    """
    root = Path(root).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    snapshot_rel_paths = _iter_snapshot_rel_paths(root)
    copied = []
    missing = []
    for rel in snapshot_rel_paths:
        src = root / rel
        if not src.is_file():
            missing.append(rel)
            continue
        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(rel)

    cfg_path = out_dir / "config.json"
    cfg_snapshot = asdict(cfg)
    cfg_snapshot["checkpoint_dir"] = str(cfg.checkpoint_dir)
    cfg_snapshot["codes_dir"] = str(cfg.codes_dir)
    cfg_snapshot["eval_render_dir"] = str(cfg.eval_render_dir)
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg_snapshot, f, indent=2, default=_json_default)

    stages_path = out_dir / "STAGE_SCHEDULE.json"
    with open(stages_path, "w", encoding="utf-8") as f:
        json.dump([_stage_spec_dict(s) for s in schedule], f, indent=2, default=_json_default)

    by_prefix = Counter(rel.split("/", 1)[0] for rel in copied)
    meta = {
        "root": str(root),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n_copied": len(copied),
        "n_missing": len(missing),
        "by_top_level": dict(sorted(by_prefix.items())),
        "copied": copied,
        "missing": missing,
    }
    with open(out_dir / "dump_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"code dump: {out_dir} ({len(copied)} files, {len(missing)} missing)")
    if missing:
        print(f"  missing: {', '.join(missing)}")
    return out_dir
