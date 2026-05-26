"""Copy core training code + config/stage snapshot at run start."""

import json
import shutil
from dataclasses import asdict, fields, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


CORE_CODE_REL_PATHS = [
    "config.py",
    "train.py",
    "training/stages.py",
    "training/apply.py",
    "training/code_dump.py",
    "training/checkpoint_io.py",
    "training/eval_render.py",
    "model/ict_model.py",
    "model/ict_deformer.py",
    "model/gaussian_avatar.py",
    "model/tracker_mlp.py",
    "model/expr_regions.py",
    "model/blendshape_support.py",
    "model/pose_weight.py",
    "losses/train_losses.py",
    "losses/pie68_jaw_landmark.py",
    "losses/mediapipe_landmark_478.py",
    "losses/deformer_regularization.py",
    "dataset/video_dataset.py",
    "dataset/image_dataset.py",
    "dataset/collate.py",
    "dataset/frame_processor.py",
    "dataset/mediapipe_cache.py",
    "rendering/__init__.py",
    "rendering/avatar_renderer.py",
    "rendering/pack.py",
    "rendering/gsplat_camera.py",
    "utils/mediapipe_blendshapes.py",
    "utils/tracker.py",
    "utils/camera.py",
    "utils/mesh_ops.py",
    "utils/sampling.py",
]


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
    Mirror ``CORE_CODE_REL_PATHS`` under ``out_dir`` and write ``config.json`` +
    ``STAGE_SCHEDULE.json``.
    """
    root = Path(root).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    missing = []
    for rel in CORE_CODE_REL_PATHS:
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

    meta = {
        "root": str(root),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "copied": copied,
        "missing": missing,
    }
    with open(out_dir / "dump_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"code dump: {out_dir} ({len(copied)} files, {len(missing)} missing)")
    if missing:
        print(f"  missing: {', '.join(missing)}")
    return out_dir
