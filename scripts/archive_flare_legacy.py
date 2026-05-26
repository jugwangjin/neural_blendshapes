#!/usr/bin/env python3
"""
Move FLARE-era scripts/configs under legacy/flare/. Run once from repo root (server/WSL):

  python scripts/archive_flare_legacy.py
"""

from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
FLARE = ROOT / "legacy" / "flare"

# Already moved manually (skip if missing): arguments.py, visualize_texture.py

ROOT_SCRIPTS = [
    "test.py",
    "load_nbshapes.py",
    "gui_by_facs.py",
    "gui_by_facs_mesh_viewer.py",
    "track_video.py",
    "track_video_runningmode.py",
    "draw_mediapipe.py",
    "run_trains.py",
    "arguments.py",
    "average_dataset_cameras.py",
    "save_ict_blendshapes.py",
    "optimize_ict_expression_to_flame.py",
    "visualize_texture.py",
]

PROCESSING_SCRIPTS = [
    "save_canonical_pose.py",
    "optimize_ict_expression_to_flame.py",
    "nicp_from_ict_to_flame.py",
    "estimate_facs.py",
    "merge_facs_to_dataset.py",
    "prepare_normals.py",
    "run_prepare_normals.py",
]

UTILS_REMOVE = [
    "utils/io.py",
    "utils/gaze_uv.py",
    "utils/eye_uv_sampling.py",
    "utils/texture_spaces.py",
    "utils/geometry.py",
]


def move(src: Path, dst: Path):
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        print(f"skip (exists): {dst.relative_to(ROOT)}")
        return False
    shutil.move(str(src), str(dst))
    print(f"mv {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")
    return True


def main():
    (FLARE / "scripts").mkdir(parents=True, exist_ok=True)
    (FLARE / "processing").mkdir(parents=True, exist_ok=True)
    (FLARE / "losses").mkdir(parents=True, exist_ok=True)
    (FLARE / "tools").mkdir(parents=True, exist_ok=True)

    n = 0
    for name in ROOT_SCRIPTS:
        if move(ROOT / name, FLARE / "scripts" / name):
            n += 1

    setup = ROOT / "scripts" / "setup_project_layout.py"
    if setup.exists():
        if move(setup, FLARE / "scripts" / "setup_project_layout.py"):
            n += 1

    for name in PROCESSING_SCRIPTS:
        if move(ROOT / "processing" / name, FLARE / "processing" / name):
            n += 1

    lm = ROOT / "losses" / "legacy_landmark_68.py"
    if move(lm, FLARE / "losses" / "legacy_landmark_68.py"):
        n += 1

    sanity = ROOT / "utils" / "sanity_region_colors.py"
    if move(sanity, FLARE / "tools" / "sanity_region_colors.py"):
        n += 1

    for d in ("configs", "configs_tmp"):
        src = ROOT / d
        if src.is_dir():
            dst = FLARE / d
            if not dst.exists():
                shutil.move(str(src), str(dst))
                print(f"mv {d}/ -> legacy/flare/{d}/")
                n += 1

    for rel in UTILS_REMOVE:
        p = ROOT / rel
        if p.is_file():
            p.unlink()
            print(f"rm {rel}")
            n += 1

    readme = FLARE / "README.md"
    if not readme.exists():
        readme.write_text(
            "# FLARE / neural-shader era (archived)\n\n"
            "Requires old `flare` package, `flame.FLAME`, `nvdiffrec`, `arguments.py`.\n"
            "Active training: repo root `train.py` only.\n\n"
            "See `legacy/README.md`.\n",
            encoding="utf-8",
        )

    print(f"done ({n} operations)")


if __name__ == "__main__":
    main()
