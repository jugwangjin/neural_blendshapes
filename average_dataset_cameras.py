"""
Read cameras from flare-style datasets (flame_params.json / merged_params.json)
and print per-scene / global averages. Optionally bake defaults to assets/.

Saved assets/default_camera.npz keys:
  intrinsics_norm (4,), K (3,3), R (3,3), t (3,), s (1,), center (3,), resolution (2,)
"""

import argparse
import json
from pathlib import Path

import numpy as np

from flare.dataset.dataset_util import _load_K_Rt_from_P
from configs_tmp.load_subjects import (
    DEFAULT_FLARE2_ROOT,
    load_all_subjects,
    resolve_data_path,
)

DEFAULT_SUBJECT = "marcel"
DEFAULT_CAMERA_NPZ = Path("assets/default_camera.npz")
DEFAULT_CAMERA_TXT = Path("assets/default_camera.txt")


def find_scene_json_files(dataset_root, scene_filter=None):
    dataset_root = Path(dataset_root)
    scenes = {}

    if scene_filter:
        for name in scene_filter:
            scene_dir = dataset_root / name
            if not scene_dir.is_dir():
                continue
            for json_name in ("flame_params.json", "merged_params.json"):
                json_path = scene_dir / json_name
                if json_path.exists():
                    scenes[scene_dir] = json_path
                    break

    for json_name in ("flame_params.json", "merged_params.json"):
        for path in sorted(dataset_root.rglob(json_name)):
            scene_dir = path.parent
            if scene_dir in scenes:
                continue
            if scene_filter is not None and scene_dir.name not in scene_filter:
                continue
            scenes[scene_dir] = path

    return [(d.name, d, scenes[d]) for d in sorted(scenes, key=lambda p: str(p))]


def scene_resolution(scene_dir, frames):
    import imageio.v2 as imageio

    for frame in frames:
        rel = frame.get("file_path", "")
        if not rel:
            continue
        rel_path = Path(rel.lstrip("./"))
        for candidate in (
            scene_dir / rel_path,
            scene_dir / f"{rel_path}.png",
            scene_dir / "image" / f"{Path(rel).name}.png",
        ):
            if candidate.suffix != ".png":
                candidate = candidate.with_suffix(".png")
            if candidate.exists():
                img = imageio.imread(candidate)
                return img.shape[0], img.shape[1]
    return None, None


def parse_cameras(scene_dir, json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    frames = data.get("frames", [])
    if not frames:
        return None

    h, w = scene_resolution(scene_dir, frames)
    if h is None:
        raise FileNotFoundError(f"No image found to infer resolution under {scene_dir}")

    focal_cxcy = data["intrinsics"]
    k_list = []
    r_list = []
    t_list = []
    center_list = []

    for frame in frames:
        world_mat = np.array(frame["world_mat"], dtype=np.float32)
        _, pose = _load_K_Rt_from_P(None, world_mat)
        r = pose[:3, :3].copy()
        r *= -1
        t = pose[:3, 3].copy()

        k = np.eye(3, dtype=np.float64)
        k[0, 0] = focal_cxcy[0] * w
        k[1, 1] = focal_cxcy[1] * h
        k[0, 2] = focal_cxcy[2] * w
        k[1, 2] = focal_cxcy[3] * h

        center = -r.T @ t

        k_list.append(k)
        r_list.append(r)
        t_list.append(t)
        center_list.append(center)

    intr = np.array(focal_cxcy, dtype=np.float64)
    return {
        "num_frames": len(frames),
        "resolution": (h, w),
        "intrinsics_norm": np.tile(intr[None], (len(frames), 1)),
        "K": np.stack(k_list),
        "R": np.stack(r_list),
        "t": np.stack(t_list),
        "center": np.stack(center_list),
    }


def summarize_cameras(cam_pack):
    k_mean = cam_pack["K"].mean(axis=0)
    r_mean = cam_pack["R"].mean(axis=0)
    t_mean = cam_pack["t"].mean(axis=0)
    center_mean = cam_pack["center"].mean(axis=0)
    intr_mean = cam_pack["intrinsics_norm"].mean(axis=0)

    return {
        "num_frames": cam_pack["num_frames"],
        "resolution": cam_pack["resolution"],
        "intrinsics_norm_mean": intr_mean,
        "K_mean": k_mean,
        "R_mean": r_mean,
        "t_mean": t_mean,
        "center_mean": center_mean,
        "K_std": cam_pack["K"].std(axis=0),
        "t_std": cam_pack["t"].std(axis=0),
        "center_std": cam_pack["center"].std(axis=0),
    }


def summary_to_default_params(summary):
    h, w = summary["resolution"]
    r = summary["R_mean"].astype(np.float32)
    t = summary["t_mean"].astype(np.float32)
    return {
        "intrinsics_norm": summary["intrinsics_norm_mean"].astype(np.float32),
        "K": summary["K_mean"].astype(np.float32),
        "R": r,
        "t": t,
        "s": np.array([1.0], dtype=np.float32),
        "center": summary["center_mean"].astype(np.float32),
        "resolution": np.array([h, w], dtype=np.int32),
        "num_frames": np.array([summary["num_frames"]], dtype=np.int32),
    }


def print_summary(label, summary):
    h, w = summary["resolution"]
    intr = summary["intrinsics_norm_mean"]
    k = summary["K_mean"]
    r = summary["R_mean"]
    t = summary["t_mean"]
    c = summary["center_mean"]

    print(f"\n{'=' * 60}")
    print(f"{label}  ({summary['num_frames']} frames, {w}x{h})")
    print(f"{'=' * 60}")
    print(
        f"intrinsics (norm) mean [fx, fy, cx, cy]: "
        f"{intr[0]:.6f} {intr[1]:.6f} {intr[2]:.6f} {intr[3]:.6f}"
    )
    print(
        f"K mean (px): "
        f"fx={k[0,0]:.2f} fy={k[1,1]:.2f} cx={k[0,2]:.2f} cy={k[1,2]:.2f}"
    )
    print(f"R mean:\n{r}")
    print(f"t mean: [{t[0]:.6f}, {t[1]:.6f}, {t[2]:.6f}]")
    print(f"s: 1.0")
    print(f"camera center mean: [{c[0]:.6f}, {c[1]:.6f}, {c[2]:.6f}]")
    print(
        f"std (t): [{summary['t_std'][0]:.6f}, {summary['t_std'][1]:.6f}, {summary['t_std'][2]:.6f}]  "
        f"std (center): [{summary['center_std'][0]:.6f}, {summary['center_std'][1]:.6f}, {summary['center_std'][2]:.6f}]"
    )


def merge_cam_packs(packs):
    return {
        "num_frames": sum(p["num_frames"] for p in packs),
        "resolution": packs[0]["resolution"],
        "intrinsics_norm": np.concatenate([p["intrinsics_norm"] for p in packs]),
        "K": np.concatenate([p["K"] for p in packs]),
        "R": np.concatenate([p["R"] for p in packs]),
        "t": np.concatenate([p["t"] for p in packs]),
        "center": np.concatenate([p["center"] for p in packs]),
    }


def save_default_camera(params, subjects_used, out_npz, out_txt):
    out_npz = Path(out_npz)
    out_txt = Path(out_txt)
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    save_dict = dict(params)
    save_dict["subjects_used"] = np.array(subjects_used, dtype=object)

    np.savez(out_npz, **save_dict)

    intr = params["intrinsics_norm"]
    k = params["K"]
    r = params["R"]
    t = params["t"]
    h, w = params["resolution"]

    lines = [
        "# default camera baked from flare_2 datasets",
        f"# subjects: {', '.join(subjects_used)}",
        f"# frames: {int(params['num_frames'][0])}",
        f"resolution_h={int(h)}",
        f"resolution_w={int(w)}",
        f"intrinsics_norm={intr[0]:.8f} {intr[1]:.8f} {intr[2]:.8f} {intr[3]:.8f}",
        f"s={float(params['s'][0]):.8f}",
        f"t={t[0]:.8f} {t[1]:.8f} {t[2]:.8f}",
        f"center={params['center'][0]:.8f} {params['center'][1]:.8f} {params['center'][2]:.8f}",
        "K=" + " ".join(f"{x:.6f}" for x in k.reshape(-1)),
        "R=" + " ".join(f"{x:.6f}" for x in r.reshape(-1)),
    ]
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nSaved default camera:")
    print(f"  {out_npz.resolve()}")
    print(f"  {out_txt.resolve()}")


def collect_subject_packs(subjects, scene_override=None):
    all_packs = []
    used_subjects = []
    for name, s in subjects.items():
        packs = run_for_root(
            s["input_dir"],
            scene_filter=scene_override or s["scenes"],
            header=f"\n{'#' * 60}\n# SUBJECT: {name}\n{'#' * 60}",
            quiet=False,
        )
        if packs:
            all_packs.extend(packs)
            used_subjects.append(name)
    return all_packs, used_subjects


def run_for_root(dataset_root, scene_filter=None, header=None, quiet=False):
    dataset_root = Path(dataset_root)
    scenes = find_scene_json_files(dataset_root, scene_filter)

    if header and not quiet:
        print(header)
    if not quiet:
        print(f"dataset_root: {dataset_root.resolve()}")
        print(f"found {len(scenes)} scene(s)" + (f" (filter: {scene_filter})" if scene_filter else ""))

    if not scenes:
        if not quiet:
            print("  (no scenes)")
        return []

    all_packs = []
    for scene_name, scene_dir, json_path in scenes:
        cam_pack = parse_cameras(scene_dir, json_path)
        if cam_pack is None:
            if not quiet:
                print(f"\n[skip] {scene_name}: no frames")
            continue
        all_packs.append(cam_pack)
        if not quiet:
            print_summary(f"scene: {scene_name}", summarize_cameras(cam_pack))

    if not quiet and all_packs:
        if len(all_packs) > 1:
            print_summary("GLOBAL (all scenes, all frames)", summarize_cameras(merge_cam_packs(all_packs)))
        else:
            print_summary("GLOBAL (= single scene)", summarize_cameras(all_packs[0]))
    return all_packs


def main():
    subjects = load_all_subjects()
    subject_names = sorted(subjects.keys())

    parser = argparse.ArgumentParser(
        description="Average cameras from flare dataset JSONs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dataset_root",
        type=str,
        nargs="?",
        default=None,
        help="Override path (default: --subject input_dir from configs_tmp)",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default=DEFAULT_SUBJECT,
        choices=subject_names,
        help="configs_tmp subject (uses input_dir + train/eval scenes)",
    )
    parser.add_argument(
        "--all-subjects",
        action="store_true",
        help="Run every subject defined in configs_tmp",
    )
    parser.add_argument(
        "--list-subjects",
        action="store_true",
        help="Print subjects from configs_tmp and exit",
    )
    parser.add_argument(
        "--flare2-root",
        type=str,
        default=str(DEFAULT_FLARE2_ROOT),
        help="flare_2 root on bean",
    )
    parser.add_argument(
        "--scenes",
        type=str,
        nargs="*",
        default=None,
        help="Scene folder names (overrides config train/eval list)",
    )
    parser.add_argument(
        "--save-defaults",
        action="store_true",
        help="Bake global mean camera to assets/default_camera.npz and .txt",
    )
    parser.add_argument(
        "--out-npz",
        type=str,
        default=str(DEFAULT_CAMERA_NPZ),
    )
    parser.add_argument(
        "--out-txt",
        type=str,
        default=str(DEFAULT_CAMERA_TXT),
    )
    args = parser.parse_args()

    if args.list_subjects:
        print(f"configs_tmp subjects ({len(subject_names)}):")
        for name in subject_names:
            s = subjects[name]
            exists = "OK" if s["input_dir"].exists() else "MISSING"
            print(
                f"  {name:20s} [{exists}]  scenes={s['scenes']}\n"
                f"    {s['input_dir']}"
            )
        return

    if args.all_subjects:
        all_packs, used_subjects = collect_subject_packs(subjects, args.scenes)
        if args.save_defaults and all_packs:
            summary = summarize_cameras(merge_cam_packs(all_packs))
            print_summary("DATASET GLOBAL (all subjects, all frames)", summary)
            save_default_camera(
                summary_to_default_params(summary),
                used_subjects,
                args.out_npz,
                args.out_txt,
            )
        elif args.save_defaults:
            print("No frames collected; defaults not saved.")
        return

    if args.dataset_root:
        root = resolve_data_path(args.dataset_root)
        scenes = args.scenes
        if scenes is None and args.subject in subjects:
            scenes = subjects[args.subject]["scenes"]
        packs = run_for_root(root, scene_filter=scenes)
        if args.save_defaults and packs:
            summary = summarize_cameras(merge_cam_packs(packs))
            save_default_camera(
                summary_to_default_params(summary),
                [str(root)],
                args.out_npz,
                args.out_txt,
            )
        return

    s = subjects[args.subject]
    packs = run_for_root(
        s["input_dir"],
        scene_filter=args.scenes or s["scenes"],
        header=f"subject={args.subject}  (from {s['config_path'].name})",
    )
    if args.save_defaults and packs:
        summary = summarize_cameras(merge_cam_packs(packs))
        save_default_camera(
            summary_to_default_params(summary),
            [args.subject],
            args.out_npz,
            args.out_txt,
        )


if __name__ == "__main__":
    main()
