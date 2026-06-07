#!/usr/bin/env python3
"""
Write ``configs_tmp/*.txt`` for subjects produced by:

  - ``processing/process_video/process_nerface.py``  → ``nf_XX/nf_XX/{train,test}/...``
  - ``processing/process_video/process_nerfblendshape.py`` → ``nbs_idN/nbs_idN/{train,test}/...``

Only ``run_name``, ``input_dir``, ``train_dir``, ``eval_dir`` need to be correct for
``sweep_configs.py`` / ``train.py``; other fields are legacy placeholders.

NeRFBlendShape (``nbs_id*``): configs can be written **before** FLARE output exists
(``--plan-nbs``, default on). IDs come from ``--nerfblendshape-input-dir`` (id1, id2, …),
``--nbs-ids``, or ``--nbs-id-range``.

Usage (repo root)::

    python generate_processed_configs.py --dry-run
    python generate_processed_configs.py --source nbs --plan-nbs
    python generate_processed_configs.py --nerfblendshape-input-dir /path/to/nerfblendshape
    python generate_processed_configs.py --nbs-id-range 1 12 --force
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

DEFAULT_OUTPUT_DIR = Path("/Bean/data/gwangjin/2024/nbshapes/flare_2")
DEFAULT_NERFBLENDSHAPE_INPUT = Path("/Bean/data/gwangjin/2024/nbshapes/nerfblendshape")
DEFAULT_CONFIGS_DIR = Path("configs_tmp")

NF_RE = re.compile(r"^nf_(\d+)$", re.IGNORECASE)
NBS_RE = re.compile(r"^nbs_id(\d+)$", re.IGNORECASE)
NBS_INPUT_ID_RE = re.compile(r"^id(\d+)$", re.IGNORECASE)

_CONFIG_BODY = """\
working_dir = .
output_dir = /Bean/log/gwangjin/2026/neural_blendshapes_4/

batch_size = 1
sample_idx_ratio = 1

iterations = 10000
upsample_iterations = [500]

lr_shader = 1e-3
lr_deformer = 1e-3
lr_jacobian = 1e-3

weight_shading = 1
weight_perceptual_loss = 1e-1
weight_mask = 1
weight_albedo_regularization = 0.01
weight_white_lgt_regularization = 0.01
weight_roughness_regularization = 0.01
weight_fresnel_coeff = 0.01
weight_normal_regularization = 1e-2
weight_laplacian_regularization = 1

weight_feature_regularization = 1e-4

weight_geometric_regularization = 1e-5
weight_normal = 0.25
weight_normal_laplacian = 1e-1
weight_landmark = 1
weight_closure = 1

weight_linearity_regularization = 1e-4

weight_flame_regularization = 10

weight_temporal_regularization = 1e-4

light_mlp_ch = 3
light_mlp_dims = [64, 64]
material_mlp_dims = [128, 128, 128, 128]
material_mlp_ch = 5

stage_iterations = [6000, 0000, 4000, 2000]
only_flame_iterations = 2000
"""


def scene_root(output_dir: Path, subject_name: str) -> Path:
    return output_dir / subject_name / subject_name


def nbs_subject_name(id_num: int) -> str:
    return f"nbs_id{id_num}"


def nf_subject_name(person_num: int) -> str:
    return f"nf_{person_num:02d}"


def split_has_frames(split_image_dir: Path) -> bool:
    return split_image_dir.is_dir() and any(split_image_dir.glob("*.png"))


def subject_is_ready(output_dir: Path, subject_name: str) -> bool:
    scene = scene_root(output_dir, subject_name)
    return split_has_frames(scene / "train" / "image") and split_has_frames(scene / "test" / "image")


def discover_nbs_ids_from_input(input_dir: Path) -> list[int]:
    """``id1``, ``id2``, … under NeRFBlendShape raw root (same as process_nerfblendshape)."""
    input_dir = input_dir.resolve()
    if not input_dir.is_dir():
        return []
    out = []
    for p in sorted(input_dir.iterdir()):
        if not p.is_dir():
            continue
        m = NBS_INPUT_ID_RE.fullmatch(p.name)
        if m is not None:
            out.append(int(m.group(1)))
    out.sort()
    return out


def discover_ready_nf_subjects(
    output_dir: Path,
    *,
    ids: list[int] | None,
) -> list[str]:
    if not output_dir.is_dir():
        return []
    id_set = set(ids) if ids else None
    found = []
    for child in sorted(output_dir.iterdir()):
        if not child.is_dir():
            continue
        m = NF_RE.fullmatch(child.name)
        if m is None:
            continue
        num = int(m.group(1))
        if id_set is not None and num not in id_set:
            continue
        if subject_is_ready(output_dir, child.name):
            found.append(child.name)
    return found


def discover_ready_nbs_subjects(
    output_dir: Path,
    *,
    ids: list[int] | None,
) -> list[str]:
    if not output_dir.is_dir():
        return []
    id_set = set(ids) if ids else None
    found = []
    for child in sorted(output_dir.iterdir()):
        if not child.is_dir():
            continue
        m = NBS_RE.fullmatch(child.name)
        if m is None:
            continue
        num = int(m.group(1))
        if id_set is not None and num not in id_set:
            continue
        if subject_is_ready(output_dir, child.name):
            found.append(child.name)
    return found


def planned_nbs_ids(
    *,
    nerfblendshape_input_dir: Path | None,
    nbs_ids: list[int] | None,
    nbs_id_range: tuple[int, int] | None,
) -> list[int]:
    if nbs_ids is not None and len(nbs_ids) > 0:
        return sorted(set(int(i) for i in nbs_ids))

    if nerfblendshape_input_dir is not None:
        from_input = discover_nbs_ids_from_input(nerfblendshape_input_dir)
        if len(from_input) > 0:
            return from_input

    if nbs_id_range is not None:
        lo, hi = nbs_id_range
        if lo > hi:
            lo, hi = hi, lo
        return list(range(lo, hi + 1))

    return []


def collect_subjects(
    output_dir: Path,
    *,
    source: str,
    nf_ids: list[int] | None,
    plan_nbs: bool,
    nerfblendshape_input_dir: Path | None,
    nbs_ids: list[int] | None,
    nbs_id_range: tuple[int, int] | None,
) -> list[tuple[str, bool]]:
    """
    Returns [(subject_name, is_planned), ...].
    ``is_planned`` True = FLARE output not required / may not exist yet.
    """
    out: list[tuple[str, bool]] = []
    seen: set[str] = set()

    if source in ("all", "nf"):
        for name in discover_ready_nf_subjects(output_dir, ids=nf_ids):
            if name not in seen:
                seen.add(name)
                out.append((name, False))

    if source in ("all", "nbs"):
        for name in discover_ready_nbs_subjects(output_dir, ids=nbs_ids):
            if name not in seen:
                seen.add(name)
                out.append((name, False))

        if plan_nbs:
            for id_num in planned_nbs_ids(
                nerfblendshape_input_dir=nerfblendshape_input_dir,
                nbs_ids=nbs_ids,
                nbs_id_range=nbs_id_range,
            ):
                name = nbs_subject_name(id_num)
                if name not in seen:
                    seen.add(name)
                    out.append((name, True))

    return out


def render_config(subject_name: str, input_dir: Path) -> str:
    return (
        f"run_name = {subject_name}\n"
        f"\n"
        f"input_dir = {input_dir.as_posix()}\n"
        f'train_dir = ["train"]\n'
        f'eval_dir = ["test"]\n'
        f"{_CONFIG_BODY}"
    )


def write_config(
    configs_dir: Path,
    subject_name: str,
    input_dir: Path,
    *,
    force: bool,
    dry_run: bool,
) -> Path:
    out_path = configs_dir / f"{subject_name}.txt"
    if out_path.is_file() and not force:
        return out_path
    text = render_config(subject_name, input_dir)
    if dry_run:
        tag = " (planned)" if not input_dir.is_dir() else ""
        print(f"[dry-run] would write {out_path}{tag}")
        print(f"  input_dir={input_dir}")
        return out_path
    configs_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(f"wrote {out_path}")
    return out_path


def main():
    p = argparse.ArgumentParser(description="Generate configs_tmp for nf_* / nbs_id* FLARE subjects")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="flare_2 root (parent of nf_XX/nf_XX and nbs_idN/nbs_idN)",
    )
    p.add_argument(
        "--configs-dir",
        type=Path,
        default=DEFAULT_CONFIGS_DIR,
        help="Directory for *.txt configs (default: configs_tmp)",
    )
    p.add_argument(
        "--source",
        choices=("all", "nf", "nbs"),
        default="all",
        help="nerface (nf_*) | nerfblendshape (nbs_id*) | both",
    )
    p.add_argument(
        "--ids",
        type=int,
        nargs="*",
        default=None,
        help="Filter nf person nums / nbs id nums (ready scan + planned nbs)",
    )
    p.add_argument(
        "--nbs-ids",
        type=int,
        nargs="*",
        default=None,
        help="Explicit NeRFBlendShape id list for planned configs (overrides --ids for nbs)",
    )
    p.add_argument(
        "--nbs-id-range",
        type=int,
        nargs=2,
        metavar=("LO", "HI"),
        default=None,
        help="Planned nbs_id LO..HI inclusive if no input-dir ids found (e.g. 1 12)",
    )
    p.add_argument(
        "--nerfblendshape-input-dir",
        type=Path,
        default=DEFAULT_NERFBLENDSHAPE_INPUT,
        help="Raw NeRFBlendShape root with id1, id2, … (for planned nbs configs)",
    )
    p.add_argument(
        "--no-plan-nbs",
        action="store_true",
        help="Only write nbs configs when train/test PNGs already exist under output-dir",
    )
    p.add_argument("--force", action="store_true", help="Overwrite existing config files")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    output_dir = args.output_dir.resolve()
    configs_dir = args.configs_dir.resolve()
    plan_nbs = not args.no_plan_nbs

    nbs_input = args.nerfblendshape_input_dir
    if nbs_input is not None:
        nbs_input = nbs_input.resolve()
        if not nbs_input.is_dir():
            nbs_input = None

    nbs_id_range = None
    if args.nbs_id_range is not None:
        nbs_id_range = (int(args.nbs_id_range[0]), int(args.nbs_id_range[1]))
    elif plan_nbs and args.source in ("all", "nbs") and args.nbs_ids is None and nbs_input is None:
        nbs_id_range = (1, 12)

    nbs_ids = args.nbs_ids if args.nbs_ids is not None and len(args.nbs_ids) > 0 else args.ids

    subjects = collect_subjects(
        output_dir,
        source=args.source,
        nf_ids=args.ids,
        plan_nbs=plan_nbs,
        nerfblendshape_input_dir=nbs_input,
        nbs_ids=nbs_ids,
        nbs_id_range=nbs_id_range,
    )

    if len(subjects) == 0:
        print(
            "No subjects to write. For nbs planned configs use --nbs-ids, --nbs-id-range, "
            "or --nerfblendshape-input-dir with id* folders."
        )
        return

    ready = [n for n, planned in subjects if not planned]
    planned = [n for n, planned in subjects if planned]
    print(f"Subjects [{args.source}] (output_dir={output_dir}):")
    if ready:
        print(f"  ready ({len(ready)}): {', '.join(ready)}")
    if planned:
        print(f"  planned nbs ({len(planned)}): {', '.join(planned)}")

    written = 0
    skipped = 0
    for name, is_planned in subjects:
        inp = scene_root(output_dir, name)
        out_path = configs_dir / f"{name}.txt"
        if out_path.is_file() and not args.force:
            print(f"skip (exists): {out_path}")
            skipped += 1
            continue
        if is_planned and not args.dry_run:
            print(f"  [planned] {name} (data may not exist yet)")
        write_config(configs_dir, name, inp, force=args.force, dry_run=args.dry_run)
        written += 1

    print(f"\nDone: {written} written, {skipped} skipped, configs_dir={configs_dir}")


if __name__ == "__main__":
    main()
