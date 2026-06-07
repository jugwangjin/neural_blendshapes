"""
Recommend ``gaussian_grow_grad2d`` / ``gaussian_grow_gradrgb`` from training ``loss_log.jsonl``.

Reads ``densify/grad2d_*`` and ``densify/gradrgb_*`` snapshots written by ``LossAnalysisLogger``
(see ``docs/implementation/loss_log_densify_grad2d_gradrgb.md``).

Usage (repo root, after a run with densify logging):

  python analyze_grad.py --output-root /path/to/run
  python analyze_grad.py --log /path/to/run/analysis/loss_log.jsonl --stage 2_coarse_mesh
  python analyze_grad.py --log loss_log.jsonl --grow-frac 0.01 0.005 --out-json analysis/grad_thr.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def _get(row: dict, *keys: str):
    for k in keys:
        if k in row:
            return row[k]
    return None


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _filter_rows(
    rows: list[dict],
    *,
    stage: str | None,
    step_min: int | None,
    step_max: int | None,
    tracking_only: bool,
) -> list[dict]:
    out = []
    for r in rows:
        if tracking_only and _get(r, "densify/tracking") not in (1, 1.0, True):
            continue
        if stage is not None and _get(r, "stage_name") != stage:
            continue
        gs = _get(r, "global_step")
        if gs is None:
            continue
        gs = int(gs)
        if step_min is not None and gs < step_min:
            continue
        if step_max is not None and gs > step_max:
            continue
        n_obs = _get(r, "densify/grad2d_n_obs", "densify/gradrgb_n_obs")
        if n_obs is not None and int(n_obs) <= 0:
            continue
        out.append(r)
    return out


def _series(rows: list[dict], key: str) -> list[float]:
    xs = []
    for r in rows:
        v = _get(r, key)
        if v is None:
            continue
        xs.append(float(v))
    return xs


def _quantile_sorted(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    q = min(max(q, 0.0), 1.0)
    idx = q * (len(ys) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return ys[lo]
    w = idx - lo
    return ys[lo] * (1.0 - w) + ys[hi] * w


def _summarize(name: str, xs: list[float]):
    if not xs:
        print(f"  [{name}] no samples")
        return
    print(
        f"  [{name}] n={len(xs)} "
        f"min={min(xs):.6e} p25={_quantile_sorted(xs, 0.25):.6e} "
        f"p50={_quantile_sorted(xs, 0.5):.6e} p75={_quantile_sorted(xs, 0.75):.6e} "
        f"p90={_quantile_sorted(xs, 0.9):.6e} max={max(xs):.6e} "
        f"mean={statistics.fmean(xs):.6e}"
    )


def _frac_above_rows(rows: list[dict], prefix: str) -> list[float]:
    out = []
    for r in rows:
        n_obs = _get(r, f"densify/{prefix}_n_obs")
        above = _get(r, f"densify/{prefix}_above_thr")
        if n_obs is None or above is None:
            continue
        n = int(n_obs)
        if n <= 0:
            continue
        out.append(int(above) / n)
    return out


def _thr_from_snapshot(p50: float, p90: float, pmax: float, target_frac: float) -> float:
    """
    Map target visible-fraction (grow candidates) to a threshold using logged p50/p90/max.

    Anchors: thr=p50 -> ~50% above; thr=p90 -> ~10%; thr~max -> ~0%.
    """
    target_frac = min(max(target_frac, 1e-6), 0.99)
    p50 = max(p50, 1e-20)
    p90 = max(p90, p50 * 1.0001)
    pmax = max(pmax, p90 * 1.0001)

    knots = [
        (p50, 0.5),
        (p90, 0.1),
        (pmax, 1e-4),
    ]
    log_f = math.log(target_frac)
    for i in range(len(knots) - 1):
        t0, f0 = knots[i]
        t1, f1 = knots[i + 1]
        lf0, lf1 = math.log(f0), math.log(f1)
        if log_f <= lf0 and log_f >= lf1:
            w = (log_f - lf0) / (lf1 - lf0) if lf1 != lf0 else 0.0
            log_t = math.log(t0) * (1.0 - w) + math.log(t1) * w
            return math.exp(log_t)
    if log_f > lf0:
        return p50
    return pmax


def _recommend_thresholds(rows: list[dict], prefix: str, grow_fracs: list[float]) -> dict:
    per_row_thr = {f: [] for f in grow_fracs}
    for r in rows:
        p50 = _get(r, f"densify/{prefix}_p50")
        p90 = _get(r, f"densify/{prefix}_p90")
        pmax = _get(r, f"densify/{prefix}_max")
        if p50 is None or p90 is None or pmax is None:
            continue
        for f in grow_fracs:
            per_row_thr[f].append(_thr_from_snapshot(float(p50), float(p90), float(pmax), f))

    out = {}
    for f in grow_fracs:
        xs = per_row_thr[f]
        out[str(f)] = {
            "median": _quantile_sorted(xs, 0.5) if xs else None,
            "p25": _quantile_sorted(xs, 0.25) if xs else None,
            "p75": _quantile_sorted(xs, 0.75) if xs else None,
            "n_rows": len(xs),
        }
    return out


def _report_signal(rows: list[dict], prefix: str, label: str, grow_fracs: list[float]) -> dict:
    print(f"\n=== {label} ({prefix}) ===")
    thr_key = f"densify/{prefix}_thr"
    thr_logged = _series(rows, thr_key)
    if thr_logged:
        print(f"  logged threshold (config at log time): median={statistics.median(thr_logged):.6e}")

    for stat in ("p50", "p90", "max", "mean"):
        _summarize(f"{stat} (per-step snapshot)", _series(rows, f"densify/{prefix}_{stat}"))

    fracs = _frac_above_rows(rows, prefix)
    _summarize("frac_above at logged thr", fracs)
    if fracs:
        print(
            f"  => at current threshold, median grow-candidate fraction ≈ "
            f"{_quantile_sorted(fracs, 0.5) * 100:.3f}% of observed splats"
        )

    rec = _recommend_thresholds(rows, prefix, grow_fracs)
    print(f"  suggested thresholds (median across steps, anchor p50/p90/max):")
    for f in grow_fracs:
        block = rec[str(f)]
        med = block["median"]
        if med is None:
            print(f"    target_frac={f:.4g}: (no data)")
            continue
        print(
            f"    target_frac={f:.4g}: thr≈{med:.6e}  "
            f"(p25={block['p25']:.6e} p75={block['p75']:.6e}, n={block['n_rows']})"
        )
    return {
        "logged_thr_median": statistics.median(thr_logged) if thr_logged else None,
        "frac_above_median": _quantile_sorted(fracs, 0.5) if fracs else None,
        "recommend": rec,
    }


def parse_cli():
    p = argparse.ArgumentParser(
        description="Suggest gaussian_grow_grad2d / gradrgb from loss_log.jsonl densify stats"
    )
    p.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Path to analysis/loss_log.jsonl (default: <output-root>/analysis/loss_log.jsonl)",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Run directory; used if --log omitted",
    )
    p.add_argument("--stage", type=str, default=None, help="Filter e.g. 2_coarse_mesh")
    p.add_argument("--step-min", type=int, default=None)
    p.add_argument("--step-max", type=int, default=None)
    p.add_argument(
        "--grow-frac",
        type=float,
        nargs="+",
        default=[0.1, 0.01, 0.001],
        help="Target fraction of observed splats above threshold (default: 0.1 0.01 0.001)",
    )
    p.add_argument(
        "--include-non-tracking",
        action="store_true",
        help="Include rows where densify/tracking=0",
    )
    p.add_argument("--out-json", type=Path, default=None, help="Write recommendation JSON")
    return p.parse_args()


def resolve_log_path(args) -> Path:
    if args.log is not None:
        return Path(args.log)
    if args.output_root is not None:
        return Path(args.output_root) / "analysis" / "loss_log.jsonl"
    return Path("analysis") / "loss_log.jsonl"


def main():
    args = parse_cli()
    log_path = resolve_log_path(args)
    if not log_path.is_file():
        raise FileNotFoundError(f"loss log not found: {log_path}")

    rows = _load_jsonl(log_path)
    filtered = _filter_rows(
        rows,
        stage=args.stage,
        step_min=args.step_min,
        step_max=args.step_max,
        tracking_only=not args.include_non_tracking,
    )

    print(f"log: {log_path}")
    print(f"rows total={len(rows)} used={len(filtered)}")
    if args.stage:
        print(f"  stage filter: {args.stage}")
    if args.step_min is not None or args.step_max is not None:
        print(f"  global_step in [{args.step_min}, {args.step_max}]")

    if not filtered:
        print("No densify rows with grad stats. Train stage 2+ with gaussian_densify_stages and log_every.")
        return

    n_cached = sum(1 for r in filtered if _get(r, "densify/snapshot_cached"))
    n_grad2d_p50 = len(_series(filtered, "densify/grad2d_p50"))
    g2d_max = _series(filtered, "densify/grad2d_max")
    grgb_max = _series(filtered, "densify/gradrgb_max")
    if n_grad2d_p50 < len(filtered):
        print(
            f"  note: only {n_grad2d_p50}/{len(filtered)} rows have grad percentiles "
            f"(older logs or sparse densify window)"
        )
    if n_cached:
        print(f"  note: {n_cached}/{len(filtered)} rows use cached pre-reset densify stats")
    if g2d_max and grgb_max:
        g2d_med = _quantile_sorted(g2d_max, 0.5)
        grgb_med = _quantile_sorted(grgb_max, 0.5)
        if g2d_med > 0 and grgb_med <= 0:
            print(
                "  warning: gradrgb stats are all zero while grad2d is non-zero — "
                "re-run with fixed densify logging (SH color grad shape) or check color.requires_grad"
            )

    grow_option = _get(filtered[-1], "densify/grow_option")
    pixel_scale = _series(filtered, "densify/pixel_scale")
    if grow_option is not None:
        print(f"  active grow_option (last row): {grow_option}")
    if pixel_scale:
        print(f"  pixel_scale median: {statistics.median(pixel_scale):.4f}")

    report = {
        "log": str(log_path),
        "n_rows_total": len(rows),
        "n_rows_used": len(filtered),
        "grow_fracs": list(args.grow_frac),
        "grad2d": _report_signal(filtered, "grad2d", "view-space grad2d", args.grow_frac),
        "gradrgb": _report_signal(filtered, "gradrgb", "color gradrgb", args.grow_frac),
    }

    g2 = report["grad2d"]["recommend"]
    gr = report["gradrgb"]["recommend"]
    f_main = str(args.grow_frac[1] if len(args.grow_frac) > 1 else args.grow_frac[0])
    print("\n=== config.py paste (review before applying) ===")
    if g2.get(f_main, {}).get("median") is not None:
        print(f"  gaussian_grow_grad2d: float = {g2[f_main]['median']:.6e}")
    if gr.get(f_main, {}).get("median") is not None:
        print(f"  gaussian_grow_gradrgb: float = {gr[f_main]['median']:.6e}")
    print(
        "  # target_frac interpretation: ~fraction of *observed* splats with "
        "running-mean grad >= thr at log snapshots"
    )
    print(
        "  # compare frac_above at old thr; switch gaussian_grow_option after comparing grad2d vs gradrgb"
    )

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with args.out_json.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=True)
        print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
