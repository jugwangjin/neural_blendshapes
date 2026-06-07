"""Per-iteration CUDA-synchronized component timing for training profiling."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path


def _cuda_sync():
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


class ComponentTimer:
    """
    Accumulate wall time per named section across training iterations.

    Call ``begin_iter()`` once per step; sections recorded only after ``warmup`` iters.
    """

    def __init__(self, warmup: int = 0):
        self.warmup = int(warmup)
        self._iter = 0
        self._recording = False
        self._sums: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    def begin_iter(self):
        self._iter += 1
        self._recording = self._iter > self.warmup

    @property
    def n_recorded(self) -> int:
        if not self._counts:
            return 0
        return min(self._counts.values())

    def section(self, name: str):
        if not self._recording:
            return nullcontext()
        return self._section_record(name)

    @contextmanager
    def _section_record(self, name: str):
        _cuda_sync()
        t0 = time.perf_counter()
        yield
        _cuda_sync()
        dt = time.perf_counter() - t0
        self._sums[name] = self._sums.get(name, 0.0) + dt
        self._counts[name] = self._counts.get(name, 0) + 1

    def averages_ms(self) -> dict[str, float]:
        out = {}
        for name, total in self._sums.items():
            n = self._counts.get(name, 0)
            if n > 0:
                out[name] = (total / n) * 1000.0
        return out

    def totals_ms(self) -> dict[str, float]:
        return {name: total * 1000.0 for name, total in self._sums.items()}

    def print_report(self, stage_name: str, n_iters: int, out_path: Path | None = None):
        avgs = self.averages_ms()
        if not avgs:
            print(f"[TimeMeasure] stage={stage_name}: no samples recorded")
            return

        order = [
            "data",
            "tracker",
            "avatar_forward",
            "render",
            "backward",
            "densify_pre_backward",
            "densify_post_backward",
            "optimizer_mesh",
            "densify_pre_optimizer",
            "optimizer_gaussian",
            "triangle_walk",
            "densify_post_optimizer",
        ]
        loss_keys = sorted(k for k in avgs if k.startswith("loss/"))
        ordered = [k for k in order if k in avgs] + loss_keys
        for k in sorted(avgs):
            if k not in ordered:
                ordered.append(k)

        lines = [
            f"[TimeMeasure] stage={stage_name} iters={n_iters} warmup={self.warmup}",
            f"  {'component':<28} {'avg_ms':>10} {'share':>8}",
        ]
        top_level = [k for k in ordered if not k.startswith("loss/")]
        denom = sum(avgs[k] for k in top_level)
        for name in ordered:
            ms = avgs[name]
            if name.startswith("loss/"):
                share = (ms / denom * 100.0) if denom > 0 else 0.0
                label = f"  {name}"
            else:
                share = (ms / denom * 100.0) if denom > 0 else 0.0
                label = f"  {name}"
            lines.append(f"{label:<28} {ms:10.3f} {share:7.1f}%")
        lines.append(f"  {'(top-level sum)':<28} {denom:10.3f}")
        print("\n".join(lines), flush=True)

        if out_path is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "stage": stage_name,
                "n_iters": n_iters,
                "warmup": self.warmup,
                "avg_ms": avgs,
                "total_ms": self.totals_ms(),
            }
            out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"[TimeMeasure] wrote {out_path}", flush=True)
