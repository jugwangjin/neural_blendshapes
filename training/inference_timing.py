"""Wall-clock timing for inference render loops (FPS in meta)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch


@dataclass
class InferenceTimer:
    device: torch.device
    samples_s: list[float] = field(default_factory=list)

    def sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def start(self) -> float:
        self.sync()
        return time.perf_counter()

    def stop(self, t0: float) -> float:
        self.sync()
        dt = time.perf_counter() - t0
        self.samples_s.append(dt)
        return dt

    def summary(self) -> dict:
        n = len(self.samples_s)
        if n == 0:
            return {"n": 0, "total_s": 0.0, "fps": 0.0, "ms_per_frame": 0.0}
        total = float(sum(self.samples_s))
        return {
            "n": n,
            "total_s": total,
            "fps": n / total,
            "ms_per_frame": 1000.0 * total / n,
        }
