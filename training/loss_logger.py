"""Training loss logging and tqdm postfix helpers."""

import json
from pathlib import Path

import torch


def tqdm_postfix(losses, global_step: int, n_show: int = 6) -> dict:
    items = [(k, losses[k].item()) for k in losses if k != "total"]
    items.sort(key=lambda x: -abs(x[1]))
    out = {"step": global_step, "loss": f"{losses['total'].item():.4f}"}
    for k, v in items[:n_show]:
        out[k] = f"{v:.4f}"
    return out


def print_losses(losses, global_step: int, stage_name: str):
    parts = [f"step={global_step}", f"loss={losses['total'].item():.4f}"]
    for k in sorted(losses.keys()):
        if k == "total":
            continue
        parts.append(f"{k}={losses[k].item():.4f}")
    print(f"[{stage_name}] " + " ".join(parts), flush=True)


class LossAnalysisLogger:
    """Step-wise JSONL loss log for post-hoc LR/loss-flow analysis."""

    def __init__(self, out_path: Path, ema_beta: float = 0.9):
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.ema_beta = float(ema_beta)
        self._ema = {}

    @staticmethod
    def _to_float(v):
        if torch.is_tensor(v):
            return float(v.detach().item())
        return float(v)

    def _update_ema(self, key: str, value: float) -> float:
        prev = self._ema.get(key)
        if prev is None:
            self._ema[key] = value
        else:
            self._ema[key] = self.ema_beta * prev + (1.0 - self.ema_beta) * value
        return self._ema[key]

    @staticmethod
    def _optimizer_lrs(optim):
        if optim is None:
            return []
        return [float(g["lr"]) for g in optim.param_groups]

    def log(
        self,
        *,
        global_step: int,
        stage_name: str,
        stage_local: int,
        losses: dict,
        mesh_optim,
        gaussian_optim,
        densify_stats: dict | None = None,
    ):
        payload = {
            "global_step": int(global_step),
            "stage_name": str(stage_name),
            "stage_local": int(stage_local),
            "mesh_lrs": self._optimizer_lrs(mesh_optim),
            "gaussian_lrs": self._optimizer_lrs(gaussian_optim),
        }
        for k, v in losses.items():
            val = self._to_float(v)
            payload[f"loss/{k}"] = val
            payload[f"ema/{k}"] = self._update_ema(k, val)

        if densify_stats:
            for k, v in densify_stats.items():
                key = k if k.startswith("densify/") else f"densify/{k}"
                if isinstance(v, (int, float, str, bool)) or v is None:
                    payload[key] = v
                    if isinstance(v, (int, float)):
                        ema_key = key.replace("densify/", "densify_ema/")
                        payload[ema_key] = self._update_ema(ema_key, float(v))
                else:
                    payload[key] = v

        with self.out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
