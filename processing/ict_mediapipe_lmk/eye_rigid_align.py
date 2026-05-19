"""Per-eye similarity ``s,T`` only (R=I): pytorch3d chamfer + front/back anchor pairs."""

import numpy as np
import torch
from pytorch3d.loss import chamfer_distance

from processing.ict_flame_similarity import apply_flame_alignment, fit_uniform_scale_translation

EYE_ALIGN_R = np.eye(3, dtype=np.float64)


def _as_anchors(a):
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 1:
        return a.reshape(1, 3)
    return a.reshape(-1, 3)


def apply_eye_similarity(vertices, s, T):
    """``x' = s * x + T`` (rotation fixed to identity)."""
    return apply_flame_alignment(np.asarray(vertices, dtype=np.float64), float(s), EYE_ALIGN_R, T)


def _init_scale_translation_from_anchors(src_anchors, tgt_anchors):
    """``s,T`` init from paired front/back anchors only (equal N)."""
    return fit_uniform_scale_translation(_as_anchors(src_anchors), _as_anchors(tgt_anchors))


def fit_eye_rigid_chamfer_torch(
    src_v,
    tgt_v,
    src_anchors,
    tgt_anchors,
    *,
    iters=300,
    lr=1e-2,
    w_chamfer=1.0,
    w_anchor=200.0,
    device="cuda",
):
    """
    Optimize ``x' = s * x + T`` on FLAME eyeball (``R = I``).

    Chamfer: ``pytorch3d.loss.chamfer_distance`` — **different** point counts OK
    (bidirectional when ``single_directional=False``).
    See https://pytorch3d.readthedocs.io/en/latest/modules/loss.html
    """
    device = torch.device(device)
    src_v = np.asarray(src_v, dtype=np.float64)
    tgt_v = np.asarray(tgt_v, dtype=np.float64)
    src_anchors = _as_anchors(src_anchors)
    tgt_anchors = _as_anchors(tgt_anchors)
    if src_anchors.shape != tgt_anchors.shape or src_anchors.shape[0] < 1:
        raise ValueError(f"anchor shape mismatch: {src_anchors.shape} vs {tgt_anchors.shape}")

    s0, T0 = _init_scale_translation_from_anchors(src_anchors, tgt_anchors)

    src = torch.tensor(src_v, dtype=torch.float32, device=device)
    tgt = torch.tensor(tgt_v, dtype=torch.float32, device=device)
    src_a = torch.tensor(src_anchors, dtype=torch.float32, device=device)
    tgt_a = torch.tensor(tgt_anchors, dtype=torch.float32, device=device)

    log_s = torch.tensor([np.log(max(float(s0), 1e-6))], dtype=torch.float32, device=device, requires_grad=True)
    T = torch.tensor(T0, dtype=torch.float32, device=device, requires_grad=True)

    opt = torch.optim.Adam([log_s, T], lr=lr)
    n_anc = int(src_anchors.shape[0])

    for it in range(iters):
        opt.zero_grad()
        s = torch.exp(log_s)[0]
        x = s * src + T
        loss_ch, _ = chamfer_distance(
            x.unsqueeze(0),
            tgt.unsqueeze(0),
            single_directional=False,
        )
        x_a = s * src_a + T
        loss_a = ((x_a - tgt_a) ** 2).sum()
        loss = w_chamfer * loss_ch + w_anchor * loss_a
        loss.backward()
        opt.step()

        if it % 100 == 0 or it == iters - 1:
            anc_d = torch.linalg.norm(x_a - tgt_a, dim=1).detach().cpu().numpy()
            parts = " ".join(f"a{i}={anc_d[i]:.5f}" for i in range(n_anc))
            print(
                f"  eye s,T iter {it:04d} loss={loss.item():.6f} "
                f"ch_p3d={loss_ch.item():.6f} anchor={loss_a.item():.6f} dist [{parts}] "
                f"(FLAME V={len(src_v)} ICT V={len(tgt_v)})"
            )

    s = float(torch.exp(log_s).detach().cpu())
    T = T.detach().cpu().numpy().reshape(3)
    aligned = apply_eye_similarity(src_v, s, T)
    aligned_anchors = apply_eye_similarity(src_anchors, s, T)
    return aligned, s, EYE_ALIGN_R.copy(), T, aligned_anchors
