"""Eye-only NICP: fit FLAME eyeball submesh to ICT eyeball submesh."""

import numpy as np
import torch


def build_edges_from_faces(faces):
    edges = set()
    for tri in faces:
        a, b, c = map(int, tri)
        edges.add(tuple(sorted((a, b))))
        edges.add(tuple(sorted((b, c))))
        edges.add(tuple(sorted((c, a))))
    return np.array(sorted(edges), dtype=np.int64)


def normalize_by_center_scale(src_v, tgt_v):
    src_c = src_v.mean(axis=0, keepdims=True)
    tgt_c = tgt_v.mean(axis=0, keepdims=True)
    src_r = np.mean(np.linalg.norm(src_v - src_c, axis=1))
    tgt_r = np.mean(np.linalg.norm(tgt_v - tgt_c, axis=1))
    scale = tgt_r / max(src_r, 1e-8)
    return (src_v - src_c) * scale + tgt_c


def chamfer_oneway(x, y):
    d2 = torch.cdist(x, y) ** 2
    return d2.min(dim=1).values.mean()


def fit_eye_nicp_torch(
    src_v,
    src_f,
    tgt_v,
    src_anchor_local=None,
    tgt_anchor_v=None,
    iters=800,
    lr=5e-3,
    w_chamfer=1.0,
    w_edge=20.0,
    w_anchor=50.0,
    device="cuda",
    src_init=None,
):
    """
    Fit src eye mesh (FLAME) to tgt eye mesh (ICT).
    src_anchor_local: FLAME iris seed verts in submesh local indices.
    tgt_anchor_v: target 3D positions for iris anchors (ICT iris verts, not FLAME).
    ``src_init``: if set, skip ``normalize_by_center_scale`` (use after rigid prescale).
    """
    if src_init is None:
        src_init = normalize_by_center_scale(src_v, tgt_v)
    else:
        src_init = np.asarray(src_init, dtype=np.float64)

    x = torch.tensor(src_init, dtype=torch.float32, device=device, requires_grad=True)
    x0 = torch.tensor(src_init, dtype=torch.float32, device=device)
    y = torch.tensor(tgt_v, dtype=torch.float32, device=device)

    edges = build_edges_from_faces(src_f)
    edges_t = torch.tensor(edges, dtype=torch.long, device=device)
    e0 = torch.linalg.norm(x0[edges_t[:, 0]] - x0[edges_t[:, 1]], dim=1).detach()

    src_anchor_local_t = None
    tgt_anchor_v_t = None
    if (
        src_anchor_local is not None
        and tgt_anchor_v is not None
        and len(src_anchor_local) > 0
        and len(tgt_anchor_v) > 0
    ):
        src_anchor_local_t = torch.tensor(src_anchor_local, dtype=torch.long, device=device)
        tgt_anchor_v_t = torch.tensor(tgt_anchor_v, dtype=torch.float32, device=device)

    opt = torch.optim.Adam([x], lr=lr)

    for it in range(iters):
        opt.zero_grad()
        loss_xy = chamfer_oneway(x, y)
        loss_yx = chamfer_oneway(y, x)
        loss_chamfer = loss_xy + loss_yx

        e = torch.linalg.norm(x[edges_t[:, 0]] - x[edges_t[:, 1]], dim=1)
        loss_edge = ((e - e0) ** 2).mean()

        loss_anchor = x.new_zeros(())
        if src_anchor_local_t is not None:
            loss_anchor = chamfer_oneway(x[src_anchor_local_t], tgt_anchor_v_t)

        loss = w_chamfer * loss_chamfer + w_edge * loss_edge + w_anchor * loss_anchor
        loss.backward()
        opt.step()

        if it % 200 == 0 or it == iters - 1:
            print(
                f"  eye NICP iter {it:04d} loss={loss.item():.6f} "
                f"ch={loss_chamfer.item():.6f} edge={loss_edge.item():.6f} "
                f"anchor={loss_anchor.item():.6f}"
            )

    return x.detach().cpu().numpy()
