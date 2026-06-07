"""Mesh semantic loss (nvdiffrast): FLARE part_label vs posed ICT mesh region raster."""

import torch

from losses.mesh_silhouette import (
    _exclude_faces_by_vertex_ids,
    _faces_for_batch,
    _front_facing_mask,
    _gl_projection_from_camera,
    _gl_view_from_camera,
    _mesh_alpha_nvdiffrast,
    _raster_ctx,
)
from losses.segmentation import loss_segmentation_l2
from rendering.mesh_face_semantic import mesh_triangle_semantic_class, vertex_semantic_onehot
from rendering.semantic import SEMANTIC_IGNORE_INDEX


def _face_keep_mask(faces, exclude_vertex_ids, n_verts, device):
    if exclude_vertex_ids is None:
        return torch.ones(faces.shape[0], dtype=torch.bool, device=device)
    if torch.is_tensor(exclude_vertex_ids):
        ex = exclude_vertex_ids.to(device=device, dtype=torch.long).reshape(-1)
    else:
        ex = torch.as_tensor(list(exclude_vertex_ids), device=device, dtype=torch.long).reshape(-1)
    if ex.numel() == 0:
        return torch.ones(faces.shape[0], dtype=torch.bool, device=device)
    ex_mask = torch.zeros(int(n_verts), dtype=torch.bool, device=device)
    ex = ex[(ex >= 0) & (ex < int(n_verts))]
    ex_mask[ex] = True
    return ~ex_mask[faces.long()].any(dim=1)


def _mesh_semantic_accum(
    mesh_xyz,
    faces,
    ict,
    camera,
    *,
    width: int,
    height: int,
    exclude_vertex_ids=None,
    cull_backfaces: bool = False,
    cull_flip: bool = False,
):
    import nvdiffrast.torch as dr

    b, v, _ = mesh_xyz.shape
    device = mesh_xyz.device
    dtype = mesh_xyz.dtype
    n_classes = 3

    face_cls = mesh_triangle_semantic_class(faces, ict, device)
    keep = (face_cls != SEMANTIC_IGNORE_INDEX) & _face_keep_mask(
        faces, exclude_vertex_ids, v, device
    )
    faces_eff = faces[keep]
    cls_eff = face_cls[keep]
    if faces_eff.shape[0] == 0:
        return torch.zeros((b, n_classes, height, width), device=device, dtype=dtype)

    if cull_backfaces:
        front = _front_facing_mask(mesh_xyz, faces_eff, camera, flip=cull_flip)
        if front.shape[0] == 1:
            sub = front[0]
        else:
            sub = front.all(dim=0)
        faces_eff = faces_eff[sub]
        cls_eff = cls_eff[sub]
        if faces_eff.shape[0] == 0:
            return torch.zeros((b, n_classes, height, width), device=device, dtype=dtype)

    v_attr = vertex_semantic_onehot(faces_eff, cls_eff, v, n_classes).to(device=device, dtype=dtype)
    v_attr = v_attr.unsqueeze(0).expand(b, -1, -1)

    proj = _gl_projection_from_camera(
        camera,
        width=width,
        height=height,
        near=0.01,
        far=100.0,
        device=device,
        dtype=dtype,
    )
    view = _gl_view_from_camera(camera, device=device, dtype=dtype)
    mvp = proj @ view
    ones = torch.ones((b, v, 1), device=device, dtype=dtype)
    posw = torch.cat([mesh_xyz, ones], dim=-1)
    clip = torch.bmm(posw, mvp.t().unsqueeze(0).expand(b, -1, -1))
    tri = faces_eff.to(device=device, dtype=torch.int32).contiguous()
    ctx = _raster_ctx(device)
    rast, _ = dr.rasterize(ctx, clip, tri, resolution=(height, width))
    accum, _ = dr.interpolate(v_attr, rast, tri)
    accum = dr.antialias(accum, rast, clip, tri)
    return accum.permute(0, 3, 1, 2).contiguous()


def loss_mesh_semantic(
    mesh_xyz,
    faces,
    ict,
    camera,
    part_label,
    *,
    image_size: int = 512,
    downsample: int = 4,
    exclude_vertex_ids=None,
    cull_backfaces: bool = False,
    cull_flip: bool = False,
):
    """
    L2 on mesh-rasterized 3-class accum vs ``flare_part_label_to_semantic_class(part_label)``.

    Differentiable w.r.t. ``mesh_xyz`` → tracker / template / deformer (bootstrap stages).
    """
    from dataset.flare_semantic import flare_part_label_to_semantic_class

    ds = max(1, int(downsample))
    w = int(camera.width) // ds
    h = int(camera.height) // ds

    pl = part_label
    if pl.ndim == 2:
        pl = pl.unsqueeze(0)
    if pl.shape[-2:] != (h, w):
        pl = torch.nn.functional.interpolate(
            pl.unsqueeze(1).float(), size=(h, w), mode="nearest"
        ).squeeze(1).long()

    target = flare_part_label_to_semantic_class(pl)
    pred = _mesh_semantic_accum(
        mesh_xyz,
        faces,
        ict,
        camera,
        width=w,
        height=h,
        exclude_vertex_ids=exclude_vertex_ids,
        cull_backfaces=cull_backfaces,
        cull_flip=cull_flip,
    )

    valid = pl != 0
    faces_keep = _face_keep_mask(faces, exclude_vertex_ids, mesh_xyz.shape[1], mesh_xyz.device)
    faces_r = _exclude_faces_by_vertex_ids(faces[faces_keep], exclude_vertex_ids)
    alpha = _mesh_alpha_nvdiffrast(mesh_xyz, faces_r, camera, width=w, height=h)
    valid = valid & (alpha[:, 0] > 0.02)
    return loss_segmentation_l2(pred, target, valid_mask=valid, ignore_index=SEMANTIC_IGNORE_INDEX)
