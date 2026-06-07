"""Stage-1 mesh silhouette loss using minimal nvdiffrast rasterization."""

import nvdiffrast.torch as dr
import torch

_RASTER_CTX = {}


def _raster_ctx(device):
    key = str(device)
    if key not in _RASTER_CTX:
        _RASTER_CTX[key] = dr.RasterizeCudaContext(device=device)
    return _RASTER_CTX[key]


def _resize_mask(mask, h, w):
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.shape[-2:] == (h, w):
        return mask
    return torch.nn.functional.interpolate(
        mask.float(), size=(h, w), mode="bilinear", align_corners=False
    )


def _front_facing_mask(mesh_xyz, faces, camera, *, eps=1e-8, flip=False):
    """
    Per-triangle front-facing test in OpenCV camera space (+Z forward, camera at origin).

    Front if unnormalized normal points toward the camera (``dot(n, -centroid) > 0``).
    """
    from utils.camera import world_to_camera

    v_cam = world_to_camera(mesh_xyz, camera)
    tri = v_cam[:, faces.long(), :]
    e1 = tri[:, :, 1, :] - tri[:, :, 0, :]
    e2 = tri[:, :, 2, :] - tri[:, :, 0, :]
    n = torch.cross(e1, e2, dim=-1)
    c = tri.mean(dim=2)
    front = (n * (-c)).sum(dim=-1) > float(eps)
    if flip:
        front = ~front
    return front


def _faces_for_batch(faces, front_mask, *, want_front: bool):
    """Select triangle rows shared across batch (``B==1`` exact; else conservative)."""
    if front_mask.shape[0] == 1:
        keep = front_mask[0] if want_front else ~front_mask[0]
    else:
        keep = front_mask.all(dim=0) if want_front else (~front_mask).any(dim=0)
    return faces[keep]


def _exclude_faces_by_vertex_ids(faces, exclude_vertex_ids):
    if exclude_vertex_ids is None:
        return faces
    if torch.is_tensor(exclude_vertex_ids):
        ex = exclude_vertex_ids.to(device=faces.device, dtype=torch.long).reshape(-1)
    else:
        ex = torch.as_tensor(list(exclude_vertex_ids), device=faces.device, dtype=torch.long).reshape(-1)
    if ex.numel() == 0:
        return faces
    n_verts = int(faces.max().item()) + 1
    ex = ex[(ex >= 0) & (ex < n_verts)]
    if ex.numel() == 0:
        return faces
    ex_mask = torch.zeros(n_verts, dtype=torch.bool, device=faces.device)
    ex_mask[ex] = True
    keep = ~ex_mask[faces.long()].any(dim=1)
    return faces[keep]


def _gl_projection_from_camera(camera, *, width: int, height: int, near: float, far: float, device, dtype):
    fx = float(camera.fx)
    fy = float(camera.fy)
    cx = float(camera.cx) * (float(width) / float(camera.width))
    cy = float(camera.cy) * (float(height) / float(camera.height))
    return torch.tensor(
        [
            [2.0 * fx / width, 0.0, 1.0 - 2.0 * cx / width, 0.0],
            [0.0, 2.0 * fy / height, 1.0 - 2.0 * cy / height, 0.0],
            [0.0, 0.0, -(far + near) / (far - near), -(2.0 * far * near) / (far - near)],
            [0.0, 0.0, -1.0, 0.0],
        ],
        device=device,
        dtype=dtype,
    )


def _gl_view_from_camera(camera, device, dtype):
    Rt = torch.eye(4, device=device, dtype=dtype)
    Rt[:3, :3] = camera.R.to(device=device, dtype=dtype)
    Rt[:3, 3] = camera.t.to(device=device, dtype=dtype)
    gl = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    )
    return gl @ Rt


def _mesh_alpha_nvdiffrast(mesh_xyz, faces, camera, *, width: int, height: int, near: float = 0.01, far: float = 100.0):
    b, v, _ = mesh_xyz.shape
    device = mesh_xyz.device
    dtype = mesh_xyz.dtype
    proj = _gl_projection_from_camera(
        camera, width=width, height=height, near=near, far=far, device=device, dtype=dtype
    )
    view = _gl_view_from_camera(camera, device=device, dtype=dtype)
    mvp = proj @ view  # [4,4]

    ones = torch.ones((b, v, 1), device=device, dtype=dtype)
    posw = torch.cat([mesh_xyz, ones], dim=-1)  # [B,V,4]
    clip = torch.bmm(posw, mvp.t().unsqueeze(0).expand(b, -1, -1))  # [B,V,4]

    tri = faces.to(device=device, dtype=torch.int32).contiguous()
    if tri.numel() == 0 or tri.shape[0] == 0:
        return torch.zeros((b, 1, height, width), device=device, dtype=dtype)
    ctx = _raster_ctx(device)
    rast, _ = dr.rasterize(ctx, clip, tri, resolution=(height, width))
    # Differentiable silhouette: interpolate per-vertex ones, then antialias.
    ones_attr = torch.ones((b, v, 1), device=device, dtype=dtype)
    alpha, _ = dr.interpolate(ones_attr, rast, tri)
    alpha = dr.antialias(alpha, rast, clip, tri)
    return alpha.permute(0, 3, 1, 2).contiguous()  # [B,1,H,W]


def render_mesh_silhouette_alpha(
    mesh_xyz,
    faces,
    camera,
    *,
    image_size: int = 512,
    downsample: int = 1,
    exclude_vertex_ids=None,
    cull_backfaces: bool = False,
    cull_flip: bool = False,
):
    """Render mesh alpha ``[B,1,H,W]`` with minimal nvdiffrast setup."""
    ds = max(1, int(downsample))
    w = int(getattr(camera, "width", image_size)) // ds
    h = int(getattr(camera, "height", image_size)) // ds
    faces_eff = _exclude_faces_by_vertex_ids(faces, exclude_vertex_ids)
    if cull_backfaces:
        front = _front_facing_mask(mesh_xyz, faces_eff, camera, flip=cull_flip)
        faces_eff = _faces_for_batch(faces_eff, front, want_front=True)
    return _mesh_alpha_nvdiffrast(mesh_xyz, faces_eff, camera, width=w, height=h)


def loss_mesh_backface_curl(mesh_xyz, faces, camera, *, image_size: int = 512, downsample: int = 4, cull_flip: bool = False):
    """
    Penalize visible back-facing triangles (curling / folded-back shell in silhouette).
    """
    ds = max(1, int(downsample))
    w = int(getattr(camera, "width", image_size)) // ds
    h = int(getattr(camera, "height", image_size)) // ds
    front = _front_facing_mask(mesh_xyz, faces, camera, flip=cull_flip)
    faces_back = _faces_for_batch(faces, front, want_front=False)
    if faces_back.shape[0] == 0:
        return mesh_xyz.new_zeros(())
    alpha_back = _mesh_alpha_nvdiffrast(mesh_xyz, faces_back, camera, width=w, height=h)
    return alpha_back.pow(2).mean()


def _mesh_silhouette_alpha_target(
    mesh_xyz,
    faces,
    camera,
    target_mask,
    *,
    downsample: int = 4,
    exclude_vertex_ids=None,
    cull_backfaces: bool = False,
    cull_flip: bool = False,
):
    """Raster mesh silhouette ``alpha`` and resized GT ``target`` (same layout as GS sil loss)."""
    if mesh_xyz.ndim != 3:
        raise ValueError(f"mesh_xyz must be [B,V,3], got {mesh_xyz.shape}")
    ds = max(1, int(downsample))
    w = int(camera.width) // ds
    h = int(camera.height) // ds
    target = _resize_mask(target_mask, h, w).to(mesh_xyz.device, dtype=mesh_xyz.dtype)
    faces_eff = _exclude_faces_by_vertex_ids(faces, exclude_vertex_ids)
    if cull_backfaces:
        front = _front_facing_mask(mesh_xyz, faces_eff, camera, flip=cull_flip)
        faces_front = _faces_for_batch(faces_eff, front, want_front=True)
        if faces_front.shape[0] == 0:
            faces_front = faces_eff
    else:
        faces_front = faces_eff
    alpha = _mesh_alpha_nvdiffrast(mesh_xyz, faces_front, camera, width=w, height=h)
    return alpha, target, faces_eff, camera


def loss_mesh_silhouette(
    mesh_xyz,
    faces,
    camera,
    target_mask,
    *,
    image_size: int = 512,
    downsample: int = 4,
    exclude_vertex_ids=None,
    cull_backfaces: bool = False,
    cull_flip: bool = False,
    backface_curl_weight: float = 0.0,
    use_edt: bool = False,
    cfg=None,
    batch=None,
):
    """
    Stage-1 mesh silhouette alignment loss from mesh raster alpha vs GT mask.

    With ``cull_backfaces=True``, only front-facing triangles contribute to the main
    silhouette term. Optional ``backface_curl_weight`` penalizes alpha from back faces.

    ``use_edt``: same EDT fields as ``loss_silhouette_edt`` (see ``silhouette.py``).
    """
    alpha, target, faces_eff, sil_camera = _mesh_silhouette_alpha_target(
        mesh_xyz,
        faces,
        camera,
        target_mask,
        downsample=downsample,
        exclude_vertex_ids=exclude_vertex_ids,
        cull_backfaces=cull_backfaces,
        cull_flip=cull_flip,
    )

    if use_edt and cfg is not None and batch is not None:
        from losses.silhouette import loss_silhouette_edt, silhouette_edt_distance_fields

        d_out, d_in = silhouette_edt_distance_fields(batch, cfg, alpha)
        if d_out is not None and d_in is not None:
            loss = loss_silhouette_edt(
                alpha,
                d_out,
                d_in,
                w_ext=float(getattr(cfg, "silhouette_edt_w_ext", 1.0)),
                w_int=float(getattr(cfg, "silhouette_edt_w_int", 1.0)),
                max_dist_px=float(getattr(cfg, "silhouette_edt_max_dist_px", 50.0)),
            )
        else:
            loss = (alpha - target).pow(2).mean()
    else:
        loss = (alpha - target).pow(2).mean()

    w_curl = float(backface_curl_weight)
    if w_curl > 0.0:
        curl = loss_mesh_backface_curl(
            mesh_xyz,
            faces_eff,
            sil_camera,
            image_size=max(sil_camera.width, sil_camera.height),
            downsample=downsample,
            cull_flip=cull_flip,
        )
        loss = loss + w_curl * curl
    return loss
