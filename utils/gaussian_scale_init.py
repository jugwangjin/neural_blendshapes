"""KNN-based Gaussian scale init (3DGS / gsplat style: log σ = log √d̄)."""

import torch

from utils.barycentric import sample_normals, sample_surface


def knn_mean_distance(xyz, k=3):
    """
    Per-point mean Euclidean distance to ``k`` nearest neighbors (excluding self).

    ``xyz`` [N, 3]. Tries ``simple_knn.distCUDA2`` (original 3DGS) then scipy cKDTree.
    """
    xyz = xyz.detach().float()
    n = xyz.shape[0]
    if n <= 1:
        return torch.full((n,), 1e-3, device=xyz.device, dtype=xyz.dtype)

    k_query = min(int(k) + 1, n)

    try:
        import simple_knn._C as simple_knn_cuda

        dist2 = simple_knn_cuda.distCUDA2(xyz.contiguous())
        return torch.sqrt(dist2.clamp(min=1e-7))
    except ImportError:
        pass

    from scipy.spatial import cKDTree

    pts = xyz.cpu().numpy()
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=k_query, workers=-1)
    if k_query == 1:
        mean_d = dists
    else:
        mean_d = dists[:, 1:].mean(axis=1)
    return torch.tensor(mean_d, device=xyz.device, dtype=xyz.dtype)


def log_scale_from_knn(xyz, k=3, scale_factor=1.0):
    """
    Isotropic log-scale [N, 3] for ``exp(log_scale)`` parameterization.

    Matches 3DGS: ``log(sqrt(clamp_min(knn_dist)))`` (optionally × ``scale_factor``).
    """
    d = knn_mean_distance(xyz, k=k) * float(scale_factor)
    log_s = torch.log(d.clamp(min=1e-7))
    return log_s.unsqueeze(1).expand(-1, 3)


def surface_gaussian_xyz(ict, face_idx, bary, h=None, h_sigma_scale=None):
    """World positions of surface Gaussians at init (canonical / neutral)."""
    device = face_idx.device
    verts = ict.template_reference_verts().to(device=device)
    faces = ict.faces.to(device=device)
    xyz_base = sample_surface(verts, faces, face_idx, bary)
    if h is None:
        return xyz_base
    from utils.mesh_ops import vertex_normals

    vn = vertex_normals(verts, faces)
    n = sample_normals(vn, faces, face_idx, bary)
    h = h.to(device=device)
    if h_sigma_scale is not None:
        h_sigma_scale = h_sigma_scale.to(device=device)
        h_eff = h * h_sigma_scale.unsqueeze(1)
    else:
        h_eff = h
    return xyz_base + h_eff * n


def init_module_log_scale(module, xyz, k=3, scale_factor=1.0):
    with torch.no_grad():
        module.log_scale.copy_(log_scale_from_knn(xyz, k=k, scale_factor=scale_factor))
