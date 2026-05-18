import torch

from model.uvh_gaussians import UVHGaussians
from utils.uv_mesh import UVMesh


def test_uvh_surface_finite():
    g = UVHGaussians(16, fixed_h=0.0)
    verts = torch.randn(50, 3)
    faces = torch.randint(0, 50, (20, 3))
    uvs = torch.rand(50, 2)
    mesh = UVMesh(verts=verts, faces=faces, verts_uvs=uvs, faces_uvs=faces)
    out = g(mesh)
    assert torch.isfinite(out["xyz"]).all()
    assert out["h"].abs().max() < 1e-5
