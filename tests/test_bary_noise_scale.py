import torch

from training.densify import per_gaussian_bary_noise_std, resolve_bary_noise_base_std


class _Cfg:
    gaussian_scene_extent = 1.0
    gaussian_percent_dense = 0.0025
    gaussian_split_bary_noise_gb_match = True
    gaussian_split_bary_noise = 0.12
    gaussian_split_bary_noise_area_normalize = True
    gaussian_split_bary_noise_area_eps = 0.0


def test_gb_match_median_face_reference():
    areas = torch.tensor([1.0, 4.0, 1.0])
    cfg = _Cfg()
    ref_scale = 0.0025
    parent = torch.tensor([ref_scale, ref_scale, ref_scale])
    fi = torch.tensor([0, 1, 2])
    std = per_gaussian_bary_noise_std(fi, areas, cfg, parent)
    base = resolve_bary_noise_base_std(areas, cfg, parent_scale_ref=ref_scale)
    assert abs(std[0].item() - base) < 1e-6
    assert abs(std[2].item() - base) < 1e-6
    assert std[1].item() < std[0].item()


def test_larger_parent_scale_increases_std():
    areas = torch.tensor([1.0, 1.0])
    cfg = _Cfg()
    fi = torch.tensor([0, 0])
    s1 = 0.0025
    s2 = 0.005
    std_lo = per_gaussian_bary_noise_std(fi, areas, cfg, torch.tensor([s1, s1]))
    std_hi = per_gaussian_bary_noise_std(fi, areas, cfg, torch.tensor([s2, s2]))
    assert std_hi[0].item() > std_lo[0].item()
    assert abs(std_hi[0].item() / std_lo[0].item() - 2.0) < 1e-5


def test_manual_base_at_ref_parent_on_uniform_faces():
    areas = torch.ones(2)
    cfg = _Cfg()
    cfg.gaussian_split_bary_noise_gb_match = False
    cfg.gaussian_split_bary_noise = 0.07
    ref_scale = 0.0025
    fi = torch.tensor([0, 0])
    std = per_gaussian_bary_noise_std(
        fi, areas, cfg, torch.full((2,), ref_scale), base_std=0.07
    )
    assert abs(std[0].item() - 0.07) < 1e-5
