"""Training config for MediaPipe → ICT → surface/eye 3DGS stack."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

SplitNames = Union[str, list[str]]


@dataclass
class Config:
    # data — default ``ImageDataset`` (``dataset/image_dataset.py``)
    # ``input_dir/{scene}/image`` — scene name or list of scenes (e.g. ``["MVI_1797", "MVI_1801"]``)
    dataset_type: str = "flare"  # ``flare``|``image`` → ImageDataset layout; ``mp_npz`` → VideoDataset
    input_dir: Path = Path("/Bean/data/gwangjin/2024/nbshapes/flare_2/justin/justin")
    train_split: SplitNames = field(default_factory=lambda: ["train"])
    eval_split: SplitNames = field(default_factory=lambda: ["test"])
    mp_cache_dir: Path = Path("cache/mediapipe")
    mask_edt_cache_dir: Path = Path("cache/mask_edt")
    rebuild_mask_edt_cache: bool = False
    segmentation_dir: Path = Path("cache/segmentation")
    face_landmarker_task: Path = Path("assets/face_landmarker.task")
    rebuild_mp_cache: bool = False
    bshapes_mode_percentile: float = 10.0
    fa_bbox_scale: float = 1.25
    distribution_sample_ratio: float = 0.75
    distribution_low_weight: float = 0.05
    distribution_high_cap: float = 1.0
    distribution_var_eps: float = 5e-2  # per-dim variance floor for MP+pose sampling weights
    # Up-weight frames with high RGB L1 EMA (hard fits: mouth interior, etc.) even if pose is near mean.
    distribution_rgb_ema_enabled: bool = False
    distribution_rgb_ema_beta: float = 0.995  # per-frame EMA of detached rgb_l1
    distribution_rgb_ema_scale: float = 1.0  # scales normalized EMA before lift
    distribution_rgb_ema_max_lift: float = 0.0  # max fraction of gap to max(dist); never exceeds max MP+pose score
    eye_blink_lo_percentile: float = 10.0

    eye_au_min_range: float = 0.4

    # assets
    ict_npy: Path = Path("assets/ict_facekit_torch.npy")
    mediapipe_name_to_ict: Path = Path("assets/mediapipe_name_to_indices.pkl")
    mp_embedding: Path = Path("assets/ict_mediapipe_landmark_indices.npz")
    camera_npz: Path = Path("assets/default_camera.npz")

    # model
    num_mp_blendshapes: int = 52
    num_ict_expressions: int = 53
    # GB-style mouth interior: socket/teeth/gum/tongue verts use jaw AU only (ICT + expr_mlp).
    mouth_interior_jaw_only_expression: bool = True
    # color_expression off on mouth interior + eye socket/sclera/occlusion Gaussians.
    color_expression_exclude_mouth_eye: bool = True
    # Per-triangle samples (uniform bary); ×1.5 vs prior face/head counts (GB-align init ablation).
    n_surface_gaussians_per_face: int = 8
    n_surface_gaussians_per_head: int = 8
    n_surface_gaussians_per_mouth_socket: int = 1
    n_surface_gaussians_mouth_interior: int = 1
    # Teeth (ICT code 7): initial h ~ Uniform(-radius, radius) along tooth normal (~1 cm).
    teeth_h_radius: float = 0.005
    n_surface_gaussians_per_teeth: int = 2  # ICT teeth tris → semantic mouth_interior
    n_surface_gaussians_per_eye_socket: int = 1
    n_surface_gaussians_per_eyeball_sclera: int = 0
    n_surface_gaussians_per_eye_occlusion: int = 6
    gaussian_scale_knn_k: int = 5
    gaussian_scale_knn_factor: float = 0.66  # GB: log(sqrt(dist²)); was 0.12
    gaussian_color_random_init: bool = True  # GB face_gs_model: rand/255 → logit DC
    # SplattingAvatar ``with_mesh_scaling``: exp(log_scale * A_pose/A_cano) per face.
    gaussian_with_mesh_scaling: bool = True
    # Hard cap on rendered Gaussian scale = factor * geometry_max_scale (0 = off).
    # Safety net so no single splat covers a whole region; reg + prune do the rest.
    gaussian_scale_max_clamp_factor: float = 0.0
    gamma_min: float = 0.2  # legacy affine only; symmetric log uses 1/gamma_max
    gamma_max: float = 5.0
    gamma_symmetric_log: bool = True  # gamma=exp(span*(2*sig-1)); span=log(gamma_max)
    additive_gamma_correction: bool = False  # coeffs = ict_raw + head_gamma residual
    au_active_sample_ratio: float = 0.3  # legacy ``mp_npz`` VideoDataset only
    au_active_thresh: float = 0.12

    # train
    use_stage_schedule: bool = True
    batch_size: int = 1
    # 0 = main-process loading only (avoids forked workers OOM on large FLARE caches).
    num_workers: int = 0
    dataloader_persistent_workers: bool = True  # reuse workers across epoch restarts
    dataloader_prefetch_factor: int = 0
    pin_memory: bool = False
    iterations: int = 60000
    lr_tracker: float = 1e-3
    lr_pose_weight: float = 1e-3
    lr_deformer: float = 1e-3
    lr_gaussian: float = 1e-3
    log_every: int = 100
    grad_clip_max_norm: float = 5.0
    seed: int = 0
    deterministic: bool = True
    output_root: Path = Path("/Bean/log/gwangjin/2026/neural_blendshapes_5/justin")
    eval_max_frames: int = 500  # 0 = all eval frames per stage-end render
    eval_render_interval: int = 10000  # periodic mid-training eval render (global steps)
    # Stage 2 early: print MP/ICT jawOpen & mouthClose when both MP coeffs are small
    mouth_debug_enabled: bool = False
    mouth_debug_stage_local_max: int = 3000
    mouth_debug_jaw_open_max: float = 0.15
    mouth_debug_mouth_close_max: float = 0.35

    @property
    def checkpoint_dir(self) -> Path:
        return self.output_root / "checkpoints"

    @property
    def codes_dir(self) -> Path:
        return self.output_root / "codes"

    @property
    def eval_render_dir(self) -> Path:
        return self.output_root / "renders"

    # loss weights (defaults; stages override). RGB/silhouette: GB config_blendshapes.py
    w_rgb: float = 1.0
    w_normal: float = 0.0  # Laplacian normal loss (reference_codes/normal.py); needs gt_normal + render["normal"]
    load_gt_normals: bool = True  # ``{scene}/normal/<stem>.png`` from prepare_normals
    rgb_ssim_lambda: float = 0.1
    w_lpips: float = 0.0
    lpips_net: str = "alex"
    w_mp_lmk: float = 14.0  # global default; stages override (bootstrap uses this tier)
    mp_lmk_iris_weight: float = 6.0  # per-point multiplier for MP iris 468–477 inside mp_lmk loss
    mp_lmk_mouth_weight: float = 5.0  # lip-only MP indices in mp_lmk loss (see MP_LIP_IDS)
    # Landmark distance: smooth_l1 | l2 | l1 | charbonnier | wing (see losses/landmark_distance.py)
    lmk_distance_metric: str = "smooth_l1"
    lmk_charbonnier_eps: float = 1e-4  # UV; smooth_l1 / charbonnier ε
    lmk_wing_w_px: float = 10.0  # paper Eq.5 default (pixel coords)
    lmk_wing_eps_px: float = 2.0  # paper default; do not use <<2px
    w_pie68_jaw: float = 5.0  # PIE-68 jaw 0:landmark_start only (FA); stages override
    w_mesh_seg: float = 0.0  # mesh nvdiffrast 3-class seg (bootstrap + early stage 2)
    pie68_jaw_score_thresh: float = 0.3
    w_silhouette: float = 10.0
    w_mp_mask: float = 10.0  # alias for silhouette (GB alpha_loss)
    silhouette_detach_covariance: bool = False  # False lets silhouette sharpen scale/rotation
    silhouette_l1: bool = False  # True: |α-mask| mean; False: L2 (GB default). L1 can sharpen α slightly
    silhouette_use_edt: bool = False  # L2 silhouette; EDT off for all stages
    mesh_silhouette_use_edt: Optional[bool] = False  # mesh sil L2 (bootstrap stage 1)
    silhouette_edt_w_ext: float = 1.0
    silhouette_edt_w_int: float = 1.0
    silhouette_edt_max_dist_px: float = 50.0
    mesh_silhouette_cull_backfaces: bool = False
    mesh_silhouette_cull_flip: bool = False  # flip if ICT winding is opposite
    mesh_backface_curl_weight: float = 0.0  # stage1 bootstrap often 0.2–0.4
    precompute_mask_edt_cache: bool = False  # only when silhouette_use_edt True
    mask_edt_downsample: int = 4
    mask_edt_normalize: bool = False  # keep pixel EDT; loss applies clamp / max_dist_px
    w_seg: float = 0.0  # gsplat semantic render vs FLARE part_label → class (see flare_semantic)
    seg_l1: bool = False  # False: L2 on semantic accum (GB-style); True: L1
    seg_alpha_min: float = 0.02  # CE only where rendered semantic_alpha exceeds this
    w_lip_mouth_leak: float = 0.25  # lip-only mouth_interior leak; keep << w_seg (stages 2–3)
    w_h: float = 0.10
    w_geometry: float = 0.1  # stages 2–3 via BASIC_LOSS; bootstrap overrides to 0
    w_opacity: float = 0.0
    w_opacity_loose: float = 0.001
    w_opacity_headneck: float = 0.01
    w_face_region: float = 0.0
    face_region_alpha_min: float = 0.02
    w_color_expr_sparse: float = 0.0
    w_color_expr_group_sparse: float = 0.0  # per blendshape k: few Gaussians when c_k active
    w_color_expr_per_gaussian: float = 0.0  # per Gaussian: few active k at each splat
    expression_support_train_mask: float = 0.25
    lambda_sparsity: float = 0.0
    w_gamma_prior: float = 0.0
    h_w_skin: float = 2.4
    h_w_nose: float = 1.4
    h_w_eye: float = 2.8
    h_w_brow: float = 1.4
    h_w_neck: float = 1.4
    h_w_cloth: float = 0.1
    h_w_misc: float = 1.4
    h_w_mouth: float = 0.0  # oral cavity: no image-space h reg (seg + lip_mouth_leak)
    h_w_hair: float = 0.015
    h_w_glasses: float = 0.006
    h_teeth_h_loss_scale: float = 1.0  # teeth GS only; lower = looser h on teeth
    h_eye_occlusion_h_loss_scale: float = 2.5  # ICT eye_occlusion GS; >1 = stronger |h|→0
    h_alpha_min: float = 0.08
    geometry_max_scale: float = 0.004
    thresh_scaling_max: float = 0.008
    thresh_scaling_ratio: float = 20.0
    opacity_target: float = 1.0
    opacity_loose_target: float = 1.0
    opacity_w_skin: float = 1.0
    opacity_w_other: float = 0.05
    w_pose_prior: float = 0.0
    w_template_smooth: float = 0.0
    w_template_laplacian: float = 0.0
    w_template_scale_prior: float = 0.0
    w_opacity_decay: float = 0.0

    # render (gsplat only)
    image_size: int = 512
    n_semantic_classes: int = 3  # others, mouth_interior, eye_occlusion
    # FLARE ``semantic/*.png`` → h-reg masks + ``part_label`` for ``w_seg`` supervision.
    load_dataset_semantic: bool = True
    # Full matting mask for RGB / silhouette / seg (no INSTA-style semantic exclude).
    tight_mask_from_semantic: bool = False
    tight_mask_exclude_parts: List[int] = field(default_factory=lambda: [16])
    tight_mask_median_ksize: int = 5
    tight_mask_exclude_dilate_iters: int = 3
    tight_mask_keep_mouth_interior: bool = True
    tight_mask_keep_mouth_dilate_iters: int = 1
    save_eval_checkpoint: bool = False  # eval_step_*.pt (point cloud / state) in render_eval_set
    sh_degree: Optional[int] = 3
    # gsplat only: "classic" (3DGS-like, matches GB diff_gaussian_rasterization) or "antialiased".
    gsplat_rasterize_mode: str = "classic"

    gaussian_face_center_init: bool = False
    gaussian_densify: bool = True
    gaussian_densify_max: int = 900000
    # Densify: stage 2 grow full stage; stage 3 grow first 5k local, then cleanup prune 5k–10k.
    gaussian_densify_stages: list[str] = field(
        default_factory=lambda: ["2_coarse_mesh", "3_expression_detail"]
    )
    # Global fallback only if ``gaussian_*_stage_local`` missing; stage 2 uses local windows.
    gaussian_densify_start_iter: int = 7501  # after bootstrap (2500+5000)
    gaussian_densify_stop_iter: int = 22500  # end of 2_coarse_mesh (7500+15000)
    # Stage-local grow+prune (clone/split + inline prune every gaussian_densify_every).
    gaussian_densify_stage_local: Dict[str, List[int]] = field(
        default_factory=lambda: {"2_coarse_mesh": [1, 15000], "3_expression_detail": [1, 5000]}
    )
    # Prune-only window (no clone/split). Stage 2: none. Stage 3: local 5000–10000.
    gaussian_cleanup_stage_local: Dict[str, List[int]] = field(
        default_factory=lambda: {"3_expression_detail": [5000, 7500]}
    )
    # GB train.py: opacity reset at interval is ``pass`` (disabled); no opacity decay.
    gaussian_cleanup_reset_stage_local: Dict[str, List[int]] = field(default_factory=dict)
    gaussian_cleanup_prune_every: int = 100
    gaussian_densify_every: int = 100
    gaussian_densify_accum_every: int = 1
    gaussian_prune_every: int = 100
    gaussian_reset_every: int = 0
    gaussian_reset_iters: List[int] = field(default_factory=list)
    # 3DGS / SplattingAvatar ``opacity_reset_interval=3000``: re-justify opacity to cull
    # over-large / over-overlapping Gaussians. Stage-2 only (avoid wiping stage-3 detail).
    gaussian_reset_stage_local: Dict[str, List[int]] = field(
        default_factory=dict
    )
    gaussian_reset_opacity_value: float = 0.01
    gaussian_opacity_decay_every: int = 0
    gaussian_opacity_decay_stage_local: Dict[str, List[int]] = field(default_factory=dict)
    gaussian_opacity_decay_factor: float = 0.98
    gaussian_opacity_decay_min: float = 1e-4
    # When True, multiply opacity by ``gaussian_opacity_decay_factor`` on each stage-3 cleanup prune step.
    gaussian_opacity_decay_during_cleanup: bool = False
    # arxiv:2404.06109 Sec. 3.5 (opacity regularization): subtract fixed amount from each
    # primitive's opacity after every densification run. 0 = disabled (default; other sweeps).
    gaussian_opacity_subtract_after_densify: float = 0.0
    # GB ``densify_grad_threshold`` (mouth interior / eye / head); face skin × ``face_scale``.
    gaussian_grow_grad2d: float = 1.5e-4
    gaussian_grow_grad2d_face_scale: float = 1.5  # face code 4 → 0.001 (GB 0.0002 × 5)
    gaussian_grow_grad2d_region_split: bool = True  # False = legacy uniform base×face_scale
    gaussian_grow_grad2d_pixel_scale: bool = True
    gaussian_densify_warmup_local: int = 500
    gaussian_grow_gradrgb: float = 2e-7
    gaussian_grow_option: str = "grad2d"
    gaussian_prune_opa: float = 0.005
    # Screen-size prune (radii > 20px) starts after local step 3000 to match GB.
    gaussian_prune_screen_after_local: int = 3000
    gaussian_max_per_face: int = -1
    # GB / 3DGS: max(σ) > ratio * camera_extent (FLAME-normalized space; ICT via NICP).
    gaussian_scene_extent: float = 1.0  # GB camera_extent
    gaussian_prune_world_scale_ratio: float = 0.075  # GB: max(scale) > 0.1 * extent
    gaussian_percent_dense: float = 0.0075  # GB percent_dense
    gaussian_prune_screen_radius: float = 15.0  # GB max_screen_size; lower → prune blurry large splats sooner
    gaussian_split_scale_divisor: float = 1.6  # GB: 0.8 * N, N=2
    gaussian_clone_opacity_correction: bool = False
    gaussian_triangle_walk_every: int = 10
    gaussian_triangle_walk_max_iter: int = 3
    gaussian_densify_walk_after_grow: bool = False
    # Split: GB ``densify_and_split`` — per-parent N(0, scale) in local frame.
    # Mesh: target_world = parent max scale; bary std via per-face sqrt(2A).
    gaussian_split_bary_noise_area_normalize: bool = True
    gaussian_split_bary_noise_area_eps: float = 0.0  # 0 = median_area * 1e-3
    gaussian_split_bary_noise_gb_match: bool = True  # target_world = parent scale (not fixed)
    gaussian_split_bary_noise: float = 0.12  # manual σ_ref when gb_match=False
    gaussian_split_h_noise: float = 0.001

    gsplat_submodule: Path = Path("gsplat")

    # packed=True saves memory but gsplat packed backward can yield NaN grads (see debug_nan.py).
    gsplat_packed: bool = False