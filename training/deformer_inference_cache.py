"""Precompute template_mlp / expr_mlp outputs for inference (per checkpoint load)."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DeformerInferenceCache:
    """Vertex-static deformer outputs reused across frames."""

    template_delta: torch.Tensor
    expression_basis: torch.Tensor | None = None
    pose_weight: torch.Tensor | None = None


@torch.no_grad()
def build_deformer_inference_cache(
    deformer,
    *,
    use_expression_basis: bool = True,
    cache_pose_weight: bool = True,
) -> DeformerInferenceCache:
    template_delta = deformer.template_delta()
    expression_basis = deformer.expression_delta_basis() if use_expression_basis else None
    pose_weight = None
    if cache_pose_weight:
        pose_weight = deformer.pose_weight_net(deformer.canonical_xyz_template)
    return DeformerInferenceCache(
        template_delta=template_delta,
        expression_basis=expression_basis,
        pose_weight=pose_weight,
    )


@torch.no_grad()
def install_deformer_inference_cache(deformer, spec) -> DeformerInferenceCache:
    """
    Attach cache on ``deformer._inference_cache`` for ``ICTDeformer.forward``.

    ``template_delta`` always cached. ``expression_basis`` when ``spec.train_expression_deform``.
    ``pose_weight`` when pose is not fixed at 1.0.
    """
    use_expr = bool(getattr(spec, "train_expression_deform", False))
    cache_pose = not bool(getattr(spec, "pose_weight_one", False))
    cache = build_deformer_inference_cache(
        deformer,
        use_expression_basis=use_expr,
        cache_pose_weight=cache_pose,
    )
    deformer._inference_cache = cache
    v = int(cache.template_delta.shape[0])
    parts = [f"template_delta [{v}, 3]"]
    if cache.expression_basis is not None:
        j = int(cache.expression_basis.shape[0])
        parts.append(f"expression_basis [{j}, {v}, 3]")
    if cache.pose_weight is not None:
        parts.append(f"pose_weight [{v}, 1]")
    print(f"deformer inference cache: {', '.join(parts)}")
    return cache


def clear_deformer_inference_cache(deformer) -> None:
    if hasattr(deformer, "_inference_cache"):
        deformer._inference_cache = None
