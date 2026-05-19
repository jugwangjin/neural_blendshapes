"""
Legacy 68-point ICT vertex landmark loss (gbuffer clip-space path).

Do not use in train.py. Active training uses baked MP embedding via
losses/mediapipe_landmark_478.py and assets/ict_mediapipe_landmark_embedding_from_metrical_tracker.npz.
"""

import torch


def loss_ict_landmarks_68_gbuffer(ict_facekit, gbuffers, views_subset):
    """
    Pick ICT mesh landmark vertices in clip space and compare to dataset landmarks.

    Requires gbuffers['deformed_verts_clip_space'] and views_subset['landmark'].
    """
    landmark_indices = ict_facekit.landmark_indices
    landmarks_on_clip_space_ = gbuffers["deformed_verts_clip_space"][:, landmark_indices]
    detected_landmarks = views_subset["landmark"]
    return (landmarks_on_clip_space_ - detected_landmarks).pow(2).mean()
