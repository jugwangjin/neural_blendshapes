# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import torch
import imageio
import numpy as np
import cv2
import skimage

import mediapipe as mp

def parse_mediapipe_output(face_landmarker_result):
    if len(face_landmarker_result.face_landmarks) == 0:
        return None, None, None
    landmarks = face_landmarker_result.face_landmarks[0]
    lmks = torch.from_numpy(np.array([[landmarks[i].x, landmarks[i].y, landmarks[i].z] for i in range(len(landmarks))]).astype(np.float32))

    blendshapes = face_landmarker_result.face_blendshapes[0]
    # print(blendshapes[0])
    bshape = torch.from_numpy(np.array([blendshapes[i].score for i in range(len(blendshapes))]).astype(np.float32))
    
    transform_matrix = torch.from_numpy(face_landmarker_result.facial_transformation_matrixes[0].astype(np.float32))

    return lmks, bshape, transform_matrix
    
###############################################################################
# Helpers/utils
###############################################################################

def _load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K/K[2,2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3,3] = (t[:3] / t[3])[:,0]

    return intrinsics, pose

def _load_mask(fn):
    alpha = imageio.imread(fn, mode='F') 
    alpha = skimage.img_as_float32(alpha)
    mask = torch.tensor(alpha / 255., dtype=torch.float32).unsqueeze(-1)
    mask[mask < 0.5] = 0.0
    # alpha = imageio.imread(fn) 
    # mask = torch.Tensor(np.array(alpha) > 127.5)[:, :, 1:2].bool().int().float()
    return mask

def _load_img(fn):
    img = imageio.imread(fn)
    if img.dtype != np.float32: # LDR image
        img = torch.tensor(img / 255, dtype=torch.float32)
        # look into this
        img[..., 0:3] = srgb_to_rgb(img[..., 0:3])
    else:
        img = torch.tensor(img, dtype=torch.float32)
    return img

def _load_semantic(fn):
    img = imageio.imread(fn, mode='F')
    h, w = img.shape
    semantics = np.zeros((h, w, 7))
    # Labels that ICT have
    # face, head/neck/, left eye, right eye, mouth interior
    # face + eyebrow + nose + upper lip + lower lip + ears +  == ICT-FaceKit.full_face_area
    # left eye == ICT_FaceKit.eyeball_left
    # right eye == ICT_FaceKit.eyeball_right
    # mouth interior == ICT_FaceKit.mouth_interior == ICT_Facekit.outh_socket + ICT_Facekit.gums_and_tongue + ICT_FaceKit.teeth
    # hair + cloth + necklace + neck == ICT_FaceKit.head_and_neck
    # What I missed
    # part_idx = {
    #     'background': 0,
    #     'skin': 1,
    #     'l_brow': 2,
    #     'r_brow': 3,
    #     'l_eye': 4,
    #     'r_eye': 5,
    #     'eye_g': 6, # eyeglasses, ignored
    #     'l_ear': 7,
    #     'r_ear': 8,
    #     'ear_r': 9,
    #     'nose': 10,
    #     'mouth': 11,
    #     'u_lip': 12,
    #     'l_lip': 13,
    #     'neck': 14,
    #     'neck_l': 15, # necklace
    #     'cloth': 16,
    #     'hair': 17,
    #     'hat': 18
    # }


    # tight face area : skin + nose + left eyebwow + right eyebrow + upper lip + lower lip 
    semantics[:, :, 0] = ((img == 1) + (img == 2) + (img == 3) + (img == 10) + (img == 12) + (img == 13)\
                        + (img == 17) + (img == 16) + (img == 15) + (img == 14) + (img==7) + (img==8) + (img==9)\
                            ) >= 1 # skin, nose, ears, neck, lips

    # except hair, neck, ....
    semantics[:, :, 1] = ((img == 1) + (img == 2) + (img == 3) + (img == 10) + (img == 12) + (img == 13)
                        ) >= 1 # skin, brows, nose, lips


    # skin, ear, nose, neck
    semantics[:, :, 2] = ((img == 1) + (img == 7) + (img == 8) + (img == 10) + (img == 14)) >= 1

    # left eyes
    semantics[:, :, 3] = ((img == 5)) >= 1

    # right eyes
    semantics[:, :, 4] = ((img == 4)) >= 1

    #inside mouth
    semantics[:, :, 5] = (img == 11) >= 1  # will it include the teeth and 

    semantics[:, :, 6] = 1. - np.sum(semantics[:, :, :-1], 2) # background

    semantics = torch.tensor(semantics, dtype=torch.float32)
    return semantics


#----------------------------------------------------------------------------
# sRGB color transforms:Code adapted from Nvdiffrec
#----------------------------------------------------------------------------

def _rgb_to_srgb(f: torch.Tensor) -> torch.Tensor:
    return torch.where(f <= 0.0031308, f * 12.92, torch.pow(torch.clamp(f, 0.0031308), 1.0/2.4)*1.055 - 0.055)

def rgb_to_srgb(f: torch.Tensor) -> torch.Tensor:
    assert f.shape[-1] == 3 or f.shape[-1] == 4
    out = torch.cat((_rgb_to_srgb(f[..., 0:3]), f[..., 3:4]), dim=-1) if f.shape[-1] == 4 else _rgb_to_srgb(f)
    assert out.shape[0] == f.shape[0] and out.shape[1] == f.shape[1] and out.shape[2] == f.shape[2]
    return out

def _srgb_to_rgb(f: torch.Tensor) -> torch.Tensor:
    return torch.where(f <= 0.04045, f / 12.92, torch.pow((torch.clamp(f, 0.04045) + 0.055) / 1.055, 2.4))

def srgb_to_rgb(f: torch.Tensor) -> torch.Tensor:
    assert f.shape[-1] == 3 or f.shape[-1] == 4
    out = torch.cat((_srgb_to_rgb(f[..., 0:3]), f[..., 3:4]), dim=-1) if f.shape[-1] == 4 else _srgb_to_rgb(f)
    assert out.shape[0] == f.shape[0] and out.shape[1] == f.shape[1] and out.shape[2] == f.shape[2]
    return out

