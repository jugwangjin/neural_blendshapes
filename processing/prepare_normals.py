# FROM  https://github.com/philgras/neural-head-avatars/python_scripts/video_to_dataset.py
# normal estimation part

# landmark detector: uses dlib, from stylegan-encoder (ffhq-alignment source)

import sys
import os
# add this file's parent directory into sys path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import os 
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
parent_dir = os.path.dirname(script_dir)
os.chdir(parent_dir)
sys.path.append(parent_dir)
import argparse

import numpy as np
from PIL import Image
import torch
from face_normals.resnet_unet import ResNetUNet
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as ttf
import bz2
import dlib

import tqdm

print(os.getcwd())
NORMAL_MODEL_PATH = os.path.join(parent_dir, "assets/face_normals.pth")
# https://github.com/boukhayma/face_normals/
# https://drive.google.com/file/d/1Qb7CZbM13Zpksa30ywjXEEHHDcVWHju_

LANDMARKS_MODEL_URL = os.path.join(parent_dir, "assets/shape_predictor_68_face_landmarks.dat.bz2")
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

def unpack_bz2(src_path):
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path

class LandmarksDetector:
    def __init__(self, predictor_model_path):
        """
        :param predictor_model_path: path to shape_predictor_68_face_landmarks.dat file
        """
        self.detector = dlib.get_frontal_face_detector() # cnn_face_detection_model_v1 also can be used
        self.shape_predictor = dlib.shape_predictor(predictor_model_path)

    def get_landmarks(self, image):
        img = dlib.load_rgb_image(image)
        dets = self.detector(img, 1)

        for detection in dets:
            face_landmarks = [(item.x, item.y) for item in self.shape_predictor(img, detection).parts()]
            yield face_landmarks

def get_face_bbox(lmks, img_size):
    """
    Computes facial bounding box as required in face_normals
    :param lmks:
    :param img_size:
    :return: (vertical_start, vertical_end, horizontal_start, horizontal_end)
    """

    umin = np.min(lmks[:, 0])
    umax = np.max(lmks[:, 0])
    vmin = np.min(lmks[:, 1])
    vmax = np.max(lmks[:, 1])

    umean = np.mean((umin, umax))
    vmean = np.mean((vmin, vmax))

    l = round(1.2 * np.max((umax - umin, vmax - vmin)))

    if l > np.min(img_size):
        l = np.min(img_size)

    us = round(np.max((0, umean - float(l) / 2)))
    ue = us + l

    vs = round(np.max((0, vmean - float(l) / 2)))
    ve = vs + l

    if ue > img_size[1]:
        ue = img_size[1]
        us = img_size[1] - l

    if ve > img_size[0]:
        ve = img_size[0]
        vs = img_size[0] - l

    us = int(us)
    ue = int(ue)

    vs = int(vs)
    ve = int(ve)

    return vs, ve, us, ue

def get_face_normals():
    model = ResNetUNet(n_class=3).cuda()
    model.load_state_dict(torch.load(NORMAL_MODEL_PATH))
    model.eval()
    return model

def get_landmark_estimator():
    landmarks_model_path = unpack_bz2(LANDMARKS_MODEL_URL)
    landmarks_detector = LandmarksDetector(landmarks_model_path)
    return landmarks_detector

def estimate_face_normals(model_normal, model_landmark, image_path, save_path):
    img = Image.open(image_path)    
    img = ttf.to_tensor(img)
    img_size = img.shape[-2:]

    lmks = None
    for i, face_landmarks in enumerate(model_landmark.get_landmarks(image_path), start=1):
        lmks = np.array(face_landmarks)
        break
        
    if lmks is None: 
        return

    t, b, l, r = get_face_bbox(lmks, img_size)
    crop = img[:, t:b, l:r]
    crop = ttf.resize(crop, 256, InterpolationMode.BICUBIC)
    crop = crop.clamp(-1, 1) * 0.5 + 0.5

    normals = model_normal(crop[None].cuda())[0]
    normals = normals / torch.sqrt(torch.sum(normals ** 2, dim=1, keepdim=True))
    rescaled_normals = ttf.resize(
        normals[0], (b - t, r - l), InterpolationMode.BILINEAR
    )
    rescaled_normals[0] *= -1
    normals = - torch.ones_like(img)
    normals[:, t:b, l:r] = rescaled_normals.cpu()

    # R = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]], device=normals.device, dtype=torch.float32)
    # normals = torch.einsum('ixy,ij->jxy', normals, R)

    normal_img = ttf.to_pil_image(normals * 0.5 + 0.5)
    normal_img.save(save_path)


def main(args):
    model_normal = get_face_normals()
    model_landmark = get_landmark_estimator()

    for vid in os.listdir(args.input):
        if not os.path.isdir(os.path.join(args.input, vid)):
            continue
        os.makedirs(os.path.join(args.input, vid, 'normal'), exist_ok=True)
        for img in tqdm.tqdm(os.listdir(os.path.join(args.input, vid, 'image')), desc=vid):
            # check if img is a file with image extensions and call estimate_face_normals
            if img.split('.')[-1] in ['jpg', 'jpeg', 'png', 'bmp']:
                estimate_face_normals(model_normal, model_landmark, os.path.join(args.input, vid, 'image', img), os.path.join(args.input, vid, 'normal', img))

    

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Prepare normals for face images')
    parser.add_argument('--input', type=str, help='Input image path')

    args = parser.parse_args()

    main(args)

