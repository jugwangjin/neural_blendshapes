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
from flare.core import Camera

import torch
import numpy as np
import json
from pathlib import Path
from .dataset import Dataset
from .dataset_util import _load_img, _load_mask, _load_semantic, _load_K_Rt_from_P
import face_alignment
import imageio
from tqdm import tqdm
import os

# Select the device
device = torch.device('cpu')
devices = 0
if torch.cuda.is_available() and devices >= 0:
    device = torch.device(f'cuda:{devices}')

class DatasetLoader(Dataset):
    def __init__(self, args, train_dir, sample_ratio, pre_load):
        self.args = args
        self.train_dir = train_dir
        self.base_dir = args.working_dir / args.input_dir
        self.pre_load = pre_load
        self.subject = self.base_dir.stem

        self.fixed_cam = torch.load('./assets/fixed_cam.pt')
        
        self.json_dict = {"frames": []}
        for dir in self.train_dir: 
            json_file = self.base_dir / dir / "merged_params.json"
  
            with open(json_file, 'r') as f:
                json_data = json.load(f)
                for item in json_data["frames"]:
                    # keep track of the subfolder
                    item.update({"dir":dir})
                self.json_dict["frames"].extend(json_data["frames"])

        if sample_ratio > 1:
            self.all_img_path = self.json_dict["frames"][::sample_ratio]
        else:
            self.all_img_path = self.json_dict["frames"]

        self.len_img = len(self.all_img_path)
        test_path = self.base_dir / self.all_img_path[0]["dir"] / Path(self.all_img_path[0]["file_path"] + ".png")
        self.resolution = _load_img(test_path).shape[0:2]

        # Load the camera intrinsics (Note that we use the same intrinsics for all cameras since it is shot on a single device)
        focal_cxcy = json_data["intrinsics"]
        self.K = torch.eye(3)
        self.K[0, 0] = focal_cxcy[0] * self.resolution[0]
        self.K[1, 1] = focal_cxcy[1] * self.resolution[1]
        self.K[0, 2] = focal_cxcy[2] * self.resolution[0]
        self.K[1, 2] = focal_cxcy[3] * self.resolution[1]

        self.face_alignment = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, flip_input=False, 
                                                            device='cuda' if torch.cuda.is_available() else 'cpu')


        self.shape_params = torch.tensor(json_data['shape_params']).float().unsqueeze(0)
        # Pre-load from disc to avoid slow png parsing
        if self.pre_load:
            self.all_images, self.all_masks, self.all_skin_mask, self.all_flame_expression, \
            self.all_facs, self.all_pose, self.all_camera, self.frames, self.all_landmarks, self.all_normals  = [], [], [], [], [], [], [], [], []
            print('loading all images from all_img_path')
            for i in tqdm(range(len(self.all_img_path))):
                img, mask, skin_mask, flame_expression, facs, flame_pose, camera, frame_name, landmark, normal = self._parse_frame(i)
                self.all_images.append(img)
                self.all_masks.append(mask)
                self.all_skin_mask.append(skin_mask)
                self.all_flame_expression.append(flame_expression)
                self.all_facs.append(facs)
                self.all_pose.append(flame_pose)
                self.all_camera.append(camera)
                self.frames.append(frame_name)
                self.all_landmarks.append(landmark)
                self.all_normals.append(normal)

            self.len_img = len(self.all_images)    
            print("loaded {:d} views".format(self.len_img))
        else:
            self.all_flame_expression = []
            for i in range(len(self.all_img_path)):
                self.all_flame_expression.append(self._parse_expression(i))
            self.loaded = {}

        self._compute_importance()

    def get_camera_mat(self):
        '''
        The lab captured data for some subjects has a slightly shifted t. Here, for each video, we get t and R. 
        R is the same and t is different for some subjects. 
        '''
        cam = []
        for dir in self.train_dir: 
            json_file = self.base_dir / dir / "flame_params.json"
            with open(json_file, 'r') as f:
                json_data = json.load(f)

                # since we have a fixed camera, we can sample t from any frame (we just choose the first frame here)
                world_mat = torch.tensor(_load_K_Rt_from_P(None, np.array(json_data["frames"][0]['world_mat']).astype(np.float32))[1], dtype=torch.float32)
                # camera matrix to openGL format 
                R = world_mat[:3, :3]
                R *= -1 
                t = world_mat[:3, 3]
                camera = Camera(self.K, R, t, device=device)
                cam.append(camera)
        return cam

    def get_mean_expression_train(self, train_dir):
        # return torch.zeros(1, 52)
        all_expression = []
        json_dict = {"frames": []}
        for dir in train_dir: 
            json_file = self.base_dir / dir / "merged_params.json"

            with open(json_file, 'r') as f:
                json_data = json.load(f)
                for item in json_data["frames"]:
                    # keep track of the subfolder
                    item.update({"dir":dir})
                json_dict["frames"].extend(json_data["frames"])    
        for i in range(len(json_dict)):    
            flame_expression = torch.tensor(json_dict["frames"][i]["expression"], dtype=torch.float32)
            all_expression.append(flame_expression[None, ...])
        return torch.mean(torch.cat(all_expression), 0, keepdim=True)
    
    def get_mean_expression(self):
        mean_expression = torch.mean(torch.concat(self.all_flame_expression), 0, keepdim=True)
        return mean_expression

    def _parse_expression(self, idx):
        json_dict = self.all_img_path[idx]
        
        flame_expression = torch.tensor(json_dict["expression"], dtype=torch.float32)
        return flame_expression[None, ...] # Add batch dimension

    def resolution(self):
        return self.resolution

    def _parse_frame(self, idx):
        json_dict = self.all_img_path[idx]
        img_path = self.base_dir / json_dict["dir"] / Path(json_dict["file_path"] + ".png")
        
        # ================ semantics =======================
        semantic_parent = img_path.parent.parent / "semantic"
        semantic_path = semantic_parent / (img_path.stem + ".png")
        semantic = _load_semantic(semantic_path)
    
        # ================ img & mask =======================
        img  = _load_img(img_path)

        mask_parent = img_path.parent.parent / "mask"
        if mask_parent.is_dir():
            mask_path = mask_parent / (img_path.stem + ".png")
            mask = _load_mask(mask_path)
        else:
            mask = img[..., 3].unsqueeze(-1)
            mask[mask < 0.5] = 0.0

            img = img[..., :3]
        
        semantic = semantic[..., :6]
        # black bg because we have perceptual loss  
        img = img * mask 
        
        # ================ flame and camera params =======================
        # flame params
        flame_pose = torch.tensor(json_dict["pose"], dtype=torch.float32)
        flame_expression = torch.tensor(json_dict["expression"], dtype=torch.float32)
        facs = torch.tensor(json_dict["facs"], dtype=torch.float32)

        camera = Camera(self.K, self.fixed_cam['R'], self.fixed_cam['t'], device=device)

        frame_name = img_path.stem

        with torch.no_grad():
            landmarks, scores, _ = self.face_alignment.get_landmarks_from_image(str(img_path), return_bboxes=True, return_landmark_score=True)

            if len(landmarks) == 0:
                landmark = torch.zeros(68, 4)
            else:
                landmark = torch.tensor(landmarks[0], dtype=torch.float32)
                landmark = landmark / img.size(1)
                score = torch.tensor(scores[0], dtype=torch.float32)
                # print(landmark.shape, score.shape)
                landmark = torch.cat([landmark, score[:, None]], dim=1).data

        return img[None, ...], mask[None, ...], semantic[None, ...], flame_expression[None, ...], facs[None, ...], flame_pose[None, ...], \
                camera, frame_name, landmark[None, ...] # Add batch dimension
    
    def __len__(self):
        return self.len_img

    def __getitem__(self, itr):
        if self.pre_load:
            img = self.all_images[itr % self.len_img]
            mask = self.all_masks[itr % self.len_img]
            skin_mask = self.all_skin_mask[itr % self.len_img]
            flame_expression = self.all_flame_expression[itr % self.len_img]
            flame_pose = self.all_pose[itr % self.len_img]
            camera = self.all_camera[itr % self.len_img]
            frame_name = self.frames[itr % self.len_img]
            landmark = self.all_landmarks[itr % self.len_img]
            normal = self.all_normals[itr % self.len_img]
        else:
            local_itr = itr % self.len_img
            if local_itr not in self.loaded:
                self.loaded[local_itr] = self._parse_frame_single(local_itr)
            img, mask, skin_mask, flame_expression, facs, flame_pose, camera, frame_name, landmark, normal = self._parse_frame_single(itr)

        # facs = (facs / self.facs_range).clamp(0, 1)

        return {
            'img' : img,
            'mask' : mask,
            'skin_mask' : skin_mask,
            'flame_pose' : flame_pose,
            'flame_expression' : flame_expression,
            'facs' : facs,
            'camera' : camera,
            'frame_name': frame_name,
            'idx': itr % self.len_img,
            'landmark': landmark,
            'normal': normal
        }


    def _compute_importance(self):
        len_img = self.len_img
        importance = [0] * len_img
        for idx in range(len_img):
            json_dict = self.all_img_path[idx % self.len_img]
            facs = torch.tensor(json_dict["facs"], dtype=torch.float32)

            importance[idx] = float(torch.sum(facs).cpu().data.numpy())

            # max_facs = torch.max(max_facs, facs)
        importance = torch.log(torch.tensor(importance) + 1.5)
        importance = importance / torch.max(importance)
        importance = list(importance.cpu().data.numpy())

        self.importance = importance
       


    def _parse_frame_single(self, idx):
        ''' helper function to parse a single frame and test/debug
        '''


        # json_dict = {}
        # for frame in self.all_img_path:
        #     if Path(frame["file_path"]) == Path('./image/' +  f'{idx}'):
        #         json_dict = frame.copy()
        # assert Path(json_dict["file_path"]) == Path('./image/' +  f'{idx}')

        json_dict = self.all_img_path[idx % self.len_img]

        img_path = self.base_dir / json_dict["dir"] / Path(json_dict["file_path"] + ".png")
        
        # ================ semantics =======================
        semantic_parent = img_path.parent.parent / "semantic"
        semantic_path = semantic_parent / (img_path.stem + ".png")
        semantic = _load_semantic(semantic_path)
    
        # ================ normal =======================
        normal_parent = img_path.parent.parent / "normal"
        normal_path = normal_parent / (img_path.stem + ".png")
        
        if os.path.exists(normal_path):
            normal = imageio.imread(normal_path)
            normal = torch.tensor(normal / 255, dtype=torch.float32)

        else:
            normal = torch.zeros(512, 512, 3)

        # normal mask is zero where all normal values are zero
        normal_mask = (torch.sum(normal, dim=2) > 0).float()

        # ================ img & mask =======================
        img  = _load_img(img_path)
        mask_parent = img_path.parent.parent / "mask"
        if mask_parent.is_dir():
            mask_path = mask_parent / (img_path.stem + ".png")
            mask = _load_mask(mask_path)
        else:
            mask = img[..., 3].unsqueeze(-1)
            mask[mask < 0.5] = 0.0

            img = img[..., :3]
        
        semantic = semantic[..., :6]

        # concat normal_mask on semantic
        semantic = torch.cat([semantic, normal_mask[..., None]], dim=-1) # shape of 512, 512, 7

        # black bg because we have perceptual loss  
        img = img * mask 
        
        # ================ flame and camera params =======================
        # flame params
        flame_pose = torch.tensor(json_dict["pose"], dtype=torch.float32)
        flame_expression = torch.tensor(json_dict["expression"], dtype=torch.float32)
        facs = torch.tensor(json_dict["facs"], dtype=torch.float32)
        
        camera = Camera(self.K, self.fixed_cam['R'], self.fixed_cam['t'], device=device)

        frame_name = img_path.stem

        with torch.no_grad():
            landmarks, scores, _ = self.face_alignment.get_landmarks_from_image(str(img_path), return_bboxes=True, return_landmark_score=True)

            if len(landmarks) == 0:
                landmark = torch.zeros(68, 4)
            else:
                landmark = torch.tensor(landmarks[0], dtype=torch.float32)
                # print(torch.amin(landmark, dim=0), torch.amax(landmark, dim=0))
                landmark = landmark / img.size(1)
                score = torch.tensor(scores[0], dtype=torch.float32)
                # print(landmark.shape, score.shape)
                landmark = torch.cat([landmark, score[:, None]], dim=1).data

        return img[None, ...], mask[None, ...], semantic[None, ...], flame_expression[None, ...], facs[None, ...], flame_pose[None, ...], \
                camera, frame_name, landmark[None, ...], normal[None, ...] # Add batch dimension