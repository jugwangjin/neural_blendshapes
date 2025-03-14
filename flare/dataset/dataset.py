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

# Select the device
device = torch.device('cpu')
devices = 0
if torch.cuda.is_available() and devices >= 0:
    device = torch.device(f'cuda:{devices}')

class Dataset(torch.utils.data.Dataset):
    """Basic dataset interface"""
    def __init__(self): 
        super().__init__()

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self):
        raise NotImplementedError

    def collate(self, batch):
        return {
            'img' : torch.cat(list([item['img'] for item in batch]), dim=0).to(device),
            'img_path' : list([item['img_path'] for item in batch]),
            'mask' : torch.cat(list([item['mask'] for item in batch]), dim=0).to(device),
            'skin_mask' : torch.cat(list([item['skin_mask'] for item in batch]), dim=0).to(device),
            'camera': list([item['camera'] for item in batch]),
            'frame_name': list([item['frame_name'] for item in batch]),
            'idx': torch.LongTensor(list([item['idx'] for item in batch])).to(device),
            'landmark' : torch.cat(list([item['landmark'] for item in batch]), dim=0).to(device),
            'mp_landmark': torch.cat(list([item['mp_landmark'] for item in batch]), dim=0).to(device),
            'mp_blendshape' : torch.cat(list([item['mp_blendshape'] for item in batch]), dim=0).to(device),
            'mp_transform_matrix' : torch.cat(list([item['mp_transform_matrix'] for item in batch]), dim=0).to(device),
            'normal' : torch.cat(list([item['normal'] for item in batch]), dim=0).to(device),
            'flame_expression' : torch.cat(list([item['flame_expression'] for item in batch]), dim=0).to(device),
            'flame_pose' : torch.cat(list([item['flame_pose'] for item in batch]), dim=0).to(device),
            'flame_camera': list([item['flame_camera'] for item in batch]),
            'img_deca': torch.cat(list([item['img_deca'] for item in batch]), dim=0).to(device),
        }