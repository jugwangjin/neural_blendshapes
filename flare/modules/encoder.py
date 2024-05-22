import torch
from torch import nn
import torchvision

import numpy as np

PI = torch.pi 
HALF_PI = torch.pi / 2

import pytorch3d.transforms as p3dt


def initialize_weights(m, gain=0.1):
    
    for name, param in m.named_parameters():
        # if 'weight' in name:
        #     nn.init.xavier_uniform_(param.data, gain=gain)
        if 'bias' in name:
            param.data.zero_()


class ResnetEncoder(nn.Module):
    def __init__(self, outsize, ict_facekit):
        super(ResnetEncoder, self).__init__()
        self.ict_facekit = ict_facekit

        self.tail = nn.Linear(7, 7)
            
        # set weights of self.tail as identity
        self.tail.weight.data = torch.eye(7)

        self.tail.weight.data[3:6] *= 0.1
        self.tail.weight.data[:2] *= -1
        


        self.blendshapes_multiplier = torch.nn.Parameter(torch.ones(53))
        self.softplus = nn.Softplus(beta=10)
        

        initialize_weights(self.tail, gain=0.01)

    def forward(self, image, views):
        blendshape = views['mp_blendshape'][..., self.ict_facekit.mediapipe_to_ict].reshape(-1, 53) * self.softplus(self.blendshapes_multiplier)
        transform_matrix = views['mp_transform_matrix'].reshape(-1, 4, 4)

        # calculate scale, translation, rotation from transform matrix
        # assume scale is the same for all axes
        scale = torch.norm(transform_matrix[:, :3, :3], dim=-1).mean(dim=-1, keepdim=True)
        translation = transform_matrix[:, :3, 3]
        translation[..., 2] = 0
        rotation_matrix = transform_matrix[:, :3, :3] / scale[:, None]
        rotation = p3dt.matrix_to_euler_angles(rotation_matrix, convention='XYZ')
        # print(rotation, translation, scale)
        features = self.tail(torch.cat([rotation, translation, scale], dim=-1))
        features = torch.cat([blendshape, features], dim=-1) # shape of features: (batch_size, 60)

        return features

    def save(self, path):
        data = {
            'state_dict': self.state_dict()
        }
        torch.save(data, path)  



