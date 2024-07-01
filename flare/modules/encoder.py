import torch
from torch import nn
import torchvision

import numpy as np

PI = torch.pi 
HALF_PI = torch.pi / 2

import pytorch3d.transforms as p3dt

class mygroupnorm(nn.Module):
    def __init__(self, num_groups, num_channels):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.groupnorm = nn.GroupNorm(num_groups, num_channels)
    def forward(self, x):
        if len(x.shape) == 3:
            b, v, c = x.shape
            x = x.reshape(b*v, c)
            x = self.groupnorm(x)
            x = x.reshape(b, v, c)
            return x
        else:
            return self.groupnorm(x)
def initialize_weights(m, gain=0.1):

    # iterate over layers, apply if it is nn.Linear

    for l in m.children():
        if isinstance(l, nn.Linear):
            nn.init.xavier_uniform_(l.weight, gain=gain)
            l.bias.data.zero_()

class GaussianActivation(nn.Module):
    def __init__(self, a=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a*torch.ones(1), trainable))

    def forward(self, x):
        return torch.exp(-x**2/(2*self.a**2))

class ResnetEncoder(nn.Module):
    def __init__(self, outsize, ict_facekit):
        super(ResnetEncoder, self).__init__()
        self.ict_facekit = ict_facekit

        # self.tail = nn.Sequential(nn.Linear(7 + 478*3 + 53 + 68*2, 128),
        #                             mygroupnorm(num_groups=16, num_channels=128),
        #                             nn.LeakyReLU(),
        #                             nn.Linear(128, 128),
        #                             mygroupnorm(num_groups=16, num_channels=128),
        #                             nn.LeakyReLU(),
        #                             nn.Linear(128, 128),
        #                             mygroupnorm(num_groups=16, num_channels=128),
        #                             nn.LeakyReLU(),
        #                             nn.Linear(128, 128),
        #                             mygroupnorm(num_groups=16, num_channels=128),
        #                             nn.LeakyReLU(),
        #                             nn.Linear(128, 128),
        #                             mygroupnorm(num_groups=16, num_channels=128),
        #                             nn.LeakyReLU(),
        #                             nn.Linear(128, 53 + 7))

        self.tail = nn.Sequential(nn.Linear(7 + 478*3 + 53 + 68*2, 128),
                                    nn.LeakyReLU(),
                                    nn.Linear(128, 128),
                                    nn.LeakyReLU(),
                                    nn.Linear(128, 128),
                                    nn.LeakyReLU(),
                                    nn.Linear(128, 53 + 7))
        

        initialize_weights(self.tail, gain=0.01)
        self.tail[-1].weight.data.zero_()
        self.tail[-1].bias.data.zero_()
            
        self.softplus = nn.Softplus(beta=torch.log(torch.tensor(2.)))
        self.scale = torch.nn.Parameter(torch.zeros(1))
        global_translation = torch.zeros(3)
        self.register_buffer('global_translation', global_translation)
        # self.global_translation = nn.Parameter(torch.zeros(3))
        # self.global_translation.data = torch.tensor([0., -0.10, 0])
        self.transform_origin = torch.nn.Parameter(torch.tensor([0., 0., 0.]))
        self.transform_origin.data = torch.tensor([0., -0.40, -0.20])

        self.identity_code = nn.Parameter(torch.zeros(self.ict_facekit.num_identity))

        self.relu = nn.ReLU()

    def forward(self, views):
        blendshape = views['mp_blendshape'][..., self.ict_facekit.mediapipe_to_ict].reshape(-1, 53).detach()
        mp_landmark = views['mp_landmark'].reshape(-1, 478*3).detach()
        transform_matrix = views['mp_transform_matrix'].reshape(-1, 4, 4).detach()
        detected_landmarks = views['landmark'].clone().detach()[:, :, :2].reshape(-1, 68*2).detach()

        scale = torch.norm(transform_matrix[:, :3, :3], dim=-1).mean(dim=-1, keepdim=True)
        translation = transform_matrix[:, :3, 3]
        rotation_matrix = transform_matrix[:, :3, :3] / scale[:, None]

        '''
        debugging
        '''
        # print(torch.norm(transform_matrix[:, :3, :3], dim=-1))
        
        # print(transform_matrix)
        # print(scale, translation, rotation_matrix)

        # # print(euler_angle.shape, translation.shape, scale.shape)
        
        # rot_mat = rotation_matrix.clone()

        # import pytorch3d.transforms as pt3d
        # euler_angle = p3dt.matrix_to_euler_angles(rotation_matrix, convention='XYZ')
        # euler_angle[:, :3] *= -1
        # rotation_matrix = pt3d.euler_angles_to_matrix(euler_angle, convention = 'XYZ')

        # print(rot_mat[:, :3] * -1, rotation_matrix)
        # exit()
        # print(rotation_matrix)

        '''
        debugging block end
        '''
        rotation_matrix[:, 2:3] *= -1
        rotation_matrix[:, :, 2:3] *= -1

        # rotation_matrix[:, :1] *= -1
        # rotation_matrix[:, :, :1] *= -1

        rotation = p3dt.matrix_to_euler_angles(rotation_matrix, convention='XYZ')
        # print(rotation)
        # to radians
        # rotation[:, :3] *= -1
        features = self.tail(torch.cat([blendshape, mp_landmark, detected_landmarks, rotation, translation, scale], dim=-1)) * 0.25
        translation = translation / 256.

        out_features = torch.zeros_like(features)
        # out_features[:, :53] = blendshape * self.softplus(features[:, :53])
        # out_features[:, 53:] = torch.cat([rotation, translation, scale], dim=-1) + features[:, 53:]
        out_features = torch.cat([blendshape, rotation, translation, scale], dim=-1) * self.softplus(features)
        
        out_features[:, -2] = 0
        out_features[:, -1] = torch.ones_like(out_features[:, -1]) * self.softplus(self.scale)

        return out_features

    def save(self, path):
        data = {
            'state_dict': self.state_dict()
        }
        torch.save(data, path)  



