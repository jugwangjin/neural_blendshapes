import torch
from torch import nn
import torchvision

import numpy as np

PI = torch.pi 
HALF_PI = torch.pi / 2

import pytorch3d.transforms as p3dt


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

        self.tail = nn.Sequential(nn.Linear(7 + 68*2, 64),
                                    nn.PReLU(),
                                    nn.Linear(64, 64),
                                    nn.PReLU(),
                                    nn.Linear(64, 64),
                                    nn.PReLU(),
                                    nn.Linear(64, 64),
                                    nn.PReLU(),
                                    nn.Linear(64, 7))

        initialize_weights(self.tail, gain=0.01)
            
        # set weights of self.tail as identity
        # self.tail.weight.data = torch.eye(7)
        # self.tail.bias.data.zero_()


        # self.tail.weight.data[3:6] *= 0.01
        # self.tail.weight.data[1:3] *= -1
# tensor([[ 0.1578, -0.4291, -0.0902],                                                                                        [145/1969]
#         [ 0.1215,  0.8031,  0.1064],                                                                                                  
#         [ 0.1357,  0.4502,  0.0228],                                                                                                  
#         [ 0.1256,  0.7359,  0.0285],                                                                                                  
#         [-0.0401,  0.0417,  0.0136]], device='cuda:0')    


# tensor([[-0.1578, -0.4291, -0.0902],                                                                                                  
#         [-0.1215,  0.8031,  0.1064],                                                                                                  
#         [-0.1357,  0.4502,  0.0228],                                                                                                  
#         [-0.1256,  0.7359,  0.0285],                                                                                                  
#         [ 0.0401,  0.0417,  0.0136]], device='cuda:0')        

        self.blendshapes_multiplier = torch.nn.Parameter(torch.zeros(53))
        self.blendshapes_bias = torch.nn.Parameter(torch.zeros(53))
        self.softplus = nn.Softplus(beta=torch.log(torch.tensor(2.)))
        
        # self.blendshapes = nn.Linear(53+7+16, 53)
        # self.blendshapes.weight.data.zero_()
        # self.blendshapes.bias.data.zero_()
        # self.blendshapes.weight.data[:53, :53] = torch.eye(53)

        self.scale = torch.nn.Parameter(torch.zeros(1))


    def forward(self, views):
        blendshape = views['mp_blendshape'][..., self.ict_facekit.mediapipe_to_ict].reshape(-1, 53) * self.softplus(self.blendshapes_multiplier)
        

        transform_matrix = views['mp_transform_matrix'].reshape(-1, 4, 4)

        detected_landmarks = views['landmark'].clone().detach()[:, :, :2].reshape(-1, 68*2)

        # calculate scale, translation, rotation from transform matrix
        # assume scale is the same for all axes
        scale = torch.norm(transform_matrix[:, :3, :3], dim=-1).mean(dim=-1, keepdim=True)
        translation = transform_matrix[:, :3, 3]
        rotation_matrix = transform_matrix[:, :3, :3] / scale[:, None]
        # indexing = [0,2,1]
        # rotation_matrix = rotation_matrix[:, indexing]

        # rotation_matrix[:, 1:2] *= -1
        # rotation_matrix[:, 2:3] *= -1
        # rotation_matrix[:, :1] *= -1
        rotation = p3dt.matrix_to_euler_angles(rotation_matrix, convention='XYZ')
        # print(rotation)
        rotation[:, 1:3] *= -1
        # print(rotation, translation, scale)
        features = self.tail(torch.cat([detected_landmarks, rotation, translation, scale], dim=-1))
        translation = translation * 0

        features += torch.cat([rotation, translation, scale], dim=-1)

        # bshapes_input = torch.cat([views['mp_blendshape'][..., self.ict_facekit.mediapipe_to_ict].reshape(-1, 53), transform_matrix.reshape(-1, 16), features], dim=-1)
        # blendshape = self.blendshapes(bshapes_input)
        
        features[:, 5] = 0
        features[:, 6] = torch.ones_like(features[:, 6]) * self.softplus(self.scale)
        features = torch.cat([blendshape, features], dim=-1) # shape of features: (batch_size, 60)

        return features

    def save(self, path):
        data = {
            'state_dict': self.state_dict()
        }
        torch.save(data, path)  



