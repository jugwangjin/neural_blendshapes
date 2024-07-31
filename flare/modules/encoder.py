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

        assert num_channels % num_groups == 0

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


class ModulationActivation(nn.Module):
    def __init__(self, alpha=None):
        super().__init__()

        self.elu = nn.ELU(alpha=alpha if alpha is not None else 1.0)
        
    def forward(self, x):
        # for elements where x>0, torch.log1p(x). For elements where x<0, self.elu(x)
        ret = torch.where(x > 0, torch.log1p(x), self.elu(x))
        return ret
        

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
        self.tail = nn.Sequential(
                    nn.Linear(7 + 68*3, 256),
                    mygroupnorm(num_groups=4, num_channels=256),
                    nn.LeakyReLU(),
                    nn.Linear(256, 256),
                    mygroupnorm(num_groups=4, num_channels=256),
                    nn.LeakyReLU(),
                    nn.Linear(256, 256),
                    mygroupnorm(num_groups=4, num_channels=256),
                    nn.LeakyReLU(),
                    nn.Linear(256, 256),
                    mygroupnorm(num_groups=4, num_channels=256),
                    nn.LeakyReLU(),
                    nn.Linear(256, 6)
        )
        
        self.bshape_modulator = nn.Sequential(
                    nn.Linear(68*3 + 53, 256),
                    # nn.Linear(68*3 + 478*3 + 53, 256),
                    mygroupnorm(num_groups=4, num_channels=256),
                    nn.LeakyReLU(),
                    nn.Linear(256, 256),
                    mygroupnorm(num_groups=4, num_channels=256),
                    nn.LeakyReLU(),
                    nn.Linear(256, 256),
                    mygroupnorm(num_groups=4, num_channels=256),
                    nn.LeakyReLU(),
                    nn.Linear(256, 256),
                    mygroupnorm(num_groups=4, num_channels=256),
                    nn.LeakyReLU(),
                    nn.Linear(256, 53)
        )

        initialize_weights(self.tail, gain=0.01)
        self.tail[-1].weight.data.zero_()
        self.tail[-1].bias.data.zero_()

        initialize_weights(self.bshape_modulator, gain=0.01)
        self.bshape_modulator[-1].weight.data.zero_()
        self.bshape_modulator[-1].bias.data.zero_()
            
        self.softplus = nn.Softplus(beta=torch.log(torch.tensor(2.)))
        self.elu = nn.ELU()
        self.scale = torch.nn.Parameter(torch.zeros(1))
        global_translation = torch.zeros(3)
        self.register_buffer('global_translation', global_translation)
        # self.global_translation = torch.nn.Parameter(torch.zeros(3))

        self.leaky_relu = nn.LeakyReLU(0.01)

        # self.transform_origin = torch.nn.Parameter(torch.tensor([0., -0.40, -0.20]))
        # self.transform_origin = torch.nn.Parameter(torch.tensor([0., 0., 0.]))
        self.transform_origin = torch.nn.Parameter(torch.tensor([0., -0.30, -0.15]))
        # self.transform_origin.data = torch.tensor([0., -0.40, -0.20])
        # self.register_buffer('transform_origin', torch.tensor([0., -0.40, -0.20]))

        self.identity_code = nn.Parameter(torch.zeros(self.ict_facekit.num_identity))

        # self.bshapes_multiplier = torch.nn.Parameter(torch.zeros(53))

        self.tanh = nn.Tanh()

        self.relu = nn.ReLU()

        self.modulation_activation = ModulationActivation()

    def forward(self, views):
        blendshape = views['mp_blendshape'][..., self.ict_facekit.mediapipe_to_ict].reshape(-1, 53).detach()
        mp_landmark = views['mp_landmark'].reshape(-1, 478*3).detach()
        transform_matrix = views['mp_transform_matrix'].reshape(-1, 4, 4).detach()
        # print(transform_matrix)
        # exit()
        detected_landmarks = views['landmark'].clone().detach()[:, :, :3].reshape(-1, 68*3).detach()

        scale = torch.norm(transform_matrix[:, :3, :3], dim=-1).mean(dim=-1, keepdim=True)
        translation = transform_matrix[:, :3, 3]
        rotation_matrix = transform_matrix[:, :3, :3] / scale[:, None]

        rotation_matrix[:, 2:3] *= -1
        rotation_matrix[:, :, 2:3] *= -1


        translation[:, 1] *= -1

        rotation = p3dt.matrix_to_euler_angles(rotation_matrix, convention='XYZ')
        translation = translation / 32.
        features = self.tail(torch.cat([detected_landmarks, rotation, translation, scale], dim=-1))
        
        translation[:, -1] = 0

        bshape_modulation = self.modulation_activation(self.bshape_modulator(torch.cat([blendshape, detected_landmarks], dim=-1)))
        blendshape = blendshape + bshape_modulation * blendshape

        scale = torch.ones_like(bshape_modulation[:, -1:]) * (self.elu(self.scale) + 1)

        rotation = rotation + features[:, :3]
        translation = translation + features[:, 3:6]

        out_features = torch.cat([blendshape, rotation, translation, scale], dim=-1)

        return out_features

    def save(self, path):
        data = {
            'state_dict': self.state_dict()
        }
        torch.save(data, path)  



