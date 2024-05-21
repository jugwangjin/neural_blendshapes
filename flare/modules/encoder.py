import torch
from torch import nn
import torchvision

import numpy as np

PI = torch.pi 
HALF_PI = torch.pi / 2


def initialize_weights(m, gain=0.1):
    
    for name, param in m.named_parameters():
        # if 'weight' in name:
        #     nn.init.xavier_uniform_(param.data, gain=gain)
        if 'bias' in name:
            param.data.zero_()


class ResnetEncoder(nn.Module):
    def __init__(self, outsize, ict_facekit):
        super(ResnetEncoder, self).__init__()
        # try:
        #     self.encoder = torchvision.models.regnet_x_1_6gf(weights='DEFAULT')
        # except:
        #     self.encoder = torchvision.models.regnet_x_1_6gf(pretrained=True)

        # self.outsize = outsize

        # self.encoder = torch.nn.Sequential(*list(self.encoder.children()))

        # feature_size = self.encoder[-1].in_features
        # self.encoder = self.encoder[:-1]

        # can I append a layer to the encoder?
        # self.encoder = nn.Sequential(
        #     self.encoder,
        #     nn.Flatten(),
        #     nn.Linear(feature_size, 128)
        # )

        self.ict_facekit = ict_facekit

        self.tail = nn.Sequential(
            nn.Linear(16, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, 6),
        )

        self.min_range = torch.nn.Parameter(torch.zeros(53))
        self.max_range = torch.nn.Parameter(torch.ones(53))

        # self.resize = nn.Upsample(size=(224, 224), mode='bilinear')
        # self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        # self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.softplus = torch.nn.Softplus(beta=10)

        initialize_weights(self.tail, gain=0.01)


    def forward(self, image, views):
        blendshape = views['mp_blendshape'][..., self.ict_facekit.mediapipe_to_ict].reshape(-1, 53)
        transform_matrix = views['mp_transform_matrix'].reshape(-1, 16)

        features = self.tail(transform_matrix)
        min_range = self.softplus(self.min_range)
        max_range = self.softplus(self.max_range)
        blendshape = (blendshape - self.min_range) / (self.max_range + 1e-6)

        features = torch.cat([blendshape, features], dim=-1)

        # features[..., :53] = torch.nn.functional.tanh(features[..., :53]) / 2. + 0.5
        
        features[..., 58] = 0

        return features

    def save(self, path):
        data = {
            'state_dict': self.state_dict()
        }
        torch.save(data, path)  



