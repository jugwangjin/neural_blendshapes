import torch
from torch import nn
import torchvision

import numpy as np

PI = torch.pi 
HALF_PI = torch.pi / 2

import pytorch3d.transforms as p3dt
import pytorch3d.ops as p3do

from . import resnet

class DECAEncoder(nn.Module):
    def __init__(self, outsize, last_op=None):
        super(DECAEncoder, self).__init__()
        feature_size = 2048
        self.encoder = resnet.load_ResNet50Model() #out: 2048
        ### regressor
        self.layers = nn.Sequential(
            nn.Linear(feature_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, outsize)
        )
        self.last_op = last_op

    def forward(self, inputs):
        with torch.no_grad():
            features = self.encoder(inputs)
        parameters = self.layers(features)
        if self.last_op:
            parameters = self.last_op(parameters)
        return parameters


class ResnetEncoder(nn.Module):
    def __init__(self, outsize, ict_facekit):
        super(ResnetEncoder, self).__init__()

        self.ict_facekit = ict_facekit
        
        self.encoder = DECAEncoder(outsize = 6 + 53, last_op = None)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.register_buffer('transform_origin', torch.tensor([0., -0.05, -0.25]))

        self.register_buffer('identity_weights', torch.zeros(self.ict_facekit.num_identity, device='cuda'))
        
        self.scale = torch.nn.Parameter(torch.zeros(1))

        self.transform = torch.nn.functional.interpolate
        
        self.load_deca_encoder()

        
    def load_deca_encoder(self):
        model_path = './assets/deca_model.tar'
        deca_ckpt = torch.load(model_path)
        encoder_state_dict = {k: v for k, v in deca_ckpt['E_flame'].items() if k.startswith('encoder.')}
        self.encoder.encoder.load_state_dict(encoder_state_dict, strict=False)
    

    def forward(self, views):
        img = views['img']

        img = self.transform(img.permute(0,3,1,2), mode='bilinear', size=(256, 256), align_corners=False)
        features = self.encoder(img)
        blendshapes = self.sigmoid(features[:, :53])
        rotation = self.tanh(features[:, 53:56]) * 1.75
        translation = features[:, 56:59]

        scale = torch.ones_like(translation[:, -1:]) * (self.elu(self.scale) + 1)

        out_features = torch.cat([blendshapes, rotation, translation, scale], dim=-1)

        return out_features


    def save(self, path):
        data = {
            'state_dict': self.state_dict()
        }
        torch.save(data, path)  



