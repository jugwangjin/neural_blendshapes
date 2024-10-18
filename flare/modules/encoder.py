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
            nn.Linear(feature_size + 53 + 3 + 68*3, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, outsize, bias = False),
        )
        self.last_op = last_op

        self.gradient_scaling = 10.

        self.encoder.register_full_backward_hook(lambda module, grad_i, grad_o: (grad_i[0] / self.gradient_scaling if grad_i[0] is not None else None, ))

        nn.init.zeros_(self.layers[-1].weight)
        # If the last layer has a bias, initialize it to zero as well
        if hasattr(self.layers[-1], 'bias') and self.layers[-1].bias is not None:
            nn.init.zeros_(self.layers[-1].bias)

        for m in self.layers.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)  # Smaller gain
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def forward(self, inputs, bshapes, rotation, landmarks):
        features = self.encoder(inputs)

        # print statistics of features
        # features = features * 2
        rotation = rotation / 2

        parameters = self.layers(torch.cat([features, bshapes, rotation, landmarks], dim=-1))
        if self.last_op:
            parameters = self.last_op(parameters)
        return parameters


class ResnetEncoder(nn.Module):
    def __init__(self, outsize, ict_facekit):
        super(ResnetEncoder, self).__init__()

        self.ict_facekit = ict_facekit
        
        self.encoder = DECAEncoder(outsize = 6 + 53, last_op = None)
        self.load_deca_encoder()
        
        # set zero bias)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.elu = nn.ELU()

        self.transform_origin = torch.nn.Parameter(torch.tensor([0., -0.05, -0.25]))
        self.translation = torch.nn.Parameter(torch.tensor([0., 0., 0.]))
        # self.register_buffer('transform_origin', torch.tensor([0., -0.05, -0.25]))

        self.register_buffer('identity_weights', torch.zeros(self.ict_facekit.num_identity, device='cuda'))
        
        self.scale = torch.nn.Parameter(torch.zeros(1))

        self.transform = torch.nn.functional.interpolate
    
        
    def load_deca_encoder(self):
        model_path = './assets/deca_model.tar'
        deca_ckpt = torch.load(model_path)
        encoder_state_dict = {k[8:]: v for k, v in deca_ckpt['E_flame'].items() if k.startswith('encoder.')}
        # print(encoder_state_dict.keys())
        self.encoder.encoder.load_state_dict(encoder_state_dict, strict=True)
        # layers_state_dict = {k[9:] : v for k, v in deca_ckpt['E_flame'].items() if k.startswith('layers.0.')}
        # print(layers_state_dict.keys())
        # self.encoder.layers[0].load_state_dict(layers_state_dict, strict=True)
        
    

    def forward(self, views):
        with torch.no_grad():
            img = views['img_deca']
            transform_matrix = views['mp_transform_matrix'].clone().detach().reshape(-1, 4, 4)
            scale = torch.norm(transform_matrix[:, :3, :3], dim=-1).mean(dim=-1, keepdim=True)
            rotation_matrix = transform_matrix[:, :3, :3]
            rotation_matrix = transform_matrix[:, :3, :3] / scale[:, None]

            rotation_matrix = rotation_matrix.permute(0, 2, 1)
            mp_rotation = p3dt.matrix_to_euler_angles(rotation_matrix, convention='XYZ')

            mp_bshapes = views['mp_blendshape'].clone().detach()
            mp_bshapes = mp_bshapes[:, self.ict_facekit.mediapipe_to_ict]
            
            detected_landmarks = views['landmark'].clone().detach()[:, :, :3]
            detected_landmarks[..., :-1] = detected_landmarks[..., :-1] # center
            detected_landmarks[..., -1] *= -1 
            detected_landmarks = detected_landmarks.reshape(-1, 68*3)
  

        features = self.encoder(img, mp_bshapes, mp_rotation, detected_landmarks)
        # blendshapes = features[:, :53] + mp_bshapes
        # blendshapes = self.sigmoid(5 * (blendshapes - 0.5))
        blendshapes = mp_bshapes + features[:, :53]
        # blendshapes = self.sigmoid(features[:, :53])
        rotation = mp_rotation + features[:, 53:56]
        # rotation = self.tanh(features[:, 53:56]) * 1.75
        translation = features[:, 56:59]
        # translation = self.translation[None].expand(features.shape[0], -1)

        scale = torch.ones_like(translation[:, -1:]) * (self.elu(self.scale) + 1)

        out_features = torch.cat([blendshapes, rotation, translation, scale], dim=-1)

        return out_features


    def train(self, mode=True):
        super().train(mode)
        # self.encoder.eval()


    def save(self, path):
        data = {
            'state_dict': self.state_dict()
        }
        torch.save(data, path)  



