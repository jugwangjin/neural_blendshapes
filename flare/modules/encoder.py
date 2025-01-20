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
    def __init__(self, last_op=None):
        super(DECAEncoder, self).__init__()
        feature_size = 2048
        self.feature_size = feature_size
        self.encoder = resnet.load_ResNet50Model() #out: 2048
        ### regressor

        self.blendshapes_prefix = nn.Sequential(
            nn.Linear(53, 128),
            nn.ReLU(),
        )

        self.bshapes_tail = nn.Sequential(
            nn.Linear(feature_size + 128, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 53)
        )

        self.rotation_prefix = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
        )

        self.rotation_tail = nn.Sequential(
            nn.Linear(feature_size, 256),
            # nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(256, 9, bias=False)
        )

        # self.translation_tail = nn.Sequential(
        #     nn.Linear(3, 32),
        #     nn.LayerNorm(32),
        #     nn.ReLU(),
        #     nn.Linear(32, 32),
        #     nn.LayerNorm(32),
        #     nn.ReLU(),
        #     nn.Linear(32, 32),
        #     nn.LayerNorm(32),
        #     nn.ReLU(),
        #     nn.Linear(32, 32),
        #     nn.LayerNorm(32),
        #     nn.ReLU(),
        #     nn.Linear(32, 6)
        # )
        
        # multiply by 0.1 to the last layers of rotation_tail and translation_tail
        self.bshapes_tail[-1].weight.data *= 0.1
        self.rotation_tail[-1].weight.data *= 0.1
        # self.translation_tail[-1].weight.data *= 0.1

        # set zero of the translation_tail bias
        # self.translation_tail[-1].bias.data = torch.zeros(6)

    
        self.last_op = last_op

        for param in self.encoder.parameters():
            param.requires_grad = False  # Freeze all encoder parameters initially

        def freeze_gradients_hook(module, inputs):
            for param in module.parameters():
                param.requires_grad = False  # Enforce freezing
        def unfreeze_gradients_hook(module, inputs):
            for param in module.parameters():
                param.requires_grad = True

        # Register the hook on the encoder to ensure it stays frozen
        self.encoder.apply(lambda m: m.register_forward_pre_hook(freeze_gradients_hook))

        self.sigmoid = nn.Sigmoid()
    
    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()


    def forward(self, inputs, bshapes, mp_translation, landmark):
        with torch.no_grad():
            encoder_features = self.encoder(inputs)
            encoder_features = encoder_features.data.detach()

        bshapes_additional_features = self.blendshapes_prefix(bshapes)
        
        bshapes_out = self.bshapes_tail(torch.cat([encoder_features, bshapes_additional_features], dim=-1))
        rotation_out = self.rotation_tail(torch.cat([encoder_features], dim=-1))

        bshapes_tail_out = bshapes_out

        bshapes_out = torch.pow(bshapes, torch.exp(bshapes_out))
        # print(rotation_out.shape, mp_translation.shape, flame_cam_t.shape, torch.cat([rotation_out, mp_translation, flame_cam_t], dim=-1).shape)
        # translation_out = self.translation_tail(torch.cat([rotation_out], dim=-1))
        # translation_out = self.translation_tail(torch.cat([rotation_out, mp_translation, landmark], dim=-1))
        
        out = torch.cat([bshapes_out, rotation_out, bshapes_tail_out], dim=-1)
        if self.last_op:
            out = self.last_op(out)
        return out

class ResnetEncoder(nn.Module):
    def __init__(self, ict_facekit):
        super(ResnetEncoder, self).__init__()

        self.ict_facekit = ict_facekit
        
        self.encoder = DECAEncoder(last_op = None)
        self.load_deca_encoder()
        
        # set zero bias)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.elu = nn.ELU()

        self.translation = torch.nn.Parameter(torch.tensor([0., 0., 0.]))

        self.register_buffer('identity_weights', torch.zeros(self.ict_facekit.num_identity, device='cuda'))
        
        self.scale = torch.nn.Parameter(torch.zeros(1))

        self.transform = torch.nn.functional.interpolate
    
        self.flame_cam_t = torch.nn.Sequential(
                            nn.Linear(3, 3)
                            )
        self.flame_cam_t[0].weight.data = torch.eye(3) * 1e-2
        self.flame_cam_t[0].bias.data = torch.zeros(3)

        self.blendshapes_multiplier = torch.nn.Parameter((torch.zeros(53)))
        self.blendshapes_offset = torch.nn.Parameter(torch.zeros(53))
        self.softplus = torch.nn.Softplus(beta=4)

        self.register_buffer('transform_origin', torch.tensor([0., -0, -0.28]))
        self.global_translation = torch.nn.Parameter(torch.zeros(3))
        
    def load_deca_encoder(self):
        model_path = './assets/deca_model.tar'
        deca_ckpt = torch.load(model_path)
        encoder_state_dict = {k[8:]: v for k, v in deca_ckpt['E_flame'].items() if k.startswith('encoder.')}
        self.encoder.encoder.load_state_dict(encoder_state_dict, strict=True)
        
    

    def forward(self, views):
        with torch.no_grad():
            img = views['img_deca']
            transform_matrix = views['mp_transform_matrix'].clone().detach().reshape(-1, 4, 4)

            mp_translation = transform_matrix[:, :3, 3]
            mp_translation[..., -1] += 28
            mp_translation = mp_translation * 0.2

            mp_bshapes = views['mp_blendshape'].clone().detach()
            mp_bshapes = mp_bshapes[:, self.ict_facekit.mediapipe_to_ict]

            # ts = torch.stack([cam.t for cam in views['flame_camera']], dim=0)

            fixed_indices = [0, 16, 36, 45, 33]
            landmark = views['landmark'][:, fixed_indices, :3].reshape(-1, 15)
        
        features = self.encoder(img, mp_bshapes, mp_translation, landmark)
        blendshapes = features[:, :53]
        rotation = features[:, 53:56] 
        translation = features[:, 56:59]

        global_translation = self.global_translation.unsqueeze(0).expand(translation.shape[0], -1)

        scale = torch.ones_like(translation[:, -1:]) * (self.elu(self.scale) + 1)

        bshapes_tail_out = features[:, 62:]

        out_features = torch.cat([blendshapes, rotation, translation, scale, global_translation, bshapes_tail_out], dim=-1)

        return out_features


    def train(self, mode=True):
        super().train(mode)
        # self.encoder.eval()


    def save(self, path):
        data = {
            'state_dict': self.state_dict()
        }
        torch.save(data, path)  



