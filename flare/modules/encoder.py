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

        self.encoder_bottleneck = nn.Sequential(
            nn.Linear(2048, 1024),  # Compress to 512 dimensions
            nn.Softplus(beta=100),
        )

        self.others_bottleneck = nn.Sequential(
            nn.Linear(53 + 3 + 3 + 204, 512),  # Compress to 512 dimensions
            nn.Softplus(beta=100),
            nn.Linear(512, 512),  # Compress to 512 dimensions
            nn.Softplus(beta=100),
            nn.Linear(512, 512),  # Compress to 512 dimensions
            nn.Softplus(beta=100),
        )

        # Final MLP to fuse all inputs and produce output
        # Concatenate the bottlenecked encoder output (512) + bshapes (53) + rotation (3) + landmarks (204)
        input_size = 1024 + 512

        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024),  # Smaller hidden sizes for efficiency
            nn.Softplus(beta=100),
            nn.Linear(1024, 512),
            nn.Softplus(beta=100),
            nn.Linear(512, outsize, bias=True)
        )
    
        self.last_op = last_op

        # Initialize weights and biases for expression deformer
        for layer in self.encoder_bottleneck:
            if isinstance(layer, nn.Linear):
                # Initialize weight and bias based on ForwardDeformer strategy
                torch.nn.init.constant_(layer.bias, 0.0) if layer.bias is not None else None
                torch.nn.init.xavier_uniform_(layer.weight)

        for layer in self.others_bottleneck:
            if isinstance(layer, nn.Linear):
                # Initialize weight and bias based on ForwardDeformer strategy
                torch.nn.init.constant_(layer.bias, 0.0) if layer.bias is not None else None
                torch.nn.init.xavier_uniform_(layer.weight)

        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                # Initialize weight and bias based on ForwardDeformer strategy
                torch.nn.init.constant_(layer.bias, 0.0) if layer.bias is not None else None
                torch.nn.init.xavier_uniform_(layer.weight)

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
    
    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()


    def forward(self, inputs, bshapes, rotation, translation, landmarks):
        rotation = rotation * 0
        translation = translation * 0
        # Pass through encoder to get features
        with torch.no_grad():
            encoder_features = self.encoder(inputs)

        # Apply bottleneck to encoder output
        bottlenecked_features = self.encoder_bottleneck(encoder_features)
        
        others_features = self.others_bottleneck(torch.cat([bshapes, rotation, translation, landmarks], dim=-1))

        # Concatenate bottlenecked encoder features with additional inputs
        concat_inputs = torch.cat([bottlenecked_features, others_features], dim=-1)

        # Pass through the final MLP
        parameters = self.layers(concat_inputs)

        # Apply last activation function if specified
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

        self.transform_origin = torch.nn.Parameter(torch.tensor([0., -0.22, -0.35]))
        self.translation = torch.nn.Parameter(torch.tensor([0., 0., 0.]))

        self.register_buffer('identity_weights', torch.zeros(self.ict_facekit.num_identity, device='cuda'))
        
        self.scale = torch.nn.Parameter(torch.zeros(1))

        self.transform = torch.nn.functional.interpolate
    
        self.flame_cam_t = torch.nn.Sequential(
                            nn.Linear(3, 3)
                            )
        self.flame_cam_t[0].weight.data = torch.eye(3) * 1e-1
        self.flame_cam_t[0].bias.data = torch.zeros(3)

        self.blendshapes_multiplier = torch.nn.Parameter((torch.zeros(53)))
        self.blendshapes_offset = torch.nn.Parameter(torch.zeros(53))
        self.softplus = torch.nn.Softplus(beta=4)
        
    def load_deca_encoder(self):
        model_path = './assets/deca_model.tar'
        deca_ckpt = torch.load(model_path)
        encoder_state_dict = {k[8:]: v for k, v in deca_ckpt['E_flame'].items() if k.startswith('encoder.')}
        self.encoder.encoder.load_state_dict(encoder_state_dict, strict=True)
        
    

    def forward(self, views):
        with torch.no_grad():
            img = views['img_deca']
            transform_matrix = views['mp_transform_matrix'].clone().detach().reshape(-1, 4, 4)
            scale = torch.norm(transform_matrix[:, :3, :3], dim=-1).mean(dim=-1, keepdim=True)

            mp_translation = transform_matrix[:, :3, 3]
            mp_translation[..., -1] += 28
            mp_translation = mp_translation * 0.2

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

            flame_cam = views['flame_camera']
            ts = []
            for cam in flame_cam:
                ts.append(cam.t)
            ts = torch.stack(ts, dim=0)

            ts[..., -1] -= 3
  

        features = self.encoder(img, mp_bshapes, mp_rotation, mp_translation, detected_landmarks)
        blendshapes = (self.softplus(self.blendshapes_multiplier)[None] + 1) * mp_bshapes + features[:, :53]
        rotation = features[:, 53:56] 
        translation = features[:, 56:59]

        flame_cam_t = self.flame_cam_t(ts)
        translation = translation + flame_cam_t

        scale = torch.ones_like(translation[:, -1:]) * (self.elu(self.scale) + 1)

        out_features = torch.cat([blendshapes, rotation, translation, scale, features[:, 56:59], features[:, :53]], dim=-1)

        return out_features


    def train(self, mode=True):
        super().train(mode)
        # self.encoder.eval()


    def save(self, path):
        data = {
            'state_dict': self.state_dict()
        }
        torch.save(data, path)  



