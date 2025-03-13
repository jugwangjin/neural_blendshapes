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
    def __init__(self, last_op=None, additive=False):
        super(DECAEncoder, self).__init__()
        feature_size = 2048
        self.feature_size = feature_size
        self.encoder = resnet.load_ResNet50Model() #out: 2048
        ### regressor

        outsize = 100 + 50 + 50 + 3 + 6 + 27

        self.layers = nn.Sequential(
            nn.Linear(feature_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, outsize)
        )

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

        self.rotation_tail = nn.Sequential(
            nn.Linear(15, 64),
            # nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            # nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            # nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            # nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 9, bias=False)
        )
        # self.translation_tail = nn.Sequential(
        #     nn.Linear(15, 32),
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
        #     nn.Linear(32, 6, bias=False)
        # )

        
        # multiply by 0.1 to the last layers of rotation_tail and translation_tail
        self.bshapes_tail[-1].weight.data *= 0.1
        self.rotation_tail[-1].weight.data *= 0.1
        # self.translation_tail[-1].weight.data *= 0.1

        self.last_op = last_op

        for param in self.encoder.parameters():
            param.requires_grad = False  # Freeze all encoder parameters initially
        for param in self.layers.parameters():
            param.requires_grad = False

        def freeze_gradients_hook(module, inputs):
            for param in module.parameters():
                param.requires_grad = False  # Enforce freezing

        def unfreeze_gradients_hook(module, inputs):
            for param in module.parameters():
                param.requires_grad = True

        # Register the hook on the encoder to ensure it stays frozen
        self.encoder.apply(lambda m: m.register_forward_pre_hook(freeze_gradients_hook))
        self.layers.apply(lambda m: m.register_forward_pre_hook(freeze_gradients_hook))

        self.sigmoid = nn.Sigmoid()
    
        self.extracted_features = {}

        self.additive = additive
        self.tanh = nn.Tanh()

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        self.layers.eval()


    def forward(self, inputs, bshapes, mp_translation, landmark):
        with torch.no_grad():
            idx = inputs['idx']
            img_path = inputs['img_path']

            features = []
            for b in range(idx.shape[0]):
                if img_path[b] not in self.extracted_features:
                    img = inputs['img_deca'][b:b+1]
                    feature = self.encoder(img)
                    self.extracted_features[img_path[b]] = feature
                else:
                    feature = self.extracted_features[img_path[b]]
                features.append(feature)
            encoder_features = torch.cat(features, dim=0)

            # shape tex exp pose cam 3 light 27
            # shape 100 # tex 50 # exp 50 # pose  6 
            pose_features = inputs['flame_pose']
            # pose_features = self.layers(encoder_features)
            # pose_features = pose_features[..., -36:-30]
            # pose_features = torch.cat([pose_features[..., :100], pose_features[..., 150:-30]], dim=-1)
            # img = inputs['img_deca']


            # encoder_features = self.encoder(img)
            encoder_features = encoder_features.data.detach()
            pose_features = pose_features.data.detach()

            

        bshapes_additional_features = self.blendshapes_prefix(bshapes)
        
        bshapes_out = self.bshapes_tail(torch.cat([encoder_features, bshapes_additional_features], dim=-1))
        rotation_out = self.rotation_tail(pose_features)

        rotation_out[..., 3:] *= 0.1
        # translation_out = self.translation_tail(pose_features)

        bshapes_tail_out = bshapes_out

        if not self.additive:
            bshapes_out = torch.pow(bshapes, torch.exp(bshapes_out))
        else:
            bshapes_out = bshapes + self.tanh(bshapes_out)
            bshapes_out = bshapes_out.clamp(0, 1)
            
        # print(rotation_out.shape, mp_translation.shape, flame_cam_t.shape, torch.cat([rotation_out, mp_translation, flame_cam_t], dim=-1).shape)
        # translation_out = self.translation_tail(torch.cat([rotation_out], dim=-1))
        # translation_out = self.translation_tail(torch.cat([rotation_out, mp_translation, landmark], dim=-1))
        
        out = torch.cat([bshapes_out, rotation_out, bshapes_tail_out], dim=-1)
        if self.last_op:
            out = self.last_op(out)
        return out

class ResnetEncoder(nn.Module):
    def __init__(self, ict_facekit, fix_bshapes=False, additive=False, disable_pose=False):
        super(ResnetEncoder, self).__init__()

        self.ict_facekit = ict_facekit
        
        self.encoder = DECAEncoder(last_op = None, additive=additive)
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

        self.transform_origin = torch.nn.Parameter(torch.tensor([0., -0, -0.28]))
        # self.register_buffer('transform_origin', torch.tensor([0., -0, -0.28]))
        self.register_buffer('global_translation', (torch.zeros(3)))
        # self.global_translation = torch.nn.Parameter(torch.zeros(3))

        self.fix_bshapes = fix_bshapes

        self.disable_pose = disable_pose
        
    def load_deca_encoder(self):
        model_path = './assets/deca_model.tar'

        deca_ckpt = torch.load(model_path)
        encoder_state_dict = {k[8:]: v for k, v in deca_ckpt['E_flame'].items() if k.startswith('encoder.')}
        self.encoder.encoder.load_state_dict(encoder_state_dict, strict=True)

        layers_state_dict = {k[7:]: v for k, v in deca_ckpt['E_flame'].items() if k.startswith('layers.')}
        print(layers_state_dict.keys())
        self.encoder.layers.load_state_dict(layers_state_dict, strict=True)
        
    

    def forward(self, views):
        with torch.no_grad():
            
            transform_matrix = views['mp_transform_matrix'].clone().detach().reshape(-1, 4, 4)
            scale = torch.norm(transform_matrix[:, :3, :3], dim=-1).mean(dim=-1, keepdim=True)

            mp_translation = transform_matrix[:, :3, 3]
            mp_translation[..., -1] += 28
            mp_translation = mp_translation * 0.2

            mp_bshapes = views['mp_blendshape'].clone().detach()
            mp_bshapes = mp_bshapes[:, self.ict_facekit.mediapipe_to_ict]

            # ts = torch.stack([cam.t for cam in views['flame_camera']], dim=0)

            fixed_indices = [0, 16, 36, 45, 33]
            landmark = views['landmark'][:, fixed_indices, :3].reshape(-1, 15)
        
            # transform_matrix = views_subset['mp_transform_matrix'].reshape(-1, 4, 4).detach()
            # scale = torch.norm(transform_matrix[:, :3, :3], dim=-1).mean(dim=-1, keepdim=True)
            # translation = transform_matrix[:, :3, 3]
            rotation_matrix = transform_matrix[:, :3, :3]
            rotation_matrix = transform_matrix[:, :3, :3] / scale[:, None]

            rotation_matrix[:, 2:3] *= -1
            rotation_matrix[:, :, 2:3] *= -1

            rotation_matrix = rotation_matrix.permute(0, 2, 1)
            rotation = p3dt.matrix_to_euler_angles(rotation_matrix, convention='XYZ')

        features = self.encoder(views, mp_bshapes, mp_translation, landmark)
        if self.fix_bshapes:
            blendshapes = mp_bshapes
        else:
            blendshapes = features[:, :53]

        if not self.disable_pose:
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



