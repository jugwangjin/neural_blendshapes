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
#         cfg.model.param_list = ['shape', 'tex', 'exp', 'pose', 'cam', 'light']
        # cfg.model.n_shape = 100
        # cfg.model.n_tex = 50
        # cfg.model.n_exp = 50
        # cfg.model.n_cam = 3
        # cfg.model.n_pose = 6
        # cfg.model.n_light = 27



    # def decompose_code(self, code, num_dict):
    #     ''' Convert a flattened parameter vector to a dictionary of parameters
    #     code_dict.keys() = ['shape', 'tex', 'exp', 'pose', 'cam', 'light']
    #     '''
    #     code_dict = {}
    #     start = 0
    #     for key in num_dict:
    #         end = start+int(num_dict[key])
    #         code_dict[key] = code[:, start:end]
    #         start = end
    #         if key == 'light':
    #             code_dict[key] = code_dict[key].reshape(code_dict[key].shape[0], 9, 3)
    #     return code_dict


        self.layers = nn.Sequential(
            nn.Linear(feature_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, outsize)
        )

        self.layers_tail = nn.Sequential(
            nn.Linear(56 + 53, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 53 + 10)
        )

        
        self.rotation_tail = nn.Sequential(
            nn.Linear(15 + 3, 64),
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
            nn.Linear(64, 10, bias=True)
        )

        # multiply by 0.1 to the last layers of rotation_tail and translation_tail
        # keep the weights, zero bias for all layers
        
        self.layers_tail[-1].weight.data *= 0.1
        self.rotation_tail[-1].weight.data *= 0.1

        self.last_op = last_op

        for param in self.encoder.parameters():
            param.requires_grad = False  # Freeze all encoder parameters initially
        # for param in self.layers.parameters():
            # param.requires_grad = False

        def freeze_gradients_hook(module, inputs):
            for param in module.parameters():
                param.requires_grad = False  # Enforce freezing

        def unfreeze_gradients_hook(module, inputs):
            for param in module.parameters():
                param.requires_grad = True

        # Register the hook on the encoder to ensure it stays frozen
        self.encoder.apply(lambda m: m.register_forward_pre_hook(freeze_gradients_hook))
        # self.layers.apply(lambda m: m.register_forward_pre_hook(freeze_gradients_hook))

        # register backward hook to self.layer, scale down gradient by 10 
        # self.layers.apply(lambda m: m.register_full_backward_hook(lambda module, grad_i, grad_o: (grad_i[0] * 1e-1 if grad_i[0] is not None else None, )))

        self.sigmoid = nn.Sigmoid()
    
        self.extracted_features = {}

        self.additive = additive
        self.tanh = nn.Tanh()

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        # self.layers.eval()


    def forward(self, inputs, bshapes, mp_translation, landmark, rotation, synthetic=False):
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


            pose_features = inputs['flame_pose']
            # shape tex exp pose cam 3 light 27
            # shape 100 # tex 50 # exp 50 # pose  6 
            # pose_features = inputs['flame_pose']
            # pose_features = self.layers(encoder_features)
            # pose_features = pose_features[..., -36:-30]
            # pose_features = torch.cat([pose_features[..., :100], pose_features[..., 150:-30]], dim=-1)
            # img = inputs['img_deca']

            # encoder_features = self.encoder(img)
            encoder_features = encoder_features.data.detach()
            pose_features = pose_features.data.detach()


        # extract only exp and pose from outputs of self.layers
        encoder_features = self.layers(encoder_features)[:, 150:206]
        # encoder_features = torch.cat([encoder_features[:, 150:200], encoder_features[:, 200:206]], dim=-1)
        
# encoder_features = torch.cat([encoder_features[:, 150:200], encoder_features[:, 200:206]], dim=-1)
        bshapes_out = self.layers_tail(torch.cat([encoder_features, bshapes], dim=-1))
        rotation_out = self.rotation_tail(torch.cat([pose_features, rotation], dim=-1))

        # rotation_out = bshapes_out[:, -10:]
        bshapes_out = bshapes_out[:, :-10]

        rotation_out[..., :3] += rotation
        rotation_out[..., 3:] *= 0.1
        
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

        # register gradient hook for transform origin and global translation
        # For nn.Parameter objects, we need to register hooks on the parameter itself
        # self.transform_origin.register_hook(lambda grad: grad * 1e-1 if grad is not None else None)
        # self.global_translation.register_hook(lambda grad: grad * 1e-1 if grad is not None else None)
        # self.scale.register_hook(lambda grad: grad * 1e-1 if grad is not None else None)

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
        

    def forward(self, views, synthetic=False):
        with torch.no_grad():
            
            transform_matrix = views['mp_transform_matrix'].clone().detach().reshape(-1, 4, 4)
            scale = torch.norm(transform_matrix[:, :3, :3], dim=-1).mean(dim=-1, keepdim=True)

            mp_translation = transform_matrix[:, :3, 3]
            mp_translation[..., -1] += 28
            mp_translation = mp_translation * 0.2

            mp_bshapes = views['mp_blendshape'].clone().detach()
            mp_bshapes = mp_bshapes[:, self.ict_facekit.mediapipe_to_ict]

            # ts = torch.stack([cam.t for cam in views['flame_camera']], dim=0)

        
            # Align detected_landmarks to landmarks_on_clip_space
            detected_landmarks = views['landmark'][..., :3].clone().detach()
            detected_landmarks[..., :2] = detected_landmarks[..., :2] * 2 - 1
            detected_landmarks[..., 2] = detected_landmarks[..., 2] * -2
            mean_z_detected_landmarks = detected_landmarks[..., 2].mean(dim=-1, keepdim=True)[0]
            detected_landmarks[..., 2] = detected_landmarks[..., 2] - mean_z_detected_landmarks


            R = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]], device=detected_landmarks.device, dtype=detected_landmarks.dtype)

            # detected_landmarks = torch.einsum('bij, bjk -> bik', detected_landmarks, camera)
            detected_landmarks = torch.einsum('bij, jk  -> bik', detected_landmarks, R)
            

            fixed_indices = [36, 45, 31, 35, 16, 0, 8] # right eye corner, left eye corner, right nose corner, left nose corner
            landmark_input = detected_landmarks[:, fixed_indices, :3].reshape(-1, 21)
            
            x_axis_indices = [16, 0, 45, 36]
            z_axis_indices = [27, 31, 35]

            z_axis_landmarks = detected_landmarks[:, z_axis_indices, :3]

            # right nose corner - left eye corner   cross  left nose corner - right eye corner
            z_axis = torch.cross(z_axis_landmarks[:, 0] - z_axis_landmarks[:, 1], z_axis_landmarks[:, 0] - z_axis_landmarks[:, 2], dim=1)
            z_axis = z_axis / torch.norm(z_axis, dim=-1, keepdim=True)

            # x_axis = torch.cross(y_axis, z_axis, dim=1)
            # x_axis = x_axis / torch.norm(x_axis, dim=-1, keepdim=True)

            x_axis_landmarks = detected_landmarks[:, x_axis_indices, :3]
            x_axis = x_axis_landmarks[:, 0] + x_axis_landmarks[:, 2] - x_axis_landmarks[:, 1] - x_axis_landmarks[:, 3]
            x_axis = x_axis / torch.norm(x_axis, dim=-1, keepdim=True)

            y_axis = torch.cross(x_axis, -z_axis, dim=1)


            quad_rotation_matrix = torch.stack([x_axis, y_axis, z_axis], dim=-1)  # [B, 3, 3]

            landmark_rotation = p3dt.matrix_to_euler_angles(quad_rotation_matrix, convention='XYZ')


    


        features = self.encoder(views, mp_bshapes, mp_translation, landmark_input, landmark_rotation, synthetic)
        if self.fix_bshapes:
            blendshapes = mp_bshapes
        else:
            blendshapes = features[:, :53]

        if not self.disable_pose:
            rotation = features[:, 53:56] 
            
        translation = features[:, 56:59]
        

        global_translation = self.global_translation.unsqueeze(0).expand(translation.shape[0], -1)

        # scale = self.elu(features[:, 59:60]) + 1
        # global_translation = features[:, 60:63]

        scale = torch.ones_like(translation[:, -1:]) * (self.elu(self.scale) + 1)

        bshapes_tail_out = features[:, 63:]

        # rotation = quad_rotation

        out_features = torch.cat([blendshapes, rotation, translation, scale, global_translation, bshapes_tail_out], dim=-1)
           #                       0:53            53:56      56:59    59:60      60:63                 63:
        return out_features


    def train(self, mode=True):
        super().train(mode)
        # self.encoder.eval()


    def save(self, path):
        data = {
            'state_dict': self.state_dict()
        }
        torch.save(data, path)  



