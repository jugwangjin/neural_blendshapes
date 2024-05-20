import numpy as np
import torch
from torch import nn
from .encoder import ResnetEncoder
from flare.modules.embedder import *

# import tinycudann as tcnn
import pytorch3d.transforms as pt3d

def initialize_weights(m, gain=0.1):
    
    for name, param in m.named_parameters():
        # if 'weight' in name:
            # nn.init.xavier_uniform_(param.data, gain=gain)
        if 'bias' in name:
            param.data.zero_()


class FACS2Deformation(nn.Module):
    # this class is a deconvolution network, that takes in the FACS codes and outputs the deformation field on UV parameterization space.
    def __init__(self, output_size, feature_size, start_size=32):
        super().__init__()
        # from 16, 16 to output_size.
        # feature size is halfed when the spatial size is doubled.
        # minimum of feature_size is 16.

        self.start_size = start_size

        size = self.start_size

        self.bilienar_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        noise_map_dim = 32
        self.noise_map = torch.nn.Parameter(torch.randn(1, noise_map_dim, start_size, start_size))
        
        assert output_size % start_size == 0, 'output_size should be divisible by start_size'

        layers = []
        layers.append(nn.Conv2d(53 + noise_map_dim, feature_size, 3, 1, padding=1))
        while size < output_size:
            layers.append(nn.Conv2d(feature_size, feature_size, 3, 1, padding=1))
            layers.append(nn.SiLU())
            layers.append(nn.Conv2d(feature_size, feature_size, 3, 1, padding=1))
            layers.append(nn.SiLU())
            layers.append(self.bilienar_upsample)
            feature_size = feature_size
            size *= 2
        
        layers.append(nn.Conv2d(feature_size, feature_size, 1, 1))
        layers.append(nn.SiLU())
        layers.append(nn.Conv2d(feature_size, 3, 1, 1))

        self.layers = nn.Sequential(*layers)

        self.layers[-1].bias.data.zero_()

    def forward(self, facs):
        B, _ = facs.shape
        # expand facs to B, C, self.start_size, self.start_size
        facs = facs[:, :, None, None].repeat(1, 1, self.start_size, self.start_size)
        return self.layers(torch.cat([facs, self.noise_map.repeat(B, 1, 1, 1)], dim=1))

class FACS2EyeRotation(nn.Module):
    def __init__(self, num_layers=1, num_hidden=16,):
        super().__init__()
        self.num_layers = num_layers
        self.num_hidden = num_hidden

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(53, num_hidden))
        self.layers.append(nn.SiLU())
        for i in range(num_layers-2):
            self.layers.append(nn.Linear(num_hidden, num_hidden))
            self.layers.append(nn.SiLU())
        self.layers.append(nn.Linear(num_hidden, 3))
    
    def forward(self, facs):
        for layer in self.layers:
            facs = layer(facs)
        return facs

class NeuralBlendshapes(nn.Module):
    def __init__(self, vertex_parts):
        super().__init__()
        self.encoder = ResnetEncoder(53+6)

        self.head_parts = ['face', 'head', 'gums', 'teeth']
        self.eyeball_parts = ['left_eyeball', 'right_eyeball']
        
        self.head_parts_indices = {'face': [], 'head': [], 'gums': [], 'teeth': []}
        self.eyeball_parts_indices = {'left_eyeball': [], 'right_eyeball': []}

        self.face_index = len(self.head_parts_indices['face'])

        for i, v in enumerate(vertex_parts):
            if v < 4:
                self.head_parts_indices[self.head_parts[v]].append(i)
            else:
                self.eyeball_parts_indices[self.eyeball_parts[v-4]].append(i)

        parts_deformer_spec = {
            'face': {'output_size': 1024, 'feature_size': 32, 'start_size': 512},
            'head': {'output_size': 256, 'feature_size': 32, 'start_size': 128},
            'gums': {'output_size': 128, 'feature_size': 32, 'start_size': 64},
            'teeth': {'output_size': 128, 'feature_size': 32, 'start_size': 64},
        }

        self.head_parts = list(set(self.head_parts) - set(['head']))
        self.part_deformers = nn.ModuleDict()
        for part in self.head_parts:
            if part in parts_deformer_spec:
                self.part_deformers[part] = FACS2Deformation(**parts_deformer_spec[part])
        [(print(part, len(self.head_parts_indices[part]))) for part in self.head_parts]

        # for eyeball, only estimate rotation and learn the origin of rotation.
        self.eyeball_rotation_origins = {}
        self.eyeball_rotation_estimator = nn.ModuleDict()
        for part in self.eyeball_parts:
            self.eyeball_rotation_origins[part] = torch.nn.Parameter(torch.tensor([0., 0., 0.]))
            self.eyeball_rotation_estimator[part] = FACS2EyeRotation()

        self.only_coords_encoder, dim = get_embedder(3, input_dims=3)
        
        self.template_deformer = nn.Sequential(
                    nn.Linear(dim, 64),
                    nn.SiLU(),
                    nn.Linear(64,64),
                    nn.SiLU(),
                    nn.Linear(64,64),
                    nn.SiLU(),
                    nn.Linear(64,64),
                    nn.SiLU(),
                    nn.Linear(64,3)
        )
        
        self.pose_weight = nn.Sequential(
                    nn.Linear(dim, 32),
                    nn.SiLU(),
                    nn.Linear(32,32),
                    nn.SiLU(),
                    nn.Linear(32,1),
                    nn.Sigmoid()
        )

        self.transform_origin = torch.nn.Parameter(torch.tensor([0., 0., 0.]))
        self.scale = torch.nn.Parameter(torch.tensor([1.]))

        self.pose_weight[-2].bias.data[0] = 1.


    def set_template(self, template, uv_template):
        self.register_buffer('template', template)     
        self.register_buffer('uv_template_for_deformer', (uv_template[0, :, :2] * 2 - 1)[None, None]) #  -1 to 1, shape of 1, 1, V, 2
        self.register_buffer('encoded_only_vertices', self.only_coords_encoder(template * 3))
        for part in self.eyeball_parts:
            self.eyeball_rotation_origins[part].data = template[self.eyeball_parts_indices[part]].mean(dim=0)


    def deform_expression(self, facs, template_deformation):
        B = facs.shape[0]
        expression_deformation = torch.zeros(B, self.template.shape[0], 3, device=facs.device)
        part_deformations = {}
        for part in self.head_parts:
            part_deformations[part] = self.part_deformers[part](facs)
            expression_deformation[:, self.head_parts_indices[part]] = torch.nn.functional.grid_sample(part_deformations[part], self.uv_template_for_deformer[:, :, self.head_parts_indices[part]].repeat(B, 1, 1, 1), align_corners=False, mode='bilinear')[:, :, 0].permute(0, 2, 1)

        for part in self.eyeball_parts:
            euler_angles = self.eyeball_rotation_estimator[part](facs) # shape of B, 3
            rotation_matrix = pt3d.euler_angles_to_matrix(euler_angles, convention = 'XYZ') # shape of B, 3, 3
            local_eye_vertices = self.template[self.eyeball_parts_indices[part]] + template_deformation[self.eyeball_parts_indices[part]] - self.eyeball_rotation_origins[part][None] # shape of V, 3
            deformed_eye_vertices = torch.einsum('vd, bdj -> bvj', local_eye_vertices, rotation_matrix) # shape of B, V, 3
            expression_deformation[:, self.eyeball_parts_indices[part]] = deformed_eye_vertices - local_eye_vertices[None]

        return expression_deformation, part_deformations


    def forward(self, image=None, views=None, features=None, image_input=True):
        # it can deal with any topology, but it requires same UV parameterization ~ and the indices for each verex. 
        # since it is not feasible currently, fix the vertices
        if image_input:
            if image.shape[1] != 3 and image.shape[3] == 3:
                image = image.permute(0, 3, 1, 2)  
            features = self.encoder(image, views)
        
        template_deformation = torch.zeros_like(self.template[..., :3])
        template_deformation = self.template_deformer(self.encoded_only_vertices)
        pose_weight = self.pose_weight(self.encoded_only_vertices)

        expression_deformation, deformation_maps = self.deform_expression(features[..., :53], template_deformation)

        expression_vertices = self.template[None][..., :3] + expression_deformation + template_deformation[None]

        deformed_mesh = self.apply_deformation(expression_vertices, features, pose_weight)

        return_dict = {} 
        return_dict['features'] = features
        return_dict['full_template_deformation'] = template_deformation
        return_dict['full_expression_deformation'] = expression_deformation
        return_dict['full_expression_mesh'] = expression_vertices
        return_dict['pose_weight'] = pose_weight
        return_dict['full_deformed_mesh'] = deformed_mesh
        return_dict['deformation_maps'] = deformation_maps

        return return_dict
        

    def apply_deformation(self, vertices, features, weights=None):
        euler_angle = features[..., 53:56]
        translation = features[..., 56:59]
        scale = self.scale

        if weights is None:
            weights = torch.ones_like(vertices[..., :1])

        B, V, _ = vertices.shape
        rotation_matrix = pt3d.euler_angles_to_matrix(euler_angle[:, None].repeat(1, V, 1) * weights, convention = 'XYZ')
        local_coordinate_vertices = (vertices  - self.transform_origin[None, None]) * scale
        deformed_mesh = torch.einsum('bvd, bvdj -> bvj', local_coordinate_vertices, rotation_matrix) + translation[:, None, :] * weights + self.transform_origin[None, None] 

        return deformed_mesh

    def save(self, path):
        data = {
            'state_dict': self.state_dict()
        }
        torch.save(data, path)  


def get_neural_blendshapes(model_path=None, train=True, vertex_parts=None, device='cuda'):
    neural_blendshapes = NeuralBlendshapes(vertex_parts)
    neural_blendshapes.to(device)

    import os
    if (os.path.exists(str(model_path))):
        print("Loading model from: ", str(model_path))
        params = torch.load(str(model_path))
        neural_blendshapes.load_state_dict(params["state_dict"], strict=False)
    elif model_path is not None:
        print('Model path is provided but the model is not found. Initializing with random weights.')
        raise Exception("Model not found")

    if train:
        neural_blendshapes.train()
    else:
        neural_blendshapes.eval()

    return neural_blendshapes        