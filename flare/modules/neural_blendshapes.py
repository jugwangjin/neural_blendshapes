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
        self.noise_map = torch.nn.Parameter(torch.randn(1, noise_map_dim, start_size, start_size) * 0.1)

        layers = []
        layers.append(nn.Conv2d(53 + noise_map_dim, feature_size, 3, 1, padding=1))
        while size < output_size:
            layers.append(nn.Conv2d(feature_size, min(16, feature_size), 3, 1, padding=1))
            layers.append(nn.SiLU())
            layers.append(nn.Conv2d(min(16, feature_size), min(16, feature_size), 3, 1, padding=1))
            layers.append(nn.SiLU())
            layers.append(self.bilienar_upsample)
            feature_size = min(16, feature_size)
            size *= 2
        
        layers.append(nn.Conv2d(min(16, feature_size), 3, 1, 1))

        self.layers = nn.Sequential(*layers)

        self.layers[-1].bias.data.zero_()

    def forward(self, facs):
        B, _ = facs.shape
        # expand facs to B, C, self.start_size, self.start_size
        facs = facs[:, :, None, None].repeat(1, 1, self.start_size, self.start_size)
        return self.layers(torch.cat([facs, self.noise_map.repeat(B, 1, 1, 1)], dim=1))


class NeuralBlendshapes(nn.Module):
    def __init__(self, vertex_parts):
        super().__init__()
        self.image_encoder = ResnetEncoder(62)

        self.head_parts = ['face', 'head', 'gums', 'teeth', 'left_eyeball', 'right_eyeball']
        
        self.head_parts_indices = {'face': [], 'head': [], 'gums': [], 'teeth': [], 'left_eyeball': [], 'right_eyeball': []}
        
        for i, v in enumerate(vertex_parts):
            self.head_parts_indices[self.head_parts[v]].append(i)

        parts_deformer_spec = {
            'face': {'output_size': 512, 'feature_size': 256, 'start_size': 32},
            'head': {'output_size': 256, 'feature_size': 128, 'start_size': 32},
            'gums': {'output_size': 128, 'feature_size': 64, 'start_size': 32},
            'teeth': {'output_size': 128, 'feature_size': 64, 'start_size': 32},
            'left_eyeball': {'output_size': 64, 'feature_size': 64, 'start_size': 32},
            'right_eyeball': {'output_size': 64, 'feature_size': 64, 'start_size': 32},
        }

        self.part_deformers = nn.ModuleDict()
        for part in self.head_parts:
            self.part_deformers[part] = FACS2Deformation(**parts_deformer_spec[part])

        print('Face:', len(self.face_indices))
        print('Head:', len(self.head_indices))
        print('Gums:', len(self.gums_indices))
        print('Teeth:', len(self.teeth_indices))
        print('Left Eyeball:', len(self.left_eyeball_indices))
        print('Right Eyeball:', len(self.right_eyeball_indices))

        
        self.only_coords_encoder, dim = get_embedder(2, input_dims=3)
        
        self.constant_deformer = nn.Sequential(
                    nn.Linear(dim, 64),
                    nn.SiLU(),
                    nn.Linear(64,32),
                    nn.SiLU(),
                    nn.Linear(32,16),
                    nn.SiLU(),
                    nn.Linear(16,4)
        )

        self.transform_origin = torch.nn.Parameter(torch.tensor([0., 0., 0.]))
        self.scale = torch.nn.Parameter(torch.tensor([1.]))

        self.constant_deformer[-1].bias.data[3] = 3.


    def set_template(self, template, uv_template, vertex_parts=None, full_shape=None, head_indices=None, eyeball_indices=None):
        self.register_buffer('template', torch.cat([template, uv_template[0]], dim=1))     
        self.register_buffer('uv_template', uv_template[0]) #  
        self.register_buffer('uv_template_for_deformer', (uv_template[0, :, :2] * 2 - 1)[None, None]) #  -1 to 1, shape of 1, 1, V, 2
        # self.register_buffer('encoded_vertices', self.coords_encoder(self.template))
        self.register_buffer('encoded_only_vertices', self.only_coords_encoder(template))

        # self.pose_weight = torch.nn.Parameter((torch.ones(template.shape[0], 1)*5))
        # self.template_deformation = torhc.nn.Parameter((torch.zeros(template.shape[0], 3)))


    def deform_expression(self, facs):
        B = facs.shape[0]
        expression_deformation = torch.zeros(B, self.template.shape[0], 3, device=facs.device)
        part_deformations = {}
        for part in self.head_parts:
            part_deformations[part] = self.part_deformers[part](facs)
            expression_deformation[:, self.head_parts_indices[part]] = torch.nn.functional.grid_sample(part_deformations[part], self.uv_template_for_deformer[:, :, self.head_parts_indices[part]].repeat(B, 1, 1, 1), align_corners=False, mode='bilinear')[:, :, 0].permute(0, 2, 1)




        return expression_deformation, part_deformations

    def forward(self, image=None, lmks=None, image_input=True, features=None, vertices=None, coords_min=None, coords_max=None):
        if image_input:
            # print(image.shape)
            if image.shape[1] != 3 and image.shape[3] == 3:
                image = image.permute(0, 3, 1, 2)
            # print(image.shape)
            features = self.image_encoder(image, lmks)
        # else:
        # print(features.shape, estim_lmks.shape)
        if vertices is None:
            vertices = self.template
            # encoded_vertices = self.encoded_vertices
            encoded_only_vertices = self.encoded_only_vertices
        else:
            # encoded_vertices = self.coords_encoder(vertices)
            encoded_only_vertices = self.only_coords_encoder(vertices[..., :3])

        bsize = features.shape[0]

        constant_deformation = self.constant_deformer(encoded_only_vertices)
        template_deformation = constant_deformation[..., :3]
        
        pose_weight = torch.nn.functional.sigmoid(constant_deformation[..., 3:])[None]

        
        expression_deformation, deformation_maps = self.deform_expression(features[..., :53])


        # expression_vertices = expression_deformation + template_deformation[None]
        expression_vertices = vertices[None][..., :3] + expression_deformation + template_deformation[None]

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
        # print(features.shape, euler_angle.shape)
        # rotation_matrix = pt3d.euler_angles_to_matrix(euler_angle, convention = 'XYZ')
        # print(rotation_matrix[0])
        # print(rotation_matrix[1])

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