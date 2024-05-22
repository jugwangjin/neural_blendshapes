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
        self.bilinear_upsample = nn.PixelShuffle(2)

        noise_map_dim = feature_size
        self.noise_map = torch.nn.Parameter(torch.randn(1, noise_map_dim, start_size, start_size))
        
        assert output_size % start_size == 0, 'output_size should be divisible by start_size'

        layers = []
        layers.append(nn.Conv2d(53 + noise_map_dim, feature_size, 3, 1, padding=1))
        while size < output_size:
            layers.append(nn.Conv2d(feature_size, feature_size, 3, 1, padding=1))
            layers.append(nn.SiLU())
            layers.append(nn.Conv2d(feature_size, feature_size, 3, 1, padding=1))
            layers.append(nn.SiLU())
            layers.append(self.bilinear_upsample)
            feature_size = int(feature_size/4)
            size *= 2
        
        layers.append(nn.Conv2d(feature_size, feature_size, 3, 1, 1))
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
    def __init__(self, num_layers=3, num_hidden=16,):
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
    def __init__(self, vertex_parts, ict_facekit):
        super().__init__()
        self.encoder = ResnetEncoder(53+6, ict_facekit)

        self.ict_facekit = ict_facekit

        self.only_coords_encoder, dim = get_embedder(2, input_dims=3)
        
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
        self.pose_weight[-2].bias.data[0] = 3.


    def set_template(self, template, uv_template):
        self.register_buffer('template', template)     
        self.register_buffer('uv_template_for_deformer', (uv_template * 2 - 1)[None, None]) #  -1 to 1, shape of 1, 1, V, 2
        self.register_buffer('encoded_only_vertices', self.only_coords_encoder(template * 3))


    def forward(self, image=None, views=None, features=None, image_input=True):
        # it can deal with any topology, but it requires same UV parameterization ~ and the indices for each verex. 
        # since it is not feasible currently, fix the vertices
        if image_input:
            if image.shape[1] != 3 and image.shape[3] == 3:
                image = image.permute(0, 3, 1, 2)  
            features = self.encoder(image, views)

        expression_deformation = self.ict_facekit(expression_weights = features[..., :53]) - self.ict_facekit.neutral_mesh_canonical

        template_deformation = torch.zeros_like(self.template[..., :3])
        template_deformation = self.template_deformer(self.encoded_only_vertices)
        pose_weight = self.pose_weight(self.encoded_only_vertices)

        expression_vertices = self.template[None][..., :3] + expression_deformation + template_deformation[None]

        deformed_mesh = self.apply_deformation(expression_vertices, features, pose_weight)

        return_dict = {} 
        return_dict['features'] = features
        return_dict['full_template_deformation'] = template_deformation
        return_dict['full_expression_deformation'] = expression_deformation
        return_dict['full_expression_mesh'] = expression_vertices
        return_dict['pose_weight'] = pose_weight
        return_dict['full_deformed_mesh'] = deformed_mesh

        return return_dict
        

    def apply_deformation(self, vertices, features, weights=None):
        euler_angle = features[..., 53:56]
        translation = features[..., 56:59]
        scale = features[..., 59:]

        # print(euler_angle.shape, translation.shape, scale.shape)

        if weights is None:
            weights = torch.ones_like(vertices[..., :1])

        B, V, _ = vertices.shape
        rotation_matrix = pt3d.euler_angles_to_matrix(euler_angle[:, None].repeat(1, V, 1) * weights, convention = 'XYZ')
        local_coordinate_vertices = (vertices  - self.transform_origin[None, None]) * scale[:, None]
        deformed_mesh = torch.einsum('bvd, bvdj -> bvj', local_coordinate_vertices, rotation_matrix) + translation[:, None, :] * weights + self.transform_origin[None, None] 

        return deformed_mesh

    def save(self, path):
        data = {
            'state_dict': self.state_dict()
        }
        torch.save(data, path)  


def get_neural_blendshapes(model_path=None, train=True, vertex_parts=None, ict_facekit=None, device='cuda'):
    neural_blendshapes = NeuralBlendshapes(vertex_parts, ict_facekit)
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