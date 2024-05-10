import numpy as np
import torch
from torch import nn
from .encoder import ResnetEncoder
from flare.modules.embedder import *

# import tinycudann as tcnn
import pytorch3d.transforms as pt3d

def initialize_weights(m, gain=0.1):
    
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.xavier_uniform_(param.data, gain=gain)
        if 'bias' in name:
            param.data.zero_()

class NeuralBlendshapes(nn.Module):
    def __init__(self, ict_facekit):
        super().__init__()
        self.image_encoder = ResnetEncoder(62)

        # self.coords_encoder, dim = get_embedder(6, input_dims=6)

        self.ict_facekit = ict_facekit

        # self.expression_deformer = nn.Sequential(
        #                             nn.Linear(dim+53, 256),
        #                             nn.SiLU(),
        #                             nn.Linear(256, 256),
        #                             nn.SiLU(),
        #                             nn.Linear(256, 128),
        #                             nn.SiLU(),
        #                             nn.Linear(128, 128),
        #                             nn.SiLU(),
        #                             nn.Linear(128, 128),
        #                             nn.SiLU(),
        #                             nn.Linear(128, 3)
        #                             )
        
        self.only_coords_encoder, dim = get_embedder(3, input_dims=3)
        
        self.template_deformer = nn.Sequential(
                    nn.Linear(dim, 64),
                    nn.SiLU(),
                    nn.Linear(64,32),
                    nn.SiLU(),
                    nn.Linear(32,32),
                    nn.SiLU(),
                    nn.Linear(32,3)
        )
        
        # self.pose_weight = nn.Sequential(
        #             nn.Linear(dim+6, 64),
        #             nn.SiLU(),
        #             nn.Linear(64,32),
        #             nn.SiLU(),
        #             nn.Linear(32,32),
        #             nn.SiLU(),
        #             nn.Linear(32,1),
        #             nn.Sigmoid(),
        # )

        self.transform_origin = torch.nn.Parameter(torch.tensor([0., 0., 0.]))
        self.scale = torch.nn.Parameter(torch.tensor([1.]))

        # initialize_weights(self.expression_deformer, gain=0.1)
        initialize_weights(self.template_deformer, gain=0.1)
        # initialize_weights(self.pose_weight, gain=0.1)

        # set bias of last nn.linear of pose_weight to 2
        # self.pose_weight[-2].bias.data = torch.tensor([2.])




    def set_template(self, template, uv_template, vertex_parts=None, full_shape=None, head_indices=None, eyeball_indices=None):
        self.register_buffer('template', torch.cat([template, uv_template[0]], dim=1))     
        self.register_buffer('encoded_only_vertices', self.only_coords_encoder(template))

        self.pose_weight = torch.nn.Parameter((torch.ones(template.shape[0], 1)*5))
        self.pose_weight.requires_grad = True


    def forward(self, image=None, lmks=None, image_input=True, features=None):
        if image_input:
            # print(image.shape)
            if image.shape[1] != 3 and image.shape[3] == 3:
                image = image.permute(0, 3, 1, 2)
            features = self.image_encoder(image, lmks)
        
    
        bsize = features.shape[0]

        expression_mesh = self.ict_facekit(expression_weights = features[..., :53], to_canonical=True) # shape of B, V, 3

        encoded_only_vertices = self.encoded_only_vertices
        template_deformation = self.template_deformer(encoded_only_vertices)
        # template_deformation = self.template_deformation

        expression_vertices = expression_mesh + template_deformation[None]

        pose_weight = torch.nn.functional.sigmoid(self.pose_weight)[None]
        
        deformed_mesh = self.apply_deformation(expression_vertices, features, pose_weight)

        return_dict = {} 
        return_dict['features'] = features

        return_dict['full_template_deformation'] = template_deformation
        return_dict['full_expression_mesh'] = expression_vertices
        return_dict['pose_weight'] = pose_weight
        return_dict['full_deformed_mesh'] = deformed_mesh

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


def get_neural_blendshapes(model_path=None, train=True, ict_facekit=None, device='cuda'):
    neural_blendshapes = NeuralBlendshapes(ict_facekit)
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