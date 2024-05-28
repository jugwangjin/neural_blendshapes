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
    def __init__(self, vertex_parts, ict_facekit):
        super().__init__()
        self.encoder = ResnetEncoder(53+6, ict_facekit)

        self.ict_facekit = ict_facekit


        # self.coords_encoder, dim = get_embedder(7, input_dims=3)


        self.expression_deformation = torch.nn.Parameter(torch.zeros_like(self.ict_facekit.expression_shape_modes))
        self.template_deformation = torch.nn.Parameter(torch.zeros_like(self.ict_facekit.canonical[0]))

        # self.eyeball_deformer = nn.Sequential(
        #                             nn.Linear(3+3+53, 64),
        #                             nn.SiLU(),
        #                             nn.Linear(64, 64),
        #                             nn.SiLU(),
        #                             nn.Linear(64, 3)
        #                             )

        self.pose_weight = nn.Sequential(
                    nn.Linear(3, 32),
                    nn.SiLU(),
                    nn.Linear(32,32),
                    nn.SiLU(),
                    nn.Linear(32,1),
                    nn.Sigmoid()
        )

        # initialize_weights(self.expression_deformer, gain=0.01)
        # initialize_weights(self.template_deformer, gain=0.01)
        # initialize_weights(self.eyeball_deformer, gain=0.01)
        initialize_weights(self.pose_weight, gain=0.01)
        self.pose_weight[-2].bias.data[0] = 3.

        self.transform_origin = torch.nn.Parameter(torch.tensor([0., 0., 0.]))



    def set_template(self, template, uv_template, vertex_parts=None, full_shape=None, head_indices=None, eyeball_indices=None):
        self.register_buffer('template', template)     
        # self.register_buffer('template', torch.cat([template, uv_template[0] - 0.5], dim=1))     

        # self.num_head_deformer = self.eyeball_index
        # self.num_eye_deformer = self.template.shape[0] - self.eyeball_index

        # self.num_face_deformer = self.face_index

        self.num_vertex = self.template.shape[0]


    def forward(self, image=None, views=None, features=None, image_input=True):
        if image_input:
            # print(image.shape)
            if image.shape[1] != 3 and image.shape[3] == 3:
                image = image.permute(0, 3, 1, 2)
            features = self.encoder(image, views)
            
        bsize = features.shape[0]

        expression_deformation = self.ict_facekit(expression_weights = features[..., :53]) - self.ict_facekit.canonical
        
        template_deformation = self.template_deformation
        pose_weight = self.pose_weight(self.ict_facekit.canonical[0])

        additional_expression_deformation = torch.einsum('bn, bnmd -> bmd', features[..., :53], self.expression_deformation.repeat(bsize, 1, 1, 1)) 
        
        # expression_deformation[:, self.eyeball_index:] += eyeball_deformation

        expression_vertices = self.ict_facekit.canonical.repeat(bsize, 1, 1)
        expression_vertices += additional_expression_deformation
        expression_vertices += expression_deformation 
        expression_vertices += template_deformation[None]

        deformed_mesh = self.apply_deformation(expression_vertices, features, pose_weight)

        return_dict = {} 
        return_dict['features'] = features

        return_dict['full_template_deformation'] = template_deformation
        return_dict['full_expression_deformation'] = expression_deformation
        return_dict['additional_expression_deformation'] = additional_expression_deformation
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