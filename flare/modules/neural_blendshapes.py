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
    def __init__(self):
        super().__init__()
        self.image_encoder = ResnetEncoder(60)

        # L = 8; F = 4; log2_T = 19; N_min = 16
        # b = np.exp(np.log(2048/N_min)/(L-1))

        # self.coord_encoder = tcnn.Encoding(
        #                 n_input_dims=3, 
        #                 encoding_config={
        #                     "otype": "Grid",
        #                     "type": "Hash",
        #                     "n_levels": L,
        #                     "n_features_per_level": F,
        #                     "log2_hashmap_size": log2_T,
        #                     "base_resolution": N_min,
        #                     "per_level_scale": b,
        #                     "interpolation": "Linear"
        #                     },      
        #                 ).to('cuda')

        self.coords_encoder, dim = get_embedder(4, input_dims=3)


        # self.expression_deformer = nn.Sequential(
        #                             nn.Linear(dim+53, 128),
        #                             nn.SiLU(),
        #                             nn.Linear(128, 128),
        #                             nn.SiLU(),
        #                             nn.Linear(128, 128),
        #                             nn.SiLU(),
        #                             nn.Linear(128, 128),
        #                             nn.SiLU(),
        #                             nn.Linear(128, 128),
        #                             nn.SiLU(),
        #                             nn.Linear(128, 3)
        #                             )
        
        # L = 8; F = 2; log2_T = 19; N_min = 16
        # b = np.exp(np.log(2048/N_min)/(L-1))

        # self.coarse_coord_encoder = tcnn.Encoding(
        #                 n_input_dims=3, 
        #                 encoding_config={
        #                     "otype": "Grid",
        #                     "type": "Hash",
        #                     "n_levels": L,
        #                     "n_features_per_level": F,
        #                     "log2_hashmap_size": log2_T,
        #                     "base_resolution": N_min,
        #                     "per_level_scale": b,
        #                     "interpolation": "Linear"
        #                     },      
        #                 ).to('cuda')


        # self.coarse_coords_encoder, dim = get_embedder(2, input_dims=5)

        # self.eye_expression_deformer = nn.Sequential(
        #                             nn.Linear(dim+53, 128),
        #                             nn.SiLU(),
        #                             nn.Linear(128, 128),
        #                             nn.SiLU(),
        #                             nn.Linear(128, 128),
        #                             nn.SiLU(),
        #                             nn.Linear(128, 3)
        #                             )

        self.template_deformer = nn.Sequential(
                    nn.Linear(dim, 64),
                    nn.SiLU(),
                    nn.Linear(64,32),
                    nn.SiLU(),
                    nn.Linear(32,32),
                    nn.SiLU(),
                    nn.Linear(32,3)
        )
        
        self.pose_weight = nn.Sequential(
                    nn.Linear(dim+7, 64),
                    nn.SiLU(),
                    nn.Linear(64,32),
                    nn.SiLU(),
                    nn.Linear(32,32),
                    nn.SiLU(),
                    nn.Linear(32,1),
                    nn.Tanh(),
        )

        self.transform_origin = torch.nn.Parameter(torch.tensor([0., 0., 0.]))

        # initialize_weights(self.expression_deformer)
        # initialize_weights(self.eye_expression_deformer)
        # initialize_weights(self.template_deformer, gain=0.01)
        # initialize_weights(self.pose_weight, gain=0.01)

                                    # ict_canonical_mesh.vertices.shape, ict_facekit.head_indices, ict_facekit.eyeball_indices)
        
    def set_template(self, template, ict_facekit):
        # assert len(template.shape) == 2, "template should be a tensor shape of (num_vertices, 3) but got shape of {}".format(template.shape)
        # face_template, eye_template = template
        # face_uv_template, eye_uv_template = uv_template
        self.register_buffer('template', template)     
        self.ict_facekit = ict_facekit
        # self.register_buffer('uv_template', uv_template)



        # self.register_buffer('face_template', torch.cat([face_template, face_uv_template], dim=1))
        # self.register_buffer('eye_template', torch.cat([eye_template, eye_uv_template], dim=1))

        self.register_buffer('encoded_vertices', self.coords_encoder(self.template))
        # self.register_buffer('coarse_encoded_vertices', self.coarse_coords_encoder(self.face_template))

        # self.register_buffer('eye_encoded_vertices', self.coarse_coords_encoder(self.eye_template))
        
        # self.full_shape = list(full_shape)

        # self.head_indices = head_indices
        # self.eyeball_indices = eyeball_indices

    def forward(self, image=None, lmks=None, image_input=True, features=None, vertices=None, coords_min=None, coords_max=None):
        if image_input:
            if image.shape[1] != 3 and image.shape[3] == 3:
                image = image.permute(0, 3, 1, 2)

            features = self.image_encoder(image, lmks)

        euler_angle = features[..., 53:56]
        translation = features[..., 56:59]
        scale = features[..., -1:]

        if vertices is None:
            vertices = self.template
            encoded_vertices = self.encoded_vertices
        else:

            encoded_vertices = self.coords_encoder(vertices)


        template_deformation = self.template_deformer(encoded_vertices)

        expression_vertices= self.ict_facekit(expression_weights = features[:, :53], to_canonical = True)

        B, V, _ = expression_vertices.shape 

        pose_weight = self.pose_weight(torch.cat([encoded_vertices[None].repeat(features.shape[0], 1, 1), \
                                                features[:, None, 53:].repeat(1, V, 1)], dim=2)) # shape of B V 1
        
        rotation_matrix = pt3d.euler_angles_to_matrix(euler_angle[:, None].repeat(1, V, 1) * pose_weight, convention = 'XYZ')

        local_coordinate_vertices = expression_vertices - self.transform_origin[None, None]

        local_coordinate_vertices = local_coordinate_vertices * scale[:, None]

        deformed_mesh = torch.einsum('bvd, bvdj -> bvj', local_coordinate_vertices, rotation_matrix) + translation[:, None, :] + self.transform_origin[None, None] + template_deformation[None]

        return_dict = {} 
        return_dict['features'] = features

        return_dict['full_template_deformation'] = template_deformation
        return_dict['full_expression_mesh'] = expression_vertices
        return_dict['pose_weight'] = pose_weight
        return_dict['full_deformed_mesh'] = deformed_mesh

        return return_dict
        
    def save(self, path):
        data = {
            'state_dict': self.state_dict()
        }
        torch.save(data, path)  


def get_neural_blendshapes(model_path=None, train=True, device='cuda'):
    neural_blendshapes = NeuralBlendshapes()
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