import torch

import numpy as np
import torch.nn as nn
import torch

from flare.modules.embedder import get_embedder

import tinycudann as tcnn





class MLPDeformer(nn.Module):
    def __init__(self, input_feature_dim = 64, head_deformer_layers = 5, head_deformer_hidden_dim = 256, head_deformer_multires = 6,
                                                eye_deformer_layers = 2, eye_deformer_hidden_dim = 128, eye_deformer_multires = 2,):
        super().__init__()

        self.input_feature_dim = input_feature_dim
        self.head_deformer_layers = head_deformer_layers
        self.head_deformer_hidden_dim = head_deformer_hidden_dim
        self.head_deformer_multires = head_deformer_multires

        self.eye_deformer_layers = eye_deformer_layers
        self.eye_deformer_hidden_dim = eye_deformer_hidden_dim
        self.eye_deformer_multires = eye_deformer_multires

        self.head_encoder, self.head_coord_dim = get_embedder(self.head_deformer_multires)
        self.eye_encoder, self.eye_coord_dim = get_embedder(self.eye_deformer_multires)
        
        self.head_deformer = MultiLayerMLP(self.head_coord_dim + self.input_feature_dim, self.head_deformer_hidden_dim, 3, self.head_deformer_layers)
        self.eye_deformer = MultiLayerMLP(self.eye_coord_dim + self.input_feature_dim, self.eye_deformer_hidden_dim, 3, self.eye_deformer_layers)


    def set_template(self, template):
        head, eye = template
        self.head_template = head
        self.eyes_template = eye

        self.head_template_encoded = self.head_encoder(head)
        self.eyes_template_encoded = self.eye_encoder(eye)


    def forward(self, features, vertices=None):
        '''
        Args:
            vertices: (torch.Tensor, torch.Tensor). shape: ((num_head_vertices, 3), (num_eye_vertices, 3))
            features: torch.Tensor. shape: (batch_size, feature_dim)

        Returns:
            deformation: torch.Tensor. shape: (batch_size, num_vertices, 3)
        '''
        if vertices is None:
            head_deformation = self.head_deformer(self.head_template_encoded, features)
            eye_deformation = self.eye_deformer(self.eyes_template_encoded, features)
            vertices = (self.head_template, self.eyes_template)
        else:
            head_deformation = self.head_deformer(self.head_encoder(vertices[0]), features)
            eye_deformation = self.eye_deformer(self.eye_encoder(vertices[1]), features)

        deformed_head = vertices[0][None] + head_deformation
        deformed_eyes = vertices[1][None] + eye_deformation

        return_dict = {}
        return_dict['head_deformation'] = head_deformation
        return_dict['eye_deformation'] = eye_deformation
        return_dict['deformed_head'] = deformed_head
        return_dict['deformed_eyes'] = deformed_eyes

        return return_dict

    def save(self, path):
        data = {
            'state_dict': self.state_dict()
        }
        torch.save(data, path)  

def get_mlp_deformer(input_feature_dim = 64, head_deformer_layers = 5, head_deformer_hidden_dim = 256, head_deformer_multires = 6,
                                            eye_deformer_layers = 2, eye_deformer_hidden_dim = 128, eye_deformer_multires = 2, model_path=None, train=True, device='cuda'):
    mlp_deformer = MLPDeformer(input_feature_dim, head_deformer_layers, head_deformer_hidden_dim, head_deformer_multires,
                                                eye_deformer_layers, eye_deformer_hidden_dim, eye_deformer_multires)
    mlp_deformer.to(device)

    import os
    if (os.path.exists(str(model_path))):
        print("Loading model from: ", str(model_path))
        params = torch.load(str(model_path))
        mlp_deformer.load_state_dict(params["state_dict"], strict=False)
    elif model_path is not None:
        print('Model path is provided but the model is not found. Initializing with random weights.')
        raise Exception("Model not found")
    
    if train:
        mlp_deformer.train()
    else:
        mlp_deformer.eval()

    return mlp_deformer