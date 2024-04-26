import numpy as np
import torch
from torch import nn
from .encoder import ResnetEncoder
import tinycudann as tcnn

class NeuralBlendshapes(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ResnetEncoder(64)

        L = 8; F = 4; log2_T = 19; N_min = 16
        b = np.exp(np.log(2048/N_min)/(L-1))

        self.encoder = tcnn.Encoding(
                        n_input_dims=3, n_output_dims=32,
                        encoding_config={
                            "otype": "Grid",
                            "type": "Hash",
                            "n_levels": L,
                            "n_features_per_level": F,
                            "log2_hashmap_size": log2_T,
                            "base_resolution": N_min,
                            "per_level_scale": b,
                            "interpolation": "Linear"
                            },      
                        )

        self.expression_deformer = tcnn.Network(
                n_input_dims=64+32, n_output_dims=3,
                network_config={
                    "otype": "CutlassMLP",
                    "activation": "SoftPlus",
                    "output_activation": "None",
                    "n_neurons": 128,
                    "n_hidden_layers": 4,
                }
                )
        
        self.template_deformer = \
            tcnn.Network(
                n_input_dims=32, n_output_dims=3,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 32,
                    "n_hidden_layers": 2,
                }
            )

        
    def set_template(self, template, coords_min, coords_max):
        assert len(template) == 2, "template should be a tensor shape of (num_vertices, 3)"
        self.tempate = template 

        self.coords_min = coords_min
        self.coords_max = coords_max

        self.range = self.coords_max - self.coords_min

        self.encoded_template = self.encoder((template - self.coords_min) / self.range) # shape of (num_vertices, 32), range to [0, 1]

    def forward(self, image, vertices=None, coords_min=None, coords_max=None):
        if image.shape[1] != 3 and image.shape[3] == 3:
            image = image.permute(0, 3, 1, 2)

        if vertices is None:
            vertices = self.template
            encoded_vertices = self.encoded_template

        else:
            coords_min = coords_min if coords_min is not None else self.coords_min
            coords_max = coords_max if coords_max is not None else self.coords_max

            range = coords_max - coords_min
            encoded_vertices = self.encoder((vertices - coords_min) / range)

        features = self.encoder(image)

        template_deformation = self.template_deformer(encoded_vertices)

        # concat feature and encoded vertices. vertices has shape of N_VERTICES, 16 and features has shape of N_BATCH, 64
        deformer_input = torch.cat([encoded_vertices[None].repeat(features.shape[0], 1, 1), \
                                    features[:, None].repeat(1, encoded_vertices.shape[0], 1)], dim=2)

        B, V, _ = deformer_input.shape

        expression_deformation = self.expression_deformer(deformer_input.view(B*V, -1)).view(B, V, -1)

        deformed_mesh = vertices + expression_deformation + template_deformation[None]

        return_dict = {} 
        return_dict['features'] = features
        return_dict['template_deformation'] = template_deformation
        return_dict['expression_deformation'] = expression_deformation
        return_dict['deformed_mesh'] = deformed_mesh


        
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