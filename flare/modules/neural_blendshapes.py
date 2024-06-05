import numpy as np
import torch
from torch import nn
from .encoder import ResnetEncoder
from flare.modules.embedder import *

# import tinycudann as tcnn
import pytorch3d.transforms as pt3d
import pytorch3d.ops as pt3o

from flare.modules.NJF_sourcemesh import SourceMesh
import os


# different activation functions
class GaussianActivation(nn.Module):
    def __init__(self, a=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a*torch.ones(1), trainable))

    def forward(self, x):
        return torch.exp(-x**2/(2*self.a**2))

def initialize_weights(m, gain=0.1):

    # iterate over layers, apply if it is nn.Linear

    for l in m.children():
        if isinstance(l, nn.Linear):
            nn.init.xavier_uniform_(l.weight, gain=gain)
            l.bias.data.zero_()


class mygroupnorm(nn.Module):
    def __init__(self, num_groups, num_channels):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.groupnorm = nn.GroupNorm(num_groups, num_channels)
    def forward(self, x):
        if len(x.shape) == 3:
            return self.groupnorm(x.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            return self.groupnorm(x)
            

class NeuralBlendshapes(nn.Module):
    def __init__(self, vertex_parts, ict_facekit, exp_dir):
        super().__init__()
        self.encoder = ResnetEncoder(53+6, ict_facekit)

        self.ict_facekit = ict_facekit
        self.tight_face_index = 6705
        self.face_index = 9409     
        self.head_index = 14062

        self.socket_index = 11248

        innards_vertices = ict_facekit.canonical[0, self.head_index:]
        exterior_vertices = ict_facekit.canonical[0, :self.head_index]

        # find closest face/head from eyeball, gums, teeth
        _, innards_displacement_index, _ = pt3o.knn_points(innards_vertices[None], exterior_vertices[None], K=1, return_nn=False)
        self.innards_displacement_index = innards_displacement_index[0, :, 0]
        
        vertices = ict_facekit.canonical[0].cpu().data.numpy()
        faces = ict_facekit.faces.cpu().data.numpy()
        
        # remove source_mesh directory
        if os.path.exists(os.path.join(str(exp_dir), 'source_mesh')):
            os.system('rm -r ' + os.path.join(str(exp_dir), 'source_mesh'))
        os.makedirs(os.path.join(str(exp_dir), 'source_mesh'), exist_ok=True)
        self.source_mesh = SourceMesh(source_ind=None, source_dir = os.path.join(str(exp_dir), 'source_mesh'), \
                          extra_source_fields=[], random_scale=1, use_wks=False, random_centering=False, cpuonly=False)
        
        
        # filter faces
        faces = ict_facekit.faces.cpu().data.numpy()
        faces = faces[faces[:, 0] < self.head_index]
        faces = faces[faces[:, 1] < self.head_index]
        faces = faces[faces[:, 2] < self.head_index]

        # faces_ = ict_facekit.faces.cpu().data.numpy()
        # faces_ = faces_[(faces_[:, 0] < self.head_index)]
        # faces_ = faces_[(faces_[:, 1] < self.head_index)]
        # faces_ = faces_[(faces_[:, 2] < self.head_index)]

        # faces_ = faces_[np.logical_not((faces_[:, 0] < self.tight_face_index) & \
        #                  (faces_[:, 1] < self.tight_face_index) & \
        #                     (faces_[:, 2] < self.tight_face_index))]

        # faces = np.concatenate([faces, faces_], axis=0)

        self.source_mesh.load(source_v = vertices[:self.head_index], source_f = faces)

        head_faces = faces

        # source mesh for tight face
        # re-filter faces
        # faces = ict_facekit.faces.cpu().data.numpy()
        # faces = faces[faces[:, 0] < self.tight_face_index]
        # faces = faces[faces[:, 1] < self.tight_face_index]
        # faces = faces[faces[:, 2] < self.tight_face_index]

        # tight_faces = faces


        # self.source_mesh_tight_face = SourceMesh(source_ind=None, source_dir = os.path.join(str(exp_dir), 'source_mesh'), \
        #                                 extra_source_fields=[], random_scale=1, use_wks=False, random_centering=False, cpuonly=False)
        # self.source_mesh_tight_face.load(source_v = vertices[:self.tight_face_index], source_f = faces)

        # self.num_tight_face_jacobian = self.source_mesh_tight_face.get_centroids_and_normals().shape[0]

        point_dim = self.source_mesh.get_point_dim()
        code_dim = 53

        self.points_encoder, encoded_point_dim = get_embedder(3, input_dims=point_dim)
        self.coords_encoder, encoded_coord_dim = get_embedder(3, input_dims=3)

        self.expression_deformer = nn.Sequential(nn.Linear(encoded_point_dim + code_dim + 9, 512),
                                                  mygroupnorm(num_groups=16, num_channels=512),
                                                  nn.PReLU(),
                                                  nn.Linear(512, 512), 
                                                  mygroupnorm(num_groups=16, num_channels=512),
                                                  nn.PReLU(),
                                                  nn.Linear(512, 512),
                                                  mygroupnorm(num_groups=16, num_channels=512),
                                                  nn.PReLU(),
                                                  nn.Linear(512, 512),
                                                  mygroupnorm(num_groups=16, num_channels=512),
                                                  nn.PReLU(),
                                                  nn.Linear(512, 512),
                                                  mygroupnorm(num_groups=16, num_channels=512),
                                                  nn.PReLU(),
                                                  nn.Linear(512, 9))
        
        self.global_translation = nn.Sequential(nn.Linear(53, 32),
                                                 mygroupnorm(num_groups=4, num_channels=32),
                                                  nn.PReLU(),
                                                 nn.Linear(32, 32),
                                                 mygroupnorm(num_groups=4, num_channels=32),
                                                  nn.PReLU(),
                                                 nn.Linear(32, 3))

        
        self.pose_weight = nn.Sequential(
                    nn.Linear(3, 32),
                    mygroupnorm(num_groups=4, num_channels=32),
                    
                                                  nn.PReLU(),
                    nn.Linear(32,32),
                    mygroupnorm(num_groups=4, num_channels=32),
                    
                                                  nn.PReLU(),
                    nn.Linear(32,1),
                    nn.Sigmoid()
        )

        # last layer to all zeros, to make zero deformation as the default            
        initialize_weights(self.expression_deformer, gain=0.01)
        self.expression_deformer[-1].weight.data.zero_()
        self.expression_deformer[-1].bias.data.zero_()

        # by default, weight to almost ones
        initialize_weights(self.pose_weight, gain=0.01)
        self.pose_weight[-2].bias.data[0] = 3.


        initialize_weights(self.global_translation, gain=0.01)
        self.global_translation[-1].bias.data.zero_()

        self.transform_origin = torch.nn.Parameter(torch.tensor([0., 0., 0.]))
        
        self.template_deformation = nn.Sequential(nn.Linear(encoded_coord_dim, 256),
                                                  mygroupnorm(num_groups=16, num_channels=256),
                                                  nn.PReLU(),
                                                  nn.Linear(256, 256), 
                                                  mygroupnorm(num_groups=16, num_channels=256),
                                                  nn.PReLU(),
                                                  nn.Linear(256, 256),
                                                  mygroupnorm(num_groups=16, num_channels=256),
                                                  nn.PReLU(),
                                                  nn.Linear(256, 256),
                                                  mygroupnorm(num_groups=16, num_channels=256),
                                                  nn.PReLU(),
                                                  nn.Linear(256, 3))

        initialize_weights(self.template_deformation, gain=0.01)
        self.template_deformation[-1].bias.data.zero_()


    def set_template(self, template, uv_template, vertex_parts=None, full_shape=None, head_indices=None, eyeball_indices=None):
        self.register_buffer('template', template)     
        # self.register_buffer('template', torch.cat([template, uv_template[0] - 0.5], dim=1))     

        self.num_face_deformer = self.face_index

        self.num_vertex = self.template.shape[0]

    '''
    TODO
    - part-wise solver? -> inefficient. 
    - eye/mouth -> simple mlp deformer.
    - or, make pseudo-connection between the parts and the whole mesh. -> it will make more not 2-manifold
    - make a virtual point, and connect the virtual point to the part.
      
    '''


    def forward(self, image=None, views=None, features=None, image_input=True):
        if image_input:
            features = self.encoder(views)
            

        bsize = features.shape[0]

        points = self.source_mesh.get_centroids_and_normals()
        # points_tight = self.source_mesh_tight_face.get_centroids_and_normals()
        encoded_points = self.points_encoder(points)
        # encoded_points_tight = self.points_encoder(points_tight)
        encoded_coords = self.coords_encoder(self.ict_facekit.canonical[0])

        template_deformation = self.template_deformation(encoded_coords)
        template_mesh = self.template + template_deformation

        pose_weight = self.pose_weight(self.ict_facekit.canonical[0])

        deformed_ict = self.ict_facekit(expression_weights = features[..., :53])
        only_ict_deformed_mesh = self.apply_deformation(deformed_ict + template_deformation[None], features, pose_weight)

        # tight_index = encoded_points_tight.shape[0]

        ict_jacobian = self.source_mesh.jacobians_from_vertices(deformed_ict[:, :self.head_index])
        # ict_jacobian_tight_face = ict_jacobian[:, :tight_index]
        

        expression_input = torch.cat([encoded_points[None].repeat(bsize, 1, 1), features[:, None, :53].repeat(1, ict_jacobian.shape[1], 1), ict_jacobian.reshape(bsize, -1, 9)], dim=2) # B V ? 
        expression_jacobian = self.expression_deformer(expression_input).reshape(bsize, -1, 3, 3)
        additional_jacobian = expression_jacobian
        # expression_jacobian = self.expression_deformer(expression_input).reshape(bsize, -1, 3, 3) + ict_jacobian
        ict_jacobian += expression_jacobian
        expression_jacobian = ict_jacobian


        expression_mesh = []
        for b in range(bsize):
            expression_mesh.append(self.source_mesh.vertices_from_jacobians(expression_jacobian[b:b+1]))
        expression_mesh = torch.cat(expression_mesh, dim=0)

        # mean translated positions
        # expression_backhead_mean = expression_mesh[:, self.face_index:self.head_index].mean(dim=1)
        # ict_backhead_mean = deformed_ict[:, self.face_index:self.head_index].mean(dim=1)
        # global_translation = ict_backhead_mean - expression_backhead_mean

        global_translation = self.global_translation(features[..., :53])
        expression_mesh += global_translation[:, None]

        face_expression_displacement = expression_mesh - deformed_ict[:, :self.head_index]
        innards_expression = face_expression_displacement[:, self.innards_displacement_index]
        

        expression_mesh = torch.cat([expression_mesh, deformed_ict[:, self.head_index:] + innards_expression], dim=1)
    
        # expression_vertices = expression_mesh # because the neutral mesh is already in the canonical space
        expression_vertices = expression_mesh + template_deformation[None] # because the neutral mesh is already in the canonical space

        deformed_mesh = self.apply_deformation(expression_vertices, features, pose_weight)


        return_dict = {} 
        return_dict['features'] = features
        return_dict['template_mesh'] = template_mesh
        return_dict['template_deformation'] = template_deformation
        return_dict['only_expression_mesh'] = expression_mesh
        return_dict['full_expression_mesh'] = expression_vertices
        return_dict['pose_weight'] = pose_weight
        return_dict['full_deformed_mesh'] = deformed_mesh
        return_dict['full_ict_deformed_mesh'] = only_ict_deformed_mesh
        return_dict['ict_jacobian'] = ict_jacobian
        return_dict['expression_jacobian'] = expression_jacobian
        return_dict['additional_jacobian'] = additional_jacobian

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

    def to(self, device):
        super().to(device)
        self.source_mesh.to(device)
        # self.source_mesh_tight_face.to(device)
        return self

def get_neural_blendshapes(model_path=None, train=True, vertex_parts=None, ict_facekit=None, exp_dir=None, device='cuda'):
    neural_blendshapes = NeuralBlendshapes(vertex_parts, ict_facekit, exp_dir)
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