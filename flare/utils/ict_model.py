import torch
import numpy as np
import pickle

class ICTFaceKitTorch(torch.nn.Module):
    def __init__(self, npy_dir = './assets/ict_facekit_torch.npy', canonical = None, mediapipe_name_to_ict = './assets/mediapipe_name_to_indices.pkl'):
        super().__init__()
        self.load_mediapipe_idx(mediapipe_name_to_ict)
        model_dict = np.load(npy_dir, allow_pickle=True).item()
        self.num_expression = model_dict['num_expression']
        self.num_identity = model_dict['num_identity']

        neutral_mesh = model_dict['neutral_mesh']
        uv_neutral_mesh = model_dict['uv_neutral_mesh']
        faces = model_dict['faces']
        uv_faces = model_dict['uv_faces']
        uvs = model_dict['uvs']

        expression_shape_modes = model_dict['expression_shape_modes']
        identity_shape_modes = model_dict['identity_shape_modes']

        self.landmark_indices = model_dict['landmark_indices']
        # self.landmark_indices = [1278, 1272, 12, 1834, 243, 781, 2199, 1447, 3661, 4390, 3022, 2484, 4036, 2253, 3490, 3496,
        #                  3496, 268, 493, 1914, 
        #                          2044, 1401, 3615, 4240, 4114, 2734, 2509, 978, 4527, 4942, 
        #                          4857, 1140, 2075, 1147, 4269, 3360, 1507, 1542, 1537, 1528, 
        #                          1518, 1511, 3742, 3751, 3756, 3721, 3725, 3732, 5708, 5695, 
        #                          2081, 0, 4275, 6200, 6213, 6346, 6461, 5518, 5957, 5841, 5702, 
        #                          5711, 5533, 6216, 6207, 6470, 5517, 5966]

        self.face_indices = model_dict['face_indices']
        self.not_face_indices = model_dict['not_face_indices']
        self.eyeball_indices = model_dict['eyeball_indices']
        self.head_indices = model_dict['head_indices']

        vertex_parts = model_dict['vertex_parts']

        parts_indices = {}
        for n in set(vertex_parts):
            parts_indices[n] = []

        for i, p in enumerate(vertex_parts):
            parts_indices[p].append(i)

        self.parts_indices = parts_indices

        vertex_labels = torch.zeros(len(vertex_parts), len(list(set(vertex_parts))))
        for i in range(len(vertex_parts)):
            vertex_labels[i, vertex_parts[i]] = 1
            # print(i, vertex_parts[i])
        self.register_buffer('vertex_labels', vertex_labels[None])
        self.vertex_parts = vertex_parts

        # print(neutral_mesh.shape, uv_neutral_mesh.shape, expression_shape_modes.shape, identity_shape_modes.shape, faces.shape, uv_faces.shape, uvs.shape, len(vertex_parts))

        vertex_parts = torch.tensor(vertex_parts)
        vertex_parts = vertex_parts / torch.amax(vertex_parts)
        
        uv_neutral_mesh = torch.cat([torch.tensor(uv_neutral_mesh, dtype=torch.float32), vertex_parts[..., None]], dim=1)

        # print(vertex_labels.shape, uv_neutral_mesh.shape)

        self.expression_names = model_dict['expression_names']
        self.identity_names = model_dict['identity_names']
        self.model_config = model_dict['model_config']

        self.register_buffer('neutral_mesh', torch.tensor(neutral_mesh, dtype=torch.float32)[None])
        self.register_buffer('uv_neutral_mesh', torch.tensor(uv_neutral_mesh, dtype=torch.float32)[None].clone().detach())


        # print(torch.min(self.uv_neutral_mesh), torch.max(self.uv_neutral_mesh))

        self.register_buffer('faces', torch.tensor(faces, dtype=torch.long))
        self.register_buffer('uv_faces', torch.tensor(uv_faces, dtype=torch.long))
        self.register_buffer('uvs', torch.tensor(uvs, dtype=torch.float32))

        self.register_buffer('expression_shape_modes', torch.tensor(expression_shape_modes, dtype=torch.float32)[None])
        self.register_buffer('identity_shape_modes', torch.tensor(identity_shape_modes, dtype=torch.float32)[None])

        expression_shape_modes_norm = torch.norm(torch.tensor(expression_shape_modes, dtype=torch.float32), dim=-1) # shape of (num_expression, num_vertices)
        expression_shape_modes_norm = expression_shape_modes_norm / (torch.amax(expression_shape_modes_norm, dim=1, keepdim=True) + 1e-8) # shape of (num_expression, num_vertices)



        # get weighted sum of expression shape modes to get the mean position of the expression


        # Fit gaussian to the expression shape modes
        

        # import open3d as o3d
        # mesh = o3d.geometry.TriangleMesh()
        # mesh.vertices = o3d.utility.Vector3dVector(neutral_mesh)
        # mesh.triangles = o3d.utility.Vector3iVector(faces)
        
        # for nn in range(expression_shape_modes_norm.shape[0]):
        #     colors = np.zeros_like(neutral_mesh)
        #     colors[:, 0] = expression_shape_modes_norm.clamp(1e-2, 1).pow(0.5)[nn]
        #     colors[:, 2] = 1 - expression_shape_modes_norm.clamp(1e-2, 1).pow(0.5)[nn]
        #     mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        #     o3d.io.write_triangle_mesh(f'./debug/new_norm_{nn}_{self.expression_names[nn]}.obj', mesh)
        # exit()


        self.register_buffer('expression_shape_modes_norm', expression_shape_modes_norm.clamp(1e-4, 1).pow(0.5))

        jaw_index = self.expression_names.tolist().index('jawOpen')
        self.jaw_index = jaw_index

        if canonical is not None:
            canonical = np.load(canonical, allow_pickle=True).item()
            self.register_buffer('identity', torch.from_numpy(canonical['identity'])[None])
            self.register_buffer('expression', torch.from_numpy(canonical['expression'])[None])
            # B of s, R, T is 1
            self.register_buffer('s', canonical['s'].cpu()) # shape of (B)
            self.register_buffer('R', canonical['R'].cpu()) # shape of (B, 3, 3)
            self.register_buffer('T', canonical['T'].cpu()) # shape of (B, 3)

        else:
            self.register_buffer('identity', torch.zeros(1, self.num_identity))
            self.register_buffer('expression', torch.zeros(1, self.num_expression))
            self.expression[0, jaw_index] = 0.75 # jaw open for canonical face
            # B of s, R, T is 1
            self.register_buffer('s', torch.ones(1).cpu()) # shape of (B)
            self.register_buffer('R', torch.eye(3)[None].cpu()) # shape of (B, 3, 3)
            self.register_buffer('T', torch.zeros(1, 3).cpu()) # shape of (B, 3)

        # with torch.no_grad():
        canonical = self.forward(expression_weights=self.expression, identity_weights=self.identity, to_canonical=True)
        self.register_buffer('canonical', canonical)
        self.register_buffer('neutral_mesh_canonical', self.to_canonical_space(self.neutral_mesh).clone().detach())

        # self.debug_indices()


    def to_canonical_space(self, mesh):
        """
        Transform the deformed mesh to canonical space.

        Args:
            mesh: Tensor of shape (B, num_vertices, 3) representing the deformed mesh.

        Returns:
            mesh: Tensor of shape (B, num_vertices, 3) representing the mesh in canonical space.
        """
        mesh = self.s * (torch.einsum('bvd, bmd -> bvm', mesh, self.R)) + self.T
        return mesh

    def forward(self, expression_weights=None, identity_weights=None, to_canonical=True):
        """
        Forward pass of the ICTFaceKitTorch model.

        Args:
            expression_weights: Tensor of shape (B, num_expression) representing the expression weights.
            identity_weights: Tensor of shape (B, num_identity) representing the identity weights.
            to_canonical: Boolean indicating whether to transform the deformed mesh to canonical space.

        Returns:
            deformed_mesh: Tensor of shape (B, num_vertices, 3) representing the deformed mesh.
        """
        if identity_weights is None:
            identity_weights = self.identity

        if expression_weights is None:
            expression_weights = self.expression

        assert len(expression_weights.size()) == 2 and len(identity_weights.size()) == 2

        bsize = identity_weights.size(0)
        # print(self.neutral_mesh.shape, self.expression_shape_modes.shape, self.identity_shape_modes.shape)
        # Compute the deformed mesh by applying expression and identity shape modes to the neutral mesh
        deformed_mesh = self.neutral_mesh + \
                        torch.einsum('bn, bnmd -> bmd', expression_weights, self.expression_shape_modes.repeat(bsize, 1, 1, 1))
                        # torch.einsum('bn, bnmd -> bmd', identity_weights, self.identity_shape_modes.repeat(bsize, 1, 1, 1))

        if to_canonical:
            # Transform the deformed mesh to canonical space
            deformed_mesh = self.to_canonical_space(deformed_mesh)

        return deformed_mesh
    

    def convert_quad_mesh_to_triangle_mesh(self, faces):
        """Converts a quad mesh represented as a faces array to a triangle mesh.

        Args:
            faces: A NumPy array of shape (F, 4), where each row represents a quad.

        Returns:
            A NumPy array of shape (F * 2, 3), where each row represents a triangle.
        """

        # Create a new triangle mesh.
        triangle_mesh = np.zeros((faces.shape[0] * 2, 3))

        # For each quad in the faces array, create two triangles by splitting the quad diagonally.
        for i in range(faces.shape[0]):
            
            triangle_mesh[i * 2] = faces[i, [0, 1, 2]]
            triangle_mesh[i * 2 + 1] = faces[i, [2, 3, 0]]

        triangle_mesh = self.remove_negative_triangles(triangle_mesh)

        return triangle_mesh


    def remove_negative_triangles(self, triangle_mesh):
        """Removes triangles from a triangle mesh that have negative (-1) elements.

        Args:
            triangle_mesh: A NumPy array of shape (F, 3), where each row represents a triangle.

        Returns:
            A NumPy array of shape (F', 3), where each row represents a triangle without negative elements.
        """
        positive_triangles = triangle_mesh[np.all(triangle_mesh >= 0, axis=1)]
        return positive_triangles


    def update_vmapping(self, vmapping):
        """
        NOT USED, BUT MAY BE USEFUL IN ANOTHER SCENARIO
        CURRENTLY, I manually update the UV and vertex mapping, instead of mapping from the result of xatlas
        Update the vertex mapping and adjust the relevant attributes accordingly.

        Args:
            vmapping: A list or array representing the new vertex mapping.

        Returns:
            None
        """
        # Update the vertex mapping
        self.v_mapping = vmapping

        # Update the attributes based on the new vertex mapping
        self.neutral_mesh = self.neutral_mesh[:, self.v_mapping]
        self.expression_shape_modes = self.expression_shape_modes[:, :, self.v_mapping]
        self.identity_shape_modes = self.identity_shape_modes[:, :, self.v_mapping]
        self.canonical = self.canonical[:, self.v_mapping]

        print('neutral_mesh: ', self.neutral_mesh.size())
        print('expression_shape_modes: ', self.expression_shape_modes.size())
        print('identity_shape_modes: ', self.identity_shape_modes.size())
        print('canonical: ', self.canonical.size())

        # Update the landmark indices based on the new vertex mapping
        vmapping_dict = {v: i for i, v in enumerate(vmapping)}
        new_landmark_indices = []
        for landmark_index in self.landmark_indices:
            new_landmark_indices.append(vmapping_dict[landmark_index])
        self.landmark_indices = new_landmark_indices

        # build original indices to face / not face / eyeball dict
        region_dict = [0] * (len(self.face_indices) + len(self.not_face_indices) + len(self.eyeball_indices))

        for i in range(len(region_dict)):
            region_dict[i] = 0 if i in self.face_indices else 1 if i in self.not_face_indices else 2

        face_indices = []
        not_face_indices = []
        eyeball_indices = []
        for i, v in enumerate(vmapping):
            if region_dict[v] == 0:
                face_indices.append(i)
            elif region_dict[v] == 1:
                not_face_indices.append(i)
            else:
                eyeball_indices.append(i)

        self.face_indices = face_indices
        self.not_face_indices = not_face_indices
        self.eyeball_indices = eyeball_indices

        # Update the head indices based on the new vertex mapping
        print('head', len(self.head_indices))
        head_indices = self.face_indices + self.not_face_indices
        self.head_indices = head_indices
        print(len(self.head_indices))
    
        # Update the facial mask based on the new vertex mapping
        facial_mask = torch.zeros(self.canonical.size(1))
        facial_mask[self.face_indices] = 1
        facial_mask[self.eyeball_indices] = 1
        self.facial_mask = facial_mask

    def load_mediapipe_idx(self, mediapipe_name_to_ict):

        with open(mediapipe_name_to_ict, 'rb') as f:
            mediapipe_indices = pickle.load(f)
            self.mediapipe_indices = mediapipe_indices

            self.mediapipe_to_ict = np.array([mediapipe_indices['browDownLeft'], mediapipe_indices['browDownRight'], mediapipe_indices['browInnerUp'], mediapipe_indices['browInnerUp'], 
                                    mediapipe_indices['browOuterUpLeft'], mediapipe_indices['browOuterUpRight'], mediapipe_indices['cheekPuff'], mediapipe_indices['cheekPuff'], 
                                    mediapipe_indices['cheekSquintLeft'], mediapipe_indices['cheekSquintRight'], mediapipe_indices['eyeBlinkLeft'], mediapipe_indices['eyeBlinkRight'], 
                                    mediapipe_indices['eyeLookDownLeft'], mediapipe_indices['eyeLookDownRight'], mediapipe_indices['eyeLookInLeft'], mediapipe_indices['eyeLookInRight'], 
                                    mediapipe_indices['eyeLookOutLeft'], mediapipe_indices['eyeLookOutRight'], mediapipe_indices['eyeLookUpLeft'], mediapipe_indices['eyeLookUpRight'], 
                                    mediapipe_indices['eyeSquintLeft'], mediapipe_indices['eyeSquintRight'], mediapipe_indices['eyeWideLeft'], mediapipe_indices['eyeWideRight'], 
                                    mediapipe_indices['jawForward'], mediapipe_indices['jawLeft'], mediapipe_indices['jawOpen'], mediapipe_indices['jawRight'], 
                                    mediapipe_indices['mouthClose'], mediapipe_indices['mouthDimpleLeft'], mediapipe_indices['mouthDimpleRight'], mediapipe_indices['mouthFrownLeft'], 
                                    mediapipe_indices['mouthFrownRight'], mediapipe_indices['mouthFunnel'], mediapipe_indices['mouthLeft'], mediapipe_indices['mouthLowerDownLeft'], 
                                    mediapipe_indices['mouthLowerDownRight'], mediapipe_indices['mouthPressLeft'], mediapipe_indices['mouthPressRight'], mediapipe_indices['mouthPucker'], 
                                    mediapipe_indices['mouthRight'], mediapipe_indices['mouthRollLower'], mediapipe_indices['mouthRollUpper'], mediapipe_indices['mouthShrugLower'], 
                                    mediapipe_indices['mouthShrugUpper'], mediapipe_indices['mouthSmileLeft'], mediapipe_indices['mouthSmileRight'], mediapipe_indices['mouthStretchLeft'], 
                                    mediapipe_indices['mouthStretchRight'], mediapipe_indices['mouthUpperUpLeft'], mediapipe_indices['mouthUpperUpRight'], mediapipe_indices['noseSneerLeft'], 
                                    mediapipe_indices['noseSneerRight'],]).astype(np.int32)        
            
            
    def debug_indices(self):
        # debug
        import open3d as o3d
        vertices = self.neutral_mesh.squeeze().cpu().numpy()
        faces = self.faces.cpu().numpy()

        # head indices
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        
        inverted_head_indices = set(range(vertices.shape[0])) - set(self.head_indices)
        mesh.remove_vertices_by_index(list(inverted_head_indices))
        o3d.io.write_triangle_mesh('debug/head.obj', mesh)
        
        # point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices)
        o3d.io.write_point_cloud('debug/head_pointcloud.ply', pcd)


        # eyeball indices
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        
        inverted_eyeball_indices = set(range(vertices.shape[0])) - set(self.eyeball_indices)
        mesh.remove_vertices_by_index(list(inverted_eyeball_indices))
        o3d.io.write_triangle_mesh('debug/eyeball.obj', mesh)

        # point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices)
        o3d.io.write_point_cloud('debug/eyeball_pointcloud.ply', pcd)

        # face indices
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces)

        inverted_face_indices = set(range(vertices.shape[0])) - set(self.face_indices)
        mesh.remove_vertices_by_index(list(inverted_face_indices))
        o3d.io.write_triangle_mesh('debug/face.obj', mesh)

        # point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices)
        o3d.io.write_point_cloud('debug/face_pointcloud.ply', pcd)

        # not face indices
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces)

        inverted_not_face_indices = set(range(vertices.shape[0])) - set(self.not_face_indices)
        mesh.remove_vertices_by_index(list(inverted_not_face_indices))
        o3d.io.write_triangle_mesh('debug/notface.obj', mesh)

        # point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices)
        o3d.io.write_point_cloud('debug/notface_pointcloud.ply', pcd)

        exit()
