import torch
import numpy as np


class ICTFaceKitTorch(torch.nn.Module):
    def __init__(self, npy_dir = './assets/ict_facekit_torch.npy', canonical = None,):
        super().__init__()

        model_dict = np.load(npy_dir, allow_pickle=True).item()
        self.num_expression = model_dict['num_expression']
        self.num_identity = model_dict['num_identity']

        neutral_mesh = model_dict['neutral_mesh']

        expression_shape_modes = model_dict['expression_shape_modes']
        identity_shape_modes = model_dict['identity_shape_modes']

        self.landmark_indices = [1225, 1888, 1052, 367, 1719, 1722, 2199, 1447, 966, 3661, 
                                 4390, 3927, 3924, 2608, 3272, 4088, 3443, 268, 493, 1914, 
                                 2044, 1401, 3615, 4240, 4114, 2734, 2509, 978, 4527, 4942, 
                                 4857, 1140, 2075, 1147, 4269, 3360, 1507, 1542, 1537, 1528, 
                                 1518, 1511, 3742, 3751, 3756, 3721, 3725, 3732, 5708, 5695, 
                                 2081, 0, 4275, 6200, 6213, 6346, 6461, 5518, 5957, 5841, 5702, 
                                 5711, 5533, 6216, 6207, 6470, 5517, 5966]

        neutral_mesh = neutral_mesh[:25351]
        faces = self.convert_quad_mesh_to_triangle_mesh(model_dict['quad_faces'][:25304])
        expression_shape_modes = expression_shape_modes[:, :25351]
        identity_shape_modes = identity_shape_modes[:, :25351]

        self.expression_names = model_dict['expression_names']
        self.identity_names = model_dict['identity_names']
        self.model_config = model_dict['model_config']

        self.register_buffer('neutral_mesh', torch.tensor(neutral_mesh, dtype=torch.float32)[None])
        self.register_buffer('faces', torch.tensor(faces, dtype=torch.long))
        self.register_buffer('expression_shape_modes', torch.tensor(expression_shape_modes, dtype=torch.float32)[None])
        self.register_buffer('identity_shape_modes', torch.tensor(identity_shape_modes, dtype=torch.float32)[None])

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

        self.identity = torch.nn.Parameter(self.identity.detach())

        # with torch.no_grad():
        canonical = self.forward(expression_weights=self.expression, identity_weights=self.identity, to_canonical=True)
        self.register_buffer('canonical', canonical)

        self.face_indices = list(range(0, 9409)) + list(range(11248, 21451)) + list(range(24591, 25351))
        self.not_face_indices = list(range(9409, 11248))
        self.eyeball_indices = list(range(21451, 24591))
        self.head_indices = self.face_indices + self.not_face_indices

        self.facial_mask = torch.zeros(self.canonical.size(1))
        self.facial_mask[self.face_indices] = 1
        self.facial_mask[self.eyeball_indices] = 1



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

        # Compute the deformed mesh by applying expression and identity shape modes to the neutral mesh
        deformed_mesh = self.neutral_mesh + \
                        torch.einsum('bn, bnmd -> bmd', expression_weights, self.expression_shape_modes.repeat(bsize, 1, 1, 1)) + \
                        torch.einsum('bn, bnmd -> bmd', identity_weights, self.identity_shape_modes.repeat(bsize, 1, 1, 1))

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

        # for i, v in enumerate(vmapping):
        #     if v in self.face_indices:
        #         self.face_indices[i] = vmapping_dict[v]
            


        # # Update the face indices based on the new vertex mapping
        # print('face', len(self.face_indices))
        # face_indices = []
        # for i in self.face_indices:
        #     face_indices.append(vmapping_dict[i])
        # self.face_indices = face_indices
        # print(len(self.face_indices))

        # # Update the not face indices based on the new vertex mapping
        # print('not face', len(self.not_face_indices))
        # not_face_indices = []
        # for i in self.not_face_indices:
        #     not_face_indices.append(vmapping_dict[i])
        # self.not_face_indices = not_face_indices
        # print(len(self.not_face_indices))

        # # Update the eyeball indices based on the new vertex mapping
        # print('eyeball', len(self.eyeball_indices))
        # eyeball_indices = []
        # for i in self.eyeball_indices:
        #     eyeball_indices.append(vmapping_dict[i])
        # self.eyeball_indices = eyeball_indices
        # print(len(self.eyeball_indices))
        
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