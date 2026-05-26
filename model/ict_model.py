import torch
import numpy as np
import pickle

import pytorch3d.transforms as pt3d
import pytorch3d.ops as pt3o

import open3d as o3d


def _torch_pose_from_npy_dict(model_dict):
    """ICT→FLAME map: rigid ``flame_alignment_*`` if baked, else coarse ``flame_similarity_s/T`` + R=I."""
    if "flame_alignment_R" in model_dict:
        s = float(model_dict["flame_alignment_s"])
        R = np.asarray(model_dict["flame_alignment_R"], dtype=np.float32).reshape(3, 3)
        T = np.asarray(model_dict["flame_alignment_T"], dtype=np.float32).reshape(3)
        rigid = True
    else:
        s = float(model_dict.get("flame_similarity_s", 1.0))
        R = np.eye(3, dtype=np.float32)
        T = np.asarray(model_dict.get("flame_similarity_T", np.zeros(3, dtype=np.float32)), dtype=np.float32)
        rigid = False
    return s, R, T, rigid


class ICTFaceKitTorch(torch.nn.Module):
    def __init__(
        self,
        npy_dir='./assets/ict_facekit_torch.npy',
        canonical=None,
        mediapipe_name_to_ict='./assets/mediapipe_name_to_indices.pkl',
    ):
        super().__init__()
        if canonical is not None:
            import warnings

            warnings.warn(
                "ICTFaceKitTorch(canonical=...) is ignored. "
                "``canonical`` mesh uses flame_alignment / flame_similarity from "
                "ict_facekit_torch.npy (same as ict_facekit_to_npy_full_head.py).",
                stacklevel=2,
            )
        model_dict = np.load(npy_dir, allow_pickle=True).item()
        self.num_expression = model_dict['num_expression']
        self.num_identity = model_dict['num_identity']
        self.load_mediapipe_idx(mediapipe_name_to_ict)

        neutral_mesh = model_dict['neutral_mesh']
        uv_neutral_mesh = model_dict['uv_neutral_mesh']
        faces = model_dict['faces']
        uv_faces = model_dict['uv_faces']
        uvs = model_dict['uvs']

        # print('\n\n\n\n\n\n')
        # print(neutral_mesh.shape, uv_neutral_mesh.shape, faces.shape, uv_faces.shape, uvs.shape)
        
        expression_shape_modes = model_dict['expression_shape_modes']
        identity_shape_modes = model_dict['identity_shape_modes']

        self.landmark_indices = model_dict['landmark_indices']
        self.landmark_start = int(model_dict.get('flame_similarity_landmark_start', 17))
        
        self.face_indices = model_dict['face_indices']
        self.not_face_indices = model_dict['not_face_indices']
        self.eyeball_indices = model_dict['eyeball_indices']
        self.head_indices = model_dict['head_indices']

        self.asset_variant = model_dict.get('asset_variant', 'legacy')
        self.asset_schema_version = int(model_dict.get('asset_schema_version', 0))
        self.vertex_count = int(model_dict.get('vertex_count', model_dict['neutral_mesh'].shape[0]))

        if 'left_eyeball_indices' not in model_dict:
            self.left_eyeball_indices = list(range(21451, 23021))
        if 'right_eyeball_indices' not in model_dict:
            self.right_eyeball_indices = list(range(23021, 24591))

        for key in (
            'surface_sample_vertex_indices',
            'mouth_interior_vertex_indices',
            'gums_tongue_indices',
            'teeth_indices',
            'mouth_socket_indices',
            'eye_socket_left_indices',
            'eye_socket_right_indices',
            'skin_face_indices',
            'head_neck_indices',
            'left_eyeball_indices',
            'right_eyeball_indices',
            'left_iris_indices',
            'right_iris_indices',
            'lacrimal_indices',
            'eye_blend_indices',
            'left_eye_occlusion_indices',
            'right_eye_occlusion_indices',
            'eyelashes_left_indices',
            'eyelashes_right_indices',
            'auxiliary_part_indices',
        ):
            if key in model_dict:
                setattr(self, key, model_dict[key])

        self._register_texture_map_assets(model_dict)

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

        vertex_parts = torch.tensor(vertex_parts)
        vertex_parts = vertex_parts / torch.amax(vertex_parts)
        
        uv_neutral_mesh = torch.cat([torch.tensor(uv_neutral_mesh, dtype=torch.float32), vertex_parts[..., None]], dim=1)

        # print(vertex_labels.shape, uv_neutral_mesh.shape)

        self.expression_names = model_dict['expression_names']
        self.identity_names = model_dict['identity_names']
        self.model_config = model_dict['model_config']

        self.register_buffer('neutral_mesh', torch.tensor(neutral_mesh, dtype=torch.float32)[None])
        self.register_buffer('uv_neutral_mesh', uv_neutral_mesh[None].clone().detach())


        # print(torch.min(self.uv_neutral_mesh), torch.max(self.uv_neutral_mesh))

        self.register_buffer('faces', torch.tensor(faces, dtype=torch.long))
        self.register_buffer('uv_faces', torch.tensor(uv_faces, dtype=torch.long))
        self.register_buffer('uvs', torch.tensor(uvs, dtype=torch.float32))

        self.register_buffer('expression_shape_modes', torch.tensor(expression_shape_modes, dtype=torch.float32)[None])
        self.register_buffer('identity_shape_modes', torch.tensor(identity_shape_modes, dtype=torch.float32)[None])

        expression_shape_modes_norm = torch.norm(torch.tensor(expression_shape_modes, dtype=torch.float32), dim=-1) # shape of (num_expression, num_vertices)
        expression_shape_modes_norm = expression_shape_modes_norm / (torch.amax(expression_shape_modes_norm, dim=1, keepdim=True) + 1e-8) # shape of (num_expression, num_vertices)

        left_ids = list(getattr(self, 'left_eyeball_indices', range(21451, 23021)))
        right_ids = list(getattr(self, 'right_eyeball_indices', range(23021, 24591)))
        self.register_buffer('left_eyeball_idx', torch.tensor(left_ids, dtype=torch.long))
        self.register_buffer('right_eyeball_idx', torch.tensor(right_ids, dtype=torch.long))
        self.register_buffer(
            'left_eyeball_center',
            torch.mean(self.neutral_mesh[:, self.left_eyeball_idx], dim=1).clone().detach(),
        )
        self.register_buffer(
            'right_eyeball_center',
            torch.mean(self.neutral_mesh[:, self.right_eyeball_idx], dim=1).clone().detach(),
        )
        
        self.left_eyeball_blendshape_indices = [self.expression_names.tolist().index('eyeLookUp_L'), self.expression_names.tolist().index('eyeLookDown_L'), 
                                                self.expression_names.tolist().index('eyeLookIn_L'), self.expression_names.tolist().index('eyeLookOut_L'), ]
        self.right_eyeball_blendshape_indices = [self.expression_names.tolist().index('eyeLookUp_R'), self.expression_names.tolist().index('eyeLookDown_R'),
                                                self.expression_names.tolist().index('eyeLookIn_R'), self.expression_names.tolist().index('eyeLookOut_R'), ]

        self.register_buffer('expression_shape_modes_norm', expression_shape_modes_norm)

        jaw_index = self.expression_names.tolist().index('jawOpen')
        self.jaw_index = jaw_index

        self.register_buffer('identity', torch.zeros(1, self.num_identity))
        self.register_buffer('expression', torch.zeros(1, self.num_expression))
        self.expression[0, jaw_index] = float(model_dict.get('flame_similarity_ict_jaw_open', 0.75))
        
        flame_s, flame_R, flame_T, self.use_flame_rigid = _torch_pose_from_npy_dict(model_dict)
        self.register_buffer("flame_s", torch.tensor([flame_s], dtype=torch.float32))
        self.register_buffer("flame_R", torch.tensor(flame_R, dtype=torch.float32).reshape(1, 3, 3))
        self.register_buffer("flame_T", torch.tensor(flame_T, dtype=torch.float32).reshape(1, 3))

        # Reference mesh: jawOpen + ``flame_alignment_s,R,T`` (or coarse ``flame_similarity_s,T``)
        # from ``ict_facekit_to_npy_full_head.py`` — same space as bake / metrical NICP.
        canonical = self.forward(
            expression_weights=self.expression,
            identity_weights=self.identity,
            apply_flame_similarity=True,
            apply_eyeball_rotation=False,
        )
        # Template / deformer reference: jawOpen + rigid FLAME map only (never NICP verts).
        self.register_buffer("canonical", canonical)
        self.register_buffer("neutral_mesh_canonical", canonical.clone().detach())
        nicp_mesh = model_dict.get("nicp_canonical_mesh")
        if nicp_mesh is not None:
            nicp_t = torch.tensor(
                np.asarray(nicp_mesh, dtype=np.float32),
                dtype=torch.float32,
            ).unsqueeze(0)
            self.register_buffer("nicp_bake_mesh", nicp_t)
            self.has_nicp_bake_mesh = True
        else:
            self.has_nicp_bake_mesh = False

    def _register_texture_map_assets(self, model_dict):
        """``usemtl`` index arrays from ``ict_facekit_to_npy_full_head.py`` (K texture maps)."""
        import warnings

        for key in ("material_names", "primary_texture_materials"):
            if key in model_dict:
                setattr(self, key, list(model_dict[key]))

        if "face_material_name" in model_dict:
            setattr(self, "face_material_name", model_dict["face_material_name"])

        for key in ("n_texture_maps", "n_geometry_charts", "eye_uv_mirror_right_u"):
            if key in model_dict:
                setattr(self, key, int(model_dict[key]))

        if "geometry_chart_part" in model_dict:
            setattr(self, "geometry_chart_part", model_dict["geometry_chart_part"])

        long_keys = ("face_texture_map_id", "face_geometry_chart_id", "face_part_id")
        float_keys = (
            "triangle_uv_atlas",
            "triangle_uv_local",
            "face_uv_tile_u",
            "face_uv_tile_v",
            "texture_map_tile",
            "uv_tile_index_v",
            "uv_tile_index_vt",
        )
        for key in long_keys:
            if key in model_dict:
                self.register_buffer(key, torch.tensor(model_dict[key], dtype=torch.long))
        for key in float_keys:
            if key in model_dict:
                self.register_buffer(key, torch.tensor(model_dict[key], dtype=torch.float32))

        if not hasattr(self, "face_material_name"):
            warnings.warn(
                "ict_facekit_torch.npy missing face_material_name — rebuild with "
                "ict_facekit_to_npy_full_head.py (generic_neutral_mesh.obj usemtl)",
                stacklevel=2,
            )
        elif self.vertex_count >= 26718:
            k = int(getattr(self, "n_texture_maps", 0))
            if k != 12:
                warnings.warn(
                    f"full_head asset: expected n_texture_maps=12 (FaceKit OBJ), got {k}",
                    stacklevel=2,
                )

    def has_texture_maps(self):
        return hasattr(self, "face_texture_map_id") and hasattr(self, "material_names")

    def texture_material_names(self):
        from utils.ict_texture_maps import all_texture_materials

        return list(all_texture_materials(self))

    def faces_for_material(self, material_name):
        from utils.ict_texture_maps import face_indices_for_material

        return face_indices_for_material(self, material_name)

    def chart_uv_from_bary(self, face_idx, bary):
        from utils.ict_texture_maps import bary_to_texture_chart_uv

        return bary_to_texture_chart_uv(face_idx, bary, self)

    def build_material_uv_mesh(self, material_name, device=None):
        from utils.ict_texture_maps import build_material_uv_mesh

        return build_material_uv_mesh(self, material_name, device=device)

    def update_eyeball_centers(self, template_mesh):
        self.register_buffer(
            'left_eyeball_center',
            torch.mean(template_mesh[None, self.left_eyeball_idx], dim=1).clone().detach(),
        )
        self.register_buffer(
            'right_eyeball_center',
            torch.mean(template_mesh[None, self.right_eyeball_idx], dim=1).clone().detach(),
        )

    @property
    def use_flame_alignment(self):
        """Alias: npy baked rigid map (``flame_alignment_R`` present)."""
        return self.use_flame_rigid

    def apply_flame_similarity(self, mesh):
        """Map deformed ICT mesh to FLAME space: ``flame_s * (mesh @ flame_R) + flame_T`` (R=I if coarse bake)."""
        if self.use_flame_rigid:
            return self.flame_s * torch.matmul(mesh, self.flame_R) + self.flame_T
        return self.flame_s * mesh + self.flame_T

    def transform_displacement(self, delta):
        """Linear part of the FLAME map on per-vertex displacements ``[..., V, 3]``."""
        if self.use_flame_rigid:
            return self.flame_s * torch.matmul(delta, self.flame_R)
        return self.flame_s * delta

    def alignment_info(self):
        """Human-readable alignment state (npy mesh is always raw ICT; transforms applied at runtime)."""
        jaw = float(self.expression[0, self.jaw_index].item())
        info = {
            "neutral_mesh_prealigned": False,
            "use_flame_rigid": bool(self.use_flame_rigid),
            "use_flame_alignment": bool(self.use_flame_rigid),
            "has_nicp_bake_mesh": bool(self.has_nicp_bake_mesh),
            "flame_similarity_ict_jaw_open": jaw,
            "flame_s": float(self.flame_s.item()),
            "flame_T": self.flame_T.reshape(-1).tolist(),
        }
        if self.use_flame_rigid:
            info["flame_R"] = self.flame_R.reshape(3, 3).tolist()
        return info

    def forward(
        self,
        expression_weights=None,
        identity_weights=None,
        to_canonical=False,
        apply_eyeball_rotation=False,
        apply_flame_similarity=True,
    ):
        """
        Forward pass of the ICTFaceKitTorch model.

        Args:
            expression_weights: Tensor of shape (B, num_expression) representing the expression weights.
            identity_weights: Tensor of shape (B, num_identity) representing the identity weights.
            to_canonical: Deprecated, no-op (kept for API compat). Use ``apply_flame_similarity``.
            apply_flame_similarity: Apply npy ``flame_alignment_*`` / ``flame_similarity_*`` (FLAME space).

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
                        torch.einsum('bn, bnmd -> bmd', expression_weights, self.expression_shape_modes.repeat(bsize, 1, 1, 1)) + \
                        torch.einsum('bn, bnmd -> bmd', identity_weights, self.identity_shape_modes.repeat(bsize, 1, 1, 1))

        if apply_flame_similarity:
            deformed_mesh = self.apply_flame_similarity(deformed_mesh)

        if not apply_eyeball_rotation:
            return deformed_mesh

        left_eyeball_rotation = torch.zeros(bsize, 3).to(expression_weights.device)
        left_eyeball_rotation[:, 0] = (expression_weights[:, self.left_eyeball_blendshape_indices[1]] - expression_weights[:, self.left_eyeball_blendshape_indices[0]]) * np.pi * 0.075
        left_eyeball_rotation[:, 1] = (expression_weights[:, self.left_eyeball_blendshape_indices[3]] - expression_weights[:, self.left_eyeball_blendshape_indices[2]]) * np.pi * 0.075
        left_eyeball_matrix = pt3d.euler_angles_to_matrix(left_eyeball_rotation, convention='XYZ')

        right_eyeball_rotation = torch.zeros(bsize, 3).to(expression_weights.device)
        right_eyeball_rotation[:, 0] = (expression_weights[:, self.right_eyeball_blendshape_indices[1]] - expression_weights[:, self.right_eyeball_blendshape_indices[0]]) * np.pi * 0.075
        right_eyeball_rotation[:, 1] = (expression_weights[:, self.right_eyeball_blendshape_indices[2]] - expression_weights[:, self.right_eyeball_blendshape_indices[3]]) * np.pi * 0.075
        right_eyeball_matrix = pt3d.euler_angles_to_matrix(right_eyeball_rotation, convention='XYZ')

        li = self.left_eyeball_idx
        ri = self.right_eyeball_idx
        left_eyeball = deformed_mesh[:, li] - self.left_eyeball_center
        left_eyeball_rotated = torch.einsum('bvd, bmd -> bvm', left_eyeball, left_eyeball_matrix)
        deformed_mesh[:, li] = deformed_mesh[:, li] + left_eyeball_rotated - left_eyeball

        right_eyeball = deformed_mesh[:, ri] - self.right_eyeball_center
        right_eyeball_rotated = torch.einsum('bvd, bmd -> bvm', right_eyeball, right_eyeball_matrix)
        deformed_mesh[:, ri] = deformed_mesh[:, ri] + right_eyeball_rotated - right_eyeball

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

    def landmark_vertices(self, mesh, region='all'):
        """
        Sample Multi-PIE 68 landmarks on ``mesh`` (B,V,3) or (V,3).

        ``region``: ``all`` | ``inner`` (FLAME-pairable [17:]) | ``jawline`` ([0:17]).
        Train: MP bary (inner face) + PIE jawline verts 0:16 via ``w_pie68_jaw`` (FA detections).
        """
        idx = self.landmark_indices
        if region == 'inner':
            idx = idx[self.landmark_start :]
        elif region == 'jawline':
            idx = idx[: self.landmark_start]
        elif region != 'all':
            raise ValueError(f"landmark_vertices region must be all|inner|jawline, got {region!r}")
        if mesh.dim() == 2:
            return mesh[idx]
        return mesh[:, idx]

    def load_mediapipe_idx(self, mediapipe_name_to_ict):
        from utils.mediapipe_blendshapes import load_mediapipe_mapping

        mp_map = load_mediapipe_mapping(
            mediapipe_name_to_ict, num_expression=self.num_expression
        )
        self.mediapipe_indices = mp_map.name_to_idx
        self.register_buffer("mediapipe_to_ict", mp_map.mediapipe_to_ict)

    def mp_blendshapes_to_expression_weights(self, mp_coeffs):
        """[B, 52] MP cache → [B, 53] via ``mp[:, mediapipe_to_ict]``."""
        from utils.mediapipe_blendshapes import mp_to_ict_expression_weights

        return mp_to_ict_expression_weights(
            mp_coeffs, self.mediapipe_to_ict, self.num_expression
        )

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
