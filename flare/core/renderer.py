# Code: https://github.com/fraunhoferhhi/neural-deferred-shading/tree/main
# Modified/Adapted by: Shrisha Bharadwaj

import nvdiffrast.torch as dr
import torch
from pytorch3d.ops import knn_points
import torch.nn.functional as F
class Renderer:
    """ Rasterization-based triangle mesh renderer that produces G-buffers for a set of views.

    Args:
        device (torch.device): Device used for rendering (must be a GPU)
        near (float): Near plane distance
        far (float): Far plane distance
    """

    def __init__(self, device, near=1, far=1000, mesh=None):
        self.glctx = dr.RasterizeCudaContext()
        self.device = device
        self.near = near
        self.far = far

    def set_near_far(self, dataset_train, samples, epsilon=0.1):
        mins = []
        maxs = []
        cameras = dataset_train.get_camera_mat()
        for cam in cameras:
            samples = samples.to(cam.device)
            samples_projected = cam.project(samples, depth_as_distance=True)
            mins.append(samples_projected[...,2].min())
            maxs.append(samples_projected[...,2].max())
        near, far = min(mins), max(maxs)
        near = near.to(self.device)
        far = far.to(self.device)
        self.near = near - (near * epsilon)
        self.far = far + (far * epsilon)
        
    @staticmethod
    def transform_pos(mtx, pos):
        t_mtx = torch.from_numpy(mtx) if not torch.torch.is_tensor(mtx) else mtx
        t_mtx = t_mtx.to(pos.device)
        # (x,y,z) -> (x,y,z,1)
        posw = torch.cat([pos, torch.ones_like(pos[:, 0:1])], axis=1)
        return torch.matmul(posw, t_mtx.t())[None, ...]
    
    @staticmethod
    def transform_pos_batch(mtx, pos):
        t_mtx = torch.from_numpy(mtx) if not torch.torch.is_tensor(mtx) else mtx
        t_mtx = t_mtx.to(pos.device)
        # (x,y,z) -> (x,y,z,1)
        posw = torch.cat([pos, torch.ones_like(pos[..., 0:1])], axis=2)
        return torch.bmm(posw, t_mtx)

    @staticmethod
    def projection(fx, fy, cx, cy, n, f, width, height, device):
        """
        Returns a gl projection matrix
        The memory order of image data in OpenGL, and consequently in nvdiffrast, is bottom-up.
        Note that cy has been inverted 1 - cy!
        """
        return torch.tensor([[2.0*fx/width,           0,       1.0 - 2.0 * cx / width,                  0],
                            [         0, 2.0*fy/height,      1.0 - 2.0 * cy / height,                  0],
                            [         0,             0,                 -(f+n)/(f-n),     -(2*f*n)/(f-n)],
                            [         0,             0,                           -1,                  0.0]], device=device) 
    @staticmethod
    def to_gl_camera(camera, resolution, n=1000, f=5000):

        projection_matrix = Renderer.projection(fx=camera.K[0,0],
                                                fy=camera.K[1,1],
                                                cx=camera.K[0,2],
                                                cy=camera.K[1,2],
                                                n=n,
                                                f=f,
                                                width=resolution[1],
                                                height=resolution[0],
                                                device=camera.device)

        Rt = torch.eye(4, device=camera.device)
        Rt[:3, :3] = camera.R
        Rt[:3, 3] = camera.t

        gl_transform = torch.tensor([[1., 0,  0,  0],
                                    [0,  1., 0,  0],
                                    [0,  0, -1., 0],
                                    [0,  0,  0,  1.]], device=camera.device)

        Rt = gl_transform @ Rt
        return projection_matrix @ Rt, Rt

    @staticmethod
    def to_gl_camera_batch(camera, resolution, n=1000, f=5000):

        # since we have a single camera (!) same intrinsics!! 
        intrinsics = camera[0].K 
        scale_x = resolution[1] / 512
        scale_y = resolution[0] / 512
        device = camera[0].device

        projection_matrix = Renderer.projection(fx=intrinsics[0,0] * scale_x,
                                                fy=intrinsics[1,1] * scale_y,
                                                cx=intrinsics[0,2] * scale_x,
                                                cy=intrinsics[1,2] * scale_y,
                                                n=n,
                                                f=f,
                                                width=resolution[1],
                                                height=resolution[0],
                                                device=device)

        gl_transform = torch.tensor([[1., 0,  0,  0],
                                    [0,  1., 0,  0],
                                    [0,  0, -1., 0],
                                    [0,  0,  0,  1.]], device=device)
        
        p_l = []
        rt_gl = []
        for cam in camera:
            Rt = torch.eye(4, device=device)
            Rt[:3, :3] = cam.R
            Rt[:3, 3] = cam.t
            Rt_gl = gl_transform @ Rt
            P = projection_matrix @ Rt_gl
            # we transpose here instead of transposing while multiplying
            p_l.append(P.t().unsqueeze(0))
            rt_gl.append(Rt_gl.unsqueeze(0))
            
        return torch.cat(p_l, dim=0), torch.cat(rt_gl, dim=0)
    
    def get_vertices_clip_space_from_view(self, views, vertices):
        batch_size = vertices.shape[0]
        # single fixed camera and for now we fix res also
        resolution = (512, 512)
        P_batch, Rt = Renderer.to_gl_camera_batch(views['flame_camera'], resolution, n=self.near, f=self.far)
        deformed_vertices_clip_space = Renderer.transform_pos_batch(P_batch, vertices)
        return deformed_vertices_clip_space

    def get_vertices_clip_space(self, gbuffers, vertices):
        P_batch = gbuffers["P_batch"]
        return Renderer.transform_pos_batch(P_batch, vertices)


    def get_vertices_clip_space_from_view(self, camera, vertices, resolution=(512, 512), n=None, f=None):
        if n is None:
            n = self.near
        if f is None:
            f = self.far

        # batch_size = vertices.shape[0]

        P_batch, Rt = Renderer.to_gl_camera_batch(camera, resolution, n=n, f=f)
        deformed_vertices_clip_space = Renderer.transform_pos_batch(P_batch, vertices)

        return deformed_vertices_clip_space


    def render_batch(self, views, deformed_vertices, deformed_normals, channels, with_antialiasing, canonical_v, canonical_idx, 
                     canonical_uv, mesh, deformed_vertices_clip_space=None, target_resolution=None):
        """ Render G-buffers from a set of views.

        Args:
            views (List[Views]): 
        """
        batch_size = deformed_vertices.shape[0]
        # single fixed camera and for now we fix res also

        resolution = (512, 512) if target_resolution is None else target_resolution

        high_res = (512, 512)
        def upsample(buffer, high_res):
            if buffer.shape[1] == high_res[0] and buffer.shape[2] == high_res[1]:
                return buffer
            # Convert from (B, H, W, C) -> (B, C, H, W)
            buffer = buffer.permute(0, 3, 1, 2)
            
            # Perform bilinear upsampling
            upsampled = F.interpolate(buffer, size=high_res, mode='bilinear', align_corners=False)
            
            # Convert back from (B, C, H, W) -> (B, H, W, C)
            return upsampled.permute(0, 2, 3, 1)

        P_batch, Rt = Renderer.to_gl_camera_batch(views, resolution, n=self.near, f=self.far)

        canonical_verts_batch = canonical_v.unsqueeze(0).repeat(batch_size, 1, 1)
        deformed_vertices_clip_space = Renderer.transform_pos_batch(P_batch, deformed_vertices) if deformed_vertices_clip_space is None else deformed_vertices_clip_space

        idx = canonical_idx.int()
        face_idx = deformed_normals["face_idx"].int()
        rast, rast_out_db = dr.rasterize(self.glctx, deformed_vertices_clip_space, idx, resolution=resolution)

        # import time

        # start = time.time()

        face_ids = rast[..., -1].long()
        # Flatten the face_ids to easily work with indexing
        face_ids_flat = face_ids.view(-1)  # Shape: (batch_size * height * width)

        # Create a segmentation map initialized to -1 (indicating background)
        segmentation_map = torch.full((batch_size, resolution[0], resolution[1]), fill_value=-1, dtype=torch.long, device=face_ids.device)

        # Flatten segmentation_map for easier indexing
        segmentation_map_flat = segmentation_map.view(-1)

        # Create a mask for valid faces (face_ids >= 0)
        invalid_mask = torch.all(rast == 0, dim=-1) # B H W mask
        # negate 
        valid_mask = ~invalid_mask
        valid_mask = valid_mask.view(-1)

        # Update segmentation map for valid face pixels using mesh.face_labels
        valid_face_ids = face_ids_flat[valid_mask]  # Extract valid face indices (1D tensor)
        segmentation_map_flat[valid_mask] = mesh.face_labels[valid_face_ids]  # Assign labels based on face indices

        # Reshape segmentation_map back to (batch_size, height, width)
        segmentation_map = segmentation_map_flat.view(batch_size, resolution[0], resolution[1])

        # print(f"Segmentation map computation time: {time.time() - start}")

        view_dir = torch.cat([v.center.unsqueeze(0) for v in views], dim=0)
        view_dir = view_dir[:, None, None, :]
        gbuffer = {}

        gbuffer["P_batch"] = P_batch

        # deformed points in G-buffer
        if "position" in channels or "depth" in channels:
            position, _ = dr.interpolate(deformed_vertices, rast, idx)
            gbuffer["position"] = dr.antialias(position, rast, deformed_vertices_clip_space, idx) if with_antialiasing else position
            # gbuffer["position"] = upsample(gbuffer["position"], high_res)

        # canonical points in G-buffer
        if "canonical_position" in channels:
            canonical_position, _ = dr.interpolate(canonical_verts_batch, rast, idx, rast_db=rast_out_db, diff_attrs='all')
            gbuffer["canonical_position"] = canonical_position
            # gbuffer["canonical_position"] = upsample(gbuffer["canonical_position"], high_res)

        # normals in G-buffer
        if "normal" in channels:
            vertex_normals, _ = dr.interpolate(deformed_normals["vertex_normals"], rast, idx)
            face_normals, _ = dr.interpolate(deformed_normals["face_normals"], rast, face_idx)
            tangent_normals, _ = dr.interpolate(deformed_normals["tangent_normals"], rast, idx)
            gbuffer["vertex_normals"] = vertex_normals
            # gbuffer["vertex_normals"] = upsample(gbuffer["vertex_normals"], high_res)
            gbuffer["face_normals"] = face_normals
            # gbuffer["face_normals"] = upsample(gbuffer["face_normals"], high_res)
            gbuffer["tangent_normals"] = tangent_normals
            # gbuffer["tangent_normals"] = upsample(gbuffer["tangent_normals"], high_res)


        # mask of mesh in G-buffer
        if "mask" in channels:
            gbuffer["mask"] = (rast[..., -1:] > 0.).float() 
            gbuffer["mask_low_res"] = gbuffer["mask"]
            gbuffer["mask"] = upsample(gbuffer["mask"], high_res)

        if 'mask' in channels and 'canonical_position' in channels and 'segmentation' in channels:            

            # segmentation: face W/O mouth
            segmentation = ((segmentation_map[..., None] == 0) | (segmentation_map[..., None] == 1)).float() # face W/O eyes and mouth.
            segmentation = dr.antialias(segmentation.float(), rast, deformed_vertices_clip_space, idx)

            gbuffer['segmentation'] = segmentation
            gbuffer['segmentation'] = upsample(gbuffer['segmentation'], high_res)

            mouth_segmentation = (segmentation_map[..., None] == 1).float() # mouth
            mouth_segmentation = dr.antialias(mouth_segmentation.float(), rast, deformed_vertices_clip_space, idx)

            gbuffer['mouth'] = mouth_segmentation
            gbuffer['mouth'] = upsample(gbuffer['mouth'], high_res)


            eyes_segmentation = (segmentation_map[..., None] == 2).float() # eyes
            eyes_segmentation = dr.antialias(eyes_segmentation.float(), rast, deformed_vertices_clip_space, idx)

            gbuffer['eyes'] = eyes_segmentation
            gbuffer['eyes'] = upsample(gbuffer['eyes'], high_res)

        try:
            uv_coordinates, _ = dr.interpolate(canonical_uv, rast, idx, rast_db=rast_out_db, diff_attrs='all')
            gbuffer["uv_coordinates"] = dr.antialias(uv_coordinates, rast, deformed_vertices_clip_space, idx) if with_antialiasing else uv_coordinates
            gbuffer["uv_coordinates"] = upsample(gbuffer["uv_coordinates"], high_res)
        except:
            pass

        # We store the deformed vertices in clip space, the transformed camera matrix and the barycentric coordinates
        # to antialias texture and mask after computing the color 
        gbuffer["rast"] = rast
        gbuffer["deformed_verts_clip_space"] = deformed_vertices_clip_space
        gbuffer["view_pos_gl"] = Rt[:, :3, 3]

        return gbuffer
    
    def render_canonical_mesh(self, view, meshes, color, with_antialiasing=True):
        """ Rasterizes the canonical mesh: for visualization purpose only

        Args:
            views (List[Views]): 
        """
        resolution = (512, 512)

        canon_verts = meshes.vertices
        idx = meshes.indices.int()
        P, Rt = Renderer.to_gl_camera(view, resolution, n=self.near, f=self.far)
        pos = Renderer.transform_pos(P, canon_verts)
        rast, _ = dr.rasterize(self.glctx, pos, idx, resolution=resolution)
        interpolated_color, _ = dr.interpolate(color[None, ...].type(torch.float32), rast, idx)

        del rast
        return interpolated_color[0]