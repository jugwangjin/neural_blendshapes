# Code: https://github.com/fraunhoferhhi/neural-deferred-shading/tree/main
# Modified/Adapted by: Shrisha Bharadwaj

import torch
import xatlas

######################################################################################
# Utils
######################################################################################
def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x*y, -1, keepdim=True)

def reflect(x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    return 2*dot(x, n)*n - x

def length(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x,x), min=eps)) # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN

def safe_normalize(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return x / length(x, eps)

######################################################################################
# Mesh Class
######################################################################################
class Mesh:
    """ Triangle mesh defined by an indexed vertex buffer.

    Args:
        vertices (tensor): Vertex buffer (Vx3)
        indices (tensor): Index buffer (Fx3)
        device (torch.device): Device where the mesh buffers are stored
    """

    def __init__(self, vertices, indices, ict_facekit=None, vertex_labels=None, face_labels=None, device='cpu'):
        self.device = device

        self.vertices = vertices.to(device, dtype=torch.float32) if torch.is_tensor(vertices) else torch.tensor(vertices, dtype=torch.float32, device=device)
        self.indices = indices.to(device, dtype=torch.int64) if torch.is_tensor(indices) else torch.tensor(indices, dtype=torch.int64, device=device) if indices is not None else None

        if self.indices is not None:
            self.compute_normals()

        self._edges = None
        self._connected_faces = None
        self._laplacian = None
        self._faces_idx = None
        self.vmapping = None
        self.ict_facekit = ict_facekit
        if ict_facekit is None:
            self._uv_coords = None
            self._uv_idx = None

            self.face_labels = None
            self.vertex_labels = None
        else:
            self._uv_coords = ict_facekit.uvs.to(device, dtype=torch.float32) if torch.is_tensor(ict_facekit.uvs) else torch.tensor(ict_facekit.uvs, dtype=torch.float32, device=device)
            self._uv_idx = ict_facekit.uv_faces.to(device, dtype=torch.int64) if torch.is_tensor(ict_facekit.uv_faces) else torch.tensor(ict_facekit.uv_faces, dtype=torch.int64, device=device)

            if face_labels is None:
                self.face_labels = torch.zeros(indices.shape[0], 5, device=device, dtype=torch.float32) # 0: face, 1: mouth, 2: eyeball
                

                if vertex_labels is None:
                    vertex_labels = torch.zeros(self.vertices.shape[0], 5, device=device, dtype=torch.float32)  # Create labels for each vertex
                    head = torch.zeros(5, device=device, dtype=torch.float32)
                    head[0] = 1
                    left_eye = torch.zeros(5, device=device, dtype=torch.float32)
                    left_eye[1] = 1
                    right_eye = torch.zeros(5, device=device, dtype=torch.float32)
                    right_eye[2] = 1
                    mouth = torch.zeros(5, device=device, dtype=torch.float32)
                    mouth[3] = 1
                    face = torch.zeros(5, device=device, dtype=torch.float32)
                    face[4] = 1
                    # Assign labels to vertices based on the given segmentation
                    vertex_labels[0:11248] += head  # Head
                    vertex_labels[11248:13294] += mouth
                    vertex_labels[13294:13678] += left_eye
                    vertex_labels[13678:14062] += right_eye
                    vertex_labels[14062:21451] += mouth
                    vertex_labels[21451:23021] += left_eye
                    vertex_labels[23021:24591] += right_eye
                    vertex_labels[0:6706] += face
                    # vertex_labels[0:11248] = 0  # Head
                    # vertex_labels[11248:13294] = 1  # Mouth
                    # vertex_labels[13294:13678] = 2  # left Eyes
                    # vertex_labels[13678:14062] = 3  # right Eyes
                    # vertex_labels[14062:21451] = 1  # Mouth
                    # vertex_labels[21451:23021] = 2  # left Eyes
                    # vertex_labels[23021:24591] = 3  # right Eyes
                
                for i in range(indices.shape[0]):
                    vertex_indices = self.indices[i]
                    for j in vertex_indices:
                        self.face_labels[i] += vertex_labels[j]
                    self.face_labels[i] /= 3

                    # # Get the labels of the three vertices
                    # vertex_label_set = vertex_labels[vertex_indices]

                    # # Assign the most frequent label to the face
                    # # In case of a tie, PyTorch's `mode()` function returns the smallest value among the tied elements
                    # face_label = torch.mode(vertex_label_set).values

                    # self.face_labels[i] = face_label
                self.vertex_labels = vertex_labels
            else:
                self.face_labels = face_labels
                self.vertex_labels = vertex_labels


    def to(self, device):
        mesh = Mesh(self.vertices.to(device), self.indices.to(device), device=device)
        mesh._edges = self._edges.to(device) if self._edges is not None else None
        mesh._connected_faces = self._connected_faces.to(device) if self._connected_faces is not None else None
        mesh._laplacian = self._laplacian.to(device) if self._laplacian is not None else None
        mesh._faces_idx = self._faces_idx.to(device) if self._faces_idx is not None else None
        mesh._uv_coords = self._uv_coords.to(device) if self._uv_coords is not None else None
        mesh._uv_idx = self._uv_idx.to(device) if self._uv_idx is not None else None
        mesh.vmapping = self.vmapping.to(device) if self.vmapping is not None else None
        return mesh

    def detach(self):
        mesh = Mesh(self.vertices.detach(), self.indices.detach(), device=self.device)
        mesh.face_normals = self.face_normals.detach()
        mesh.vertex_normals = self.vertex_normals.detach()
        mesh._edges = self._edges.detach() if self._edges is not None else None
        mesh._connected_faces = self._connected_faces.detach() if self._connected_faces is not None else None
        mesh._laplacian = self._laplacian.detach() if self._laplacian is not None else None
        mesh._faces_idx = self._faces_idx.detach() if self._faces_idx is not None else None
        mesh._uv_coords = self._uv_coords.detach() if self._uv_coords is not None else None
        mesh._uv_idx = self._uv_idx.detach() if self._uv_idx is not None else None
        mesh.vmapping = self.vmapping.detach() if self.vmapping is not None else None
        return mesh

    def with_vertices(self, vertices):
        """ Create a mesh with the same connectivity but with different vertex positions. 
        After each optimization step, the vertices are updated.

        Args:
            vertices (tensor): New vertex positions (Vx3)
        """
        assert len(vertices) == len(self.vertices)
        mesh_new = Mesh(vertices, self.indices, ict_facekit = self.ict_facekit, vertex_labels=self.vertex_labels, face_labels = self.face_labels, device=self.device)
        mesh_new._edges = self._edges
        mesh_new._connected_faces = self._connected_faces
        mesh_new._laplacian = self._laplacian
        mesh_new._faces_idx = self._faces_idx
        mesh_new._uv_coords = self._uv_coords
        mesh_new._uv_idx = self._uv_idx
        mesh_new.vmapping = self.vmapping
        return mesh_new 


    def get_vertices_face_normals(self, vertices):
        """ Calculates vertex and face normals and returns them.
        Args:
            vertices (tensor): New vertex positions (Vx3)
        """
        a = vertices[self.indices][:, 0, :]
        b = vertices[self.indices][:, 1, :]
        c = vertices[self.indices][:, 2, :]
        face_normals = torch.nn.functional.normalize(torch.linalg.cross(b - a, c - a), p=2, dim=-1) 

        # Compute the vertex normals
        vertex_normals = torch.zeros_like(vertices)
        vertex_normals = vertex_normals.index_add(0, self.indices[:, 0], face_normals)
        vertex_normals = vertex_normals.index_add(0, self.indices[:, 1], face_normals)
        vertex_normals = vertex_normals.index_add(0, self.indices[:, 2], face_normals)
        vertex_normals = torch.nn.functional.normalize(vertex_normals, p=2, dim=-1) 
    
        return vertex_normals.contiguous(), face_normals.contiguous()

    def fetch_all_normals(self, deformed_vertices, mesh):
        """ All normals are returned: Vertex, face, tangent space normals along with indices of faces. 
        Args:
            deformed vertices (tensor): New vertex positions (Vx3)
            mesh (Mesh class with these new updated vertices)
        """
        d_normals = {"vertex_normals":[], "face_normals":[], "tangent_normals":[]}
        for d_vert in deformed_vertices:
            vertex_normals, face_normals = mesh.get_vertices_face_normals(d_vert)
            tangents = mesh.compute_tangents(d_vert, vertex_normals)
            d_normals["vertex_normals"].append(vertex_normals.unsqueeze(0))
            d_normals["face_normals"].append(face_normals.unsqueeze(0))
            d_normals["tangent_normals"].append(tangents.unsqueeze(0))

        d_normals["vertex_normals"] = torch.cat(d_normals["vertex_normals"], axis=0)
        d_normals["face_normals"] = torch.cat(d_normals["face_normals"], axis=0)
        d_normals["tangent_normals"] = torch.cat(d_normals["tangent_normals"], axis=0)
        d_normals["face_idx"] = self._faces_idx
        
        return d_normals
    
    ######################################################################################
    # Basic Mesh Operations
    ######################################################################################

    @property
    def edges(self):
        if self._edges is None:
            from flare.utils.geometry import find_edges
            self._edges = find_edges(self.indices)
        return self._edges

    @property
    def connected_faces(self):
        if self._connected_faces is None:
            from flare.utils.geometry import find_connected_faces
            self._connected_faces = find_connected_faces(self.indices)
        return self._connected_faces

    @property
    def laplacian(self):
        if self._laplacian is None:
            from flare.utils.geometry import compute_laplacian_uniform
            self._laplacian = compute_laplacian_uniform(self)
        return self._laplacian

    @property
    def face_idx(self):
        return (torch.arange(0, self.indices.shape[0], dtype=torch.int64, device=self.device)[:, None]).repeat(1, 3)
    
    def compute_normals(self):
        # Compute the face normals
        a = self.vertices[self.indices][:, 0, :]
        b = self.vertices[self.indices][:, 1, :]
        c = self.vertices[self.indices][:, 2, :]
        self.face_normals = torch.nn.functional.normalize(torch.linalg.cross(b - a, c - a), p=2, dim=-1) 

        # Compute the vertex normals
        vertex_normals = torch.zeros_like(self.vertices)
        vertex_normals = vertex_normals.index_add(0, self.indices[:, 0], self.face_normals)
        vertex_normals = vertex_normals.index_add(0, self.indices[:, 1], self.face_normals)
        vertex_normals = vertex_normals.index_add(0, self.indices[:, 2], self.face_normals)
        self.vertex_normals = torch.nn.functional.normalize(vertex_normals, p=2, dim=-1)    

    @torch.no_grad()
    def xatlas_uvmap(self):
        import numpy as np
        # Create uvs with xatlas
        v_pos = self.vertices.detach().cpu().numpy()
        t_pos_idx = self.indices.detach().cpu().numpy()
        vmapping, indices, uvs = xatlas.parametrize(v_pos, t_pos_idx)

        # Convert to tensors
        indices_int64 = indices.astype(np.uint64, casting='same_kind').view(np.int64)
        vmapping_int64 = vmapping.astype(np.uint64, casting='same_kind').view(np.int64)
        vmapping = torch.tensor(vmapping_int64, dtype=torch.int64, device=self.device)

        uvs = torch.tensor(uvs, dtype=torch.float32, device=self.device)
        faces = torch.tensor(indices_int64, dtype=torch.int64, device=self.device)

        self._uv_coords = uvs
        self._uv_idx = faces
        self.vmapping = vmapping

        # self.vertices = self.vertices[vmapping]
        # self.indices = faces
        print(self._uv_coords.shape, self.vertices.shape)

    def compute_connectivity(self):
        # self.xatlas_uvmap()
        if self._uv_coords is None:
            self.xatlas_uvmap()

        self._faces_idx = self.face_idx
        self._edges = self.edges
        self._connected_faces = self.connected_faces
        self._laplacian = self.laplacian

    ######################################################################################
    # Compute tangent space from texture map coordinates
    # Follows http://www.mikktspace.com/ conventions
    # Taken from:https://github.com/NVlabs/nvdiffrec
    ######################################################################################
    
    def compute_tangents(self, vertices, vertex_normals):        
        vn_idx = [None] * 3
        pos = [None] * 3
        tex = [None] * 3
        for i in range(0,3):
            # NOTE: VERIFY BY GIVING INDICES TO VERTICES ONCE
            pos[i] = vertices[self.indices[:, i]]
            tex[i] = self._uv_coords[self._uv_idx[:, i]]
            vn_idx[i] = self.indices[:, i]

        tangents = torch.zeros_like(vertex_normals)
        tansum   = torch.zeros_like(vertex_normals)

        # Compute tangent space for each triangle
        uve1 = tex[1] - tex[0]
        uve2 = tex[2] - tex[0]
        pe1  = pos[1] - pos[0]
        pe2  = pos[2] - pos[0]
        
        nom   = (pe1 * uve2[..., 1:2] - pe2 * uve1[..., 1:2])
        denom = (uve1[..., 0:1] * uve2[..., 1:2] - uve1[..., 1:2] * uve2[..., 0:1])
        
        # Avoid division by zero for degenerated texture coordinates
        tang = nom / torch.where(denom > 0.0, torch.clamp(denom, min=1e-6), torch.clamp(denom, max=-1e-6))

        # Update all 3 vertices
        for i in range(0,3):
            idx = vn_idx[i][:, None].repeat(1,3)
            tangents.scatter_add_(0, idx, tang)                # tangents[n_i] = tangents[n_i] + tang
            tansum.scatter_add_(0, idx, torch.ones_like(tang)) # tansum[n_i] = tansum[n_i] + 1
        tangents = tangents / tansum

        # Normalize and make sure tangent is perpendicular to normal
        tangents = safe_normalize(tangents)
        tangents = safe_normalize(tangents - dot(tangents, vertex_normals) * vertex_normals)

        if torch.is_anomaly_enabled():
            assert torch.all(torch.isfinite(tangents))

        return tangents