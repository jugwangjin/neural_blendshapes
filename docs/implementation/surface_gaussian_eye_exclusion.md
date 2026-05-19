# Surface vs eye Gaussian layout

## Policy

| Region | Surface layout (`k_face` / `k_eye_socket` / …) | Eye module (UV) |
|--------|-----------------------------------------------|-----------------|
| Skin, head/neck, mouth socket, **eye socket**, gums | yes | no |
| Eyeball verts (parts #7–#8) | no | yes (`M_Sclera*` + `M_Eyeball*` charts) |
| Teeth | no | no |
| `M_Sclera*`, `M_Iris*`, `M_Eyeball*`, lashes/occlusion | excluded from **surface** only | eye UV sampling |

Eye-socket is sampled like mouth-socket on the mesh (`k_eye_socket`). Only the **eyeball** uses texture-space Gaussians.

## `surface_sample_vertex_indices` (npy bake)

Includes `eye_socket_left` + `eye_socket_right`. Older npy files: `surface_allowed_vertices()` adds socket ids at runtime.

## Eye pose

`EyeTextureGaussians` must pass **posed** `verts` into `surface_points_from_uvh(..., mesh_verts=verts)`. UV→face/bary is cached on neutral topology; 3D positions follow deformer output each frame.

## Eye UV sampling (front hemisphere)

- Triangle set: eyeball part #7/#8, no `M_Iris*`; `hemisphere_only=True` keeps tris with any vertex in the forward half-space (`dot(v-center, forward) >= min_front_dot`, default `0`).
- `mode="hemisphere"`: **area-weighted bary** on that patch (covers the full front half of the sphere mesh, not just the small `M_Sclera` UV disk).
- `mode="hemisphere_snap"`: legacy uniform 3D dirs + nearest-tri snap (clusters if chart is small).
- `hemisphere_only=False`: whole eyeball (front + back).
