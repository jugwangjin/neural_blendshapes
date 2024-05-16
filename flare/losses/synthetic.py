import numpy as np
from flare.dataset import dataset_util
import torch
import mediapipe as mp
import cv2

channels_gbuffer = ['mask', 'position', 'normal', "canonical_position"]
lgt = None

convert_uint = lambda x: np.clip(np.rint(dataset_util.rgb_to_srgb(x).detach().cpu().numpy() * 255.0), 0, 255).astype(np.uint8)

HALF_PI = torch.pi / 2
def synthetic_loss(views_subset, neural_blendshapes, renderer, shader, mediapipe, ict_facekit, canonical_mesh, batch_size, device):
    # sample random feature
    with torch.no_grad():
        random_facs = torch.zeros(batch_size, neural_blendshapes.encoder.outsize, device=device)
        for b in range(batch_size):
            weights = torch.tensor([1/i for i in range(1, 53)])
            random_integer = torch.multinomial(weights, 1).item() + 1
            random_indices = torch.randint(0, 53, (random_integer,))
            if torch.rand(1) > 0.5:
                random_indices = torch.cat([random_indices, torch.tensor([10])])
            if torch.rand(1) > 0.5:
                random_indices = torch.cat([random_indices, torch.tensor([11])])
            random_indices = random_indices.unique()
            # sample 0 to 1 for each indices
            random_facs[b, random_indices] = torch.rand_like(random_facs[b, random_indices])
        random_pose = torch.rand_like(random_facs) * torch.pi - HALF_PI

        random_features = torch.cat([random_facs[..., :53], random_pose[..., 53:]], dim=-1)
        random_features[..., 58] = 0
    
        return_dict = neural_blendshapes(image_input=False, features=random_features)

        mesh = canonical_mesh.with_vertices(return_dict["full_template_deformation"] + canonical_mesh.vertices)

        deformed_vertices = return_dict["full_deformed_mesh"]

        d_normals = mesh.fetch_all_normals(deformed_vertices, mesh)

        gbuffers = renderer.render_batch(views_subset['camera'], deformed_vertices.contiguous(), d_normals, 
                                channels=channels_gbuffer, with_antialiasing=True, 
                                canonical_v=mesh.vertices, canonical_idx=mesh.indices, canonical_uv=ict_facekit.uv_neutral_mesh) 
        
        pred_color_masked, cbuffers, gbuffer_mask = shader.shade(gbuffers, views_subset, mesh, True, lgt)
        
        uint_imgs = convert_uint(pred_color_masked) # B, C, H, W

        encoder_input = dataset_util.srgb_to_rgb(torch.from_numpy(uint_imgs)/255.).permute(0, 3, 1, 2).to(device)
        

    losses = []
    for i in range(batch_size):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(uint_imgs[i], cv2.COLOR_RGB2BGR))
        face_landmarker_result = mediapipe.detect(mp_image)
        mp_landmark, mp_blendshape, mp_transform_matrix = dataset_util.parse_mediapipe_output(face_landmarker_result)
        if mp_landmark is None:
            losses.append(torch.tensor(0.0, device=device))
            continue
        views = {}
        views['mp_landmark'] = mp_landmark[None].to(device)
        views['mp_blendshape'] = mp_blendshape[None].to(device)
        views['mp_transform_matrix'] = mp_transform_matrix[None].to(device)
        encoder_out = neural_blendshapes.encoder(encoder_input[i:i+1], views)

        loss = torch.mean((encoder_out - random_features[i:i+1]) ** 2)
        losses.append(loss)

    return torch.stack(losses)
