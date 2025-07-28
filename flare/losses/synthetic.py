import numpy as np
from flare.dataset import dataset_util
import torch
import mediapipe as mp
import cv2

channels_gbuffer = ['mask', 'position', 'normal', "canonical_position"]
lgt = None
import face_alignment

import os
import imageio
elu = torch.nn.ELU()

convert_uint = lambda x: np.clip(np.rint(dataset_util.rgb_to_srgb(x).detach().cpu().numpy() * 255.0), 0, 255).astype(np.uint8)
from skimage.transform import estimate_transform, warp, resize, rescale

face_alignment = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, flip_input=False, 
                                                    device='cuda' if torch.cuda.is_available() else 'cpu')

dset_scale = 1.25                                                    
HALF_PI = torch.pi / 2
def synthetic_loss(views_subset, neural_blendshapes, renderer, shader, mediapipe, ict_facekit, canonical_mesh, batch_size, deformed_vertices_key, lgt, device, save_debug=False):
    # sample random feature
    with torch.no_grad():
        random_facs = torch.zeros(batch_size, 53, device=device)
        for b in range(batch_size):
            weights = torch.tensor([1/(i*2) for i in range(1, 53)])
            random_integer = torch.multinomial(weights, 1).item() + 1
            random_indices = torch.randint(0, 53, (random_integer,))
            if torch.rand(1) > 0.5:
                random_indices = torch.cat([random_indices, torch.tensor([10])])
            if torch.rand(1) > 0.5:
                random_indices = torch.cat([random_indices, torch.tensor([11])])
            random_indices = random_indices.unique()
            # sample 0 to 1 for each indices
            random_facs[b, random_indices] = torch.rand_like(random_facs[b, random_indices])

        random_rotation = torch.randn(batch_size, 3, device=device)
        random_rotation[:, 2] *= 0.5
        # Normalize so that sum of absolute values equals half pi
        abs_sum = torch.sum(torch.abs(random_rotation), dim=1, keepdim=True)
        degrees = torch.rand(2, device=device) 
        random_rotation = random_rotation / abs_sum * HALF_PI * degrees[:, None] * 0.9

        # x range in [-0.1, 0.1], y and z range in [-0.05, 0.05]
        random_translation = torch.rand(batch_size, 3, device=device)
        random_translation[..., 0] = random_translation[..., 0] * 0.1 - 0.05
        random_translation[..., 1] = random_translation[..., 1] * 0.05 - 0.025
        random_translation[..., 2] = random_translation[..., 2] * 0.05 - 0.025

        # half for the random global translation
        random_global_translation = torch.rand(batch_size, 3, device=device)
        random_global_translation[..., 0] = random_global_translation[..., 0] * 0.1 - 0.05
        random_global_translation[..., 1] = random_global_translation[..., 1] * 0.05 - 0.025
        random_global_translation[..., 2] = random_global_translation[..., 2] * 0.05 - 0.025



                # get the scale from neural_blendshapes.encoder


        scale = torch.ones_like(random_translation[:, -1:]) * (elu(neural_blendshapes.encoder.scale) + 1)

        random_features = torch.cat([random_facs[..., :53], random_rotation, random_translation,  scale, random_global_translation], dim=-1)

        return_dict = neural_blendshapes(image_input=False, features=random_features)

        deformed_vertices = return_dict[deformed_vertices_key+'_posed']

        mesh = canonical_mesh

        deformed_vertices = return_dict[deformed_vertices_key+'_posed']

        d_normals = mesh.fetch_all_normals(deformed_vertices, mesh)

        gbuffers = renderer.render_batch(views_subset['camera'][:batch_size], deformed_vertices.contiguous(), d_normals, 
                                channels=channels_gbuffer, with_antialiasing=True, 
                            canonical_v=mesh.vertices, canonical_idx=mesh.indices, canonical_uv=ict_facekit.uv_neutral_mesh,
                            mesh=mesh) 
        
        for k in views_subset:
            views_subset[k] = views_subset[k][:batch_size]

        del random_facs, random_rotation, random_translation, scale, random_global_translation

        pred_color_masked, _, _ = shader.shade(gbuffers, views_subset, mesh, True, lgt)
        
        uint_imgs = convert_uint(pred_color_masked) # B, C, H, W

        # Save the images, for debugging. 
        # Directory is : tmp_synthetic in current directory
        # use imageio
        
        if save_debug:
            os.makedirs('tmp_synthetic', exist_ok=True)
            for i in range(batch_size):
                imageio.imwrite(f'tmp_synthetic/img_{i}.png', uint_imgs[i])


        # rgb_float_imgs = dataset_util.srgb_to_rgb(torch.from_numpy(uint_imgs)/255.).permute(0, 3, 1, 2).to(device)

        # uint_imgs = rgb_float_imgs.permute(0, 2, 3, 1).cpu().numpy()
        
        del gbuffers, pred_color_masked, mesh

    losses = []
    for i in range(batch_size):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(uint_imgs[i], cv2.COLOR_RGB2BGR))
        face_landmarker_result = mediapipe.detect(mp_image)
        mp_landmark, mp_blendshape, mp_transform_matrix = dataset_util.parse_mediapipe_output(face_landmarker_result)

        if mp_landmark is None:
            losses.append(torch.tensor(0.0, device=device))
            continue

        landmarks, scores, _ = face_alignment.get_landmarks_from_image(uint_imgs[i], return_bboxes=True, return_landmark_score=True)
        if landmarks is None or len(landmarks) == 0:
            losses.append(torch.tensor(0.0, device=device))
            continue

        landmark = torch.tensor(landmarks[0], dtype=torch.float32)

        img_deca = np.array(uint_imgs[i])


        kpt = landmark.cpu().numpy()[:, :2]
        left = np.min(kpt[:,0]); right = np.max(kpt[:,0]); 
        top = np.min(kpt[:,1]); bottom = np.max(kpt[:,1])
        bbox = [left,top, right, bottom]
        
        old_size, center = bbox2point(left, right, top, bottom, type='kpt68')
        size = int(old_size*dset_scale)
        src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
        DST_PTS = np.array([[0,0], [0,224 - 1], [224 - 1, 0]])

        tform = estimate_transform('similarity', src_pts, DST_PTS)
        img_deca = img_deca / 255.
        img_deca = warp(img_deca, tform.inverse, output_shape=(224, 224))

        img_deca = torch.tensor(img_deca, dtype=torch.float32).permute(2,0,1) # H, W, C -> C, H, W

        landmark = landmark / uint_imgs.shape[2]
        if landmark.size(-1) == 3:
            landmark[..., 2] = landmark[..., 2] - torch.mean(landmark[..., 2], dim=0, keepdim=True)
        score = torch.tensor(scores[0], dtype=torch.float32)
        # print(landmark.shape, score.shape)
        landmark = torch.cat([landmark, score[:, None]], dim=1).data


        views = {}
        views['mp_landmark'] = mp_landmark[None].to(device)
        views['mp_blendshape'] = mp_blendshape[None].to(device)
        views['mp_transform_matrix'] = mp_transform_matrix[None].to(device)
        views['landmark'] = landmark[None].to(device)
        views['img_deca'] = img_deca[None].to(device)
        views['camera'] = views_subset['camera'][i:i+1]
        encoder_out = neural_blendshapes.encoder(views, synthetic=True)

        loss = torch.mean((encoder_out[:, :63] - random_features[i:i+1]) ** 2)
        losses.append(loss)

    return torch.mean(torch.stack(losses))



def bbox2point(left, right, top, bottom, type='bbox'):
    ''' bbox from detector and landmarks are different
    '''
    if type=='kpt68':
        old_size = (right - left + bottom - top)/2*1.1
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])
    elif type=='bbox':
        old_size = (right - left + bottom - top)/2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0  + old_size*0.12])
    else:
        raise NotImplementedError
    return old_size, center