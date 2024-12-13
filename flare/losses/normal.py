import torch

    # tight face area : skin + nose + left eyebwow + right eyebrow + upper lip + lower lip 
    # semantics[:, :, 0] = ((img == 1) + (img == 2) + (img == 3) + (img == 10) + (img == 12) + (img == 13)) >= 1 

    # hair, ears, neck, clothes, 
    # semantics[:, :, 1] = ((img == 17) + (img == 16) + (img == 15) + (img == 14) + (img==7) + (img==8) + (img==9)) >= 1

    # mouth cavity 
    # semantics[:, :, 2] = (img==11) >= 1

    # semantics[:, :, 3] = 1. - np.sum(semantics[:, :, :5], 2) # background

    # mesh 0 face
    # mesh 1 head and neck
    # mesh 2 mouth socket
    # mesh 3 eye socket left
    # mesh 4 eye socket right
    # mesh 5 gums and tongue
    # mesh 6 teeth
    # mesh 7 eyeball left
    # mesh 8 eyeball right


    # need to exclude eyeball, mouth cavity for stability

import torch.nn.functional as F


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


def normal_loss(gbuffers, views_subset, gbuffer_mask, device):
    laplacian_kernel = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], device=device, dtype=torch.float32).view(1,1,3,3).repeat(3,3,1,1) / 4.

    gt_normal = views_subset["normal"] # B, H, W, 3
    gt_normal = gt_normal.permute(0, 3, 1, 2) * 2 - 1 # shape of B, 3, H, W
    gt_normal_laplacian = torch.nn.functional.conv2d(gt_normal, laplacian_kernel, padding=1)
    mask = ((torch.sum(views_subset["skin_mask"][..., 1:2], dim=-1) * views_subset["skin_mask"][..., -1]) > 0).float() # shape of B H W

    mask = mask[:, None] * gbuffer_mask.permute(0,3,1,2)

    num_valid_pixel = torch.sum(mask)
    if num_valid_pixel < 1:
        return torch.tensor(0.0, device=device)

    camera = views_subset["camera"] # list of cameras. 
    camera = torch.stack([c.R.T for c in camera], dim=0).to(device)
    R = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]], device=device, dtype=torch.float32)

    estimated_normal = gbuffers["normal"] # shape of B, H, W, 3
    
    estimated_normal = torch.einsum('bhwc, cj->bhwj', torch.einsum('bhwc, bcj->bhwj', estimated_normal, camera), R.T)
    estimated_normal = estimated_normal.permute(0, 3, 1, 2) # shape of B, 3, H, W
    estimated_normal_laplacian = torch.nn.functional.conv2d(estimated_normal, laplacian_kernel, padding=1)

    normal_laplacian_loss = torch.mean(torch.abs(gt_normal_laplacian - estimated_normal_laplacian) * mask)

    return normal_laplacian_loss 


def eyeball_normal_loss_function(gbuffers, views_subset, gbuffer_mask, device):
    # get the mask. 
    # gt eye seg
    gt_eye_seg = views_subset["skin_mask"][..., 3:4]
    rendered_eye_seg = gbuffers["eyes"]


    position = upsample(gbuffers["position"], high_res)
    normal = gbuffers["normal"] # shape of B, H, W, 3

    with torch.no_grad():
        # camera space normal
        camera = views_subset["camera"] # list of cameras.
        camera = torch.stack([c.R.T for c in camera], dim=0).to(device)
        R = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]], device=device, dtype=torch.float32)

        normal_cam = torch.einsum('bhwc, cj->bhwj', torch.einsum('bhwc, bcj->bhwj', normal, camera), R.T) # shape of B, H, W, 3

        mask_cam   = (normal_cam[..., 2] < -1e-2).float()
        normal = normal * (1 - mask_cam[..., None]) + normal * mask_cam [..., None] * -1

        target_position = position - normal * 1e-1

    mask = ((1 - gt_eye_seg) * rendered_eye_seg).float()

    # # save mask
    # import cv2
    # mask_ = mask.cpu().data.numpy()
    # mask_ = mask_ * 255
    # mask_ = mask_.astype("uint8")
    # for i in range(mask_.shape[0]):
    #     cv2.imwrite(f"debug/mask_{i}.png", mask_[i])
    
    # # also save gt_eye_seg and rendered_eye_seg
    # gt_eye_seg_ = gt_eye_seg.cpu().data.numpy()
    # gt_eye_seg_ = gt_eye_seg_ * 255
    # gt_eye_seg_ = gt_eye_seg_.astype("uint8")
    # for i in range(gt_eye_seg_.shape[0]):
    #     cv2.imwrite(f"debug/gt_eye_seg_{i}.png", gt_eye_seg_[i])

    # rendered_eye_seg_ = rendered_eye_seg.cpu().data.numpy()
    # rendered_eye_seg_ = rendered_eye_seg_ * 255
    # rendered_eye_seg_ = rendered_eye_seg_.astype("uint8")
    # for i in range(rendered_eye_seg_.shape[0]):
    #     cv2.imwrite(f"debug/rendered_eye_seg_{i}.png", rendered_eye_seg_[i])

    # exit()

    eye_loss = torch.mean(torch.pow(position - target_position, 2) * mask) 

    return eye_loss

    # for 4:5, mouth.
    gt_mouth_seg = views_subset["skin_mask"][..., 4:5]
    rendered_mouth_seg = gbuffers["mouth"]

    mask = ((1 - gt_mouth_seg) * rendered_mouth_seg).float()
    with torch.no_grad():
        target_position = position - normal * 1e-2
    mouth_loss = torch.mean(torch.pow(position - target_position, 2) * mask) 
    return eye_loss + mouth_loss


def inverted_normal_loss_function(gbuffers, views_subset, gbuffer_mask, device):
    # get camera space normal
    camera = views_subset["camera"] # list of cameras.

    position = upsample(gbuffers["position"], high_res)
    normal = gbuffers["normal"] # shape of B, H, W, 3

    # camera space normal
    camera = torch.stack([c.R.T for c in camera], dim=0).to(device)
    R = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]], device=device, dtype=torch.float32)

    normal = torch.einsum('bhwc, cj->bhwj', torch.einsum('bhwc, bcj->bhwj', normal, camera), R.T) # shape of B, H, W, 3

    # for where z negative
    # deformed_verts_clip_space should move to its world coordinate normal direction 
    with torch.no_grad():
        mask = (normal[..., 2] < -1e-2).float()

        mask = mask[..., None] * gbuffer_mask
        target_position = position + gbuffers["normal"] 

    loss = torch.mean(torch.pow(position - target_position, 2) * mask) 
    return loss

    # save z component
    import cv2
    
    normal_ = (normal[..., 2] + 1).clamp(0)
    normal_ = normal_.cpu().data.numpy()
    normal_ = normal_ * 127
    normal_ = normal_.astype("uint8")
    for i in range(normal_.shape[0]):
        cv2.imwrite(f"debug/normal_{i}_axis_2.png", normal_[i])

    normal_ = (normal[..., 1] + 1).clamp(0)
    normal_ = normal_.cpu().data.numpy()
    normal_ = normal_ * 127
    normal_ = normal_.astype("uint8")
    for i in range(normal_.shape[0]):
        cv2.imwrite(f"debug/normal_{i}_axis_1.png", normal_[i])

    normal_ = (normal[..., 0] + 1).clamp(0)
    normal_ = normal_.cpu().data.numpy()
    normal_ = normal_ * 127
    normal_ = normal_.astype("uint8")
    for i in range(normal_.shape[0]):
        cv2.imwrite(f"debug/normal_{i}_axis_0.png", normal_[i])
    
    normal_ = normal * 0.5 + 0.5
    normal_ = normal_.cpu().data.numpy()
    normal_ = normal_ * 255
    normal_ = normal_.astype("uint8")
    for i in range(normal_.shape[0]):
        cv2.imwrite(f"debug/normal_{i}_rgb.png", normal_[i])

    normal_ = gbuffers["normal"]
    normal_ = normal_ * 0.5 + 0.5
    normal_ = normal_.cpu().data.numpy()
    normal_ = normal_ * 255
    normal_ = normal_.astype("uint8")
    for i in range(normal_.shape[0]):
        cv2.imwrite(f"debug/normal_{i}_rgb_world.png", normal_[i])

