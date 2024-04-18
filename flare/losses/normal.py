import torch


def normal_loss(gbuffers, views_subset, gbuffer_mask, device):
    with torch.no_grad():
        gt_normal = views_subset["normal"] # B, H, W, 3
        laplacian_kernel = torch.tensor([[1, 1, 1], [1, -8, 1], [1, 1, 1]], device=device, dtype=torch.float32).view(1,1,3,3).repeat(3,3,1,1)
        gt_normal = gt_normal.permute(0, 3, 1, 2) # shape of B, 3, H, W
        gt_normal_laplacian = torch.nn.functional.conv2d(gt_normal, laplacian_kernel, padding=1)
        mask = ((torch.sum(views_subset["skin_mask"][..., :3], dim=-1) * views_subset["skin_mask"][..., -1]) > 0).float() # shape of B H W
        # print(gbuffer_mask.shape)
        
        mask = mask[:, None] * gbuffer_mask.permute(0,3,1,2)
    
        camera = views_subset["camera"] # list of cameras. 
        camera = torch.stack([c.R.T for c in camera], dim=0).to(device)
        R = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]], device=device, dtype=torch.float32)

    estimated_normal = gbuffers["normal"] # shape of B, H, W, 3
    
    estimated_normal = 0.5 * (torch.einsum('bhwc, cj->bhwj', torch.einsum('bhwc, bcj->bhwj', estimated_normal, camera), R.T) + 1)
    estimated_normal = estimated_normal.permute(0, 3, 1, 2) # shape of B, 3, H, W
    estimated_normal_laplacian = torch.nn.functional.conv2d(estimated_normal, laplacian_kernel, padding=1)

    normal_loss = torch.mean(torch.abs(gt_normal - estimated_normal) * mask)
    normal_laplacian_loss = torch.mean(torch.abs(gt_normal_laplacian - estimated_normal_laplacian) * mask)

    # here, gt_normal and estimated_normal are of shape B 3 H W and have range of [0, 1]. save them to debug directory
    # print(gt_normal.shape, estimated_normal.shape)
    # print(gt_normal[0, 0], estimated_normal[0, 0])
    # import imageio
    # import numpy as np
    
    # for i in range(estimated_normal.shape[0]):
    #     imageio.imwrite(f"debug/normal_gt_{i}.png", np.uint8(gt_normal[i].permute(1,2,0).cpu().data.numpy() * 255))
    #     imageio.imwrite(f"debug/normal_estimated_{i}.png", np.uint8(estimated_normal[i].permute(1,2,0).cpu().data.numpy() * 255))
    #     imageio.imwrite(f"debug/input_image{i}.png", np.uint8(views_subset["img"][i].cpu().data.numpy() * 255))
    #     imageio.imwrite(f"debug/mask{i}.png", np.uint8(mask[i].permute(1,2,0).repeat(1,1,3).cpu().data.numpy() * 255))
    # # exit()
    return normal_loss, normal_laplacian_loss 
