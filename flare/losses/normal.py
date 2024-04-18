import torch


def normal_loss(gbuffers, views_subset, gbuffer_mask, device):
    print(views_subset.keys())
    print(gbuffers.keys())
    exit()
    with torch.no_grad():
        gt_normal = views_subset["normal"] # B, H, W, 3
        laplacian_kernel = torch.tensor([[1, 1, 1], [1, -8, 1], [1, 1, 1]], device=device, dtype=torch.float32).view(1,1,3,3).repeat(3,3,1,1)
        gt_normal = gt_normal.permute(0, 3, 1, 2) # shape of B, 3, H, W
        gt_normal_laplacian = torch.nn.functional.conv2d(gt_normal, laplacian_kernel, padding=1)
        mask = ((torch.sum(views_subset["skin_mask"][..., :3], dim=-1) * views_subset["skin_mask"][..., -1]) > 0).float() # shape of B H W
        # print(gbuffer_mask.shape)
        
        mask = mask[:, None] * gbuffer_mask.permute(0,3,1,2)
        num_valid_pixel = torch.sum(mask)
        if num_valid_pixel < 1:
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
    
        camera = views_subset["camera"] # list of cameras. 
        camera = torch.stack([c.R.T for c in camera], dim=0).to(device)
        R = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]], device=device, dtype=torch.float32)

    estimated_normal = gbuffers["normal"] # shape of B, H, W, 3
    
    estimated_normal = 0.5 * (torch.einsum('bhwc, cj->bhwj', torch.einsum('bhwc, bcj->bhwj', estimated_normal, camera), R.T) + 1)
    estimated_normal = estimated_normal.permute(0, 3, 1, 2) # shape of B, 3, H, W
    estimated_normal_laplacian = torch.nn.functional.conv2d(estimated_normal, laplacian_kernel, padding=1)


    normal_loss = torch.sum(torch.abs(gt_normal - estimated_normal) * mask) / (num_valid_pixel + 1e-6)
    normal_laplacian_loss = torch.sum(torch.abs(gt_normal_laplacian - estimated_normal_laplacian) * mask) / (num_valid_pixel + 1e-6)

    return normal_loss, normal_laplacian_loss 
