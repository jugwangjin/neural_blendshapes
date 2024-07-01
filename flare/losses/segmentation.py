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

from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import knn_points

seg_map_to_vertex_labels = {}
seg_map_to_vertex_labels[0] = [0, 1]
# seg_map_to_vertex_labels[1] = [3,7]
# seg_map_to_vertex_labels[2] = [4,8]
# seg_map_to_vertex_labels[1] = [1, -1]

def segmentation_loss(views_subset, gbuffers, parts_indices, canonical_vertices, img_size=512):

    bsize = views_subset['skin_mask'].shape[0]

    gt_segs = views_subset['skin_mask'] # Shape of B, H, W, 6

    canonical_positions = gbuffers['canonical_position'] * gbuffers["mask"].detach().sum(dim=-1, keepdim=True)

    deformed_verts_clip_space_ = gbuffers['deformed_verts_clip_space']
    deformed_verts_clip_space = deformed_verts_clip_space_[..., :3] / deformed_verts_clip_space_[..., 3:]

    seman_losses = torch.tensor(0.0, device=gt_segs.device)
    stat_losses = torch.tensor(0.0, device=gt_segs.device)
    
    for b, gt_seg in enumerate(gt_segs):
        seman_loss = torch.tensor(0.0, device=gt_seg.device)
        stat_loss = torch.tensor(0.0, device=gt_seg.device)

        with torch.no_grad():
            temp = canonical_positions[b].view(-1, 3)
            nonzero_rows = torch.abs(temp).sum(dim=1) > 0
            valid_canonical_positions = temp[nonzero_rows]

            _, valid_idx, _ = knn_points(valid_canonical_positions[None], canonical_vertices[None], K=1, return_nn=False)
            valid_idx = torch.unique(valid_idx)
            
        for i in seg_map_to_vertex_labels.keys():

            gt_seg_pixels = (torch.nonzero(gt_seg[:,:,i]) / (img_size - 1)) * 2 - 1

            part_index = list(range(11248))
            
            if len(gt_seg_pixels) == 0 or len(part_index) == 0:
                continue

            valid_indices = list(set(valid_idx.cpu().numpy().tolist()) & set(part_index))
            seg_pixels_on_clip_space = deformed_verts_clip_space[b, valid_indices, :2]
            
            cd_loss, _ = chamfer_distance(gt_seg_pixels[None], seg_pixels_on_clip_space[None])
            # print(cd_loss.shape)
            seman_loss += cd_loss.mean()

            gt_seg_pixels_mean = gt_seg_pixels.mean(dim=0)
            rendered_seg_pixels_mean = seg_pixels_on_clip_space.mean(dim=0)

            stat_loss += (gt_seg_pixels_mean - rendered_seg_pixels_mean).pow(2).mean()

        seman_losses += (seman_loss / float(len(seg_map_to_vertex_labels)))
        stat_losses += (stat_loss / float(len(seg_map_to_vertex_labels)))

    return seman_losses/bsize, stat_losses/bsize
 

    