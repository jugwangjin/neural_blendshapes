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
seg_map_to_vertex_labels[0] = [0]
# seg_map_to_vertex_labels[1] = [3,7]
# seg_map_to_vertex_labels[2] = [4,8]
seg_map_to_vertex_labels[1] = [1, -1]

def segmentation_loss(views_subset, gbuffers, parts_indices, canonical_vertices, img_size=512):
    # full face gt 2d points -> sample where gt_seg is 0
    bsize = views_subset['skin_mask'].shape[0]

    gt_segs = views_subset['skin_mask'] # Shape of B, H, W, 6
    # rendered_segs = gbuffers['vertex_labels'] # Shape of B, H, W, 9
    # print(gt_segs.shape, rendered_segs.shape)
    canonical_positions = gbuffers['canonical_position'] * views_subset["skin_mask"][..., :2].sum(dim=-1, keepdim=True)

    vertices_on_clip_space = gbuffers['deformed_verts_clip_space'].clone()
    vertices_on_clip_space = vertices_on_clip_space[..., :3] / torch.clamp(vertices_on_clip_space[..., 3:], min=1e-8)

    seman_losses = torch.tensor(0.0, device=gt_segs.device)
    stat_losses = torch.tensor(0.0, device=gt_segs.device)
    for b, gt_seg in enumerate(gt_segs):
        # sample 2d pixel positions where gt_segs==i
        seman_loss = torch.tensor(0.0, device=gt_seg.device)
        stat_loss = torch.tensor(0.0, device=gt_seg.device)

        with torch.no_grad():
            # canonical_positions = gbuffers['canonical_position'][b] * views_subset["skin_mask"][b, ..., :5].sum(dim=-1, keepdim=True)
            nonzero_rows = torch.abs(canonical_positions[b].view(-1, 3)).sum(dim=1) > 0
            valid_canonical_positions = canonical_positions[b].view(-1, 3)[nonzero_rows]
            valid_canonical_positions = torch.unique(valid_canonical_positions, dim=0)

            _, valid_idx, _ = knn_points(valid_canonical_positions[None], canonical_vertices[None], K=1, return_nn=False)
            valid_idx = torch.unique(valid_idx)
            # print(valid_idx.shape)

        for i in range(len(seg_map_to_vertex_labels)):
            gt_seg_pixels = (torch.nonzero(gt_seg[:,:,i]) / (img_size - 1)) * 2 - 1
            # print(rendered_seg.shape)
            part_index = []
            
            for n in seg_map_to_vertex_labels[i]:
                if n == -1:
                    part_index += list(range(6706, 9409))
                    continue
                else:
                    part_index += parts_indices[n]
            #     print(n, len(parts_indices[n]))
            # print(len(part_index))
            # part index should be intersection of valid_idx and part index
            part_index = list(set(part_index) & set(valid_idx.cpu().numpy().tolist()))

            if len(gt_seg_pixels) == 0 or len(part_index) == 0:
                continue

            seg_pixels_on_clip_space = vertices_on_clip_space[b, part_index, :2]
            # print(i, gt_seg_pixels.shape, rendered_seg_pixels.shape)
            # calculate chamfer distance
            cd_loss, _ = chamfer_distance(gt_seg_pixels[None], seg_pixels_on_clip_space[None])
            seman_loss += cd_loss
            # print(batch_loss)
            # additionally, get mean and std of them and add to loss
            gt_seg_pixels_mean = gt_seg_pixels.mean(dim=0)
            rendered_seg_pixels_mean = seg_pixels_on_clip_space.mean(dim=0)

            stat_loss += (gt_seg_pixels_mean - rendered_seg_pixels_mean).pow(2).mean()

        seman_losses += (seman_loss / float(len(seg_map_to_vertex_labels)))
        stat_losses += (stat_loss / float(len(seg_map_to_vertex_labels)))

    return seman_losses/bsize, stat_losses/bsize
 

    