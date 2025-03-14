## Code: https://github.com/zhengyuf/IMavatar
# Modified/Adapted by: Shrisha Bharadwaj
import argparse
import json
import math
import os
import os.path as osp

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from skimage.io import imread
# global
from tqdm import tqdm
# import face_alignment
import imageio
perc_loss_net = None
sifid_net = None
from functools import partial

print_flushed = partial(print, flush=True)

def keypoint(fa_2d, pred, gt):
    pred_2dkey = fa_2d.get_landmarks(pred)[0]
    gt_2dkey = fa_2d.get_landmarks(gt)[0]
    key_2derror = np.mean(np.sqrt(np.sum((pred_2dkey - gt_2dkey) ** 2, 1)))
    return key_2derror

def img_mse(pred, gt, mask=None, error_type='mse', return_all=False, use_mask=False):
    """
    MSE and variants
    Input:
        pred        :  bsize x 3 x h x w
        gt          :  bsize x 3 x h x w
        error_type  :  'mse' | 'rmse' | 'mae' | 'L21'
    MSE/RMSE/MAE between predicted and ground-truth images.
    Returns one value per-batch element
    pred, gt: bsize x 3 x h x w
    """
    assert pred.dim() == 4
    bsize = pred.size(0)

    if error_type == 'mae':
        all_errors = (pred-gt).abs()
    elif error_type == 'L21':
        all_errors = torch.norm(pred-gt, dim=1)
    elif error_type == "L1":
        all_errors = torch.norm(pred - gt, dim=1, p=1)
    else:
        all_errors = (pred-gt).square()

    if mask is not None and use_mask:
        assert mask.size(1) == 1

        nc = pred.size(1)
        # nnz = torch.sum(mask.reshape(bsize, -1), 1) * nc
        nnz = torch.sum(torch.ones_like(mask.reshape(bsize, -1)), 1) * nc
        all_errors = mask.expand(-1, nc, -1, -1) * all_errors
        errors = all_errors.reshape(bsize, -1).sum(1) / nnz
    else:
        errors = all_errors.reshape(bsize, -1).mean(1)

    if error_type == 'rmse':
        errors = errors.sqrt()

    if return_all:
        return errors, all_errors
    else:
        return errors

def img_psnr(pred, gt, mask=None, rmse=None):
    # https://github.com/huster-wgm/Pytorch-metrics/blob/master/metrics.py
    if torch.max(pred) > 128:   max_val = 255.
    else:                       max_val = 1.

    if rmse is None:
        rmse = img_mse(pred, gt, mask, error_type='rmse')

    EPS = 1e-8
    return 20 * torch.log10(max_val / (rmse+EPS))

def _gaussian(w_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - w_size//2)**2/float(2*sigma**2)) for x in range(w_size)])
    return gauss/gauss.sum()

def _create_window(w_size, channel=1):
    _1D_window = _gaussian(w_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, w_size, w_size).contiguous()
    return window

def img_ssim(pred, gt, w_size=11, full=False):
    # https://github.com/huster-wgm/Pytorch-metrics/blob/master/metrics.py
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).


    if torch.max(pred) > 128:   max_val = 255
    else:                       max_val = 1

    if torch.min(pred) < -0.5:  min_val = -1
    else:                       min_val = 0

    L = max_val - min_val

    padd = 0
    (_, channel, height, width) = pred.size()
    window = _create_window(w_size, channel=channel).to(pred.device)

    mu1 = F.conv2d(pred, window, padding=padd, groups=channel)
    mu2 = F.conv2d(gt, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(gt * gt, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(pred * gt, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if False:
    # if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret

def perceptual(pred, gt, mask=None, with_grad=False, use_mask=False):
    """
    https://richzhang.github.io/PerceptualSimilarity/
    """

    def _run(pred, gt, mask):
        assert pred.dim() == 4
        assert gt.dim() == 4

        global perc_loss_net
        if perc_loss_net is None:
            import lpips
            perc_loss_net = lpips.LPIPS(net='alex').to(pred.device)
            perc_loss_net.eval()
            # not good:
            # perc_loss_net = lpips.LPIPS(net='vgg').to(pred.device)

        if mask is not None and use_mask:
            pred = pred * mask
            gt = gt * mask

        errors = perc_loss_net(pred, gt, normalize=True)
        return errors

    if with_grad:
        return _run(pred, gt, mask)
    else:
        with torch.no_grad():
            return _run(pred, gt, mask)

def run(output_dir, gt_dir, subfolders, load_npz=False):
    ############
    # testing extreme poses
    ############
    # frames_ = np.loadtxt("out/cluster_poses/extreme_pose_03_yao.txt")

    if load_npz:
        path_result_npz = os.path.join(output_dir, "results.npz")
        results = np.load(path_result_npz)
        mse_l = results['mse_l']
        rmse_l = results['rmse_l']
        mae_l = results['mae_l']
        perceptual_l = results['perceptual_l']
        psnr_l = results['psnr_l']
        ssim_l = results['ssim_l']
        keypoint_l = results['keypoint_l']
    else:
        res = 512
        files = os.listdir(os.path.join(output_dir))
        if 'rgb_erode_dilate' in files:
            pred_file_name = 'rgb_erode_dilate'
        elif 'rgb_not_optimized_filled' in files:
            pred_file_name = 'rgb_not_optimized_filled'
        elif 'rgb_optim_filled' in files:
            pred_file_name = 'rgb_optim_filled'
        elif 'rgb_r_filled' in files:
            pred_file_name = 'rgb_r_filled' #'0_0' #'rgb_optim_filled'
        elif 'rgb_r' in files:
            pred_file_name = 'rgb_r'
        elif 'rgb' in files:
            pred_file_name = 'rgb'
        elif '0_0' in files:
            pred_file_name = '0_0'
        elif 'overlay' in files:
            pred_file_name = 'overlay'
        print(pred_file_name)
        # pred_file_name = 'rgb_r_filled'
        use_mask = True
        only_face_interior = False
        no_cloth_mask = True

        def _load_img(imgpath):
            image = imread(imgpath).astype(np.float32)
            if image.shape[-2] != res:
                image = cv2.resize(image, (res, res))
            image = image / 255.
            if image.ndim >= 3:
                image = image[:, :, :3]
            # 256, 256, 3
            return image

        def _to_tensor(image):
            if image.ndim == 3:
                image = image.transpose(2, 0, 1)
            image = torch.as_tensor(image).unsqueeze(0)
            # 1, 3, 256, 256
            return image

        mse_l = np.zeros(0)
        rmse_l = np.zeros(0)
        mae_l = np.zeros(0)
        perceptual_l = np.zeros(0)
        ssim_l = np.zeros(0)
        psnr_l = np.zeros(0)
        l1_l = np.zeros(0)
        keypoint_l = np.zeros(0)

        # Keep track of where the images come from
        result_subfolders = list()
        result_filenames = list()

        for subfolder_i in range(len(subfolders)):
            subfolder = subfolders[subfolder_i]
            instance_dir = os.path.join(gt_dir, subfolder)

            assert os.path.exists(instance_dir), "Data directory is empty {}".format(instance_dir)
            cam_file = '{0}/flame_params.json'.format(instance_dir)
            with open(cam_file, 'r') as f:
                camera_dict = json.load(f)

            frames = camera_dict['frames']

            # fa_2d = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cuda')

            expressions = {os.path.basename(frame['file_path']): frame["expression"] for frame in camera_dict["frames"]}

            files = [f for f in os.listdir(os.path.join(output_dir, pred_file_name)) if f.endswith('.png')]

            files_nopng = [f for f in files if f.endswith('.png')]

            start_from = min([int(f[:-4]) for f in files_nopng])
            files_nopng = [str(int(f[:-4]) - start_from + 1) for f in files_nopng]
            print("image index start from: ", start_from)
            assert len(set(files_nopng).intersection(set(expressions.keys()))) == len(files_nopng)
            
            # files = files_nopng
            print(files_nopng, len(frames))
            for i in tqdm(range(len(files_nopng))):
                filename = files[i]
                filename_nopad =  str(int(files_nopng[i]) - start_from + 1) + ".png"
                gt_single_pad = f'{(int(files_nopng[i]) - start_from + 1)}.png'
                # for i in tqdm(range(len(frames_))):
                #     gt_single_pad = f'{int(frames_[i])+1:04d}.png'
                #     filename = f'{int(frames_[i]):05d}.png'
                #     # filename = f'{int(frames_[i])+1}.png'
                #     filename_nopad = f'{int(frames_[i])+1}.png'
                
                pred_path = os.path.join(output_dir, pred_file_name, filename)
                pred_for_key = imread(pred_path)
                pred_for_key = pred_for_key[..., :3]
                if pred_for_key.shape[-2] != res:
                    pred_for_key = cv2.resize(pred_for_key, (res, res))

                gt_path = osp.join(os.path.join(gt_dir, subfolder, "image", gt_single_pad))
                mask_path = osp.join(os.path.join(gt_dir, subfolder, "mask", filename_nopad))

                gt_for_key = imread(gt_path)
                gt_for_key = gt_for_key[..., :3]
                if gt_for_key.shape[-2] != res:
                    gt_for_key = cv2.resize(gt_for_key, (res, res))

                pred = _load_img(pred_path)
                gt = _load_img(gt_path)
                mask = _load_img(mask_path)

                # Our prediction has white background, so do the same for GT
                gt_masked = gt * mask + 1.0 * (1 - mask)
                if use_mask:
                    pred_masked = pred * mask + 1.0 * (1 - mask)
                    pred = pred_masked
                gt = gt_masked

                if no_cloth_mask:
                    def load_semantic(path, img_res):
                        img = imageio.imread(path, mode='F')
                        img = cv2.resize(img, (int(img_res), int(img_res)))
                        return img
                    semantic_path = osp.join(os.path.join(gt_dir, subfolder, "semantic", filename_nopad))
                    semantics = load_semantic(semantic_path, img_res=res)
                    mask_cloth = np.logical_or(semantics == 16, semantics == 15)
                    mask[mask_cloth] = 0.
                w, h, d = gt.shape
                gt = gt.reshape(-1, d)
                # gt[np.sum(gt, 1) == 0., :] = 1 # if background is black, change to white
                gt = gt.reshape(w, h, d)
                try:
                    gt_2d_key = ((np.array(frames[int(files[i][:-4]) - start_from]['flame_keypoints'])[None, :, :] + 1.0) * res / 2).astype(int)
                except:
                    print(int(files[i][:-4]) - start_from + 1, len(frames))
                gt_2d_key = gt_2d_key[:, :68, :]
                # key_error = keypoint(fa_2d, pred_for_key, gt_for_key)

                ### insta has some blank frames, check them and skip those frames
                if (pred * mask).sum() == 0:
                    continue

                if only_face_interior:
                    lmks = gt_2d_key[0]  # 68, 2

                    hull = cv2.convexHull(lmks)
                    hull = hull.squeeze().astype(np.int32)

                    mask = np.zeros(pred_for_key.shape, dtype=np.uint8)
                    mask = cv2.fillPoly(mask, pts=[hull], color=(1, 1, 1))

                pred = _to_tensor(pred)
                gt = _to_tensor(gt)
                mask = _to_tensor(mask)
                mask = mask[:, [0], :, :]


                l1, error_mask = img_mse(pred, gt, mask=mask, error_type='l1', use_mask=use_mask, return_all=True)

                if not osp.exists(osp.join(output_dir, "err_l1")):
                    os.mkdir(osp.join(output_dir, "err_l1"))
                cv2.imwrite(osp.join(output_dir, "err_l1", filename), 255 * error_mask[0].permute(1,2,0).cpu().numpy())

                if not osp.exists(osp.join(output_dir, "gt")):
                    os.mkdir(osp.join(output_dir, "gt"))
                cv2.imwrite(osp.join(output_dir, "gt", filename), cv2.cvtColor(255 * gt[0].permute(1,2,0).cpu().numpy(), cv2.COLOR_RGB2BGR))

                mse = img_mse(pred, gt, mask=mask, error_type='mse', use_mask=use_mask)
                rmse = img_mse(pred, gt, mask=mask, error_type='rmse', use_mask=use_mask)
                mae = img_mse(pred, gt, mask=mask, error_type='mae', use_mask=use_mask)
                perc_error = perceptual(pred, gt, mask, use_mask=use_mask)

                assert mask.size(1) == 1
                if use_mask:
                    mask = mask.bool()
                    pred_masked = pred.clone()
                    gt_masked = gt.clone()
                    pred_masked[~mask.expand_as(pred_masked)] = 0
                    gt_masked[~mask.expand_as(gt_masked)] = 0

                    ssim = img_ssim(pred_masked, gt_masked)
                    psnr = img_psnr(pred_masked, gt_masked, rmse=rmse)

                else:
                    ssim = img_ssim(pred, gt)
                    psnr = img_psnr(pred, gt, rmse=rmse)

                if i % 200 == 0:
                    print("{}\t{}\t{}\t{}\t{}".format(np.mean(mae_l), np.mean(perceptual_l), np.mean(ssim_l), np.mean(psnr_l), np.mean(keypoint_l)))
                mse_l = np.append(mse_l, mse)
                rmse_l = np.append(rmse_l, rmse)
                mae_l = np.append(mae_l, mae)
                perceptual_l = np.append(perceptual_l, perc_error)
                ssim_l = np.append(ssim_l, ssim)
                psnr_l = np.append(psnr_l, psnr)
                l1_l = np.append(l1_l, l1)
                # keypoint_l = np.append(keypoint_l, key_error)


                result_subfolders.append(subfolder)
                result_filenames.append(filename_nopad)

        result = {
            "subfolders": result_subfolders,
            "filenames": result_filenames,
            "mse_l": mse_l.copy(),
            "rmse_l": rmse_l.copy(),
            "mae_l": mae_l.copy(),
            "perceptual_l": perceptual_l.copy(),
            "ssim_l": ssim_l.copy(),
            "psnr_l": psnr_l.copy(),
            "l1_l": l1_l.copy(),
            # "keypoint_l": keypoint_l.copy()
        }
        base_result_name = "results"
        if no_cloth_mask:
            base_result_name = "results_no_cloth"
        path_result_npz = os.path.join(output_dir, "{}_{}.npz".format(base_result_name, pred_file_name))
        path_result_csv = os.path.join(output_dir, "{}_{}.csv".format(base_result_name, pred_file_name))
        np.savez(path_result_npz, **result)
        pd.DataFrame.from_dict(result).to_csv(path_result_csv)
        print("Written result to ", path_result_npz)

    print("{}\t{}\t{}\t{}".format(np.mean(mae_l), np.mean(perceptual_l), np.mean(ssim_l), np.mean(psnr_l)))
    return np.mean(mae_l), np.mean(perceptual_l), np.mean(ssim_l), np.mean(psnr_l)









def run_one_folder(output_dir, gt_dir, save_dir, is_insta, no_cloth):
    ############
    # testing extreme poses
    ############
    # frames_ = np.loadtxt("out/cluster_poses/extreme_pose_03_yao.txt")

    res = 512
    files = os.listdir(os.path.join(output_dir))
    use_mask = True
    only_face_interior = False
    no_cloth_mask = no_cloth

    def _load_img(imgpath):
        image = imread(imgpath).astype(np.float32)
        if image.shape[-2] != res:
            image = cv2.resize(image, (res, res))
        image = image / 255.
        if image.ndim >= 3:
            image = image[:, :, :3]
        # 256, 256, 3
        return image

    def _to_tensor(image):
        if image.ndim == 3:
            image = image.transpose(2, 0, 1)
        image = torch.as_tensor(image).unsqueeze(0)
        # 1, 3, 256, 256
        return image

    mse_l = np.zeros(0)
    rmse_l = np.zeros(0)
    mae_l = np.zeros(0)
    perceptual_l = np.zeros(0)
    ssim_l = np.zeros(0)
    psnr_l = np.zeros(0)
    l1_l = np.zeros(0)
    keypoint_l = np.zeros(0)

    # Keep track of where the images come from
    result_subfolders = list()
    result_filenames = list()

    import tqdm

    filenames = os.listdir(output_dir)

    filenames_padded = {}
    for filename in filenames:
        if not filename.endswith(".png") and not filename.endswith(".jpg"):
            continue
        filenames_padded[filename] = str(int(filename[:-4]) + 1) + ".png"
    
    filenames = [filename for filename in filenames if filename.endswith('.png')]
    # sort filenames by the padded number
    filenames = sorted(filenames, key=lambda x: int(x[:-4]))

    n = 0

    for file in tqdm.tqdm(filenames):
        if not file.endswith(".png") and not file.endswith(".jpg"):
            continue
        n += 1
        if n > 300:
            break
        try:
            pred_image = os.path.join(output_dir, file)
            # inverse of zfill?
            # i need to remove the padding from file
            if is_insta:
                # need to add 1 on file
                file_no_pad = str(int(file[:-4]) + 1) + ".png"
                # print(file, file_no_pad)
                file_no_pad = file_no_pad.lstrip('0')
            else:
                file_no_pad = file.lstrip('0')
            gt_image = os.path.join(gt_dir, 'image', file_no_pad)
            mask_path = os.path.join(gt_dir, 'mask', file_no_pad)


            pred = _load_img(pred_image)
            gt = _load_img(gt_image)
            mask = _load_img(mask_path)

            # Our prediction has white background, so do the same for GT
            gt_masked = gt * mask + 1.0 * (1 - mask)
            if use_mask:
                pred_masked = pred * mask + 1.0 * (1 - mask)
                pred = pred_masked
            gt = gt_masked

            if no_cloth_mask:
                def load_semantic(path, img_res):
                    img = imageio.imread(path, mode='F')
                    img = cv2.resize(img, (int(img_res), int(img_res)))
                    return img
                semantic_path = osp.join(os.path.join(gt_dir, "semantic", file_no_pad))
                semantics = load_semantic(semantic_path, img_res=res)
                mask_cloth = np.logical_or(semantics == 16, semantics == 15)
                mask[mask_cloth] = 0.
            w, h, d = gt.shape
            gt = gt.reshape(-1, d)
            # gt[np.sum(gt, 1) == 0., :] = 1 # if background is black, change to white
            gt = gt.reshape(w, h, d)

            # key_error = keypoint(fa_2d, pred_for_key, gt_for_key)

            ### insta has some blank frames, check them and skip those frames
            if (pred * mask).sum() == 0:
                continue

            pred = _to_tensor(pred)
            gt = _to_tensor(gt)
            mask = _to_tensor(mask)
            mask = mask[:, [0], :, :]


            l1, error_mask = img_mse(pred, gt, mask=mask, error_type='l1', use_mask=use_mask, return_all=True)
        
            if not osp.exists(osp.join(save_dir, "err_l1")):
                os.mkdir(osp.join(save_dir, "err_l1"))
            # cv2.imwrite(osp.join(save_dir, "err_l1", file), 255 * error_mask[0].permute(1,2,0).cpu().numpy())

            if not osp.exists(osp.join(save_dir, "gt")):
                os.mkdir(osp.join(save_dir, "gt"))
            # cv2.imwrite(osp.join(save_dir, "gt", file), cv2.cvtColor(255 * gt[0].permute(1,2,0).cpu().numpy(), cv2.COLOR_RGB2BGR))

            mse = img_mse(pred, gt, mask=mask, error_type='mse', use_mask=use_mask)
            rmse = img_mse(pred, gt, mask=mask, error_type='rmse', use_mask=use_mask)
            mae = img_mse(pred, gt, mask=mask, error_type='mae', use_mask=use_mask)
            perc_error = perceptual(pred, gt, mask, use_mask=use_mask)

            assert mask.size(1) == 1
            if use_mask:
                mask = mask.bool()
                pred_masked = pred.clone()
                gt_masked = gt.clone()
                pred_masked[~mask.expand_as(pred_masked)] = 0
                gt_masked[~mask.expand_as(gt_masked)] = 0

                ssim = img_ssim(pred_masked, gt_masked)
                psnr = img_psnr(pred_masked, gt_masked, rmse=rmse)

            else:
                ssim = img_ssim(pred, gt)
                psnr = img_psnr(pred, gt, rmse=rmse)

            mse_l = np.append(mse_l, mse)
            rmse_l = np.append(rmse_l, rmse)
            mae_l = np.append(mae_l, mae)
            perceptual_l = np.append(perceptual_l, perc_error)
            ssim_l = np.append(ssim_l, ssim)
            psnr_l = np.append(psnr_l, psnr)
            l1_l = np.append(l1_l, l1)
            # keypoint_l = np.append(keypoint_l, key_error)
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue



    result = {
        "filenames": result_filenames,
        "mse_l": mse_l.copy(),
        "rmse_l": rmse_l.copy(),
        "mae_l": mae_l.copy(),
        "perceptual_l": perceptual_l.copy(),
        "ssim_l": ssim_l.copy(),
        "psnr_l": psnr_l.copy(),
        "l1_l": l1_l.copy(),
        # "keypoint_l": keypoint_l.copy()
    }
    base_result_name = "results"
    if no_cloth_mask:
        base_result_name = "results_no_cloth"
    path_result_npz = os.path.join(save_dir, "{}_{}.npy".format(base_result_name, file))
    path_result_csv = os.path.join(save_dir, "{}_{}.csv".format(base_result_name, file))
    # np.save(path_result_npz, **result)
    # pd.DataFrame.from_dict(result).to_csv(path_result_csv)
    print("Written result to ", path_result_npz)

    print("{}\t{}\t{}\t{}".format(np.mean(mae_l), np.mean(perceptual_l), np.mean(ssim_l), np.mean(psnr_l)))

    if args.name != None:
        with open(os.path.join(save_dir, f"{args.name}_metrics.txt"), "w") as f:
            f.write(f"MAE: {np.mean(mae_l)}\nPerceptual: {np.mean(perceptual_l)}\nSSIM: {np.mean(ssim_l)}\nPSNR: {np.mean(psnr_l)}")

    return np.mean(mae_l), np.mean(perceptual_l), np.mean(ssim_l), np.mean(psnr_l)





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data_dir', type=str, help='.')
    parser.add_argument('--gt_dir', type=str, help='.')
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--is_insta', action='store_true')
    parser.add_argument('--no_cloth', action='store_true')
    parser.add_argument('--name', type=str)

    args = parser.parse_args()
    # if os.path.exists(os.path.join(args.save_dir, f"{args.name}_metrics.txt")):
    #     print("Already ran this folder")
    #     exit()

    if not os.path.exists(args.data_dir):
        print("Data directory does not exist")
        exit()

    os.makedirs(args.save_dir, exist_ok=True)
    _,_, _, _ = run_one_folder(args.data_dir, args.gt_dir, args.save_dir, args.is_insta, args.no_cloth)

    