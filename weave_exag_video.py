import os
import glob

def main(args):

    directory = args.parent
    name = args.name
    video_name = args.video_name

    
    # glob: 
# marcel_MVI_1802_2_gt_0.png
# marcel_MVI_1802_2_no_personalization_rgb_0.png
# marcel_MVI_1802_2_rgb_0.png

    #in {parent}/{name} directory, there are images with name: 
    # {name}_{video_name}_{frame_number}_{type}_0.png
    # I would like to glob each type - gt, no_personalization_rgb_0, rgb_0, weave to three videos

    # for each type, I would like to sort by frame number
    # then weave them together
    # save the video in {parent}/{name}_{type}.mp4
# justin_test_2_gt_0.png
    gt_images = glob.glob(f'{directory}/{name}/{name}_{video_name}_*_gt_0.png')
    gt_images = [gt_image for gt_image in gt_images if 'gaze' not in  os.path.basename(gt_image) and 'rgb' not in  os.path.basename(gt_image) and not 'normal' in os.path.basename(gt_image)]
    gt_images.sort(key=lambda x: int(x.split('_')[-3]))

    mouth_exag_images = glob.glob(f'{directory}/{name}/{name}_{video_name}_*_exag_mouth_twice_rgb_0.png')
    mouth_exag_images.sort(key=lambda x: int(x.split('_')[-6]))
    brows_exag_images = glob.glob(f'{directory}/{name}/{name}_{video_name}_*_exag_brows_twice_rgb_0.png')
    brows_exag_images.sort(key=lambda x: int(x.split('_')[-6]))

    resolution = (512, 512)


    # weave them together
    # save the video in {parent}/{name}_{type}.mp4
    import cv2

    writer = cv2.VideoWriter(f'{directory}/{name}_{video_name}_gt.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, resolution)
    for gt_image in gt_images:
        print(gt_image)
        frame_number = gt_image.split('_')[-3]

        gt = cv2.imread(gt_image)
        gt = cv2.resize(gt, resolution)
        writer.write(gt)
    writer.release()

    writer = cv2.VideoWriter(f'{directory}/{name}_{video_name}_exag_mouth_twice.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, resolution)
    for mouth_exag_image in mouth_exag_images:
        print(mouth_exag_image)
        frame_number = mouth_exag_image.split('_')[-3]

        mouth_exag = cv2.imread(mouth_exag_image)
        mouth_exag = cv2.resize(mouth_exag, resolution)
        writer.write(mouth_exag)

    writer.release()

    writer = cv2.VideoWriter(f'{directory}/{name}_{video_name}_exag_brows_twice.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, resolution)
    for brows_exag_image in brows_exag_images:
        print(brows_exag_image)
        frame_number = brows_exag_image.split('_')[-3]

        brows_exag = cv2.imread(brows_exag_image)
        brows_exag = cv2.resize(brows_exag, resolution)
        writer.write(brows_exag)

    writer.release()

        


    return

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--parent', type=str, default='videos')
    parser.add_argument('--name', type=str, default='justin')
    parser.add_argument('--video_name', type=str, default='test')

    args = parser.parse_args()
    main(args)