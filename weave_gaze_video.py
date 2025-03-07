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
# marcel_MVI_1802_0_fixed_gaze_rgb_0
    no_personalization_images = glob.glob(f'{directory}/{name}/{name}_{video_name}_*_fixed_gaze_rgb_0.png')
    no_personalization_images = [no_personalization_image for no_personalization_image in no_personalization_images if 'gt' not in no_personalization_image and 'normal' not in no_personalization_image]
    no_personalization_images.sort(key=lambda x: int(x.split('_')[-5]))

    rgb_images = glob.glob(f'{directory}/{name}/{name}_{video_name}_*_rgb_0.png')
    # print(rgb_images)
    # print(len(rgb_images))
    # remove if it is gt or no_personalization
    rgb_images = [image for image in rgb_images if 'gt' not in os.path.basename(image) and 'gaze' not in os.path.basename(image) and 'normal' not in os.path.basename(image)]
    # print(rgb_images)
    # print(len(rgb_images))
    # marcel_MVI_1802_1_rgb_0.png
    rgb_images.sort(key=lambda x: int(x.split('_')[-3]))
    
    resolution = (512, 512)

    print(len(gt_images), len(no_personalization_images), len(rgb_images))

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

    writer = cv2.VideoWriter(f'{directory}/{name}_{video_name}_no_personalization.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, resolution)
    for no_personalization_image in no_personalization_images:
        print(no_personalization_image)
        frame_number = no_personalization_image.split('_')[-3]

        no_personalization = cv2.imread(no_personalization_image)
        no_personalization = cv2.resize(no_personalization, resolution)
        writer.write(no_personalization)

    writer.release()

    writer = cv2.VideoWriter(f'{directory}/{name}_{video_name}_rgb.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, resolution)
    for rgb_image in rgb_images:
        print(rgb_image)
        frame_number = rgb_image.split('_')[-3]

        rgb = cv2.imread(rgb_image)
        rgb = cv2.resize(rgb, resolution)
        writer.write(rgb)

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