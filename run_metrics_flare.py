import os
import subprocess
import multiprocessing


def main():

    data_dir = '/Bean/data/gwangjin/2024/nbshapes/flare_2'
    exps_dir = os.path.join('/Bean/log/gwangjin/2025/nbshapes_comparisons/')
    save_dir = 'figures/metrics'
    # get flare image directories
    flare_images_dir = f'{{}}/images_evaluation/qualitative_results/rgb'
    
    skip = True



    # run flares

    for directory in os.listdir(data_dir):
        print(directory)
        test_set = 'test'
        if directory == 'yufeng':
            test_set = 'MVI_1812'
        elif directory == 'marcel':
            test_set = 'MVI_1802'

        gt_dir = os.path.join(data_dir, directory, directory, test_set)
        images_dir = os.path.join(exps_dir, 'flare', directory, 'images_evaluation', 'qualitative_results', 'rgb')

        name = 'flare_' + directory

        command = f'python flare/metrics/metrics.py --gt_dir {gt_dir} --data_dir {images_dir} --save_dir {save_dir} --no_cloth --name {name}'

        print(command)
        os.system(command)



if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')

        exit(0)
