import os
import subprocess
import multiprocessing


def main():

    data_dir = '/Bean/data/gwangjin/2024/nbshapes/flare'
    exps_dir = os.path.join('/Bean/log/gwangjin/2024/nbshapes_comparisons/')
    save_dir = 'figures/metrics'
    # get flare image directories
    flare_images_dir = f'{{}}/images_evaluation/qualitative_results/rgb'
    
    skip = True



    for directory in os.listdir(data_dir):
        print(directory)
        test_set = 'test'
        if directory == 'yufeng':
            test_set = 'MVI_1812'
        elif directory == 'marcel':
            test_set = 'MVI_1802'

        gt_dir = os.path.join(data_dir, directory, directory, test_set)
        images_dir = os.path.join(exps_dir, 'ours_enc_v10', directory, 'images_evaluation', 'qualitative_results', 'rgb')

        name = 'oursv10_' + directory

        command = f'python flare/metrics/metrics.py --gt_dir {gt_dir} --data_dir {images_dir} --save_dir {save_dir} --no_cloth --name {name}'

        print(command)
        os.system(command)

    if not skip:
        for directory in os.listdir(data_dir):
            print(directory)
            test_set = 'test'
            if directory == 'yufeng':
                test_set = 'MVI_1812'
            elif directory == 'marcel':
                test_set = 'MVI_1802'

            gt_dir = os.path.join(data_dir, directory, directory, test_set)
            images_dir = os.path.join(exps_dir, 'ours_enc_v6', directory, 'images_evaluation', 'qualitative_results', 'rgb')

            name = 'oursv6_' + directory

            command = f'python flare/metrics/metrics.py --gt_dir {gt_dir} --data_dir {images_dir} --save_dir {save_dir} --no_cloth --name {name}'



            print(command)

            os.system(command)

            # run the command on multiprocessing 
            #add to subprocesses
            # subprocesses.append(subprocess.Popen(command, shell=True))







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

        # run pointavatars
        for directory in os.listdir(data_dir):
            print(directory)
            test_set = 'test'
            if directory == 'yufeng':
                test_set = 'MVI_1812'
            elif directory == 'marcel':
                test_set = 'MVI_1802'

            gt_dir = os.path.join(data_dir, directory, directory, test_set)
            
            # getting images_dir
            experiment_dir = os.path.join(exps_dir, 'pointavatar', directory, directory, 'pointavatar')
            for subdir in os.listdir(experiment_dir):
                if os.path.isdir(os.path.join(experiment_dir, subdir)):
                    for subsubdir in os.listdir(os.path.join(experiment_dir, subdir, 'eval')):
                        if os.path.isdir(os.path.join(experiment_dir, subdir, 'eval', subsubdir)):
                            for epochs in sorted(os.listdir(os.path.join(experiment_dir, subdir, 'eval', subsubdir))):
                                if epochs.startswith('epoch'):
                                    images_dir = os.path.join(experiment_dir, subdir, 'eval', subsubdir, epochs, 'rgb_erode_dilate')
                                    break   
            
            name = 'pointavatar_' + directory

            command = f'python flare/metrics/metrics.py --gt_dir {gt_dir} --data_dir {images_dir} --save_dir {save_dir} --no_cloth --name {name}'

            print(command)

            os.system(command)


        # run imavatar
        for directory in os.listdir(data_dir):
            print(directory)
            test_set = 'test'
            if directory == 'yufeng':
                test_set = 'MVI_1812'
            elif directory == 'marcel':
                test_set = 'MVI_1802'

            gt_dir = os.path.join(data_dir, directory, directory, test_set)

            # getting images_dir
            experiment_dir = os.path.join(exps_dir, 'imavatar', directory, directory, 'IMavatar')
            for subdir in os.listdir(experiment_dir):
                if os.path.isdir(os.path.join(experiment_dir, subdir)):
                    for subsubdir in os.listdir(os.path.join(experiment_dir, subdir, 'eval')):
                        if os.path.isdir(os.path.join(experiment_dir, subdir, 'eval', subsubdir)):
                            for epochs in sorted(os.listdir(os.path.join(experiment_dir, subdir, 'eval', subsubdir))):
                                if epochs.startswith('epoch'):
                                    images_dir = os.path.join(experiment_dir, subdir, 'eval', subsubdir, epochs, 'rgb')
                                    break

            name = 'imavatar_' + directory

            command = f'python flare/metrics/metrics.py --gt_dir {gt_dir} --data_dir {images_dir} --save_dir {save_dir} --no_cloth --name {name}'

            print(command)

            os.system(command)

        # run gbshapes

        for directory in os.listdir(data_dir):
            print(directory)
            test_set = 'test'
            if directory == 'yufeng':
                test_set = 'MVI_1812'
            elif directory == 'marcel':
                test_set = 'MVI_1802'

            gt_dir = os.path.join(data_dir, directory, directory, test_set)
            name = directory
            # getting images_dir
            if directory == 'yufeng':
                name = 'imavatar_yuf_v2'
            elif directory == 'marcel':
                name = 'imavatar_mar_v2'
            elif directory == 'subject_3':
                name = 'imavatar_sub3_v2'
                
            gt_dir = os.path.join(data_dir, directory, directory, test_set)

            if not os.path.exists(os.path.join(exps_dir, 'gbshapes', name, 'test/split_40000')):
                continue


            images_dir = os.path.join(exps_dir, 'gbshapes', name, 'test/split_40000')
            new_images_dir = os.path.join(exps_dir, 'gbshapes', name, 'test/split_40000_only_pred')
            os.makedirs(new_images_dir, exist_ok=True)

            images = [img for img in os.listdir(images_dir) if img.endswith('.png') and img.startswith('pred') and not img.startswith('pred_mask')]

            images = sorted(images, key=lambda x: int(x.split('.')[0].split('_')[-1]))

            images_pair = []

            for i, image in enumerate(images):
                images_pair.append((os.path.join(images_dir, image), os.path.join(new_images_dir, f'{i:05d}.png')))

            for image_pair in images_pair:
                os.system(f'cp {image_pair[0]} {image_pair[1]}')

            name = 'gbshapes_' + directory

            command = f'python flare/metrics/metrics.py --gt_dir {gt_dir} --data_dir {new_images_dir} --save_dir {save_dir} --no_cloth --is_insta --name {name}'

            print(command)

            os.system(command)

        # run nha
        for directory in os.listdir(data_dir):
            print(directory)

            test_set = 'test'
            if directory == 'yufeng':
                test_set = 'MVI_1812'
            elif directory == 'marcel':
                test_set = 'MVI_1802'

            gt_dir = os.path.join(data_dir, directory, directory, test_set)

            name = directory
            # getting images_dir
            if directory == 'yufeng':
                name = 'imavatar_yuf_v2'
            elif directory == 'marcel':
                name = 'imavatar_mar_v2'
            elif directory == 'subject_3':
                name = 'imavatar_sub3_v2'

            gt_dir = os.path.join(data_dir, directory, directory, test_set)

            if not name.startswith('nha_person'):
                versions = sorted(os.listdir(os.path.join(exps_dir, 'nha', name, 'lightning_logs')), key=lambda x: int(x.split('_')[-1]))

                latest_version = versions[-1]
                if directory == 'obama':
                    latest_version = 'version_2'

                images_dir = os.path.join(exps_dir, 'nha', name, 'lightning_logs', latest_version, 'NovelViewSynthesisResults', 'val', '0_0')
            else:
                
                images_dir = os.path.join(exps_dir, 'nha', name)

                # get images list, sort by name(int)

            images = [img for img in os.listdir(images_dir) if img.endswith('.png')]
            images = sorted(images, key=lambda x: int(x.split('.')[0]))
            
            if directory.startswith('nha_person'):
                # images = [img for img in os.listdir(images_dir) if img.endswith('.png')]

                images = images[-750:]

            images_pair = []

            if directory.startswith('nha_person'):
                new_images_dir = images_dir.replace('nha_person_', 'new_nha_person_')
            
            else:
                new_images_dir = images_dir.replace('0_0', '0_0_new')
            os.makedirs(new_images_dir, exist_ok=True)

            for i, image in enumerate(images):
                images_pair.append((os.path.join(images_dir, image), os.path.join(new_images_dir, f'{i:05d}.png')))

            for image_pair in images_pair:
                os.system(f'cp {image_pair[0]} {image_pair[1]}')

            name = 'nha_' + directory

            command = f'python flare/metrics/metrics.py --gt_dir {gt_dir} --data_dir {new_images_dir} --save_dir {save_dir} --no_cloth --name {name}'

            print(command)

            os.system(command)


        # run insta

        for directory in os.listdir(data_dir):
            print(directory)

            directory_ = directory

            test_set = 'test'
            if directory == 'yufeng':
                test_set = 'MVI_1812'
            elif directory == 'marcel':
                test_set = 'MVI_1802'

            gt_dir = os.path.join(data_dir, directory, directory, test_set)

            # getting images_dir
            if directory == 'yufeng':
                directory = 'imavatar_yuf_v2_dataset'
            elif directory == 'marcel':
                directory = 'imavatar_mar_v2_dataset'
            elif directory == 'subject_3':
                directory = 'imavatar_sub3_v2_dataset'

            elif directory == 'nha_person_0000':
                directory = 'nha_person_0000_dataset'
            elif directory == 'nha_person_0004':
                directory = 'nha_person_0004_dataset'


            images_dir = os.path.join('/Bean/data/gwangjin/2024/nbshapes/insta', directory, 'experiments', 'insta', 'debug', 'overlay')

            name = 'insta_' + directory_

            command = f'python flare/metrics/metrics.py --gt_dir {gt_dir} --data_dir {images_dir} --save_dir {save_dir} --no_cloth --is_insta --name {name}'

            print(command)

            os.system(command)





if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')

        exit(0)
