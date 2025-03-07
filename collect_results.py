import os
import sys
import shutil


if __name__=='__main__':
    ratio = 75
    collection_dir = 'figures/comparison'
    os.makedirs(collection_dir, exist_ok=True)
    main_dir = '/Bean/log/gwangjin/2024/nbshapes_comparisons'
    

    ours_images_dir = f'{{}}/images_evaluation/qualitative_results/rgb'

    for exp in os.listdir(os.path.join(main_dir, 'ours_enc_v10')):
        images_dir = os.path.join(main_dir, 'ours_enc_v10', exp, 'images_evaluation/qualitative_results/rgb')        
        # collect every 50th
        try:
            for i, img in enumerate(sorted([img for img in os.listdir(images_dir) if not img.endswith('.db')])):
                if i % ratio == 0:
                    shutil.copy(os.path.join(images_dir, img), os.path.join(collection_dir, f'{exp}_{i:05d}_v10_ours.png'))
        except:
            pass


    # flare
    flare_images_dir = f'{{}}/images_evaluation/qualitative_results/rgb'
    gbshapes_images_dir = f'{{}}/gbshapes/{{}}/test/split_40000'

    for exp in os.listdir(os.path.join(main_dir, 'flare')):
        images_dir = os.path.join(main_dir, 'flare', exp, 'images_evaluation/qualitative_results/rgb')        
        # collect every 50th
        for i, img in enumerate(sorted([img for img in os.listdir(images_dir) if not img.endswith('.db')])):
            if i % ratio == 0:
                shutil.copy(os.path.join(images_dir, img), os.path.join(collection_dir, f'{exp}_{i:05d}_flare.png'))


    for exp in os.listdir(os.path.join(main_dir, 'gbshapes')):
        if exp.startswith('imavatar') and not exp.endswith('v2'):
            continue
        
        if not os.path.exists(os.path.join(main_dir, 'gbshapes', exp, 'test/split_40000')):
            continue

        images_dir = os.path.join(main_dir, 'gbshapes', exp, 'test/split_40000')
        # only select images with pred_, and not with mask
        images_list = [img for img in os.listdir(images_dir) if 'pred_' in img and 'mask' not in img]
        # sort images list, by: name[5:].replace('.png', '') into integer, 
        images_list = sorted([img for img in images_list if not img.endswith('.db')], key=lambda x: int(x[5:].replace('.png', '')))
        # can I get original image name from pred_image name?
        
        if exp == 'imavatar_mar_v2':
            exp = 'marcel'
        elif exp == 'imavatar_yuf_v2':
            exp = 'yufeng'
        elif exp == 'imavatar_sub3_v2':
            exp = 'subject_3'

        for i, img in enumerate(images_list):
            if i % ratio == 0:
                shutil.copy(os.path.join(images_dir, img), os.path.join(collection_dir, f'{exp}_{i:05d}_gbshapes.png'))

    for exp in os.listdir(os.path.join(main_dir, 'imavatar')):
        # \\bean.postech.ac.kr\log\gwangjin\2024\nbshapes_comparisons\imavatar\justin\justin\IMavatar\train\eval\test\epoch_20\rgb
        experiment_dir = os.path.join(main_dir, 'imavatar', exp, exp, 'IMavatar')
        for subdir in os.listdir(experiment_dir):
            if os.path.isdir(os.path.join(experiment_dir, subdir)):
                for subsubdir in os.listdir(os.path.join(experiment_dir, subdir, 'eval')):
                    if os.path.isdir(os.path.join(experiment_dir, subdir, 'eval', subsubdir)):
                        for epochs in sorted(os.listdir(os.path.join(experiment_dir, subdir, 'eval', subsubdir))):
                            if epochs.startswith('epoch'):
                                epoch = epochs.split('_')[-1]
                                imgs = os.listdir(os.path.join(experiment_dir, subdir, 'eval', subsubdir, epochs, 'rgb'))
                                # sort, by: name.replace('.png', '') into integer
                                imgs = sorted([img for img in imgs if not img.endswith('.db')], key=lambda x: int(x.replace('.png', '')))

                                for i, img in enumerate(imgs):
                                    if i % ratio == 0:
                                        # with zero padding 5d for index file name
                                        shutil.copy(os.path.join(experiment_dir, subdir, 'eval', subsubdir, epochs, 'rgb', img), os.path.join(collection_dir, f'{exp}_{i:05d}_imavatar_{epoch}.png'))



    # for exp in os.listdir(os.path.join(main_dir, 'imavatar')):
    #     # \\bean.postech.ac.kr\log\gwangjin\2024\nbshapes_comparisons\imavatar\justin\justin\IMavatar\train\eval\test\epoch_20\rgb
    #     experiment_dir = os.path.join(main_dir, 'imavatar', exp, exp, 'IMavatar')
    #     for subdir in os.listdir(experiment_dir):
    #         if os.path.isdir(os.path.join(experiment_dir, subdir)):
    #             for subsubdir in os.listdir(os.path.join(experiment_dir, subdir, 'eval')):
    #                 if os.path.isdir(os.path.join(experiment_dir, subdir, 'eval', subsubdir)):
    #                     for epochs in sorted(os.listdir(os.path.join(experiment_dir, subdir, 'eval', subsubdir))):
    #                         try:
    #                             if epochs.startswith('epoch') and epochs.endswith('60'):
    #                                 imgs = os.listdir(os.path.join(experiment_dir, subdir, 'eval', subsubdir, epochs, 'rgb'))
    #                                 # sort, by: name.replace('.png', '') into integer
    #                                 imgs = sorted([img for img in imgs if not img.endswith('.db')], key=lambda x: int(x.replace('.png', '')))

    #                                 for i, img in enumerate(imgs):
    #                                     if i % ratio == 0:
    #                                         # with zero padding 5d for index file name
    #                                         shutil.copy(os.path.join(experiment_dir, subdir, 'eval', subsubdir, epochs, 'rgb', img), os.path.join(collection_dir, f'{exp}_{i:05d}_imavatar_60.png'))
    #                         except:
    #                             continue



    for exp in os.listdir(os.path.join(main_dir, 'pointavatar')):
        # \\bean.postech.ac.kr\log\gwangjin\2024\nbshapes_comparisons\imavatar\justin\justin\IMavatar\train\eval\test\epoch_20\rgb
        experiment_dir = os.path.join(main_dir, 'pointavatar', exp, exp, 'pointavatar')
        for subdir in os.listdir(experiment_dir):
            if os.path.isdir(os.path.join(experiment_dir, subdir)):
                for subsubdir in os.listdir(os.path.join(experiment_dir, subdir, 'eval')):
                    if os.path.isdir(os.path.join(experiment_dir, subdir, 'eval', subsubdir)):
                        for epochs in sorted(os.listdir(os.path.join(experiment_dir, subdir, 'eval', subsubdir))):
                            if epochs.startswith('epoch'):
                                imgs = os.listdir(os.path.join(experiment_dir, subdir, 'eval', subsubdir, epochs, 'rgb_erode_dilate'))
                                # sort, by: name.replace('.png', '') into integer
                                imgs = sorted([img for img in imgs if not img.endswith('.db')], key=lambda x: int(x.replace('.png', '')))

                                for i, img in enumerate(imgs):
                                    if i % ratio == 0:
                                        # with zero padding 5d for index file name
                                        shutil.copy(os.path.join(experiment_dir, subdir, 'eval', subsubdir, epochs, 'rgb_erode_dilate', img), os.path.join(collection_dir, f'{exp}_{i:05d}_pointavatar.png'))


    
    for exp in os.listdir(os.path.join(main_dir, 'nha')):
        try:
            if exp.startswith('imavatar') and not exp.endswith('v2'):
                continue
                
            if exp.startswith('nha_person'):
                continue
            
            if exp.startswith('new_nha_person'):
                images_list = os.listdir(os.path.join(main_dir, 'nha', exp))
                exp = exp.replace('new_', '')

            else:
                experiment_dir = os.path.join(main_dir, 'nha', exp, 'lightning_logs')
                # there are version_0, version_1, ... multiple subdirectories
                # pick the latest version
                versions = sorted(os.listdir(experiment_dir), key=lambda x: int(x.split('_')[-1]))
                latest_version = versions[-1]
                if exp == 'obama':
                    latest_version = 'version_2'
                # \\bean.postech.ac.kr\log\gwangjin\2024\nbshapes_comparisons\nha\imavatar_mar_v2\lightning_logs\version_0\NovelViewSynthesisResults\val\0_0

                if not exp.startswith('nha_person'):
                    images_list = os.listdir(os.path.join(experiment_dir, latest_version, 'NovelViewSynthesisResults', 'val', '0_0'))
                else:
                    images_list = os.listdir(os.path.join(main_dir, 'nha', 'new_'+exp))
            
            # sort images list, by: name[5:].replace('.png', '') into integer,
            images_list = sorted([img for img in images_list if not img.endswith('.db')], key=lambda x: int(x.replace('.png', '')))

            if exp == 'imavatar_mar_v2':
                exp = 'marcel'
            elif exp == 'imavatar_yuf_v2':
                exp = 'yufeng'
            elif exp == 'imavatar_sub3_v2':
                exp = 'subject_3'

            for i, img in enumerate(images_list):
                if i % ratio == 0:
                    shutil.copy(os.path.join(experiment_dir, latest_version, 'NovelViewSynthesisResults', 'val', '0_0', img), os.path.join(collection_dir, f'{exp}_{i:05d}_nha.png'))
        except:
            continue
    
    insta_dir = os.path.join('/Bean/data/gwangjin/2024/nbshapes/insta/')

    for exp in os.listdir(insta_dir):

        if exp.startswith('imavatar') and not exp.endswith('v2'):
            continue

        if not os.path.exists(os.path.join(insta_dir, exp, 'experiments', 'insta', 'debug', 'overlay')):
            continue

        images_list = os.listdir(os.path.join(insta_dir, exp, 'experiments', 'insta', 'debug', 'overlay'))
        # sort images list, by: name[5:].replace('.png', '') into integer,
        images_list = sorted([img for img in images_list if not img.endswith('.db')], key=lambda x: int(x.replace('.png', '')))

        for i, img in enumerate(images_list):
            if i % ratio == 0:
                shutil.copy(os.path.join(insta_dir, exp, 'experiments', 'insta', 'debug', 'overlay', img), os.path.join(collection_dir, f'{exp}_{i:05d}_insta.png'))


        if exp == 'imavatar_mar_v2':
            exp = 'marcel'
        elif exp == 'imavatar_yuf_v2':
            exp = 'yufeng'
        elif exp == 'imavatar_sub3_v2':
            exp = 'subject_3'


    # collect gt
    gt_dir = os.path.join('/Bean/data/gwangjin/2024/nbshapes/flare')

    for exp in os.listdir(gt_dir):
        test_dir = 'test'
        if exp == 'marcel':
            test_dir = 'MVI_1802'
        elif exp == 'yufeng':
            test_dir = 'MVI_1812'

        images_dir = os.path.join(gt_dir, exp, exp, test_dir, 'image')
        # collect every 50th
        for i, img in enumerate(sorted([img for img in os.listdir(images_dir) if not img.endswith('.db')], key=lambda x: int(x.replace('.png', '')))):
            if i % ratio == 0:
                shutil.copy(os.path.join(images_dir, img), os.path.join(collection_dir, f'{exp}_{i:05d}_gt.png'))


    ratio = 150
    collection_dir = 'figures/comparison_mesh'            
    os.makedirs(collection_dir, exist_ok=True)
    # flare

    flare_images_dir = f'{{}}/images_evaluation/qualitative_results/rgb'
    gbshapes_images_dir = f'{{}}/gbshapes/{{}}/test/split_40000'

    for exp in os.listdir(os.path.join(main_dir, 'flare')):
        images_dir = os.path.join(main_dir, 'flare', exp, 'images_evaluation/qualitative_results/rgb')        
        # collect every 50th
        for i, img in enumerate(sorted([img for img in os.listdir(images_dir) if not img.endswith('.db')])):
            if i % ratio == 0:
                shutil.copy(os.path.join(images_dir, img), os.path.join(collection_dir, f'{exp}_{i:05d}_flare.png'))
                shutil.copy(os.path.join(images_dir.replace('rgb', 'normal'), img), os.path.join(collection_dir, f'{exp}_{i:05d}_flare_normal.png'))

    # ours
    ours_images_dir = f'{{}}/images_evaluation/qualitative_results/rgb'

    for exp in os.listdir(os.path.join(main_dir, 'ours_enc_v13')):
        images_dir = os.path.join(main_dir, 'ours_enc_v13', exp, 'images_evaluation/qualitative_results/rgb')        
        # collect every 50th
        try:
            for i, img in enumerate(sorted([img for img in os.listdir(images_dir) if not img.endswith('.db')])):
                if i % ratio == 0:
                    shutil.copy(os.path.join(images_dir, img), os.path.join(collection_dir, f'{exp}_{i:05d}_v13_ours.png'))
                    shutil.copy(os.path.join(images_dir.replace('rgb', 'normal'), img), os.path.join(collection_dir, f'{exp}_{i:05d}_v13_ours_normal.png'))
        except:
            pass

    # ours
    ours_images_dir = f'{{}}/images_evaluation/qualitative_results/rgb'

    for exp in os.listdir(os.path.join(main_dir, 'ours_enc_v6')):
        images_dir = os.path.join(main_dir, 'ours_enc_v6', exp, 'images_evaluation/qualitative_results/rgb')        
        # collect every 50th
        try:
            for i, img in enumerate(sorted([img for img in os.listdir(images_dir) if not img.endswith('.db')])):
                if i % ratio == 0:
                    shutil.copy(os.path.join(images_dir, img), os.path.join(collection_dir, f'{exp}_{i:05d}_ours.png'))
                    shutil.copy(os.path.join(images_dir.replace('rgb', 'normal'), img), os.path.join(collection_dir, f'{exp}_{i:05d}_ours_normal.png'))
        except:
            pass

    # ours
    ours_images_dir = f'{{}}/images_evaluation/qualitative_results/rgb'

    for exp in os.listdir(os.path.join(main_dir, 'ours_enc_v10')):
        images_dir = os.path.join(main_dir, 'ours_enc_v10', exp, 'images_evaluation/qualitative_results/rgb')        
        # collect every 50th
        try:
            for i, img in enumerate(sorted([img for img in os.listdir(images_dir) if not img.endswith('.db')])):
                if i % ratio == 0:
                    shutil.copy(os.path.join(images_dir, img), os.path.join(collection_dir, f'{exp}_{i:05d}_v10_ours.png'))
                    shutil.copy(os.path.join(images_dir.replace('rgb', 'normal'), img), os.path.join(collection_dir, f'{exp}_{i:05d}_ours_normal.png'))
        except:
            pass

    # ours
    ours_images_dir = f'{{}}/images_evaluation/qualitative_results/rgb'

    for exp in os.listdir(os.path.join(main_dir, 'ours_enc_v3')):
        images_dir = os.path.join(main_dir, 'ours_enc_v3', exp, 'images_evaluation/qualitative_results/rgb')        
        # collect every 50th
        try:
            for i, img in enumerate(sorted([img for img in os.listdir(images_dir) if not img.endswith('.db')])):
                if i % ratio == 0:
                    shutil.copy(os.path.join(images_dir, img), os.path.join(collection_dir, f'{exp}_{i:05d}_v3_ours.png'))
                    shutil.copy(os.path.join(images_dir.replace('rgb', 'normal'), img), os.path.join(collection_dir, f'{exp}_{i:05d}_v3_ours_normal.png'))
        except:
            pass


    # collect gt
    gt_dir = os.path.join('/Bean/data/gwangjin/2024/nbshapes/flare')

    for exp in os.listdir(gt_dir):
        test_dir = 'test'
        if exp == 'marcel':
            test_dir = 'MVI_1802'
        elif exp == 'yufeng':
            test_dir = 'MVI_1812'

        images_dir = os.path.join(gt_dir, exp, exp, test_dir, 'image')
        # collect every 50th
        for i, img in enumerate(sorted([img for img in os.listdir(images_dir) if not img.endswith('.db')], key=lambda x: int(x.replace('.png', '')))):
            if i % ratio == 0:
                shutil.copy(os.path.join(images_dir, img), os.path.join(collection_dir, f'{exp}_{i:05d}_gt.png'))




    ratio = 150
    collection_dir = 'figures/ours_collection'            
    os.makedirs(collection_dir, exist_ok=True)
    # flare
    
    # collect gt
    gt_dir = os.path.join('/Bean/data/gwangjin/2024/nbshapes/flare')

    for exp in os.listdir(gt_dir):
        test_dir = 'test'
        if exp == 'marcel':
            test_dir = 'MVI_1802'
        elif exp == 'yufeng':
            test_dir = 'MVI_1812'

        images_dir = os.path.join(gt_dir, exp, exp, test_dir, 'image')
        # collect every 50th
        for i, img in enumerate(sorted([img for img in os.listdir(images_dir) if not img.endswith('.db')], key=lambda x: int(x.replace('.png', '')))):
            if i % ratio == 0:
                shutil.copy(os.path.join(images_dir, img), os.path.join(collection_dir, f'{exp}_{i:05d}_gt.png'))



    ours_images_dir = f'{{}}/images_evaluation/qualitative_results/rgb'

    for exp in os.listdir(os.path.join(main_dir, 'ours_enc_v6')):
        images_dir = os.path.join(main_dir, 'ours_enc_v6', exp, 'images_evaluation/qualitative_results/rgb')        
        # collect every 50th
        try:
            for i, img in enumerate(sorted([img for img in os.listdir(images_dir) if not img.endswith('.db')])):
                if i % ratio == 0:
                    shutil.copy(os.path.join(images_dir, img), os.path.join(collection_dir, f'{exp}_{i:05d}_ours.png'))
                    shutil.copy(os.path.join(images_dir.replace('rgb', 'normal'), img), os.path.join(collection_dir, f'{exp}_{i:05d}_ours_normal.png'))
        except:
            pass



    ours_images_dir = f'{{}}/images_evaluation/qualitative_results/rgb'

    for exp in os.listdir(os.path.join(main_dir, 'ours_enc_v3')):
        images_dir = os.path.join(main_dir, 'ours_enc_v3', exp, 'images_evaluation/qualitative_results/rgb')        
        # collect every 50th
        try:
            for i, img in enumerate(sorted([img for img in os.listdir(images_dir) if not img.endswith('.db')])):
                if i % ratio == 0:
                    shutil.copy(os.path.join(images_dir, img), os.path.join(collection_dir, f'{exp}_{i:05d}_v3_ours.png'))
                    shutil.copy(os.path.join(images_dir.replace('rgb', 'normal'), img), os.path.join(collection_dir, f'{exp}_{i:05d}_v3_ours_normal.png'))
        except:
            pass

    ours_images_dir = f'{{}}/images_evaluation/qualitative_results/rgb'

    for exp in os.listdir(os.path.join(main_dir, 'ours_enc_v10')):
        images_dir = os.path.join(main_dir, 'ours_enc_v10', exp, 'images_evaluation/qualitative_results/rgb')        
        # collect every 50th
        try:
            for i, img in enumerate(sorted([img for img in os.listdir(images_dir) if not img.endswith('.db')])):
                if i % ratio == 0:
                    shutil.copy(os.path.join(images_dir, img), os.path.join(collection_dir, f'{exp}_{i:05d}_v10_ours.png'))
                    shutil.copy(os.path.join(images_dir.replace('rgb', 'normal'), img), os.path.join(collection_dir, f'{exp}_{i:05d}_v10_ours_normal.png'))
        except:
            pass

    for exp in os.listdir(os.path.join(main_dir, 'ours_enc_v13')):
        images_dir = os.path.join(main_dir, 'ours_enc_v13', exp, 'images_evaluation/qualitative_results/rgb')        
        # collect every 50th
        try:
            for i, img in enumerate(sorted([img for img in os.listdir(images_dir) if not img.endswith('.db')])):
                if i % ratio == 0:
                    shutil.copy(os.path.join(images_dir, img), os.path.join(collection_dir, f'{exp}_{i:05d}_v13_ours.png'))
                    shutil.copy(os.path.join(images_dir.replace('rgb', 'normal'), img), os.path.join(collection_dir, f'{exp}_{i:05d}_v13_ours_normal.png'))
        except:
            pass
