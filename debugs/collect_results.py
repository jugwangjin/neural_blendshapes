import os
import sys
import shutil


if __name__=='__main__':
    main_dir = '/Bean/log/gwangjin/2024/neural_blendshapes'
    images_dir = 'stage_1/images/grid'

    os.makedirs('debug/hyperparam', exist_ok=True)
    for exps in os.listdir(main_dir):
        if not exps.endswith('_0') or not exps.startswith('jaco'):
            continue

        exp_name = exps[:-2]

        ict_img = os.path.join(main_dir, exps, images_dir, 'grid_900_ict.png')
        full_img = os.path.join(main_dir, exps, images_dir, 'grid_900.png')

        try:
            # copy ict_img, full_img to hyperparam, with exp_name

            shutil.copy(ict_img, f'debug/hyperparam/{exp_name}_ict.png')
            shutil.copy(full_img, f'debug/hyperparam/{exp_name}.png')
        except:
            pass