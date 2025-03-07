import os


for name in os.listdir('./configs_tmp'):
    name = name.replace('.txt', '')

    train = 'train' 

    if name == 'marcel':
        train = 'MVI_1797'
    elif name == 'yufeng':
        train = 'MVI_1814'


    command = f'CUDA_VISIBLE_DEVICES=6 python render_tracking_effects.py --model_name {name} --video_name {train} --output_dir_name figures/supp_tracking'  

    print(command)
    os.system(command)