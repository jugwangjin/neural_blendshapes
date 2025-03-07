import os


for name in os.listdir('./configs_tmp'):
    name = name.replace('.txt', '')

    train = 'test' 

    if name == 'marcel':
        train = 'MVI_1802'
    elif name == 'yufeng':
        train = 'MVI_1812'

    output_dir = '/Bean/log/gwangjin/2024/nbshapes_comparisons/ours_enc_v13/'
    if name == 'yufeng':
        output_dir = '/Bean/log/gwangjin/2024/nbshapes_comparisons/ours_enc_v14/'
    else:
        continue
    command = f'CUDA_VISIBLE_DEVICES=6 python make_tracking_video.py --model_name {name} --video_name {train} --output_dir_name videos/driving/{name} --model_dir {output_dir}' 

    print(command)
    os.system(command)