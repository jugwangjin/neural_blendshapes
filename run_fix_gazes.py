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
    output_dir = '/Bean/log/gwangjin/2024/nbshapes_comparisons/ours_enc_v13/'
    command = f'CUDA_VISIBLE_DEVICES=5 python make_fixing_gaze_video.py --model_name {name} --video_name {train} --output_dir_name videos/gaze/{name}'  

    print(command)
    os.system(command)

# CUDA_VISIBLE_DEVICES=7 python train.py --config configs/ablation_correction.txt --run_name marcel_w_additive_2 --skip_wandb --additive --only_flame_iterations 1000 stage_iterations 8000 0 3000 3000