import os


try:
    for name in os.listdir('./configs_tmp'):
        name = name.replace('.txt', '')

        train = 'test' 

        if name == 'marcel':
            train = 'MVI_1797'
        elif name == 'yufeng':
            train = 'MVI_1802'


        command = f'CUDA_VISIBLE_DEVICES=6 python render_tracking_video.py --model_name {name} --video_name {train} --output_dir_name figures/enc_v13 --model_dir /Bean/log/gwangjin/2024/nbshapes_comparisons/ours_enc_v13' 

        print(command)
        os.system(command)
        # \\bean.postech.ac.kr\log\gwangjin\2024\nbshapes_comparisons\ours_enc_v13
except Exception as e:
    print(e)
    raise e

finally:
    print('done')