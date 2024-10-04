import os
import subprocess

if __name__ == '__main__':
    # 돌리는 요소
    # config, run_name, eval_dir, input_dir, target_model_path, transfer_out_name.)

    # set: config, run_name,   /  eval_dir, input_dir, target_model_path,  /   transfer_out_name
    output_dir = '/Bean/log/gwangjin/2024/neural_blendshapes'

    names = ['marcel', 'yufeng', 'sub3', 'nha_0', 'nha_1', 'nerface_1']

    config_files = {}
    config_files['marcel'] = './configs/mar.txt'
    config_files['yufeng'] = './configs/yuf.txt'
    config_files['sub3'] = './configs/sub3.txt'
    config_files['nha_0'] = './configs/nha_0.txt'
    config_files['nha_1'] = './configs/nha_1.txt'
    config_files['nerface_1'] = './configs/nerface_1.txt'

    run_names = {}
    run_names['marcel'] = 'mar_albedo_insta_shading_2'
    run_names['yufeng'] = 'yuf_albedo_insta_shading_2'
    run_names['sub3'] = 'sub3_albedo_insta_shading_2'
    run_names['nha_0'] = 'nha_0'
    run_names['nha_1'] = 'nha_1'
    run_names['nerface_1'] = 'nerface_1'

    eval_dirs = {}
    eval_dirs['marcel'] = ['MVI_1802', 'MVI_1799', 'MVI_1797', 'MVI_1801']
    eval_dirs['yufeng'] = ['MVI_1812', 'MVI_1814', 'MVI_1810', 'MVI_1811']
    eval_dirs['sub3'] = ['train', 'test']
    eval_dirs['nha_0'] = ['train', 'test']
    eval_dirs['nha_1'] = ['train', 'test']
    eval_dirs['nerface_1'] = ['train', 'test']

    target_model_paths = {}
    for k in run_names.keys():
        target_model_paths[k] = os.path.join(output_dir, run_names[k], 'stage_1', 'network_weights', 'neural_blendshapes.pt')


    # now we combination
    # config, run_name, lambda_, transfer_out_name, target_model_dir, eval_dir, input_dir.

    # divide them by two sets 
    # config+run_name
    # transfer_out_name+eval_dir+target_model_dir_eval_dir
    # config+run_name
    for source_name in names:
        config = config_files[source_name]
        run_name = run_names[source_name]

        for target_name in names:
            if source_name == target_name:
                continue
            target_model_path = target_model_paths[target_name]
            for eval_dir in eval_dirs[target_name]:
                if target_name in ['marcel', 'yufeng', 'sub3']:
                    input_dir = os.path.join('/Bean/data/gwangjin/2024', target_name, eval_dir)
                else:
                    input_dir = os.path.join('/Bean/data/gwangjin/2024/nbshape_additional/imavatar_videos', target_name, eval_dir)
                input_dir = os.path.join('/Bean/data/gwangjin/2024', target_name, target_name)
                transfer_out_name = f'{source_name}_{target_name}_{eval_dir}'
                lambda_ = 0

                command = f'CUDA_VISIBLE_DEVICES=7 python expression_transfer.py --config {config} --run_name {run_name} --lambda_ {lambda_} --transfer_out_name {transfer_out_name} --target_model_dir {target_model_path} --eval_dir {eval_dir} --input_dir {input_dir}'

                print(command)
                # instead of os.system, use subprocess
                subprocess.run(command, shell=True)
                # does it wait for the command to finish?


                # run the command
                

# python expression_transfer.py --config configs/yuf.txt --run_name yuf_albedo_insta_shading_2 --lambda_ 0 --transfer_out_name subject_3 --target_model_dir /Bean/log/gwangjin/2024/neural_blendshapes/sub3_albedo_insta_shading_2/stage_1/network_weights/neural_blendshapes.pt --eval_dir test --input_dir /Bean/data/gwangjin/2024/subject_3/subject_3/
