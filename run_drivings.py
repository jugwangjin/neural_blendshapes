import os


pairs= []
pairs.append(('obama', 'wojtek_1'))
# pairs.append(('obama', 'bala'))
# pairs.append(('obama', 'justin'))
pairs.append(('justin', 'bala'))
# pairs.append(('justin', 'wojtek_1'))
# pairs.append(('justin', 'malte_1'))//
# pairs.append(('malte_1', 'wojtek_1'))
# pairs.append(('malte_1', 'bala'))
# pairs.append(('malte_1', 'justin'))
# pairs.append(('yufeng', 'justin'))
# pairs.append(('yufeng', 'bala'))


for source, target in pairs:
    name = source 

    train = 'test'

    if name == 'marcel':
        train = 'MVI_1802'
    elif name == 'yufeng':
        train = 'MVI_1812'

    output_dir = '/Bean/log/gwangjin/2024/nbshapes_comparisons/ours_enc_v13/'
    

    # parser.add_argument('--model_name_2', type=str, default='marcel', help='Name of the run in model_dir')
    # parser.add_argument('--output_dir_2', type=str, default='marcel', help='Name of the run in model_dir')

    output_dir_2 = '/Bean/log/gwangjin/2024/nbshapes_comparisons/ours_enc_v13/'
    model_name_2 = target

    command = f'CUDA_VISIBLE_DEVICES=6 python make_other_person_driving.py --model_name {name} --video_name {train} --output_dir_name videos/cross_driving/{name}_{target} --model_dir {output_dir} --model_name_2 {model_name_2} --output_dir_2 {output_dir_2}'
    # print(command)
    # continue
    print(command)
    os.system(command)