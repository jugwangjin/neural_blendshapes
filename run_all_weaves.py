import os


for name in os.listdir('./configs_tmp'):
    name = name.replace('.txt', '')

    train = 'test' 

    if name == 'marcel':
        train = 'MVI_1802'
    elif name == 'yufeng':
        train = 'MVI_1812'

    # parser.add_argument('--parent', type=str, default='videos')
    # parser.add_argument('--name', type=str, default='justin')
    # parser.add_argument('--video_name', type=str, default='test')


    command = f'python weave_driving_video.py --parent videos/driving --name {name} --video_name {train}'
    
    print(command)
    os.system(command)

    
    command = f'python weave_gaze_video.py --parent videos/gaze --name {name} --video_name {train}'
    
    print(command)
    os.system(command)

    command = f'python weave_exag_video.py --parent videos/exag --name {name} --video_name {train}'

    print(command)
    os.system(command)