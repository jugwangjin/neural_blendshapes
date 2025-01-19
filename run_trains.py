import os
import subprocess
import multiprocessing
from queue import Queue
import argparse
import shutil

def worker(gpu_id, command_queue):
    while not command_queue.empty():
        command, directory = command_queue.get()
        OUTPUT_DIR_ROOT = '/Bean/log/gwangjin/2024/nbshapes_comparisons/ours/'
        # print(f"Running on GPU {gpu_id}: {command}")
        # print(os.path.exists(os.path.join(OUTPUT_DIR_ROOT, directory, 'currently_training.txt')), os.path.exists(os.path.join(OUTPUT_DIR_ROOT, directory, 'final_eval.txt')))
        # if not os.path.exists(os.path.join(OUTPUT_DIR_ROOT, directory, 'currently_training.txt')) and not os.path.exists(os.path.join(OUTPUT_DIR_ROOT, directory, 'final_eval.txt')):

        # write a dummy txt file, that will be used to check if the training is ongoing
        os.makedirs(os.path.join(OUTPUT_DIR_ROOT, directory), exist_ok=True)
        with open(os.path.join(OUTPUT_DIR_ROOT, directory, 'currently_training.txt'), 'w') as f:
            f.write('')
        

        command = command.format(gpu_id)
        print(f"Running on GPU {gpu_id}: {command}")
        subprocess.run(command, shell=True)

        os.remove(os.path.join(OUTPUT_DIR_ROOT, directory, 'currently_training.txt'))
        command_queue.task_done()

def run_trackings(gpu_ids):
    commands = []

# input_dir = /Bean/data/gwangjin/2024/nbshapes/flare/yufeng/yufeng
# train_dir = ["MVI_1814", "MVI_1810"]
# eval_dir = ["MVI_1812"]

# input_dir = /data1/gwangjin/flare/marcel2
# train_dir = ["MVI_1797", "MVI_1801"]
# eval_dir = ["MVI_1802"]
# working_dir = .
# run_name = fully_neural_v1

# input_dir = /data1/gwangjin/flare/marcel2
# train_dir = ["MVI_1797", "MVI_1801"]
# eval_dir = ["MVI_1802"]
# working_dir = .
# output_dir = /Bean/log/gwangjin/2024/neural_blendshapes
    
        
    INPUT_DIR_ROOT = '/Bean/data/gwangjin/2024/nbshapes/flare/'
    OUTPUT_DIR_ROOT = '/Bean/log/gwangjin/2024/nbshapes_comparisons/ours_enc_v2/'
    
    directories = os.listdir(INPUT_DIR_ROOT)
    # reverse the order
    directories = sorted(directories, reverse=True)

    # shuffle the directories
    import random
    random.shuffle(directories)
    
    for directory in directories:
            # if not directory.startswith('imavatar_'):
            #       continue
            # if directory in ['person_0000', 'person_0004']:
            #       continue
            # if directory != 'imavatar_sub3' and directory != 'imavatar_mar':
            #       continue

            # if not directory.endswith('v2'):
            #       continue

        input_dir = os.path.join(INPUT_DIR_ROOT, directory, directory)
        
        if directory == 'yufeng':
            train_dir = '["MVI_1814", "MVI_1810"]'
            eval_dir = '["MVI_1812"]'
        elif directory == 'marcel':
            train_dir = '["MVI_1797", "MVI_1801"]'
            eval_dir = '["MVI_1802"]'
        else:
            train_dir = '["train"]'
            eval_dir = '["test"]'

        tmp_configs_dir = './configs_tmp'
        os.makedirs(tmp_configs_dir, exist_ok=True)
        base_conf_file = './configs/mar.txt'
        with open(base_conf_file, 'r') as f:
            conf = f.read()
            conf = conf.replace('input_dir = /Bean/data/gwangjin/2024/nbshapes/flare/marcel/marcel', f'input_dir = {input_dir}')
            conf = conf.replace('output_dir = /Bean/log/gwangjin/2024/neural_blendshapes', f'output_dir = {OUTPUT_DIR_ROOT}')    
            conf = conf.replace('run_name = fully_neural_v1', f'run_name = {directory}')
            conf = conf.replace('train_dir = ["MVI_1797", "MVI_1801"]', f'train_dir = {train_dir}')
            conf = conf.replace('eval_dir = ["MVI_1802"]', f'eval_dir = {eval_dir}')


            conf_file = os.path.join(tmp_configs_dir, f'{directory}.txt')
            with open(conf_file, 'w') as f:
                f.write(conf)

        # out_path = os.path.join(input_dir, directory, 'tracking_results')
        # shutil.rmtree(out_path, ignore_errors=True)
        # os.makedirs(out_path, exist_ok=True)

        
        command = (f"CUDA_VISIBLE_DEVICES={{}} python train.py --config {conf_file} --compute_mode", directory)
        commands.append(command)


    # Create a queue for each GPU
    gpu_queues = {gpu_id: Queue() for gpu_id in gpu_ids}

    # Distribute commands evenly across GPUs
    for i, command in enumerate(commands):
        gpu_id = gpu_ids[i % len(gpu_ids)]

        gpu_queues[gpu_id].put(command)
        print(gpu_id, command)

    # print(gpu_queues)
    # exit()

    # Start a worker process for each GPU

    try:
        processes = []
        for gpu_id in gpu_ids:
            p = multiprocessing.Process(target=worker, args=(gpu_id, gpu_queues[gpu_id]))
            p.start()
            processes.append(p)

        # Wait for all processes to finish
        for p in processes:
            p.join()
    except KeyboardInterrupt as e:
        # send keyboard interrupt to all processes
        for p in processes:
            p.terminate()
    
    except Exception as e:
        print(e)
        for p in processes:
            p.terminate()


    finally:
        # remove all currently_training.txt files
        for directory in directories:
            if os.path.exists(os.path.join(OUTPUT_DIR_ROOT, directory, 'currently_training.txt')):
                print(f"Removing {os.path.join(OUTPUT_DIR_ROOT, directory, 'currently_training.txt')}")
                os.remove(os.path.join(OUTPUT_DIR_ROOT, directory, 'currently_training.txt'))
        for p in processes:
            p.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', type=int, nargs='+', required=True)
    args = parser.parse_args()
    run_trackings(args.gpu_ids)