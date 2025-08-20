import os
import subprocess
import multiprocessing
from queue import Queue
import argparse
import shutil
import time

def worker(gpu_id, command_queue):
    while not command_queue.empty():
        command, directory = command_queue.get()
        OUTPUT_DIR_ROOT = '/Bean/log/gwangjin/2025/nbshapes_iccv_unit_wise_higher_linear/'
        os.makedirs(OUTPUT_DIR_ROOT, exist_ok=True)
        
        print(f"Processing {directory} on GPU {gpu_id}")


        # Check if manipulation is already complete or if model exists
        if os.path.exists(os.path.join(OUTPUT_DIR_ROOT, directory, 'stage_1', 'network_weights', 'neural_blendshapes_latest.pt')):

            if not os.path.exists(os.path.join(OUTPUT_DIR_ROOT, directory, 'currently_manipulating.txt')):

                # Create output directory for figures
                figures_dir = os.path.join(OUTPUT_DIR_ROOT, directory, 'figures', 'supp_edit_higher_linear')
                os.makedirs(figures_dir, exist_ok=True)

                # write a dummy txt file to track manipulation progress
                os.makedirs(os.path.join(OUTPUT_DIR_ROOT, directory), exist_ok=True)
                with open(os.path.join(OUTPUT_DIR_ROOT, directory, 'currently_manipulating.txt'), 'w') as f:
                    f.write('')
                
                start_time = time.time()

                try:
                    command = command.format(gpu_id)
                    print(f"Running manipulation on GPU {gpu_id}: {command}")
                    subprocess.run(command, shell=True)
                    
                    # Mark manipulation as complete
                    with open(os.path.join(OUTPUT_DIR_ROOT, directory, 'manipulation_complete.txt'), 'w') as f:
                        f.write(f'Completed at {time.time()}\n')
                        
                except Exception as e:
                    print(e)
                    print(f"Failed to run manipulation on GPU {gpu_id}: {command}")
                    
                    # remove the currently_manipulating.txt file
                    if os.path.exists(os.path.join(OUTPUT_DIR_ROOT, directory, 'currently_manipulating.txt')):
                        os.remove(os.path.join(OUTPUT_DIR_ROOT, directory, 'currently_manipulating.txt'))
                    
                    # dump the error to a file
                    with open(os.path.join(OUTPUT_DIR_ROOT, directory, 'manipulation_error.txt'), 'w') as f:
                        f.write(str(e))

                    # dump error to error directory as well
                    error_dir = os.path.join(OUTPUT_DIR_ROOT, 'manipulation_errors')
                    os.makedirs(error_dir, exist_ok=True)
                    
                    current_time = time.time()
                    with open(os.path.join(error_dir, f'{directory}_manipulation.txt'), 'w') as f:
                        f.write(f'Error: {e}\n')
                        f.write(f'Running time: {current_time - start_time}\n')
                        f.write(f'Current time: {current_time}\n')  
                        f.write(f'Start time: {start_time}\n')
                        f.write(f'Arguments: {command}\n')
                        f.write(f'GPU: {gpu_id}\n')
                        f.write(f'Directory: {directory}\n')
                        f.write(f'Command: {command}\n')

                finally:
                    if os.path.exists(os.path.join(OUTPUT_DIR_ROOT, directory, 'currently_manipulating.txt')):
                        os.remove(os.path.join(OUTPUT_DIR_ROOT, directory, 'currently_manipulating.txt'))
        else:
            print(f"Skipping {directory}: manipulation already complete or model not found")
            
        command_queue.task_done()


def run_manipulations(gpu_ids):
    commands = []
    
    OUTPUT_DIR_ROOT = '/Bean/log/gwangjin/2025/nbshapes_iccv_unit_wise_higher_linear/'
    
    # Get all directories that have trained models
    directories = []
    if os.path.exists(OUTPUT_DIR_ROOT):
        for directory in os.listdir(OUTPUT_DIR_ROOT):
            model_path = os.path.join(OUTPUT_DIR_ROOT, directory, 'stage_1', 'network_weights', 'neural_blendshapes_latest.pt')
            if os.path.exists(model_path):
                directories.append(directory)
    
    # Sort directories
    directories = sorted(directories)
    print(f"Found {len(directories)} trained models to process")
    os.makedirs('figures/supp_edit', exist_ok=True)

    if directory in ['marcel']:
        video_name = 'MVI_1797'
    elif directory in ['yufeng']:
        video_name = 'MVI_1814'
    else:
        video_name = 'test'

    for directory in directories:
        # Create command for manipulation
        command = (f"CUDA_VISIBLE_DEVICES={{}} python manipulate_expressions_simple_edits.py "
                  f"--model_dir {OUTPUT_DIR_ROOT} "
                  f"--model_name {directory} "
                  f"--video_name {video_name} "
                  f"--index 0 "
                  f"--output_dir_name figures/supp_edit_higher_linear", directory)
        commands.append(command)

    if not commands:
        print("No models found to process")
        return

    # Create a queue for each GPU
    gpu_queues = {gpu_id: Queue() for gpu_id in gpu_ids}

    # Distribute commands evenly across GPUs
    for i, command in enumerate(commands):
        gpu_id = gpu_ids[i % len(gpu_ids)]
        gpu_queues[gpu_id].put(command)
        print(f"Assigned {command[1]} to GPU {gpu_id}")

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
        print("Keyboard interrupt received, terminating processes...")
        for p in processes:
            p.terminate()
    
    except Exception as e:
        print(f"Error: {e}")
        for p in processes:
            p.terminate()
        raise e

    finally:
        for p in processes:
            p.join()


def remove_all_currently_manipulating_txts():
    OUTPUT_DIR_ROOT = '/Bean/log/gwangjin/2025/nbshapes_iccv_unit_wise_higher_linear/'
    os.makedirs(OUTPUT_DIR_ROOT, exist_ok=True)
    for directory in os.listdir(OUTPUT_DIR_ROOT):
        if os.path.exists(os.path.join(OUTPUT_DIR_ROOT, directory, 'currently_manipulating.txt')):
            print(f"Removing {os.path.join(OUTPUT_DIR_ROOT, directory, 'currently_manipulating.txt')}")
            os.remove(os.path.join(OUTPUT_DIR_ROOT, directory, 'currently_manipulating.txt'))


def check_manipulation_status():
    OUTPUT_DIR_ROOT = '/Bean/log/gwangjin/2025/nbshapes_iccv_unit_wise_higher_linear/'
    total_models = 0
    completed_models = 0
    running_models = 0
    error_models = 0
    
    if os.path.exists(OUTPUT_DIR_ROOT):
        for directory in os.listdir(OUTPUT_DIR_ROOT):
            model_path = os.path.join(OUTPUT_DIR_ROOT, directory, 'stage_1', 'network_weights', 'neural_blendshapes_latest.pt')
            if os.path.exists(model_path):
                total_models += 1
                
                if os.path.exists(os.path.join(OUTPUT_DIR_ROOT, directory, 'manipulation_complete.txt')):
                    completed_models += 1
                elif os.path.exists(os.path.join(OUTPUT_DIR_ROOT, directory, 'currently_manipulating.txt')):
                    running_models += 1
                elif os.path.exists(os.path.join(OUTPUT_DIR_ROOT, directory, 'manipulation_error.txt')):
                    error_models += 1
    
    print(f"Manipulation Status:")
    print(f"  Total models: {total_models}")
    print(f"  Completed: {completed_models}")
    print(f"  Running: {running_models}")
    print(f"  Errors: {error_models}")
    print(f"  Pending: {total_models - completed_models - running_models - error_models}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', type=int, nargs='+', required=True, help='GPU IDs to use for processing')
    parser.add_argument('--status', action='store_true', help='Check manipulation status only')
    parser.add_argument('--clean', action='store_true', help='Remove all currently_manipulating.txt files')
    args = parser.parse_args()
    
    if args.status:
        check_manipulation_status()
    elif args.clean:
        remove_all_currently_manipulating_txts()
    else:
        try:
            run_manipulations(args.gpu_ids)
        except Exception as e:
            print(f"Error: {e}")
        finally:
            remove_all_currently_manipulating_txts() 