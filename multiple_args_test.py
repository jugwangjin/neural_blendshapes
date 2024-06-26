import os

if __name__ == '__main__':
    base_command = "CUDA_VISIBLE_DEVICES=3 python train.py --config configs/mar.txt --skip_eval --skip_wandb "

    # run base command, with one of those weight args set to 0
    # make proper --run_name, and --wandb_name for each run
    #  --iterations == 1000
    params_list = ["landmark", "closure", "mask", "segmentation"]

    for i in range(len(params_list)):
        for j in range(i+1, len(params_list)):
            command = base_command + f" --run_name jaco_{params_list[i]}_{params_list[j]}_0 --wandb_name {params_list[i]}_{params_list[j]}_0 --weight_{params_list[i]} 0 --weight_{params_list[j]} 0 --iterations 1000"
            # if not os.path.exists(f'./out/{params_list[i]}_{params_list[j]}_0/stage_1/network_weights/neural_blendshapes_latest.pt'):
            os.system(command)
       
            for k in range(j+1, len(params_list)):
                command = base_command + f" --run_name jaco_{params_list[i]}_{params_list[j]}_{params_list[k]}_0 --wandb_name {params_list[i]}_{params_list[j]}_{params_list[k]}_0 --weight_{params_list[i]} 0 --weight_{params_list[j]} 0 --weight_{params_list[k]} 0 --iterations 1000"
                # if not os.path.exists(f'./out/{params_list[i]}_{params_list[j]}_{params_list[k]}_0/stage_1/network_weights/neural_blendshapes_latest.pt'):
                os.system(command)
        command = base_command + f" --run_name jaco_{params_list[i]}_0 --wandb_name {params_list[i]}_0 --weight_{params_list[i]} 0 --iterations 1000"
        # if not os.path.exists(f'./out/{params_list[i]}_0/stage_1/network_weights/neural_blendshapes_latest.pt'):
        os.system(command)
        