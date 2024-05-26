import os

if __name__ == '__main__':
    base_command = "python train.py --config configs/mar_local.txt "

    # run base command, with one of those weight args set to 0
    # make proper --run_name, and --wandb_name for each run
    #  --iterations == 10000
    params_list = ["mask", "laplacian_regularization", "shading", "perceptual_loss", "landmark", "closure", "feature_regularization", "cbuffers_regularization", "segmentation", "semantic_stat"]

    for i in range(len(params_list)):
        for j in range(i+1, len(params_list)):
            command = base_command + f"--run_name {params_list[i]}_{params_list[j]}_0 --wandb_name {params_list[i]}_{params_list[j]}_0 --weight_{params_list[i]} 0 --weight_{params_list[j]} 0 --iterations 10000"
            os.system(command)
       
            for k in range(j+1, len(params_list)):
                command = base_command + f"--run_name {params_list[i]}_{params_list[j]}_{params_list[k]}_0 --wandb_name {params_list[i]}_{params_list[j]}_{params_list[k]}_0 --weight_{params_list[i]} 0 --weight_{params_list[j]} 0 --weight_{params_list[k]} 0 --iterations 10000"
                os.system(command)
        command = base_command + f"--run_name {params_list[i]}_0 --wandb_name {params_list[i]}_0 --weight_{params_list[i]} 0 --iterations 10000"
        os.system(command)