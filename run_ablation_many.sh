
CUDA_VISIBLE_DEVICES=1 python train.py --config configs/ablation_correction_justin.txt --run_name justin_w_o_pers_3 --skip_wandb --fix_bshapes 
CUDA_VISIBLE_DEVICES=1 python train.py --config configs/ablation_correction_justin.txt --run_name justin_w_o_pers_no_pose_3 --skip_wandb --fix_bshapes --disable_pose 


CUDA_VISIBLE_DEVICES=1 python train.py --config configs/ablation_correction_justin.txt --run_name justin_w_pers_3 --skip_wandb 

CUDA_VISIBLE_DEVICES=2 python train.py --config configs/ablation_correction_justin.txt --run_name justin_w_additive_3 --skip_wandb --additive 


CUDA_VISIBLE_DEVICES=1 python train.py --config configs/ablation_correction_wojtek_1.txt --run_name wojtek_1_w_o_pers_3 --skip_wandb --fix_bshapes  &
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/ablation_correction_wojtek_1.txt --run_name wojtek_1_w_o_pers_no_pose_3 --skip_wandb --fix_bshapes --disable_pose 


CUDA_VISIBLE_DEVICES=2 python train.py --config configs/ablation_correction_wojtek_1.txt --run_name wojtek_1_w_pers_3 --skip_wandb 
CUDA_VISIBLE_DEVICES=2 python train.py --config configs/ablation_correction_wojtek_1.txt --run_name wojtek_1_w_additive_3 --skip_wandb --additive 


CUDA_VISIBLE_DEVICES=1 python train.py --config configs/ablation_correction_0004.txt --run_name 0004_w_o_pers_3 --skip_wandb --fix_bshapes 
CUDA_VISIBLE_DEVICES=1 python train.py --config configs/ablation_correction_0004.txt --run_name 0004_w_o_pers_no_pose_3 --skip_wandb --fix_bshapes --disable_pose 


CUDA_VISIBLE_DEVICES=1 python train.py --config configs/ablation_correction_0004.txt --run_name 0004_w_pers_3 --skip_wandb 
CUDA_VISIBLE_DEVICES=2 python train.py --config configs/ablation_correction_0004.txt --run_name 0004_w_additive_3 --skip_wandb --additive 