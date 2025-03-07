
CUDA_VISIBLE_DEVICES=1 python train.py --config configs/ablation_correction.txt --run_name marcel_w_o_pers_3 --skip_wandb --fix_bshapes &
CUDA_VISIBLE_DEVICES=2 python train.py --config configs/ablation_correction.txt --run_name marcel_w_o_pers_no_pose_3 --skip_wandb --fix_bshapes --disable_pose &

