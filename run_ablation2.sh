CUDA_VISIBLE_DEVICES=1 python train.py --config configs/ablation_correction.txt --run_name marcel_w_pers_3 --skip_wandb &
CUDA_VISIBLE_DEVICES=2 python train.py --config configs/ablation_correction.txt --run_name marcel_w_additive_3 --skip_wandb --additive &