CUDA_VISIBLE_DEVICES=5 python train.py --config configs/mar.txt --run_name marcel_w_pers --skip_wandb &

CUDA_VISIBLE_DEVICES=6 python train.py --config configs/mar.txt --run_name marcel_w_additive --skip_wandb --additive &
CUDA_VISIBLE_DEVICES=7 python train.py --config configs/mar.txt --run_name marcel_w_o_pers --skip_wandb --fix_bshapes &
