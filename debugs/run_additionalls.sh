CUDA_VISIBLE_DEVICES=5 python train.py --config configs/nha_0.txt --run_name nha_0  --iterations 15000 --lambda_ 0  &
CUDA_VISIBLE_DEVICES=6 python train.py --config configs/nha_1.txt --run_name nha_1  --iterations 15000 --lambda_ 0  &
CUDA_VISIBLE_DEVICES=7 python train.py --config configs/nerface_1.txt --run_name nerface_1  --iterations 15000 --lambda_ 0  &
