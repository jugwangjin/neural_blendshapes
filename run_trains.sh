CUDA_VISIBLE_DEVICES=5 python train.py --config configs/mar.txt --run_name mar_albedo  --iterations 15000 --lambda_ 0  &
CUDA_VISIBLE_DEVICES=6 python train.py --config configs/yuf.txt --run_name yuf_albedo  --iterations 15000 --lambda_ 0  &
CUDA_VISIBLE_DEVICES=7 python train.py --config configs/sub3.txt --run_name sub3_albedo  --iterations 15000 --lambda_ 0  &
