CUDA_VISIBLE_DEVICES=0 python train.py --config configs/mar.txt --run_name test_lambda_value --lambda_ 4 &
CUDA_VISIBLE_DEVICES=1 python train.py --config configs/mar.txt --run_name test_lambda_value --lambda_ 8 &
CUDA_VISIBLE_DEVICES=2 python train.py --config configs/mar.txt --run_name test_lambda_value --lambda_ 16 &
CUDA_VISIBLE_DEVICES=3 python train.py --config configs/mar.txt --run_name test_lambda_value --lambda_ 32
