CUDA_VISIBLE_DEVICES=1 python train.py --config configs/mar.txt --run_name alt_base &
CUDA_VISIBLE_DEVICES=2 python train.py --config configs/mar.txt --run_name alt_low_lap --weight_laplacian_regularization 1e1 &
CUDA_VISIBLE_DEVICES=3 python train.py --config configs/mar.txt --run_name alt_low_mask --weight_mask 2.0
CUDA_VISIBLE_DEVICES=1 python train.py --config configs/mar.txt --run_name alt_high_landmark --weight_landmark 10.0 &
CUDA_VISIBLE_DEVICES=2 python train.py --config configs/mar.txt --run_name alt_high_seg --weight_segmentation 1.0 &
CUDA_VISIBLE_DEVICES=3 python train.py --config configs/mar.txt --run_name alt_high_mlp --light_mlp_dims [128, 128] --material_mlp_dims [256, 256, 256, 256] &
CUDA_VISIBLE_DEVICES=4 python train.py --config configs/mar.txt --run_name alt_high_lr --lr_encoder 5e-4 --lr_shader 1e-3 --lr_deformer 1e-3
