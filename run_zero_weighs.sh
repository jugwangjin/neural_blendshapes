CUDA_VISIBLE_DEVICES=0 python train.py --config configs/mar.txt  --lambda_ 0 --compute_mode --stage_iterations 1000 502 1 1 1 1 --skip_eval --skip_wandb --weight_mask 0 --run_name 1205_mask_0 &

CUDA_VISIBLE_DEVICES=1 python train.py --config configs/mar.txt  --lambda_ 0 --compute_mode --stage_iterations 1000 502 1 1 1 1 --skip_eval --skip_wandb --weight_shading 0 --weight_perceptual_loss 0 --run_name 1205_shading_0 &

CUDA_VISIBLE_DEVICES=2 python train.py --config configs/mar.txt  --lambda_ 0 --compute_mode --stage_iterations 1000 502 1 1 1 1 --skip_eval --skip_wandb --weight_landmark 0 --run_name 1205_lmk_0 &
    
CUDA_VISIBLE_DEVICES=3 python train.py --config configs/mar.txt  --lambda_ 0 --compute_mode --stage_iterations 1000 502 1 1 1 1 --skip_eval --skip_wandb --weight_white_lgt_regularization 0 --run_name 1205_reg_0 --weight_laplacian_regularization 0 --weight_linearity_regularization 0 --weight_flame_regularization 0 --weight_roughness_regularization 0 --weight_albedo_regularization 0 --weight_fresnel_coeff 0 & 

CUDA_VISIBLE_DEVICES=4 python train.py --config configs/mar.txt  --lambda_ 0 --compute_mode --stage_iterations 1000 502 1 1 1 1 --skip_eval --skip_wandb --weight_feature_regularization 0 --run_name 1205_feat_0 &
CUDA_VISIBLE_DEVICES=5 python train.py --config configs/mar.txt  --lambda_ 0 --compute_mode --stage_iterations 1000 502 1 1 1 1 --skip_eval --skip_wandb --weight_normal_laplacian 0 --run_name 1205_norm_0 &


CUDA_VISIBLE_DEVICES=6 python train.py --config configs/mar.txt  --lambda_ 0 --compute_mode --stage_iterations 1000 502 1 1 1 1 --skip_eval --skip_wandb --weight_flame_regularization 0 --run_name 1205_flame_0 &


CUDA_VISIBLE_DEVICES=7 python train.py --config configs/mar.txt  --lambda_ 0 --compute_mode --stage_iterations 1000 502 1 1 1 1 --skip_eval --skip_wandb --weight_white_lgt_regularization 0 --run_name 1205_reg_wo_flame_0 --weight_laplacian_regularization 0 --weight_linearity_regularization 0 --weight_roughness_regularization 0 --weight_albedo_regularization 0 --weight_fresnel_coeff 0 & 



CUDA_VISIBLE_DEVICES=0 python train.py --config configs/mar.txt  --lambda_ 0 --compute_mode --run_name flare_shader_mar12
