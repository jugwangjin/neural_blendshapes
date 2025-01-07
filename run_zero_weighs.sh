CUDA_VISIBLE_DEVICES=0 python train.py --config configs/mar.txt  --lambda_ 0 --compute_mode --stage_iterations 2000 1 1 1 --skip_eval --skip_wandb --weight_mask 0 --run_name param_deb_mask_0 --compute_mode 

CUDA_VISIBLE_DEVICES=0 python train.py --config configs/mar.txt  --lambda_ 0 --compute_mode --stage_iterations 2000 1 1 1 --skip_eval --skip_wandb --weight_shading 0 --weight_perceptual_loss 0 --run_name param_deb_shading_0  --compute_mode 

CUDA_VISIBLE_DEVICES=0 python train.py --config configs/mar.txt  --lambda_ 0 --compute_mode --stage_iterations 2000 1 1 1 --skip_eval --skip_wandb --weight_white_lgt_regularization 0 --run_name param_deb_reg_0 --weight_laplacian_regularization 0 --weight_linearity_regularization 0 --weight_flame_regularization 0 --weight_roughness_regularization 0 --weight_albedo_regularization 0 --weight_fresnel_coeff 0  --compute_mode 

CUDA_VISIBLE_DEVICES=0 python train.py --config configs/mar.txt  --lambda_ 0 --compute_mode --stage_iterations 2000 1 1 1 --skip_eval --skip_wandb --weight_feature_regularization 0 --run_name param_deb_feat_0  --compute_mode 

CUDA_VISIBLE_DEVICES=0 python train.py --config configs/mar.txt  --lambda_ 0 --compute_mode --stage_iterations 2000 1 1 1 --skip_eval --skip_wandb --weight_normal_laplacian 0 --run_name param_deb_norm_0  --compute_mode 

CUDA_VISIBLE_DEVICES=0 python train.py --config configs/mar.txt  --lambda_ 0 --compute_mode --stage_iterations 2000 1 1 1 --skip_eval --skip_wandb --weight_temporal_regularization 0 --run_name param_deb_temp_0  --compute_mode 

CUDA_VISIBLE_DEVICES=0 python train.py --config configs/mar.txt  --lambda_ 0 --compute_mode --stage_iterations 2000 1 1 1 --skip_eval --skip_wandb --weight_white_lgt_regularization 0 --run_name param_deb_reg_wo_flame_0 --weight_laplacian_regularization 0 --weight_linearity_regularization 0 --weight_roughness_regularization 0 --weight_albedo_regularization 0 --weight_fresnel_coeff 0  --compute_mode 


