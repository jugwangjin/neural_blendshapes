cd /Jarvis/workspace/gwangjin/2024/blendshape/neural_blendshapes
/Jarvis/workspace/gwangjin/clear_cache.sh 
CUDA_VISIBLE_DEVICES=2 python train.py --config configs/mar.txt --run_name simp_shading_linear_v8 --iterations 15000 --lambda_ 0 &
cd ../neural_blendshapes_simp_shading/
cd /Jarvis/workspace/gwangjin/2024/blendshape/neural_blendshapes_simp_shading
/Jarvis/workspace/gwangjin/clear_cache.sh 
CUDA_VISIBLE_DEVICES=3 python train.py --config configs/mar.txt --run_name simp_shading_linear_uv_v8 --iterations 15000 --lambda_ 0 &