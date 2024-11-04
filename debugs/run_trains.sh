CUDA_VISIBLE_DEVICES=2 python train.py --config configs/mar.txt --run_name mar_albedo_insta_shading_5  --iterations 15000 --lambda_ 0  &
CUDA_VISIBLE_DEVICES=3 python train.py --config configs/yuf.txt --run_name yuf_albedo_insta_shading_5  --iterations 15000 --lambda_ 0  &
CUDA_VISIBLE_DEVICES=4 python train.py --config configs/sub3.txt --run_name sub3_albedo_insta_shading_5  --iterations 15000 --lambda_ 0  &
