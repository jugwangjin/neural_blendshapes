import os
import sys
import shutil


if __name__=='__main__':
    ratio = 75
    collection_dir = 'figures/ours_meshes'
    os.makedirs(collection_dir, exist_ok=True)
    main_dir = '/Bean/log/gwangjin/2024/nbshapes_comparisons'
    

    # ours
    ours_mesh_dir = f'{{}}/stage_1/meshes'

    for exp in os.listdir(os.path.join(main_dir, 'ours_enc_v6')):
        # glob mesh_*_temp.obj
        mesh_dir = os.path.join(main_dir, 'ours_enc_v6', exp, 'stage_1', 'meshes')
        meshes = sorted([mesh for mesh in os.listdir(mesh_dir) if mesh.endswith('_temp.obj')], key=lambda x: int(x[5:10]), reverse=True)
        
        shutil.copy(os.path.join(mesh_dir, meshes[0]), os.path.join(collection_dir, f'{exp}.obj'))