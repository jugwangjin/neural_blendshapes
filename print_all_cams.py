# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from arguments import config_parser

import numpy as np
import torch

import os
os.environ["GLOG_minloglevel"] ="2"
import numpy as np
import torch


import json
from flare.dataset.dataset_util import _load_K_Rt_from_P
from tqdm import tqdm

def main(args):
    base_dir = args.working_dir / args.input_dir
    all_cameras = {}
    
    # Get all subdirectories
    subdirs = [d for d in base_dir.iterdir() if d.is_dir()]
    
    for subdir in tqdm(subdirs, desc="Processing directories"):
        try:
            json_file = subdir / "flame_params.json"
            
            if not json_file.exists():
                print(f"No flame_params.json in {subdir}")
                continue
                
            with open(json_file, 'r') as f:
                json_data = json.load(f)
                
            # Iterate through all frames
            for frame_idx, frame in enumerate(json_data["frames"]):
                world_mat = np.array(frame['world_mat']).astype(np.float32)
                world_mat = torch.tensor(_load_K_Rt_from_P(None, world_mat)[1], dtype=torch.float32)
                
                # Convert to openGL format
                R = world_mat[:3, :3].cpu().numpy()
                R *= -1
                t = world_mat[:3, 3].cpu().numpy()
                
                key = f"R:{R.tobytes()}, t:{t.tobytes()}"
                
                if key not in all_cameras:
                    all_cameras[key] = {
                        'subdir': subdir.name,
                        'frame_idx': frame_idx,
                        'R': R,
                        't': t,
                        'count': 1,
                        'datasets': {subdir.name}
                    }
                else:
                    all_cameras[key]['count'] += 1
                    all_cameras[key]['datasets'].add(subdir.name)
        except Exception as e:
            print(f"Error processing {subdir}: {e}")


    print(f"\nFound {len(all_cameras)} unique camera configurations:")
    for i, (key, cam_data) in enumerate(all_cameras.items()):
        print(f"\nCamera {i+1}:")
        print(f"Found in datasets: {sorted(cam_data['datasets'])}")
        print(f"Total occurrences: {cam_data['count']}")
        print(f"First seen in: {cam_data['subdir']} (frame {cam_data['frame_idx']})")
        print("R:")
        print(cam_data['R'])
        print("t:")
        print(cam_data['t'])

    return

if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()


    main(args)