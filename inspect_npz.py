import numpy as np
import os

path = "assets/ict_mediapipe_landmark_embedding_from_metrical_tracker.npz"
print("Exists:", os.path.exists(path))
if os.path.exists(path):
    d = np.load(path, allow_pickle=True)
    print("Keys:", d.files)
    for k in d.files:
        val = d[k]
        if hasattr(val, 'shape'):
            print(f"  {k}: shape={val.shape}, dtype={val.dtype}")
            if len(val) > 0:
                print(f"    sample={val[:5]}")
        else:
            print(f"  {k}: {val}")
