import os 
import sys
import tqdm

MODEL_DIR = '/data1/gwangjin/face_landmarker.task'
IMAGE_DIR = '/data1/gwangjin/ffhq_256p'

import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision

import numpy as np

def result_to_blendshape(result):
    bs = result.face_blendshapes[0]

    [print(bs[i].category_name, bs[i].index, bs[i].score,) for i in range(len(bs))]

    # print(bs[0].index, bs[0].category_name)
    # blendshape_dict = {}
    # for i in range(len(bs)):
        # blendshape_dict[bs[i].category_name] = bs[i].index
    # print(blendshape_dict)
    # import pickle
    # with open('mediapipe_name_to_indices.pkl', 'wb') as f:
        # pickle.dump(blendshape_dict, f)
    # bshape = np.zeros(52)
    # print(bs)

    bshape = [bs[i].score for i in range(len(bs))]
    print(len(bshape))
    print([print(bshape[i], bs[i].score, bs[i].category_name, bs[i].index) for i in range(len(bshape))])
    return bshape

    bshape = {}
    for i in range(len(bs)):
        bshape[bs[i].category_name] = bs[i].score
        # print(bs[i].category_name, bs[i].score)
        # bshape[bs[i].index] = bs[i].score
    # bshape = [bs[i].score for i in range(len(bs))]
    return bshape

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_DIR),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1,
    output_face_blendshapes=True,
    )
    
img = sys.argv[1]

bshapes = []
data = {}
data['frames'] = []
with FaceLandmarker.create_from_options(options) as landmarker:
  # The landmarker is initialized. Use it here.
  # ...
    # for d in os.listdir(IMAGE_DIR):
        # if not os.path.isdir(os.path.join(IMAGE_DIR, d)):
            # continue
        # for fn in tqdm.tqdm(os.listdir(os.path.join(IMAGE_DIR, d))):
            
            # if not fn.endswith('.png'):
                # continue
    direc = os.path.join(img, 'image')
    
    for f in sorted(os.listdir(direc)):    
        if not (f.endswith('.png') or f.endswith('.jpg')):
            continue
        try:
            file_path = "./image/"+os.path.splitext(os.path.basename(f))[0]
            os.system(f'cp {os.path.join(img, "image", os.path.basename(f))} debug/input_image.png')
            mp_image = mp.Image.create_from_file(os.path.join(direc, f))
            face_landmarker_result = landmarker.detect(mp_image)
            # print(face_landmarker_result)
            # print(face_landmarker_result)

            bshape = result_to_blendshape(face_landmarker_result)
            # print(bshape)
            data['frames'].append({
                'file_path': file_path,
                'expression': bshape
            })
            exit()
            # if len(data['frames']) > 4:
            #     break
            # exit()
            # print(bshape)
            # with open(os.path.join(direc, os.path.splitext(f)[0]+'.txt'), 'w') as f:
                # for key in bshape:
                    # print(key, bshape[key])
                    # f.write(key + '\t\t\t' + '{:.4f}'.format(bshape[key]) + '\n')
                # f.write(str(bshape))
            # exit()
        except Exception as e:
            print(direc, f, e)
import json
with open(os.path.join(img, 'facs_params.json'), 'w') as f:
    json.dump(data, f)
            # if num_img > 10000:
            #         break
import numpy as np             


