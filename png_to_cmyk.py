from PIL import Image
import os

import argparse



def png_to_cmyk(input_file):
    if os.path.isdir(input_file):
        for file in os.listdir(input_file):
            if file.endswith('.png'):
                png_to_cmyk(os.path.join(input_file, file))
            if os.path.isdir(os.path.join(input_file, file)):
                png_to_cmyk(os.path.join(input_file, file))
        return
    else:
        if not input_file.endswith('.png'):
            return
        output_file = input_file.replace('.png', '.jpg')

    # first, read with opencv 
    # if alpha channel exists, remove it with white background
    import cv2
    import numpy as np
    image = cv2.imread(input_file, cv2.IMREAD_UNCHANGED)
    if image.shape[2] == 4:
        alpha_map = image[:, :, 3]
        image = image[:, :, :3]
        background = np.ones_like(image) * 255

        image = alpha_map * image + (1 - alpha_map) * background

    # convert to rgb
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #convert to Image object
    image = Image.fromarray(image)





    cmyk_image = image.convert('CMYK')

    # JPG로 저장 (CMYK 컬러스페이스로 지정됨)
    cmyk_image.save(output_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()  
    parser.add_argument('--input_file', '-i', type=str, required=True)

    args = parser.parse_args()

    png_to_cmyk(args.input_file)