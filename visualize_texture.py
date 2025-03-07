import cv2
import os

def main():

    low_res_img = './debug/texture_map/marcel_kd.png'
    high_res_img = './debug/texture_map/marcel_kd_HAT_SRx4_ImageNet-pretrain.png'

    # upsample both to 4096, 4096 with nearest neighbor

    low_res = cv2.imread(low_res_img)
    high_res = cv2.imread(high_res_img)

    low_res = cv2.resize(low_res, (4096, 4096), interpolation=cv2.INTER_NEAREST)
    high_res = cv2.resize(high_res, (4096, 4096), interpolation=cv2.INTER_NEAREST)

    # save as png

    cv2.imwrite('./debug/texture_map/low_res.png', low_res)
    cv2.imwrite('./debug/texture_map/high_res.png', high_res)

    # crop at 1000, 600 with size of 200, 200
    x = 1000
    y = 600

    size = 800

    low_res_crop = low_res[y:y+size, x:x+size]
    high_res_crop = high_res[y:y+size, x:x+size]

    cv2.imwrite('./debug/texture_map/low_res_crop.png', low_res_crop)
    cv2.imwrite('./debug/texture_map/high_res_crop.png', high_res_crop)

    return



if __name__ == '__main__':
    main()
