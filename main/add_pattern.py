import cv2
import numpy as np


def main():
    pattern_image_path = 'main/basketball1_adobespark.png'
    pattern_image_alpha = cv2.imread(pattern_image_path, -1).astype(np.float64)
    pattern_image = cv2.imread(pattern_image_path).astype(np.float64)
    _, _, _, alpha = cv2.split(pattern_image_alpha)
    cv2.imwrite('alpha.jpg', alpha.astype(np.uint8))
    pattern_image[alpha == 0] = 0
    patch_shape = 128
    print('resize pattern image')
    pattern_image_resize = cv2.resize(pattern_image, (patch_shape, patch_shape))
    base_image_path = 'main/00000002.jpg'
    base_image = cv2.imread(base_image_path).astype(np.float64)
    print('base_image_loaded')
    merged_image = base_image.copy()
    x0 = 200
    y0 = 200
    merged_image[y0:y0+patch_shape, x0:x0+patch_shape] += pattern_image_resize / 10
    merged_image_uint8 = merged_image.astype(np.uint8)
    save_path = 'merged_image.jpg'
    assert cv2.imwrite(save_path, merged_image_uint8)
    print(save_path)


if __name__ == '__main__':
    print('start')
    main()
