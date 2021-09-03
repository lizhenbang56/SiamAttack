import cv2
import torch
import numpy as np
from skimage.metrics import structural_similarity


def cal_SSIM(path):
    perturbation = torch.load(path, map_location='cpu')
    perturbation_shape = perturbation.shape[-1]
    perturbation = perturbation[0].permute(1, 2, 0).numpy().astype(np.float64)

    """读入基准图像"""
    if perturbation_shape == 127:
        base = cv2.imread('/home/yyshi/zhbli/projects/Universal-Targeted-Attacks-for-Siamese-Visual-Tracking/snapshots_imperceptible_patch/64/visualization/256/GOT-10k_Val/GOT-10k_Val_000003/1_clean_template_img.jpg').astype(np.float64)
    elif perturbation_shape == 128:
        base = cv2.imread('/home/yyshi/zhbli/projects/Universal-Targeted-Attacks-for-Siamese-Visual-Tracking/snapshots_imperceptible_patch/64/visualization/256/GOT-10k_Val/GOT-10k_Val_000003/2_clean_search_img.jpg').astype(np.float64)
    """读入基准图像"""
    
    img = base.copy()
    img[:perturbation_shape, :perturbation_shape, :] += perturbation
    img[img>255] = 255
    img[img<0] = 0

    ssim = structural_similarity(img[:perturbation_shape, :perturbation_shape].astype(np.uint8), base[:perturbation_shape, :perturbation_shape].astype(np.uint8), multichannel=True)
    cv2.imwrite('/tmp/base.jpg', base)
    cv2.imwrite('/tmp/ssim.jpg', img)
    print(ssim)


def main():
    path = '/home/yyshi/zhbli/projects/Universal-Targeted-Attacks-for-Siamese-Visual-Tracking/snapshots_imperceptible_patch/128/x_1024'
    cal_SSIM(path)


if __name__ == '__main__':
    main()