import cv2
import torch
import numpy as np
# from skimage.measure import compare_ssim
from skimage.metrics import structural_similarity
from paths import ROOT_PATH


def cal_SSIM(path_BG, path_R):
    perturbation_BG = torch.load(path_BG, map_location='cpu')
    perturbation_R = torch.load(path_R, map_location='cpu')
    perturbation_shape = 64

    """读入基准图像"""
    base = cv2.imread('/home/yyshi/zhbli/projects/Universal-Targeted-Attacks-for-Siamese-Visual-Tracking/snapshots_imperceptible_patch/64/visualization/512/GOT-10k_Val/GOT-10k_Val_000001/2_clean_search_img.jpg')
    """读入基准图像"""

    img = base.copy()
    bgr = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
    bgr[:, :2, :perturbation_shape, :perturbation_shape] += perturbation_BG[0]  # BG, 补丁
    bgr[:, -1, :, :] += perturbation_R[0, 0]

    """张量格式的1CHW转为cv2格式的HWC"""
    bgr = bgr[0].permute(1, 2, 0).numpy().astype(np.uint8)  # 若为浮点数，则SSIM很小。
    """张量格式的1CHW转为cv2格式的HWC"""

    ssim = structural_similarity(bgr[:perturbation_shape, :perturbation_shape],
                                 base[:perturbation_shape, :perturbation_shape],
                                 multichannel=True)
    cv2.imwrite('/tmp/base.jpg', base)
    cv2.imwrite('/tmp/ssim.jpg', bgr)
    print(ssim)


def main():
    path_BG = '/home/yyshi/zhbli/projects/Universal-Targeted-Attacks-for-Siamese-Visual-Tracking/snapshots_imperceptible_patch/64/x_BG_512'
    path_R = '/home/yyshi/zhbli/projects/Universal-Targeted-Attacks-for-Siamese-Visual-Tracking/snapshots_imperceptible_patch/64/x_R_512'
    cal_SSIM(path_BG, path_R)


if __name__ == '__main__':
    main()
