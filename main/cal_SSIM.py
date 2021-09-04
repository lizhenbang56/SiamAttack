import cv2
import torch
import numpy as np
# from skimage.measure import compare_ssim
from skimage.metrics import structural_similarity
from paths import ROOT_PATH
from videoanalyst.utils.bgr_ycbcr import bgr2ycbcr_pytorch, ycbcr2bgr_pytorch


def cal_SSIM(path_CbCr, path_Y):
    perturbation_CbCr = torch.load(path_CbCr, map_location='cpu')
    perturbation_Y = torch.load(path_Y, map_location='cpu')
    perturbation_shape = 64

    """读入基准图像"""
    base = cv2.imread('/home/yyshi/zhbli/projects/Universal-Targeted-Attacks-for-Siamese-Visual-Tracking/snapshots_imperceptible_patch/64/visualization/512/GOT-10k_Val/GOT-10k_Val_000001/2_clean_search_img.jpg')
    """读入基准图像"""

    img = base.copy()
    bgr = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
    ycbcr = bgr2ycbcr_pytorch(bgr)
    delta_h = 0
    delta_w = 0
    ycbcr[:, 0, :, :] += perturbation_Y[0, 0]
    ycbcr[:, 1:, delta_h:delta_h+perturbation_shape, delta_w:delta_w+perturbation_shape] += perturbation_CbCr[0]
    bgr1 = ycbcr2bgr_pytorch(ycbcr)

    """张量格式的1CHW转为cv2格式的HWC"""
    bgr1 = bgr1[0].permute(1, 2, 0).numpy().astype(np.uint8)  # 若为浮点数，则SSIM很小。
    """张量格式的1CHW转为cv2格式的HWC"""

    ssim = structural_similarity(bgr1[delta_h:delta_h+perturbation_shape, delta_w:delta_w+perturbation_shape], 
                                 base[delta_h:delta_h+perturbation_shape, delta_w:delta_w+perturbation_shape], 
                                 multichannel=True)
    cv2.imwrite('/tmp/base.jpg', base)
    cv2.imwrite('/tmp/ssim.jpg', bgr1)
    print(ssim)


def main():
    iteration_num = 4096
    path_CbCr = '/home/yyshi/zhbli/projects/Universal-Targeted-Attacks-for-Siamese-Visual-Tracking/snapshots_imperceptible_patch/64/x_{}_CbCr'.format(iteration_num)
    path_Y = '/home/yyshi/zhbli/projects/Universal-Targeted-Attacks-for-Siamese-Visual-Tracking/snapshots_imperceptible_patch/64/x_{}_Y'.format(iteration_num)
    cal_SSIM(path_CbCr, path_Y)


if __name__ == '__main__':
    main()
