import cv2
import torch
import numpy as np
# from skimage.measure import compare_ssim
from skimage.metrics import structural_similarity
from paths import ROOT_PATH
from videoanalyst.utils.bgr_ycbcr import bgr2ycbcr_pytorch, ycbcr2bgr_pytorch


def cal_SSIM(path_z):
    perturbation = torch.load(path_z, map_location='cpu')

    """读入基准图像"""
    base = 128 * np.ones((127, 127, 3), dtype=np.float64)
    """读入基准图像"""

    img = base.copy()
    bgr = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
    ycbcr = bgr2ycbcr_pytorch(bgr)
    ycbcr += perturbation
    bgr1 = ycbcr2bgr_pytorch(ycbcr)

    """张量格式的1CHW转为cv2格式的HWC"""
    bgr1 = bgr1[0].permute(1, 2, 0).numpy().astype(np.uint8)  # 若为浮点数，则SSIM很小。
    """张量格式的1CHW转为cv2格式的HWC"""

    ssim = structural_similarity(bgr1, base, multichannel=True)
    cv2.imwrite('/tmp/base.jpg', base)
    cv2.imwrite('/tmp/ssim.jpg', bgr1)
    print(ssim)


def main():
    path_z = '/home/yyshi/zhbli/projects/Universal-Targeted-Attacks-for-Siamese-Visual-Tracking/snapshots_imperceptible_patch/64/z_512'
    cal_SSIM(path_z)


if __name__ == '__main__':
    main()
