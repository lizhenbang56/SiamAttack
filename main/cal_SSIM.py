import cv2
import torch
import numpy as np
# from skimage.measure import compare_ssim
from skimage.metrics import structural_similarity
from paths import ROOT_PATH
from videoanalyst.utils.bgr_hsv import BGR_HSV

def cal_SSIM(path):
    convertor = BGR_HSV()
    perturbation = torch.load(path, map_location='cpu')
    # perturbation = np.ascontiguousarray(perturbation[0].data.cpu().numpy().transpose(1, 2, 0))
    base = 127 * np.ones((64, 64, 3)).astype(np.uint8)
    img = base.copy()

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb = torch.from_numpy(rgb.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
    hsv = convertor.rgb_to_hsv(rgb)
    hsv = hsv + perturbation
    bgr1 = convertor.hsv_to_bgr(hsv)

    """张量格式的1CHW转为cv2格式的HWC"""
    bgr1 = bgr1[0].permute(1, 2, 0).numpy()
    """张量格式的1CHW转为cv2格式的HWC"""

    # ssim = compare_ssim(bgr1, base, multichannel=True)
    ssim = structural_similarity(bgr1, base, multichannel=True)
    cv2.imwrite('/tmp/ssim.jpg', bgr1)
    print(ssim)


def main():
    x_path = '/home/yyshi/zhbli/projects/Universal-Targeted-Attacks-for-Siamese-Visual-Tracking/snapshots_imperceptible_patch/64/x_128'
    cal_SSIM(x_path)


if __name__ == '__main__':
    main()