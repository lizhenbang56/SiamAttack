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
    perturbation_shape = perturbation.shape[-1]

    """读入基准图像"""
    # base = 128 * np.ones((127, 127, 3)).astype(np.uint8)
    if perturbation_shape == 127:
        base = cv2.imread('/home/yyshi/zhbli/projects/Universal-Targeted-Attacks-for-Siamese-Visual-Tracking/snapshots_imperceptible_patch/64/visualization/256/GOT-10k_Val/GOT-10k_Val_000001/1_clean_template_img.jpg')
    elif perturbation_shape == 64:
        base = cv2.imread('/home/yyshi/zhbli/projects/Universal-Targeted-Attacks-for-Siamese-Visual-Tracking/snapshots_imperceptible_patch/64/visualization/256/GOT-10k_Val/GOT-10k_Val_000001/2_clean_search_img.jpg')
    """读入基准图像"""
    
    img = base.copy()

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb = torch.from_numpy(rgb.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
    hsv = convertor.rgb_to_hsv(rgb)
    hsv[:, :, :perturbation_shape, :perturbation_shape] += perturbation
    bgr1 = convertor.hsv_to_bgr(hsv)

    """张量格式的1CHW转为cv2格式的HWC"""
    bgr1 = bgr1[0].permute(1, 2, 0).numpy().astype(np.uint8)  # 若为浮点数，则SSIM很小。
    """张量格式的1CHW转为cv2格式的HWC"""

    ssim = structural_similarity(bgr1[:perturbation_shape, :perturbation_shape], base[:perturbation_shape, :perturbation_shape], multichannel=True)
    cv2.imwrite('/tmp/base.jpg', base)
    cv2.imwrite('/tmp/ssim.jpg', bgr1)
    print(ssim)


def main():
    path = '/home/yyshi/zhbli/projects/Universal-Targeted-Attacks-for-Siamese-Visual-Tracking/snapshots_imperceptible_patch/64/x_1024'
    cal_SSIM(path)


if __name__ == '__main__':
    main()