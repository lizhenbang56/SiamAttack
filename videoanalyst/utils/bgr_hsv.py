import cv2
import torch
import numpy as np
from torch import nn


class BGR_HSV(nn.Module):
    """
    Pytorch implementation of BGR convert to HSV, and HSV convert to BGR,
    BGR or HSV's shape: (B * C * H * W)
    HSV's range: [0, 1)
    BGR's range: [0, 255]
    """
    def __init__(self, eps=1e-8):
        super(BGR_HSV, self).__init__()
        self.eps = eps

    def rgb_to_hsv(self, img):
        """将0~255的RGB转为0~1的RGB"""
        img = img / 255
        """将0~255的RGB转为0~1的RGB"""

        hue = torch.Tensor(img.shape[0], img.shape[2],
                           img.shape[3]).to(img.device)

        hue[img[:, 2] == img.max(1)[0]] = 4.0 + ((img[:, 0]-img[:, 1]) / (
            img.max(1)[0] - img.min(1)[0] + self.eps))[img[:, 2] == img.max(1)[0]]
        hue[img[:, 1] == img.max(1)[0]] = 2.0 + ((img[:, 2]-img[:, 0]) / (
            img.max(1)[0] - img.min(1)[0] + self.eps))[img[:, 1] == img.max(1)[0]]
        hue[img[:, 0] == img.max(1)[0]] = (0.0 + ((img[:, 1]-img[:, 2]) / (
            img.max(1)[0] - img.min(1)[0] + self.eps))[img[:, 0] == img.max(1)[0]]) % 6

        hue[img.min(1)[0] == img.max(1)[0]] = 0.0
        hue = hue/6

        saturation = (img.max(1)[0] - img.min(1)[0]) / \
            (img.max(1)[0] + self.eps)
        saturation[img.max(1)[0] == 0] = 0

        value = img.max(1)[0]

        hue = hue.unsqueeze(1)
        saturation = saturation.unsqueeze(1)
        value = value.unsqueeze(1)
        hsv = torch.cat([hue, saturation, value], dim=1)
        return hsv

    def hsv_to_bgr(self, hsv):
        """
        return
            bgr: 0~255
        """
        h, s, v = hsv[:, 0, :, :], hsv[:, 1, :, :], hsv[:, 2, :, :]
        # 对出界值的处理
        h = h % 1
        s = torch.clamp(s, 0, 1)
        v = torch.clamp(v, 0, 1)

        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)

        hi = torch.floor(h * 6)
        f = h * 6 - hi
        p = v * (1 - s)
        q = v * (1 - (f * s))
        t = v * (1 - ((1 - f) * s))

        hi0 = hi == 0
        hi1 = hi == 1
        hi2 = hi == 2
        hi3 = hi == 3
        hi4 = hi == 4
        hi5 = hi == 5

        r[hi0] = v[hi0]
        g[hi0] = t[hi0]
        b[hi0] = p[hi0]

        r[hi1] = q[hi1]
        g[hi1] = v[hi1]
        b[hi1] = p[hi1]

        r[hi2] = p[hi2]
        g[hi2] = v[hi2]
        b[hi2] = t[hi2]

        r[hi3] = p[hi3]
        g[hi3] = q[hi3]
        b[hi3] = v[hi3]

        r[hi4] = t[hi4]
        g[hi4] = p[hi4]
        b[hi4] = v[hi4]

        r[hi5] = v[hi5]
        g[hi5] = p[hi5]
        b[hi5] = q[hi5]

        r = r.unsqueeze(1)
        g = g.unsqueeze(1)
        b = b.unsqueeze(1)
        bgr = torch.cat([b, g, r], dim=1)
        bgr = bgr * 255
        return bgr


if __name__ == '__main__':
    convertor = BGR_HSV()
    bgr = cv2.imread('/home/etvuz/projects/adversarial_attack/video_analyst/snapshots_imperceptible_patch/64/visualization/512/GOT-10k_Val/GOT-10k_Val_000001/56_adv_search_img.jpg')
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    """cv2格式的HWC转为张量格式的1CHW"""
    rgb = torch.from_numpy(rgb.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
    """cv2格式的HWC转为张量格式的1CHW"""

    hsv = convertor.rgb_to_hsv(rgb)
    bgr1 = convertor.hsv_to_bgr(hsv)

    """张量格式的1CHW转为cv2格式的HWC"""
    bgr1 = bgr1[0].permute(1, 2, 0).numpy()
    """张量格式的1CHW转为cv2格式的HWC"""

    cv2.imwrite('/tmp/bgr1.jpg', bgr1)