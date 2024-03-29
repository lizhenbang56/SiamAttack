import os
import cv2
import torch
import numpy as np
from skimage.measure import compare_ssim


def visualize_uap(patch, name):
    """
    patch: [batch, 3, 303, 303], torch.tensor
    """
    img = np.ascontiguousarray(patch[0].data.cpu().numpy().transpose(1, 2, 0))
    if name == 'x':
        base_img = np.ones_like(img)
    elif name == 'z':
        base_img = 127.0 * np.ones_like(img)
    else:
        assert False, name
    img = np.clip(img + base_img, 0, 255).astype(np.uint8)

    """START：计算 mse"""
    mse = np.mean((img - base_img) ** 2)
    """END：计算 mse"""

    """START：计算 ssim"""
    ssim = compare_ssim(img, base_img, multichannel=True)
    """END：计算 ssim"""

    print('{}: mse={:.2f}, ssim={:.2f}'.format(name, mse, ssim))

    """START：保存可视化结果"""
    save_root = os.path.join(root, 'visualization', str(num), 'uap')
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    save_path = os.path.join(save_root, '{}.jpg'.format(name))
    assert cv2.imwrite(save_path, img)
    """END：保存可视化结果"""

    return


def main():
    x_path = os.path.join(os.path.join(root, 'x_{}'.format(num)))
    z_path = os.path.join(os.path.join(root, 'z_{}'.format(num)))
    print(x_path, z_path)
    x = torch.load(x_path)
    z = torch.load(z_path)
    min_z = torch.min(z).item()
    max_z = torch.max(z).item()
    min_x = torch.min(x).item()
    max_x = torch.max(x).item()
    print('min_z={:.1f}, max_z={:.1f}, min_x={:.1f}, max_x={:.1f}'.format(min_z, max_z, min_x, max_x))
    visualize_uap(x, 'x')
    visualize_uap(z, 'z')
    return


if __name__ == '__main__':
    root = '/home/etvuz/projects/adversarial_attack/video_analyst/snapshots/train_set=fulldata_FGSM_cls=1_ctr=1_reg=1_l2_z=0.005_l2_x=1e-05_lr_z=0.1_lr_x=0.5'
    num = 32768
    main()
