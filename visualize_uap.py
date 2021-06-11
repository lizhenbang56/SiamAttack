import os
import cv2
import torch
import numpy as np
# from skimage import measure


def visualize_uap(patch, name, root, num, phase):
    """
    patch: [batch, 3, 303, 303], torch.tensor
    """
    img = np.ascontiguousarray(patch[0].data.cpu().numpy().transpose(1, 2, 0))
    """START：计算 ssim"""
    if name == 'x':
        base_303 = 127 * np.ones((303, 303, 3))
        img_303 = base_303.copy()
        if phase == 'AP':
            img_303[:64,:64,:] = img
        elif phase == 'UAP':
            base_303 += img
        elif phase in ['64', 'FFT']:
            img_303[:64,:64,:] += img
        else:
            assert False, phase
        img_303 = np.clip(img_303, 0, 255).astype(np.uint8)
        # ssim = measure.compare_ssim(img_303, base_303, multichannel=True)
    else:
        base_img = 127.0 * np.ones_like(img)
        img = np.clip(img + base_img, 0, 255).astype(np.uint8)
        # ssim = measure.compare_ssim(img, base_img, multichannel=True)
    """END：计算 ssim"""

    # print('{}, ssim={:.2f}'.format(name, ssim))

    """START：保存可视化结果"""
    save_root = os.path.join(root, 'visualization', str(num), 'uap')
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    save_path = os.path.join(save_root, '{}.jpg'.format(name))
    print(save_path)
    if phase == 'UAP' and name == 'x':
        assert cv2.imwrite(save_path, img_303)
    else:
        assert cv2.imwrite(save_path, img)
    """END：保存可视化结果"""

    return 0


def vis_uap(root, num, phase):
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
    ssim_x = visualize_uap(x, 'x', root=root, num=num, phase=phase)
    ssim_z = visualize_uap(z, 'z', root=root, num=num, phase=phase)
    return ssim_z, ssim_x


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    root_ = 'snapshots_imperceptible_patch/64'
    num_ = 8192
    vis_uap(root_, num_, phase='64')
