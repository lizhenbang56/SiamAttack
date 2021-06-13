import cv2
from numpy.fft import ifft
import torch
import numpy as np


def generate_gaussian(size):
    nrows = ncols = size
    sigmax, sigmay = 20, 20  # 高斯函数的标准差，越小越模糊。已验证。
    cy, cx = nrows/2, ncols/2
    x = np.linspace(0, nrows, nrows)
    y = np.linspace(0, ncols, ncols)
    X, Y = np.meshgrid(x, y)
    gmask = np.exp(-(((X-cx)/sigmax)**2 + ((Y-cy)/sigmay)**2))
    return gmask
    

def load_img(video_name):
    img_path = '/home/yyshi/zhbli/projects/Universal-Targeted-Attacks-for-Siamese-Visual-Tracking/snapshots_imperceptible_patch/FFT/visualization/512/GOT-10k_Val/{}/1_clean_template_img.jpg'.format(video_name)
    return cv2.imread(img_path).astype(np.float32)


def load_perturbation_fft():
    perturbation_fft_path = '/home/yyshi/zhbli/projects/Universal-Targeted-Attacks-for-Siamese-Visual-Tracking/snapshots_imperceptible_patch/FFT/z_512'
    perturbation_fft_tensor = torch.load(perturbation_fft_path, map_location='cpu')[0]
    perturbation_fft_array = perturbation_fft_tensor.data.numpy().transpose((1,2,0))
    return perturbation_fft_array


def main():
    img = load_img('GOT-10k_Val_000158')
    perturbation_fft = load_perturbation_fft()  # 复数
    filter = generate_gaussian(127)
    for color_channel in range(3):
        perturbation_fft[:,:,color_channel] = perturbation_fft[:,:,color_channel] * filter
        perturbation_fft[:,:,color_channel] = np.fft.ifftshift(perturbation_fft[:,:,color_channel])
        perturbation_fft[:,:,color_channel] = np.fft.ifft2(perturbation_fft[:,:,color_channel])
        img[:,:,color_channel] = img[:,:,color_channel] + perturbation_fft[:,:,color_channel].astype(np.float32) * 10
    adv_template_img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite('/tmp/0_vis_trained_fft.jpg', adv_template_img)


if __name__ == '__main__':
    main()