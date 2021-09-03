import numpy as np
import cv2
import os
import torch
from paths import ROOT_PATH
from videoanalyst.engine.trainer.trainer_impl.regular_trainer import restrict_tensor, generate_perturbation_x, apply_perturbation



def visualize_template_img(adv_template_img, save_name):
    if len(adv_template_img.shape) == 2:
        adv_template_img = np.ascontiguousarray(adv_template_img.cpu().numpy())
    else:
        adv_template_img = np.ascontiguousarray(adv_template_img[0].cpu().numpy().transpose(1, 2, 0))
    adv_template_img = np.clip(adv_template_img, 0, 255).astype(np.uint8)
    save_path = '/tmp/0_{}.jpg'.format(save_name)
    assert cv2.imwrite(save_path, adv_template_img)
    print('img saved to: ', save_path)
    return


def load_img_tensor(video_name, img_name):
    """
    return
        img_tensor: BCHW
    """
    img_path = '/home/yyshi/zhbli/projects/Universal-Targeted-Attacks-for-Siamese-Visual-Tracking/datasets/GOT-10k/val/GOT-10k_Val_000003/00000001.jpg'
    img_np = cv2.imread(img_path)
    img_tensor = torch.from_numpy(img_np.transpose(2,0,1)).unsqueeze(0).to(torch.float32)
    return img_tensor


def load_perturbation_fft_tensor(x_or_z, loop_num):
    perturbation_fft_path = '/home/yyshi/zhbli/projects/Universal-Targeted-Attacks-for-Siamese-Visual-Tracking/snapshots_imperceptible_patch/FFT/{}_{}'.format(x_or_z, loop_num)
    perturbation_fft_tensor = torch.load(perturbation_fft_path, map_location='cpu')
    return perturbation_fft_tensor



def main():
    img_tensor = load_img_tensor(video_name='GOT-10k_Val_000001', img_name='2_clean_search_img')
    perturbation_fft_tensor = load_perturbation_fft_tensor(x_or_z='x', loop_num=256)
    for color_channel in range(3):
        dtype = img_tensor.dtype
        perturbed_x_one_channel_mask = generate_perturbation_x(perturbation_fft_tensor, None, color_channel, dtype, None, None, None, None, requires_grad=False)
        img_tensor[0, color_channel, :, :] = apply_perturbation(img_tensor[0, color_channel, :, :], perturbed_x_one_channel_mask, 1, 1, 64, 64)
    visualize_template_img(img_tensor, 'final')


if __name__ == '__main__':
    main()