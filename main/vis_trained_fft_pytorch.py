import numpy as np
import cv2
import os
import torch
from paths import ROOT_PATH
from videoanalyst.pipeline.tracker_impl.siamfcpp_track import generate_gaussian
from videoanalyst.engine.trainer.trainer_impl.regular_trainer import restrict_tensor



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


def generate_gaussian_tensor(size):
    mask_np = generate_gaussian(size)
    return torch.from_numpy(mask_np)


def load_img_tensor(video_name, img_name):
    img_path = '/home/yyshi/zhbli/projects/Universal-Targeted-Attacks-for-Siamese-Visual-Tracking/datasets/GOT-10k/val/GOT-10k_Val_000003/00000001.jpg'
    img_np = cv2.imread(img_path)
    img_tensor = torch.from_numpy(img_np.transpose(2,0,1)).unsqueeze(0).to(torch.float32)
    return img_tensor


def load_perturbation_fft_tensor(x_or_z, loop_num):
    perturbation_fft_path = '/home/yyshi/zhbli/projects/Universal-Targeted-Attacks-for-Siamese-Visual-Tracking/snapshots_imperceptible_patch/FFT/{}_{}'.format(x_or_z, loop_num)
    perturbation_fft_tensor = torch.load(perturbation_fft_path, map_location='cpu')
    return perturbation_fft_tensor


def save_fft(fft, name):
    log_fft = np.log(1 + np.abs(fft)) * 20
    cv2.imwrite('/tmp/0_{}.jpg'.format(name), log_fft.astype(np.uint8))


def vis_fft(filter, fft):
    """可视化频域图
    data: 2d tensor
    """
    filter = filter.numpy()
    fft2 = fft.cpu().data.numpy()
    shift2center = np.fft.fftshift(fft2)
    save_fft(shift2center, 'shift')
    masked_fft = filter * shift2center
    save_fft(masked_fft, 'masked_fft')
    assert cv2.imwrite('/tmp/0_mask.jpg', (filter*255).astype(np.uint8))
    return


def main():
    img_tensor = load_img_tensor(video_name='GOT-10k_Val_000001', img_name='2_clean_search_img')
    perturbation_fft_tensor = load_perturbation_fft_tensor(x_or_z='x', loop_num=512)
    visualize_template_img(perturbation_fft_tensor+128, save_name='raw_perturbation')
    filter = generate_gaussian_tensor(64)
    for color_channel in range(3):
        perturbed_z_one_channel = torch.fft.fft2(perturbation_fft_tensor[0, color_channel, :, :])
        vis_fft(filter, perturbed_z_one_channel)
        perturbed_z_one_channel_mask = restrict_tensor(torch.fft.ifft2(torch.fft.ifftshift(filter *     perturbed_z_one_channel)).to(torch.float32))
        no_mask =                      restrict_tensor(torch.fft.ifft2(torch.fft.ifftshift(             perturbed_z_one_channel)).to(torch.float32))
        sub_mask =                     restrict_tensor(torch.fft.ifft2(torch.fft.ifftshift((1-filter) * perturbed_z_one_channel)).to(torch.float32))
        visualize_template_img(perturbed_z_one_channel_mask+128, 'masked_perturbation')
        visualize_template_img(no_mask+128,            save_name='no_mask')
        visualize_template_img(sub_mask+128,           save_name='sub_mask')
        img_tensor[:, color_channel, 64:128, 64:128] = perturbed_z_one_channel_mask.unsqueeze(0) + img_tensor[:, color_channel, 64:128, 64:128].data
    visualize_template_img(img_tensor, 'final')


if __name__ == '__main__':
    main()