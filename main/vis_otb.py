import os
import cv2
import glob
import torch
import numpy as np


def vis(image, box, save_dir, img_id, name, color):
    x1, y1, w, h = [int(p) for p in box]
    x2 = x1 + w
    y2 = y1 + h
    image = cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)
    save_path = os.path.join(save_dir, str(img_id) + '_' + name + '.jpg')
    cv2.imwrite(save_path, image)


def perturb_image(image, box, patch):
    x1, y1, w, h = [int(p) for p in box]
    x2 = min(x1 + w, image.shape[1])
    y2 = min(y1 + h, image.shape[0])
    patch = cv2.resize(patch, (x2-x1, y2-y1))
    image = image.astype(np.float32)
    image[y1:y2, x1:x2, :] += patch
    image[image > 255] = 255
    image[image < 0] = 0
    image = image.astype(np.uint8)
    return image


def main():
    # 设置保存路径
    save_dir = '/tmp/OTB2015_Singer2'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 读取补丁
    patch_path = '/home/etvuz/projects/adversarial_attack/video_analyst/snapshots_imperceptible_patch/64/x_8192'
    patch = torch.load(patch_path, map_location='cpu')[0].numpy().transpose((1, 2, 0))

    # 读取 GT
    GT_path = '/home/etvuz/projects/adversarial_attack/video_analyst/datasets/OTB/OTB2015/Singer2/groundtruth_rect.txt'
    GT = np.loadtxt(GT_path, delimiter=',')

    # 读取 FGT
    FGT_path = '/home/etvuz/projects/adversarial_attack/video_analyst/snapshots_imperceptible_patch/64/FGT/OTB_2015/siamfcpp_googlenet/8192/Singer2.txt'
    FGT = np.loadtxt(FGT_path, delimiter=',')

    # 读取预测结果
    PRED_path = '/home/etvuz/projects/adversarial_attack/video_analyst/snapshots_imperceptible_patch/64/result/OTB_2015/siamfcpp_googlenet/8192/Singer2.txt'
    PRED = np.loadtxt(PRED_path, delimiter=',')

    # 读取图像路径
    IMAGE_path = sorted(glob.glob('/home/etvuz/projects/adversarial_attack/video_analyst/datasets/OTB/OTB2015/Singer2/img/*.jpg'))

    # 初始化图像索引
    img_id = 1

    # 遍历每幅图像
    for image_path, gt, fgt, pred in zip(IMAGE_path, GT, FGT, PRED):
        # 读取每帧图像数据
        image = cv2.imread(image_path)

        # 可视化每帧 GT
        vis(image.copy(), gt, save_dir, img_id, 'GT', (0, 255, 0))

        # 可视化每帧 FGT
        vis(image.copy(), fgt, save_dir, img_id, 'FGT', (255, 255, 0))

        # 根据 FGT 粘贴扰动
        perturbed_image = perturb_image(image.copy(), fgt, patch)

        # 可视化预测结果
        vis(perturbed_image, pred, save_dir, img_id, 'PRED', (0, 0, 255))

        # 更新图像索引
        print(img_id)
        img_id += 1


if __name__ == '__main__':
    main()
