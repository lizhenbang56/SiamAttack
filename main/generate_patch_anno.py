"""
现存的问题：可视化的结果是，补丁的宽高已知且固定。但实际上补丁的宽高收到缩放影响，因此在跟踪时才能确定。
版本改进20201222：因为我们训练了一个可以缩放的补丁，所以希望GT相对原始图像是尺寸固定的。这样可以避免补丁越来越大/小。
改进版本20201229：轨迹随机化
改进版本20210103：为其他数据集生成 FGT。
"""
import os
import random
import copy
import imageio
import numpy as np
from PIL import Image, ImageDraw

from videoanalyst.evaluation.got_benchmark.datasets import GOT10k, OTB
from videoanalyst.pipeline.utils import (xywh2cxywh, xywh2xyxy)


def get_crop(target_sz,
             z_size,
             context_amount=0.5):
    wc = target_sz[0] + context_amount * sum(target_sz)
    hc = target_sz[1] + context_amount * sum(target_sz)
    s_crop = np.sqrt(wc * hc)
    scale = z_size / s_crop
    return scale


def visualize(anno, patch_anno, resolution, video_index):
    img_list = []
    img = Image.new('RGB', resolution, (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for anno_, patch_anno_ in zip(anno, patch_anno):
        anno_cxy = xywh2cxywh(anno_)[:2]
        patch_anno_cxy = xywh2cxywh(patch_anno_)[:2]
        print(patch_anno_cxy)
        w=2
        draw.ellipse([tuple(anno_cxy - w), tuple(anno_cxy + w)], fill='green')
        draw.ellipse([tuple(patch_anno_cxy - w), tuple(patch_anno_cxy + w)], fill='red')
        img_list.append(copy.deepcopy(img))
    file_path = "/tmp/anno/{}.gif".format(video_index)
    print(file_path)
    imageio.mimsave(file_path, img_list, fps=20)
    return


def check_boundary(box_xywh, resolution):
    img_w, img_h = resolution
    w, h = box_xywh[2:]
    x1, y1, x2, y2 = xywh2xyxy(box_xywh)

    if x1 < 0 or y1 < 0 or x1 >= img_w or y1 >= img_h:
        return [img_w-w, img_h-h, w, h]
    else:
        return box_xywh


def get_init_patch_box_in_original_img(gt_xywh_in_original_img):
    """
    :param gt_xywh_in_original_img: xywh
    """
    """START：读取原始边框"""
    gt_x1_ori, gt_y1_ori, gt_w_ori, gt_h_ori = gt_xywh_in_original_img
    """END：读取原始边框"""

    """START：计算补丁在原始图像的边框"""
    patch_x1_ori = gt_x1_ori - gt_w_ori
    patch_y1_ori = gt_y1_ori - gt_h_ori
    patch_gt_xywh_ori = [patch_x1_ori, patch_y1_ori, gt_w_ori, gt_h_ori]
    """END：计算补丁在原始图像的边框"""

    return patch_gt_xywh_ori


def generate_random_trajectory(start_y, length, bound_y):
    y = start_y
    result = []
    for _ in range(length):
        if y < 0:
            y += random.randint(0, 32)
        elif y > bound_y:
            y += random.randint(-32, 0)
        else:
            y += random.randint(-32, 32)
        result.append(y)
    return np.array(result)


def run_per_video(anno, resolution):
    """
    resolution: wh
    return: [xywh], 每帧的 FGT
    """
    """START：确定在第一帧中的起点"""
    patch_gt_xywh_start_in_original_img = get_init_patch_box_in_original_img(anno[0])
    x1_start, y1_start, patch_w, patch_h = patch_gt_xywh_start_in_original_img
    x2_start = x1_start + patch_w
    y2_start = y1_start + patch_h
    """END：确定在第一帧中的起点"""

    """START：确定补丁与物体的相对位置关系"""
    im_w, im_h = resolution
    direction = None
    if patch_w >= patch_h and y1_start >= im_h - y2_start:
        direction = 'up'
    elif patch_w >= patch_h and y1_start < im_h - y2_start:
        direction = 'down'
    elif patch_w < patch_h and x1_start >= im_w - x2_start:
        direction = 'left'
    elif patch_w < patch_h and x1_start < im_w - x2_start:
        direction = 'right'
    else:
        assert patch_gt_xywh_start_in_original_img
    delta = 16
    """END：确定补丁与物体的相对位置关系"""

    patch_anno_x1 = []
    patch_anno_y1 = []
    patch_anno_w = np.linspace(patch_w, 64, len(anno))
    patch_anno_h = np.linspace(patch_h, 64, len(anno))
    for i, gt in enumerate(anno):
        x1, y1, w, h = gt
        x2 = x1 + w
        y2 = y1 + h
        if direction == 'up':
            fgt_x1y1 = [x1, y1-delta-patch_anno_h[i]]
        elif direction == 'down':
            fgt_x1y1 = [x1, y2+delta]
        elif direction == 'left':
            fgt_x1y1 = [x1-delta-patch_anno_w[i], y1]
        elif direction == 'right':
            fgt_x1y1 = [x2+delta, y1]
        else:
            assert False, direction
        assert fgt_x1y1 is not None
        patch_anno_x1.append(fgt_x1y1[0])
        patch_anno_y1.append(fgt_x1y1[1])
    patch_anno_x1 = np.array(patch_anno_x1)
    patch_anno_y1 = np.array(patch_anno_y1)

    patch_anno = np.stack((patch_anno_x1, patch_anno_y1, patch_anno_w, patch_anno_h)).T

    return np.array(patch_anno)


def generate_patch_anno(save_root, visualize_flag):
    """"""
    """START：生成指定数据库"""
    if dataset_name == 'GOT-10k_Val':
        dataset = GOT10k('/home/zhbli/Dataset/data2/got10k', subset='val', return_meta=False)
    elif dataset_name == 'OTB_2015':
        dataset = OTB('/home/etvuz/projects/adversarial_attack/video_analyst/datasets/OTB/OTB2015',
                      version=2015, download=False)
    else:
        assert False, dataset_name
    """END：生成指定数据库"""

    for s, (img_paths, anno) in enumerate(dataset):
        # anno: xywh
        resolution = Image.open(img_paths[0]).size  # wh
        patch_anno = run_per_video(anno, resolution)  # xywh

        """START：将补丁中心点保存为 txt"""
        save_dir = os.path.join(save_root, dataset_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        seq_name = dataset.seq_names[s]
        save_path = os.path.join(save_dir, seq_name + '.txt')
        print(save_path)
        np.savetxt(save_path, patch_anno, fmt='%d', delimiter=',')
        # 对 OTB2015 数据集 的 Jogging 视频进行特殊处理
        if seq_name == 'Jogging.1.txt':
            np.savetxt(os.path.join(save_dir, 'Jogging.txt'), patch_anno, fmt='%d', delimiter=',')
        if seq_name == 'Skating2.1.txt':
            np.savetxt(os.path.join(save_dir, 'Skating2.txt'), patch_anno, fmt='%d', delimiter=',')
        """END：将补丁中心点保存为 txt"""

        """START：可视化"""
        if visualize_flag:
            visualize(anno, patch_anno, resolution, s+1)
        """END：可视化"""


def test_get_crop():
    scale_x = get_crop(
        target_sz=(256, 256),  # 原始边框宽高
        z_size=127,
        context_amount=0.5,
    )
    return scale_x


if __name__ == '__main__':
    dataset_name = 'OTB_2015'
    generate_patch_anno(save_root='/home/etvuz/projects/adversarial_attack/patch_anno', visualize_flag=False)
    # test_get_crop()
