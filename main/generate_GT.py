import os
import cv2
import sys
import random
import numpy as np
from numpy.core.function_base import linspace
from paths import ROOT_PATH


def run_per_video(video_name, old_GT_dir, save_root):
    old_GT_path = os.path.join(old_GT_dir, video_name, 'groundtruth.txt')
    GT = np.loadtxt(old_GT_path, delimiter=',')  # xywh
    frame_num = len(GT)
    x, y, w, h = GT[0]
    img = cv2.imread(os.path.join(old_GT_dir, video_name, '00000001.jpg'))
    H, W, C = img.shape

    """设定终点"""
    # 若高大于宽，则往left或right方向移动
    if h >= w:
        directions = ['left', 'right']
    else:
        directions = ['top', 'bottom']

    direction = random.choice(directions)
    if direction == 'top':  # y = 0
        y_new = 0
        x_new = random.randint(0, W - w)
    elif direction == 'bottom':  # y = H - h
        y_new = H - h
        x_new = random.randint(0, W - w)
    elif direction == 'left':  # x = 0
        x_new = 0
        y_new = random.randint(0, H - h)
    elif direction == 'right':  # x = W - w
        x_new = W - w
        y_new = random.randint(0, H - h)
    else:
        assert False
    """设定终点"""

    """插值"""
    xs = np.linspace(x, x_new, num=frame_num)
    ys = np.linspace(y, y_new, num=frame_num)
    ws = np.linspace(w, w, num=frame_num)  # 生成的边框宽和高每帧都是不变的，与初始帧相同。
    hs = np.linspace(h, h, num=frame_num)
    new_GT = np.stack([xs, ys, ws, hs], axis=1)
    """插值"""

    """保存txt"""
    save_dir = os.path.join(save_root, video_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'groundtruth.txt')
    np.savetxt(save_path, new_GT, delimiter=',', fmt='%.4f')
    print(save_path)
    """保存txt"""


def main():
    """定义路径"""
    old_GT_dir = os.path.join(sys.path[0], 'datasets/GOT-10k/val')
    save_dir = os.path.join(sys.path[0], 'datasets/GOT-10k-val-GT')
    """定义路径"""

    video_names = open(os.path.join(old_GT_dir, 'list.txt'), 'r').read().splitlines()

    for video_name in video_names:
        run_per_video(video_name, old_GT_dir, save_dir)


if __name__ == '__main__':
    main()
