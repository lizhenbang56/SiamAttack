import os
import glob
import time
import imageio
import numpy as np
from PIL import Image, ImageDraw

from videoanalyst.pipeline.utils import xywh2xyxy, xywh2cxywh
from videoanalyst.evaluation.got_benchmark.utils.metrics import rect_iou
from videoanalyst.evaluation.got_benchmark.experiments.got10k import ExperimentGOT10k


def visualize(pred, gt, fgt, video_name, dataset_dir, overwrite):
    img_list = []
    imgs = sorted(glob.glob(os.path.join(dataset_dir, '*.jpg')))

    """START：声明透明轨迹图像"""
    trajectory = Image.new('RGBA', Image.open(imgs[0]).size)
    draw_trajectory = ImageDraw.Draw(trajectory)
    """END：声明透明轨迹图像"""

    for pred_, gt_, fgt_, img_path in zip(pred, gt, fgt, imgs):
        img = Image.open(img_path)
        img = img.convert("RGBA")
        draw = ImageDraw.Draw(img)

        """START：中心点"""
        pred_cxy = xywh2cxywh(pred_)[:2]
        gt_cxy = xywh2cxywh(gt_)[:2]
        fgt_cxy = xywh2cxywh(fgt_)[:2]
        w = 8
        draw.ellipse([tuple(pred_cxy-w), tuple(pred_cxy+w)], fill='green')
        draw.ellipse([tuple(gt_cxy-w), tuple(gt_cxy+w)], fill='red')
        draw.ellipse([tuple(fgt_cxy-w), tuple(fgt_cxy+w)], fill='yellow')
        """END：中心点"""

        """START：可视化矩形框"""
        pred_xyxy = xywh2xyxy(pred_)
        gt_xyxy = xywh2xyxy(gt_)
        fgt_xyxy = xywh2xyxy(fgt_)
        draw.rectangle(pred_xyxy, fill=None, outline='green', width=4)
        draw.rectangle(gt_xyxy, fill=None, outline='red', width=4)
        draw.rectangle(fgt_xyxy, fill=None, outline='yellow', width=4)
        """END：可视化矩形框"""

        """START：更新轨迹图像"""
        w = 2
        draw_trajectory.ellipse([tuple(pred_cxy - w), tuple(pred_cxy + w)], fill='green')
        draw_trajectory.ellipse([tuple(gt_cxy - w), tuple(gt_cxy + w)], fill='red')
        draw_trajectory.ellipse([tuple(fgt_cxy - w), tuple(fgt_cxy + w)], fill='yellow')
        """END：更新轨迹图像"""

        """START：图像融合"""
        img = Image.alpha_composite(img, trajectory)
        """END：图像融合"""

        # img = img.resize((200, 150))
        img_list.append(img)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
    file_path = os.path.join(save_dir, video_name + '_{}'.format(timestamp) + '.gif')
    if os.path.isfile(file_path) and not overwrite:
        print('Already exists:', end=' ')
    else:
        img_list[0].save(file_path, save_all=True, append_images=img_list[1:], quality=50)
    print(file_path)
    return


def cal_ao(gt, pred):
    seq_ious = [rect_iou(pred[1:], gt[1:])]
    ao, sr, speed, _ = got10k_tool._evaluate(ious=np.concatenate(seq_ious), times=[])
    return ao


def run_per_video(video_name, gt, overwrite):
    phase = video_name.split('_')[1].lower()
    dataset_dir = os.path.join(dataset_root, phase, video_name)
    pred_txt_path = os.path.join(pred_dir, video_name, video_name + "_001.txt")
    fgt_txt_path = os.path.join(fgt_root, video_name + '.txt')
    pred = np.loadtxt(pred_txt_path, delimiter=',')
    fgt = np.loadtxt(fgt_txt_path, delimiter=',')
    visualize(pred, gt, fgt, video_name, dataset_dir, overwrite)


def visualize_txt_result(overwrite):
    video_names = sorted(os.listdir(pred_dir))
    for video_name, gt in zip(video_names, got10k_tool.dataset):
        run_per_video(video_name, gt[1], overwrite)


if __name__ == '__main__':
    root = '/home/etvuz/projects/adversarial_attack/'
    dataset_root = os.path.join(root, 'video_analyst/datasets/GOT-10k')
    pred_dir = os.path.join(root, 'video_analyst/logs/GOT-Benchmark/result/GOT-10k/siamfcpp_googlenet')
    save_dir = '/tmp/uap_vis_txt'
    fgt_root = os.path.join(root, 'patch_anno')
    got10k_tool = ExperimentGOT10k(dataset_root, subset='val')
    visualize_txt_result(overwrite=True)
