import os
import glob
import numpy as np
from PIL import Image, ImageDraw

from videoanalyst.pipeline.utils import xywh2xyxy, xywh2cxywh
from videoanalyst.evaluation.got_benchmark.utils.metrics import rect_iou
from videoanalyst.evaluation.got_benchmark.experiments.got10k import ExperimentGOT10k
from videoanalyst.evaluation.got_benchmark.experiments.otb import ExperimentOTB


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
        if img.size != trajectory.size:
            print('Wrong image size', img_path, img.size, trajectory.size)
            img = img.resize(trajectory.size)
        img = Image.alpha_composite(img, trajectory)
        """END：图像融合"""

        img_list.append(img)

    """START：保存"""
    save_dir = None
    if save_mode == '.png':
        save_dir = os.path.join(save_root, video_name)
    elif save_mode == '.gif':
        save_dir = save_root
    else:
        assert False, save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(save_dir)
    if save_mode == '.gif':
        file_path = os.path.join(save_dir, video_name + save_mode)
        if os.path.isfile(file_path) and not overwrite:
            print('Already exists:', end=' ')
        img_list[0].save(file_path, save_all=True, append_images=img_list[1:], quality=50)
    elif save_mode == '.png':
        for i, img in enumerate(img_list):
            save_path = os.path.join(save_dir, str(i) + save_mode)
            img.save(save_path)
    """END：保存"""

    return


def cal_ao(gt, pred):
    seq_ious = [rect_iou(pred[1:], gt[1:])]
    ao, sr, speed, _ = dataset_tool._evaluate(ious=np.concatenate(seq_ious), times=[])
    return ao


def run_per_video(video_name, gt, overwrite):
    if dataset_name == 'OTB_2015':
        video_name = video_name.split('.')[0]
        dataset_dir = os.path.join(dataset_root, video_name, 'img')
        pred_txt_path = os.path.join(pred_dir, video_name + '.txt')
    elif dataset_name == 'GOT10k_Val':
        phase = video_name.split('_')[1].lower()
        dataset_dir = os.path.join(dataset_root, phase, video_name)
        pred_txt_path = os.path.join(pred_dir, video_name, video_name + "_001.txt")
    else:
        assert False, dataset_name
    fgt_txt_path = os.path.join(fgt_root, video_name + '.txt')
    pred = np.loadtxt(pred_txt_path, delimiter=',')
    fgt = np.loadtxt(fgt_txt_path, delimiter=',')
    visualize(pred, gt, fgt, video_name, dataset_dir, overwrite)


def visualize_txt_result(overwrite):
    video_names = sorted(os.listdir(pred_dir))
    for video_name, gt in zip(video_names, dataset_tool.dataset):
        run_per_video(video_name, gt[1], overwrite)


if __name__ == '__main__':
    """START：设定数据集名称"""
    dataset_name = 'OTB_2015'
    print('Dataset Name:', dataset_name)
    save_mode = '.gif'  # .png or .gif
    """END：设定数据集名称"""

    root = '/home/etvuz/projects/adversarial_attack/'
    if dataset_name == 'OTB_2015':
        dataset_root = os.path.join(root, 'video_analyst/datasets/OTB/OTB2015')
        dataset_tool = ExperimentOTB(dataset_root, version=2015)
        pred_dir = os.path.join(root, 'video_analyst/logs/GOT-Benchmark/result/otb2015/siamfcpp_googlenet')
    elif dataset_name == 'GOT-10k_Val':
        dataset_root = os.path.join(root, 'video_analyst/datasets/GOT-10k')
        dataset_tool = ExperimentGOT10k(dataset_root, subset='val')
        pred_dir = os.path.join(root, 'video_analyst/snapshots/train_set=fulldata_FGSM_cls=1_ctr=1_reg=1_l2_z=0.005_l2_x=1e-05_lr_z=0.1_lr_x=0.5/result/32768')
    else:
        assert False, dataset_name
    save_root = os.path.join(
        root,
        'video_analyst/snapshots/train_set=fulldata_FGSM_cls=1_ctr=1_reg=1_l2_z=0.005_l2_x=1e-05_lr_z=0.1_lr_x=0.5/'
        '{}/visualization/32768/txt'.format(dataset_name))
    fgt_root = os.path.join(root, 'patch_anno', dataset_name)
    visualize_txt_result(overwrite=True)
