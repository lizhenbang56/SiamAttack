import sys
import os
import copy
import glob
import numpy as np
from PIL import Image, ImageDraw

sys.path.append('/home/etvuz/projects/adversarial_attack/video_analyst')
from videoanalyst.pipeline.utils import xywh2xyxy, xywh2cxywh
from videoanalyst.evaluation.got_benchmark.utils.metrics import rect_iou
from videoanalyst.evaluation.got_benchmark.experiments.got10k import ExperimentGOT10k
from videoanalyst.evaluation.got_benchmark.experiments.otb import ExperimentOTB


def print_class():
    file_paths = sorted(glob.glob(os.path.join('/home/zhbli/Dataset/data2/got10k/val/GOT-10k_Val_*/meta_info.ini')))
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            content = f.readlines()
            print(content, file_path)


def draw_history_trajectory(img, gt_cxy_hist, pred_cxy_hist):
    """"""
    """START：声明透明轨迹图像"""
    trajectory = Image.new('RGBA', (target_w, target_h))
    draw_trajectory = ImageDraw.Draw(trajectory)
    """END：声明透明轨迹图像"""

    for gt_cxy, pred_cxy in zip(gt_cxy_hist, pred_cxy_hist):
        draw_trajectory.ellipse([tuple(gt_cxy - wp), tuple(gt_cxy + wp)], fill='red')
        draw_trajectory.ellipse([tuple(pred_cxy - wp), tuple(pred_cxy + wp)], fill='green')

    """START：原始图像与轨迹图像融合"""
    if img.size != trajectory.size:
        print('Wrong image size', img.size, trajectory.size)
        img = img.resize(trajectory.size)
    img = Image.alpha_composite(img, trajectory)
    """END：原始图像与轨迹图像融合"""

    return img


def visualize(pred, gt, video_name, dataset_dir, overwrite):
    img_list = []
    img_gt_list = []
    img_fgt_list = []
    img_pred_list = []
    imgs = sorted(glob.glob(os.path.join(dataset_dir, '*.jpg')))

    """START：设定输出图像尺寸"""
    ori_w, ori_h = Image.open(imgs[0]).size
    scale_w = target_w / ori_w
    scale_h = target_h / ori_h

    """START：声明透明轨迹图像"""
    trajectory = Image.new('RGBA', (target_w, target_h))
    draw_trajectory = ImageDraw.Draw(trajectory)
    """END：声明透明轨迹图像"""

    """START：绘制 GT 完整轨迹"""
    gt_cxy_list = []  # 0 代表初始帧
    pred_cxy_list = []
    for gt_, pred_ in zip(gt, pred):
        gt_cxy = xywh2cxywh(gt_)[:2] * (scale_w, scale_h)
        pred_cxy = xywh2cxywh(pred_)[:2] * (scale_w, scale_h)
        draw_trajectory.ellipse([tuple(gt_cxy - wp), tuple(gt_cxy + wp)], fill='red')
        gt_cxy_list.append(gt_cxy)
        pred_cxy_list.append(pred_cxy)
    """END：绘制 GT/FGT 完整轨迹"""

    """START：主体绘制循环"""
    for pred_, gt_, img_path in zip(pred, gt, imgs):
        img = Image.open(img_path)
        img = img.convert("RGBA").resize((target_w, target_h))
        draw = ImageDraw.Draw(img)
        img_gt = copy.deepcopy(img)  # 仅具有 gt 框
        img_pred = copy.deepcopy(img)  # 仅具有 pred 框
        draw_gt = ImageDraw.Draw(img_gt)
        draw_pred = ImageDraw.Draw(img_pred)

        # """START：绘制当前帧的边框的中心点"""
        # pred_cxy = (xywh2cxywh(pred_)[:2] * (scale_w, scale_h)).astype(np.int)
        # gt_cxy = (xywh2cxywh(gt_)[:2] * (scale_w, scale_h)).astype(np.int)
        # fgt_cxy = (xywh2cxywh(fgt_)[:2] * (scale_w, scale_h)).astype(np.int)
        # draw.ellipse([tuple(pred_cxy-wp), tuple(pred_cxy+wp)], fill='green')
        # draw.ellipse([tuple(gt_cxy-wp), tuple(gt_cxy+wp)], fill='red')
        # draw.ellipse([tuple(fgt_cxy-wp), tuple(fgt_cxy+wp)], fill='yellow')
        # """END：绘制当前帧的边框的中心点"""

        """START：可视化矩形框"""
        pred_xyxy = list(xywh2xyxy(pred_) * (scale_w, scale_w, scale_w, scale_h))
        gt_xyxy = list(xywh2xyxy(gt_) * (scale_w, scale_w, scale_w, scale_h))
        draw.rectangle(pred_xyxy, fill=None, outline='green', width=w)
        draw.rectangle(gt_xyxy, fill=None, outline='red', width=w)
        draw_gt.rectangle(gt_xyxy, fill=None, outline='red', width=w)
        draw_pred.rectangle(pred_xyxy, fill=None, outline='green', width=w)
        """END：可视化矩形框"""

        # """START：更新轨迹图像"""
        # draw_trajectory.ellipse([tuple(pred_cxy - wp), tuple(pred_cxy + wp)], fill='green')
        # draw_trajectory.ellipse([tuple(gt_cxy - wp), tuple(gt_cxy + wp)], fill='red')
        # draw_trajectory.ellipse([tuple(fgt_cxy - wp), tuple(fgt_cxy + wp)], fill='yellow')
        # """END：更新轨迹图像"""

        # """START：原始图像与轨迹图像融合"""
        # if img.size != trajectory.size:
        #     print('Wrong image size', img_path, img.size, trajectory.size)
        #     img = img.resize(trajectory.size)
        # img = Image.alpha_composite(img, trajectory)
        # """END：原始图像与轨迹图像融合"""

        img_list.append(img)
        img_gt_list.append(img_gt)
        img_pred_list.append(img_pred)
    """END：主体绘制循环"""

    """START：特定帧融合倒数30帧轨迹"""
    frame_start = img_list[1]
    mid_frame_name = 50  # 从1开始
    frame_mid = draw_history_trajectory(img_list[mid_frame_name-1],
                                        gt_cxy_list[mid_frame_name-30:mid_frame_name+1],
                                        pred_cxy_list[mid_frame_name-30:mid_frame_name+1])
    frame_end = draw_history_trajectory(img_list[-1],
                                        gt_cxy_list[-30:-1],
                                        pred_cxy_list[-30:-1])
    end_frame_name = len(img_list)
    only_save_special_frames = True
    """START：特定帧融合倒数30帧轨迹"""

    """START：保存"""
    if save_mode == '.png':
        save_dir = os.path.join(save_root, video_name)
    elif save_mode == '.gif':
        save_dir = save_root
    else:
        assert False, save_mode
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(save_dir)
    if save_mode == '.gif':
        file_path = os.path.join(save_dir, video_name + save_mode)
        if os.path.isfile(file_path) and not overwrite:
            print('Already exists:', end=' ')
        img_list[0].save(file_path, save_all=True, append_images=img_list[1:], quality=50)
    elif save_mode == '.png':
        if only_save_special_frames:
            frame_start.save(os.path.join(save_dir, 'special_' + str(2) + save_mode))
            frame_mid.save(os.path.join(save_dir, 'special_' + str(mid_frame_name) + save_mode))
            frame_end.save(os.path.join(save_dir, 'special_' + str(end_frame_name) + save_mode))
            return
        for i, data in enumerate(zip(img_list, img_gt_list, img_fgt_list, img_pred_list)):
            img, img_gt, img_fgt, img_pred = data
            save_path = os.path.join(save_dir, str(i+1))
            img.save(save_path+save_mode)
            img_gt.save(save_path+'_gt'+save_mode)
            img_fgt.save(save_path+'_fgt'+save_mode)
            img_pred.save(save_path+'_pred'+save_mode)
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
    elif dataset_name == 'GOT-10k_Val':
        phase = video_name.split('_')[1].lower()
        dataset_dir = os.path.join(dataset_root, phase, video_name)
        pred_txt_path = os.path.join(pred_dir, video_name, video_name + "_001.txt")
    else:
        assert False, dataset_name
    pred = np.loadtxt(pred_txt_path, delimiter=',')
    visualize(pred, gt, video_name, dataset_dir, overwrite)


def visualize_txt_result(overwrite):
    video_names = sorted(os.listdir(pred_dir))
    for video_name, gt in zip(video_names, dataset_tool.dataset):
        # if video_name != 'GOT-10k_Val_000006':
        #     continue
        run_per_video(video_name, gt[1], overwrite)


# if __name__ == '__main__':
#     print_class()


if __name__ == '__main__':
    """START：设定数据集名称"""
    dataset_name = 'GOT-10k_Val'
    print('Dataset Name:', dataset_name)
    save_mode = '.png'  # .png or .gif
    # 缩放后的图像尺寸
    target_w = 256 * 2
    target_h = 144 * 2
    w = 4  # 线的宽度
    wp = 2  # 点的半径
    """END：设定数据集名称"""

    root = '/home/etvuz/projects/adversarial_attack/'
    if dataset_name == 'OTB_2015':
        dataset_root = os.path.join(root, 'video_analyst/datasets/OTB/OTB2015')
        dataset_tool = ExperimentOTB(dataset_root, version=2015)
        pred_dir = os.path.join(root, 'video_analyst/logs/GOT-Benchmark/result/otb2015/siamfcpp_googlenet')
    elif dataset_name == 'GOT-10k_Val':
        dataset_root = os.path.join(root, 'video_analyst/datasets/GOT-10k')
        dataset_tool = ExperimentGOT10k(dataset_root, subset='val')
        pred_dir = os.path.join(root, 'video_analyst/snapshots_imperceptible_patch/64/result/siamfcpp_googlenet/2048')
    else:
        assert False, dataset_name
    save_root = os.path.join(
        root,
        '/home/etvuz/projects/adversarial_attack/video_analyst/snapshots_imperceptible_patch/64/visualization/2048_txt')
    visualize_txt_result(overwrite=True)
