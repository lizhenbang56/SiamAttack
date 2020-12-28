import os
import glob
import time
import imageio
import numpy as np
from PIL import Image, ImageDraw

from videoanalyst.pipeline.utils import xywh2xyxy
from videoanalyst.evaluation.got_benchmark.utils.metrics import rect_iou
from videoanalyst.evaluation.got_benchmark.experiments.got10k import ExperimentGOT10k


def visualize(pred, video_name, dataset_dir, overwrite):
    img_list = []
    imgs = sorted(glob.glob(os.path.join(dataset_dir, '*.jpg')))
    for pred_, img_path in zip(pred, imgs):
        img = Image.open(img_path)
        draw = ImageDraw.Draw(img)
        patch_anno_xyxy = xywh2xyxy(pred_)
        draw.rectangle(patch_anno_xyxy, None, 'green', width=8)
        img = img.resize((400, 300))
        img_list.append(img)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
    file_path = os.path.join(save_dir, video_name + '_{}'.format(timestamp) + '.gif')
    if os.path.isfile(file_path) and not overwrite:
        print('Already exists:', end=' ')
    else:
        imageio.mimsave(file_path, img_list, fps=20)
    print(file_path)
    return


def cal_ao(gt, pred):
    seq_ious = [rect_iou(pred[1:], gt[1:])]
    ao, sr, speed, _ = got10k_tool._evaluate(ious=np.concatenate(seq_ious), times=[])
    return ao


def run_per_video(video_name, overwrite):
    phase = video_name.split('_')[1].lower()
    dataset_dir = os.path.join(dataset_root, phase, video_name)
    pred_txt_path = os.path.join(pred_dir, video_name, video_name + "_001.txt")
    pred = np.loadtxt(pred_txt_path, delimiter=',')
    visualize(pred, video_name, dataset_dir, overwrite)


def visualize_txt_result(overwrite):
    video_names = sorted(os.listdir(pred_dir))
    for video_name in video_names:
        run_per_video(video_name, overwrite)


if __name__ == '__main__':
    root = '/home/etvuz/projects/adversarial_attack/'
    dataset_root = os.path.join(root, 'video_analyst/datasets/GOT-10k')
    pred_dir = os.path.join(root, 'video_analyst/logs/GOT-Benchmark/result/GOT-10k/siamfcpp_googlenet')
    save_dir = '/tmp/uap_vis_txt'
    got10k_tool = ExperimentGOT10k(dataset_root, subset='val')
    visualize_txt_result(overwrite=True)
