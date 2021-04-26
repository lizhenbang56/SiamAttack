import sys
sys.path.append('/home/etvuz/projects/adversarial_attack/video_analyst')
import os
import ast
import glob
import argparse
import numpy as np

from videoanalyst.evaluation.got_benchmark.utils.metrics import rect_iou
from videoanalyst.evaluation.got_benchmark.datasets import GOT10k
from videoanalyst.evaluation.got_benchmark.experiments.got10k import ExperimentGOT10k
from videoanalyst.evaluation.got_benchmark.experiments.otb import ExperimentOTB
from videoanalyst.evaluation.got_benchmark.experiments.lasot import ExperimentLaSOT


def parse_args():
    parser = argparse.ArgumentParser(description='Eval')
    parser.add_argument('--dataset_name', type=str, default='GOT-10k_Val')  # 'OTB_2015' 'LaSOT' 'GOT-10k_Val'
    parser.add_argument('--loop_num', type=int, default=2048)
    parser.add_argument('--backbone_name', type=str, default='googlenet')
    parser.add_argument('--tracker_name', type=str, default='32')
    return parser.parse_args()


def eval_got10k_val():
    """"""
    fgt_paths = sorted(glob.glob(os.path.join(FGT_root, "*.txt")))
    pred_paths = sorted(glob.glob(os.path.join(result_root, '*/*_001.txt')))

    seq_names = dataset.seq_names
    covers = {s: dataset[s][2]['cover'][1:] for s in seq_names}

    fgt_ious = {}
    gt_ious = {}
    times = {}
    for s, data in enumerate(zip(pred_paths, fgt_paths, dataset)):
        pred_path, fgt_path, (_, gt_xywh, meta) = data
        pred_xywh = np.loadtxt(pred_path, delimiter=',')
        fgt_xywh = np.loadtxt(fgt_path, delimiter=',')
        assert pred_xywh.shape == fgt_xywh.shape == gt_xywh.shape
        seq_name = experimentGOT10k.dataset.seq_names[s]
        bound = ast.literal_eval(meta['resolution'])

        fgt_seq_ious = [rect_iou(pred_xywh[1:], fgt_xywh[1:], bound=bound)]
        fgt_seq_ious = [t[covers[seq_name] > 0] for t in fgt_seq_ious]
        fgt_seq_ious = np.concatenate(fgt_seq_ious)
        fgt_ious[seq_name] = fgt_seq_ious

        gt_seq_ious = [rect_iou(pred_xywh[1:], gt_xywh[1:], bound=bound)]
        gt_seq_ious = [t[covers[seq_name] > 0] for t in gt_seq_ious]
        gt_seq_ious = np.concatenate(gt_seq_ious)
        gt_ious[seq_name] = gt_seq_ious

        """START：计算时间"""
        time_file = os.path.join(result_root, seq_name, '%s_time.txt' % seq_name)
        if os.path.exists(time_file):
            seq_times = np.loadtxt(time_file, delimiter=',')
            seq_times = seq_times[~np.isnan(seq_times)]
            seq_times = seq_times[seq_times > 0]
            if len(seq_times) > 0:
                times[seq_name] = seq_times
        """END：计算时间"""
    fgt_ious = np.concatenate(list(fgt_ious.values()))
    gt_ious = np.concatenate(list(gt_ious.values()))
    try:
        times = np.concatenate(list(times.values()))
    except Exception:
        times = []
    fgt_ao, fgt_sr_50, fgt_speed, fgt_succ_curve = experimentGOT10k._evaluate(fgt_ious, times)
    gt_ao, gt_sr_50, gt_speed, gt_succ_curve = experimentGOT10k._evaluate(gt_ious, times)
    
    fgt_str = 'FGT AO={:.3f} SR50={:.3f}'.format(fgt_ao, fgt_sr_50)
    gt_str = 'GT AO={:.3f} SR50={:.3f}'.format(gt_ao, gt_sr_50)
    return gt_str, fgt_str


def eval_otb_2015(false_ground_truth):
    experiment = ExperimentOTB('/home/etvuz/projects/adversarial_attack/video_analyst/datasets/OTB/OTB2015',
                               version=2015,
                               result_dir=result_root,
                               report_dir=report_root,
                               FGT=false_ground_truth, phase='eval', fgt_dir=FGT_root)
    eval_result = experiment.report(['siamfcpp_googlenet'])['siamfcpp_googlenet']['overall']
    if false_ground_truth:
        phase = 'FGT'
    else:
        phase = 'GT'

    print_str = '{} Success={:.3f}, Precision={:.3f}, {} FPS'.format(phase, eval_result['success_score'],
                                                            eval_result['precision_score'],
                                                            int(eval_result['speed_fps']))
    return print_str
 

def eval_lasot():
    experiment = ExperimentLaSOT('/home/etvuz/projects/adversarial_attack/video_analyst/datasets/LaSOT',
                                 subset='test',
                                 return_meta=False,
                                 report_dir=report_root,
                                 result_dir=result_root, phase='eval')
    eval_result = experiment.report(['siamfcpp_googlenet'])['siamfcpp_googlenet']['overall']
    
    gt_str = 'GT Success score={:.3f} Success rate={:.3f} Precision score={:.3f} Norm precision score={:.3f} {} FPS'.format(
            eval_result['success_score'],
            eval_result['success_rate'],
            eval_result['precision_score'],
            eval_result['normalized_precision_score'],
            int(eval_result['speed_fps']))
    
    fgt_paths = sorted(glob.glob(os.path.join(FGT_root, '*.txt')))
    annos = {}
    for path in fgt_paths:
        video_name = path.split('/')[-1].split('.')[0]
        annos[video_name] = np.loadtxt(path, delimiter=',')
    for k, v in experiment.dataset.seq_datas.items():
        v['anno'] = annos[k]
    eval_result = experiment.report(['siamfcpp_googlenet'])['siamfcpp_googlenet']['overall']
    
    fgt_str = 'FGT Success score={:.3f} Success rate={:.3f} Precision score={:.3f} Norm precision score={:.3f} {} FPS'.format(
        eval_result['success_score'],
        eval_result['success_rate'],
        eval_result['precision_score'],
        eval_result['normalized_precision_score'],
        int(eval_result['speed_fps']))
    
    return gt_str, fgt_str


if __name__ == '__main__':
    args = parse_args()
    dataset_name = args.dataset_name
    loop_num = args.loop_num
    args.backbone_name = 'siamfcpp_' + args.backbone_name
    backbone_name = args.backbone_name

    """设置文件夹路径"""
    root = '/home/etvuz/projects/adversarial_attack'
    result_root = os.path.join(root, 'video_analyst/snapshots_imperceptible_patch/{}/result/{}/{}/{}'.format(args.tracker_name, args.dataset_name, args.backbone_name, args.loop_num))
    report_root = os.path.join(root, 'video_analyst/snapshots_imperceptible_patch/{}/report/{}/{}/{}'.format(args.tracker_name, args.dataset_name, args.backbone_name, args.loop_num))
    FGT_root = os.path.join(root, 'video_analyst/snapshots_imperceptible_patch/{}/FGT/{}/{}/{}'.format(args.tracker_name, args.dataset_name, args.backbone_name, args.loop_num))
    """设置文件夹路径"""    
    
    if dataset_name == 'OTB_2015':
        fgt_str = eval_otb_2015(false_ground_truth=True)
        gt_str = eval_otb_2015(false_ground_truth=False)
    elif dataset_name == 'LaSOT':
        gt_str, fgt_str = eval_lasot()
    elif dataset_name == 'GOT-10k_Val':
        dataset = GOT10k(os.path.join(root, 'video_analyst/datasets/GOT-10k'), subset='val', return_meta=True)
        experimentGOT10k = ExperimentGOT10k(os.path.join(root, 'video_analyst/datasets/GOT-10k'), subset='val', result_dir=result_root, phase='eval', report_dir=report_root)
        gt_str, fgt_str = eval_got10k_val()
    else:
        assert False, dataset_name

    """评估结果展示与保存"""
    if not os.path.exists(report_root):
        os.makedirs(report_root)
    save_path = os.path.join(report_root, 'report.txt')
    report_str = gt_str + '\n' + fgt_str
    with open(save_path, 'w') as f:
        f.write(report_str)
    print(report_str)
    print(save_path)
    """评估结果展示与保存"""
