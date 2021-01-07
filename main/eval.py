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
    parser.add_argument('--dataset_name', type=str, default='GOT-10k_Val')
    parser.add_argument('--loop_num', type=int, default=32768)
    parser.add_argument('--backbone_name', type=str)
    parser.add_argument('--trainset', default='fulldata', type=str)
    parser.add_argument('--optimize_mode', default='FGSM', type=str)
    parser.add_argument('--cls_weight', type=float)
    parser.add_argument('--ctr_weight', type=float)
    parser.add_argument('--reg_weight', type=float)
    parser.add_argument('--l2_z_weight', default=0.005, type=float)
    parser.add_argument('--l2_x_weight', default=0.00001, type=float)
    parser.add_argument('--lr_z', default=0.1, type=float)
    parser.add_argument('--lr_x', default=0.5, type=float)
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
    times = np.concatenate(list(times.values()))
    fgt_ao, fgt_sr_50, fgt_speed, fgt_succ_curve = experimentGOT10k._evaluate(fgt_ious, times)
    gt_ao, gt_sr_50, gt_speed, gt_succ_curve = experimentGOT10k._evaluate(gt_ious, times)
    print('FGT AO={:.3f} SR50={:.3f}'.format(fgt_ao, fgt_sr_50))
    print('GT AO={:.3f} SR50={:.3f}'.format(gt_ao, gt_sr_50))
    return


def eval_otb_2015(false_ground_truth):
    experiment = ExperimentOTB('/home/etvuz/projects/adversarial_attack/video_analyst/datasets/OTB/OTB2015',
                               version=2015,
                               result_dir=os.path.join(root, 'video_analyst/logs/GOT-Benchmark/result'),
                               FGT=false_ground_truth)
    eval_result = experiment.report(['siamfcpp_googlenet'])['siamfcpp_googlenet']['overall']
    if false_ground_truth:
        phase = 'FGT'
    else:
        phase = 'GT'
    print('{} Success={:.3f}, Precision={:.3f}, {} FPS'.format(phase, eval_result['success_score'],
                                                            eval_result['precision_score'],
                                                            int(eval_result['speed_fps'])))


def eval_lasot():
    experiment = ExperimentLaSOT('/home/etvuz/projects/adversarial_attack/video_analyst/datasets/LaSOT',
                                 subset='test',
                                 return_meta=False,
                                 result_dir=os.path.join(root, 'video_analyst/logs/GOT-Benchmark/result_attack'))
    eval_result = experiment.report(['siamfcpp_googlenet'])['siamfcpp_googlenet']['overall']
    print(
        'GT Success score={:.3f} Success rate={:.3f} Precision score={:.3f} Norm precision score={:.3f} {} FPS'.format(
            eval_result['success_score'],
            eval_result['success_rate'],
            eval_result['precision_score'],
            eval_result['normalized_precision_score'],
            int(eval_result['speed_fps'])))
    fgt_paths = sorted(glob.glob('/home/etvuz/projects/adversarial_attack/patch_anno/LaSOT/*.txt'))
    annos = {}
    for path in fgt_paths:
        video_name = path.split('/')[-1].split('.')[0]
        annos[video_name] = np.loadtxt(path, delimiter=',')
    for k, v in experiment.dataset.seq_datas.items():
        v['anno'] = annos[k]
    eval_result = experiment.report(['siamfcpp_googlenet'])['siamfcpp_googlenet']['overall']
    print('FGT Success score={:.3f} Success rate={:.3f} Precision score={:.3f} Norm precision score={:.3f} {} FPS'.format(
        eval_result['success_score'],
        eval_result['success_rate'],
        eval_result['precision_score'],
        eval_result['normalized_precision_score'],
        int(eval_result['speed_fps'])))


if __name__ == '__main__':
    args = parse_args()
    dataset_name = args.dataset_name
    loop_num = args.loop_num
    backbone_name = args.backbone_name
    root = '/home/etvuz/projects/adversarial_attack'
    if dataset_name == 'OTB_2015':
        result_root = os.path.join(root, 'video_analyst/logs/GOT-Benchmark/result/otb2015/siamfcpp_googlenet')
        eval_otb_2015(false_ground_truth=True)
        eval_otb_2015(false_ground_truth=False)
    elif dataset_name == 'LaSOT':
        result_root = os.path.join(root, 'video_analyst/logs/GOT-Benchmark/result_attack/LaSOT/siamfcpp_googlenet')
        eval_lasot()
    elif dataset_name == 'GOT-10k_Val':
        save_name = 'train_set={}_{}_cls={}_ctr={}_reg={}_l2_z={}_l2_x={}_lr_z={}_lr_x={}'.format(
            args.trainset, args.optimize_mode, args.cls_weight, args.ctr_weight, args.reg_weight,
            args.l2_z_weight, args.l2_x_weight, args.lr_z, args.lr_x)
        result_root = os.path.join(
            root,
            'video_analyst/snapshots/{}/'
            'result/GOT-10k/{}/{}'.format(save_name, backbone_name, loop_num))
        dataset = GOT10k(os.path.join(root, 'video_analyst/datasets/GOT-10k'), subset='val', return_meta=True)
        experimentGOT10k = ExperimentGOT10k(os.path.join(root, 'video_analyst/datasets/GOT-10k'), subset='val')
        FGT_root = os.path.join(root, 'patch_anno', dataset_name)
        eval_got10k_val()
    else:
        assert False, dataset_name
