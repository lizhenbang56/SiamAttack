# -*- coding: utf-8 -*-
from paths import ROOT_PATH  # isort:skip
import os
import argparse
import os.path as osp

from loguru import logger

import torch

from videoanalyst.config.config import cfg as root_cfg
from videoanalyst.config.config import specify_task
from videoanalyst.engine.builder import build as tester_builder
from videoanalyst.model import builder as model_builder
from videoanalyst.pipeline import builder as pipeline_builder
from videoanalyst.utils import complete_path_wt_root_in_cfg


def make_parser():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--dataset_name', type=str, default='GOT-10k_Val')  # 'GOT-10k_Val' 'OTB_2015' 'LaSOT'
    parser.add_argument('--loop_num', type=int, default=16384)
    parser.add_argument('--do_attack', type=str, default='true')
    parser.add_argument('--trainset', default='fulldata', type=str)
    parser.add_argument('--optimize_mode', default='FGSM', type=str)
    parser.add_argument('--cls_weight', type=float, default=1.0)
    parser.add_argument('--ctr_weight', type=float, default=1.0)
    parser.add_argument('--reg_weight', type=float, default=1.0)
    parser.add_argument('--l2_z_weight', default=0.005, type=float)
    parser.add_argument('--l2_x_weight', default=0.00001, type=float)
    parser.add_argument('--lr_z', default=0.1, type=float)
    parser.add_argument('--lr_x', default=0.5, type=float)
    parser.add_argument('--gpu_id', default='1,2,3,4', type=str)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--backbone', type=str, default='googlenet')  # 'alexnet' 'shufflenetv2x1_0'
    parser.add_argument('--phase', type=str, default='OURS')
    return parser


def build_siamfcpp_tester(task_cfg):
    # build model
    model = model_builder.build("track", task_cfg.model)
    # build pipeline
    pipeline = pipeline_builder.build("track", task_cfg.pipeline, model)
    # build tester
    testers = tester_builder("track", task_cfg.tester, "tester", pipeline)
    return testers


def build_sat_tester(task_cfg):
    # build model
    tracker_model = model_builder.build("track", task_cfg.tracker_model)
    tracker = pipeline_builder.build("track",
                                     task_cfg.tracker_pipeline,
                                     model=tracker_model)
    segmenter = model_builder.build('vos', task_cfg.segmenter)
    # build pipeline
    pipeline = pipeline_builder.build('vos',
                                      task_cfg.pipeline,
                                      segmenter=segmenter,
                                      tracker=tracker)
    # build tester
    testers = tester_builder('vos', task_cfg.tester, "tester", pipeline)
    return testers


if __name__ == '__main__':
    """设置参数"""
    parser = make_parser()
    parsed_args = parser.parse_args()
    print("迭代次数：{}".format(parsed_args.loop_num))
    # 设置配置文件路径
    if parsed_args.dataset_name == 'LaSOT':
        parsed_args.config = 'experiments/siamfcpp/test/lasot/siamfcpp_{}-lasot.yaml'.format(parsed_args.backbone)
    elif parsed_args.dataset_name == 'GOT-10k_Val':
        parsed_args.config = 'experiments/siamfcpp/test/got10k/siamfcpp_{}-got.yaml'.format(parsed_args.backbone)
    elif parsed_args.dataset_name == 'OTB_2015':
        parsed_args.config = 'experiments/siamfcpp/test/otb/siamfcpp_{}-otb.yaml'.format(parsed_args.backbone)
    else:
        assert False, parsed_args.dataset_name
    """设置参数"""

    """设置GPU"""
    os.environ['CUDA_VISIBLE_DEVICES'] = parsed_args.gpu_id
    """设置GPU"""

    # experiment config
    exp_cfg_path = osp.realpath(parsed_args.config)
    root_cfg.merge_from_file(exp_cfg_path)
    logger.info("Load experiment configuration at: %s" % exp_cfg_path)

    # resolve config
    root_cfg = complete_path_wt_root_in_cfg(root_cfg, ROOT_PATH)
    root_cfg = root_cfg.test
    task, task_cfg = specify_task(root_cfg)
    task_cfg.freeze()

    torch.multiprocessing.set_start_method('spawn', force=True)

    if task == 'track':
        testers = build_siamfcpp_tester(task_cfg)
    elif task == 'vos':
        testers = build_sat_tester(task_cfg)
    for tester in testers:
        tester._pipeline.loop_num = parsed_args.loop_num
        tester._pipeline.phase = parsed_args.phase
        if parsed_args.do_attack == 'true':
            tester._pipeline.do_attack = True
        elif parsed_args.do_attack == 'false':
            tester._pipeline.do_attack = False
        else:
            assert False, parsed_args.do_attack
        tester._pipeline.dataset_name = parsed_args.dataset_name

        if parsed_args.phase == 'OURS':
            if parsed_args.cls_weight == 1.0 and parsed_args.ctr_weight == 1.0 and parsed_args.reg_weight == 1.0:
                tester._pipeline.save_name = str(parsed_args.patch_size)
            elif parsed_args.cls_weight == 1.0 and parsed_args.ctr_weight == 0.0 and parsed_args.reg_weight == 0.0:
                tester._pipeline.save_name = str(parsed_args.patch_size) + '_ctr100'
            elif parsed_args.cls_weight == 0.0 and parsed_args.ctr_weight == 1.0 and parsed_args.reg_weight == 0.0:
                tester._pipeline.save_name = str(parsed_args.patch_size) + '_ctr010'
            elif parsed_args.cls_weight == 0.0 and parsed_args.ctr_weight == 0.0 and parsed_args.reg_weight == 1.0:
                tester._pipeline.save_name = str(parsed_args.patch_size) + '_ctr001'
            else:
                assert False, parsed_args
        else:
            tester._pipeline.save_name = parsed_args.phase
        tester._pipeline.load_attack()
        tester.test()
