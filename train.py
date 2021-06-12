# -*- coding: utf-8 -*-
from main.paths import ROOT_PATH  # isort:skip
import os
import argparse
import os.path as osp
import sys

import cv2
from loguru import logger

import torch

from videoanalyst.config.config import cfg as root_cfg
from videoanalyst.config.config import specify_task
from videoanalyst.data import builder as dataloader_builder
from videoanalyst.engine import builder as engine_builder
from videoanalyst.model import builder as model_builder
from videoanalyst.optim import builder as optim_builder
from videoanalyst.utils import Timer, complete_path_wt_root_in_cfg, ensure_dir

cv2.setNumThreads(1)

# torch.backends.cudnn.enabled = False

# pytorch reproducibility
# https://pytorch.org/docs/stable/notes/randomness.html#cudnn
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def make_parser():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('-cfg',
                        '--config',
                        default='experiments/siamfcpp/train/fulldata/157_1gpu.yaml',
                        type=str,
                        help='path to experiment configuration')
    parser.add_argument(
        '-r',
        '--resume',
        default="models/siamfcpp/siamfcpp-googlenet-got-md5_e182dc4c3823427022eccf7313d740a7.pkl",
        help=r"completed epoch's number, latest or one model path")
    parser.add_argument('--signal_img_debug', default=False, type=bool)
    parser.add_argument('--uap_resume', default=False, type=bool)
    parser.add_argument('--cls_weight', default=1.0, type=float)
    parser.add_argument('--ctr_weight', default=1.0, type=float)
    parser.add_argument('--reg_weight', default=1.0, type=float)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--gpu_id', type=str, default='1,5,6,7')
    parser.add_argument('--phase', type=str, default='FFT')
    return parser


if __name__ == '__main__':
    # parsing
    parser = make_parser()
    parsed_args = parser.parse_args()

    """指定GPU"""
    os.environ['CUDA_VISIBLE_DEVICES'] = parsed_args.gpu_id
    """指定GPU"""

    # experiment config
    exp_cfg_path = osp.realpath(parsed_args.config)
    root_cfg.merge_from_file(exp_cfg_path)
    # resolve config
    root_cfg = complete_path_wt_root_in_cfg(root_cfg, ROOT_PATH)
    root_cfg = root_cfg.train
    task, task_cfg = specify_task(root_cfg)
    task_cfg.freeze()
    # log config
    log_dir = osp.join(task_cfg.exp_save, task_cfg.exp_name, "logs")
    ensure_dir(log_dir)
    logger.configure(
        handlers=[
            dict(sink=sys.stderr, level="INFO"),
            dict(sink=osp.join(log_dir, "train_log.txt"),
                 enqueue=True,
                 serialize=True,
                 diagnose=True,
                 backtrace=True,
                 level="INFO")
        ],
        extra={"common_to_all": "default"},
    )
    # backup config
    logger.info("Load experiment configuration at: %s" % exp_cfg_path)
    logger.info(
        "Merged with root_cfg imported from videoanalyst.config.config.cfg")
    cfg_bak_file = osp.join(log_dir, "%s_bak.yaml" % task_cfg.exp_name)
    with open(cfg_bak_file, "w") as f:
        f.write(task_cfg.dump())
    logger.info("Task configuration backed up at %s" % cfg_bak_file)
    # device config
    if task_cfg.device == "cuda":
        world_size = task_cfg.num_processes
        assert torch.cuda.is_available(), "please check your devices"
        assert torch.cuda.device_count(
        ) >= world_size, "cuda device {} is less than {}".format(
            torch.cuda.device_count(), world_size)
        devs = ["cuda:{}".format(i) for i in range(world_size)]
    else:
        devs = ["cpu"]
    # build model
    model = model_builder.build(task, task_cfg.model)
    model.set_device(devs[0])
    # load data
    with Timer(name="Dataloader building", verbose=True):
        dataloader = dataloader_builder.build(task, task_cfg.data, patch_size=parsed_args.patch_size, phase=parsed_args.phase)
    # build optimizer
    optimizer = optim_builder.build(task, task_cfg.optim, model)
    # build trainer
    trainer = engine_builder.build(task, task_cfg.trainer, "trainer", optimizer,
                                   dataloader)
    trainer.set_device(devs)
    trainer.resume(parsed_args.resume)
    # trainer.init_train()

    """START：声明通用扰动"""
    if not parsed_args.uap_resume:
        uap_z = torch.zeros((1, 3, 127, 127)).to(torch.complex64)
        if parsed_args.phase in ['OURS', 'FFT']:
            patch_x = torch.zeros(1, 3, parsed_args.patch_size, parsed_args.patch_size).to(torch.complex64)  # 因为是相加，所以初始化为0
        elif parsed_args.phase == 'AP':
            patch_x = 127 * torch.ones(1, 3, parsed_args.patch_size, parsed_args.patch_size)
        elif parsed_args.phase in ['UAP']:
            patch_x = torch.zeros(1, 3, 303, 303)
        optimizer = torch.optim.AdamW([patch_x, uap_z], lr=0.1, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0,
                                      amsgrad=False)  # bug in pytorch1.8.1
        
        # uap_z.requires_grad = True
        # patch_x.requires_grad = True
        # optimizer = torch.optim.SGD([patch_x, uap_z], lr=0.1)
    else:
        uap_num = 4096
        uap_x_path = '/tmp/uap_v1.1/x_{}'.format(uap_num)
        uap_z_path = '/tmp/uap_v1.1/z_{}'.format(uap_num)
        patch_x = torch.load(uap_x_path)
        uap_z = torch.load(uap_z_path)
        print('load: ', uap_x_path, uap_z_path)
    real_iter_num = 0
    dataset_name = parsed_args.config.split('/')[-2]
    """END：声明通用扰动"""

    logger.info("Start training")
    while not trainer.is_completed():
        patch_x, uap_z, real_iter_num = trainer.train(patch_x, uap_z, real_iter_num, parsed_args.signal_img_debug,
                                                      visualize=parsed_args.uap_resume, optimizer=optimizer, dataset_name=dataset_name,
                                                      params={'cls_weight': parsed_args.cls_weight,
                                                              'ctr_weight': parsed_args.ctr_weight,
                                                              'reg_weight': parsed_args.reg_weight,
                                                              'patch_size': parsed_args.patch_size,
                                                              'phase': parsed_args.phase})
        trainer.save_snapshot()
    # export final model
    trainer.save_snapshot(model_param_only=True)
    logger.info("Training completed.")