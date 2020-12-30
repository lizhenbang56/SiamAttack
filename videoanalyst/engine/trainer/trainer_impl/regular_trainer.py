# -*- coding: utf-8 -*
import copy
from collections import OrderedDict
import os
from loguru import logger
from tqdm import tqdm

import torch
from torch import nn

from videoanalyst.utils import (Timer, ensure_dir, move_data_to_device,
                                unwrap_model)
from videoanalyst.utils.visualize_training import visualize_training, visualize_patched_img

from ..trainer_base import TRACK_TRAINERS, TrainerBase


def fgsm_attack(uap, data_grad, epsilon):
    sign_data_grad = data_grad.sign()
    new_uap = uap - epsilon*sign_data_grad
    return new_uap


def get_patch_grad(tensor, pos, patch_w, patch_h):
    grad = torch.zeros((1,3,patch_h,patch_w), device=tensor.device)
    for t, p in zip(tensor, pos):
        x1, y1, x2, y2 = p
        grad += t[:, y1:y2, x1:x2]
    return grad / tensor.shape[0]


@TRACK_TRAINERS.register
class RegularTrainer(TrainerBase):
    r"""
    Trainer to test the vot dataset, the result is saved as follows
    exp_dir/logs/$dataset_name$/$tracker_name$/baseline
                                    |-$video_name$/ floder of result files
                                    |-eval_result.csv evaluation result file

    Hyper-parameters
    ----------------
    devices: List[str]
        list of string
    """
    extra_hyper_params = dict(
        minibatch=1,
        nr_image_per_epoch=1,
        max_epoch=1,
        snapshot="",
    )

    def __init__(self, optimizer, dataloader, monitors=[]):
        r"""
        Crete tester with config and pipeline

        Arguments
        ---------
        optimizer: ModuleBase
            including optimizer, model and loss
        dataloder: DataLoader
            PyTorch dataloader object. 
            Usage: batch_data = next(dataloader)
        """
        super(RegularTrainer, self).__init__(optimizer, dataloader, monitors)
        # update state
        self._state["epoch"] = -1  # uninitialized
        self._state["initialized"] = False
        self._state["devices"] = torch.device("cuda:0")

    def init_train(self, ):
        torch.cuda.empty_cache()
        # move model & loss to target devices
        devs = self._state["devices"]
        self._model.train()
        # load from self._state["snapshot_file"]
        self.load_snapshot()
        # parallelism with Data Parallel (DP)
        if len(self._state["devices"]) > 1:
            self._model = nn.DataParallel(self._model, device_ids=devs)
            logger.info("Use nn.DataParallel for data parallelism")
        super(RegularTrainer, self).init_train()
        logger.info("{} initialized".format(type(self).__name__))

    def train(self, patch_x, uap_z, real_iter_num, signal_img_debug, visualize):
        """"""
        """START：设定参数"""
        cls_weight = 2
        ctr_weight = 0
        reg_weight = 0
        l2_z_weight = 0.001
        lr = 0.5
        optimize_mode = 'optimizer'
        print('{}_cls_weight={}, ctr_weight={}, reg_weight={}, l2_z_weight={}, lr={}'.format(optimize_mode, cls_weight, ctr_weight, reg_weight, l2_z_weight, lr))
        """END：设定参数"""

        optimizer = None

        if not self._state["initialized"]:
            self.init_train()
        self._state["initialized"] = True

        self._state["epoch"] += 1
        epoch = self._state["epoch"]
        num_iterations = self._hyper_params["num_iterations"]

        # udpate engine_state
        self._state["max_epoch"] = self._hyper_params["max_epoch"]
        self._state["max_iteration"] = num_iterations

        self._optimizer.modify_grad(epoch)
        pbar = tqdm(range(num_iterations))
        self._state["pbar"] = pbar
        self._state["print_str"] = ""

        training_data_raw = None

        time_dict = OrderedDict()
        for iteration, _ in enumerate(pbar):
            self._state["iteration"] = iteration
            with Timer(name="data", output_dict=time_dict):

                """START：获取 training data"""
                if not signal_img_debug:
                    if iteration == 0:
                        training_data = next(self._dataloader)
                        training_data_raw = training_data.copy()
                    else:
                        training_data = training_data_raw
                else:
                    training_data = next(self._dataloader)

                if not signal_img_debug or training_data is None:
                    training_data = next(self._dataloader)
                if signal_img_debug:
                    training_data_raw = training_data.copy()
                """END：获取 training data"""

            training_data = move_data_to_device(training_data,
                                                self._state["devices"][0])

            schedule_info = self._optimizer.schedule(epoch, iteration)
            self._optimizer.zero_grad()

            # forward propagation
            with Timer(name="fwd", output_dict=time_dict):

                """START：初始化通用扰动"""
                patch_w = patch_h = 64
                if patch_x is None:
                    patch_x = torch.normal(mean=(128.0*torch.ones(1,3,patch_h,patch_w))).to(self._state["devices"][0])
                    uap_z = torch.zeros((1, 3, training_data['im_z'].shape[2], training_data['im_z'].shape[3])).to(self._state["devices"][0])
                    optimizer = torch.optim.AdamW([patch_x, uap_z], lr=0.1, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0, amsgrad=False)
                else:
                    patch_x = patch_x.to(self._state["devices"][0])
                    uap_z = uap_z.to(self._state["devices"][0])
                if optimizer is None:
                    optimizer = torch.optim.AdamW([patch_x, uap_z], lr=0.1, betas=(0.9, 0.999), eps=1e-08,
                                                  weight_decay=0.0, amsgrad=False)
                """END：初始化通用扰动"""

                """START：设置输入图像为可获取梯度"""
                uap_z.requires_grad = True
                patch_x.requires_grad = True
                """END：设置输入图像为可获取梯度"""

                """START：备份干净图像"""
                im_z_ori = copy.deepcopy(training_data['im_z'].data)
                im_x_ori = copy.deepcopy(training_data['im_x'].data)
                """END：备份干净图像"""

                """START：在搜索图像添加补丁"""
                patch_pos = []
                for idx, xyxy in enumerate(training_data['bbox_x']):
                    x1, y1, x2, y2 = [int(var) for var in xyxy]
                    cx = (x2+x1)/2
                    cy = (y2+y1)/2
                    xx1 = int(cx-patch_w/2)
                    yy1 = int(cy-patch_h/2)
                    xx2 = xx1 + patch_w
                    yy2 = yy1 + patch_h
                    patch_pos.append([xx1, yy1, xx2, yy2])
                    try:
                        training_data['im_x'][idx, :, yy1:yy2, xx1:xx2] = patch_x.to(self._state["devices"][0])
                    except:
                        print("Error paste patch")
                        continue
                if visualize:
                    visualize_patched_img(training_data['im_x'], name='patched_train_search')
                """END：在搜索图像添加补丁"""

                """START：将扰动叠加至输入图像"""
                training_data['im_z'] = uap_z + training_data['im_z'].data
                """END：将扰动叠加至输入图像"""

                """START：网络前向传播"""
                self._model.eval()  # !!!非常重要。否则造成训练测试不一致。我们根本不训练网络。
                optimizer.zero_grad()
                predict_data = self._model(training_data)
                """END：网络前向传播"""

                """START：可视化训练数据"""
                if visualize:
                    visualize_training(training_data, predict_data)
                """END：可视化训练数据"""

                """START：计算损失"""
                training_losses, extras = OrderedDict(), OrderedDict()
                for loss_name, loss in self._losses.items():
                    training_losses[loss_name], extras[loss_name] = loss(
                        predict_data, training_data)
                norm_x_loss = torch.mean((training_data['im_x'] - im_x_ori).pow(2))
                norm_z_loss = torch.mean((training_data['im_z'] - im_z_ori).pow(2))
                cls_loss = training_losses['cls']
                ctr_loss = training_losses['ctr']
                reg_loss = training_losses['reg']
                total_loss = cls_weight*cls_loss + \
                             ctr_weight*ctr_loss + \
                             reg_weight*reg_loss + \
                             l2_z_weight*norm_z_loss + \
                             0*norm_x_loss
                """END：计算损失"""

                """START：模型梯度清空"""
                self._model.zero_grad()
                """END：模型梯度清空"""

                """START：梯度反传"""
                total_loss.backward()
                """END：梯度反传"""

                if optimize_mode == 'FGSM':
                    """START：收集相对于输入图像的梯度"""
                    im_z_grad = torch.mean(uap_z.grad.data, dim=0, keepdims=True)
                    patch_grad = torch.mean(patch_x.grad.data, dim=0, keepdims=True)
                    """END：收集相对于输入图像的梯度"""

                    """START：根据梯度得到新的扰动"""
                    patch_x = fgsm_attack(patch_x, patch_grad, lr)
                    uap_z = fgsm_attack(uap_z, im_z_grad, lr)
                    patch_x = patch_x.detach()
                    uap_z = uap_z.detach()
                    """END：根据梯度得到新的扰动"""
                elif optimize_mode == 'optimizer':
                    optimizer.step()
                else:
                    assert False, optimize_mode

            trainer_data = dict(
                schedule_info=schedule_info,
                training_losses=training_losses,
                extras=extras,
                time_dict=time_dict,
                norm_x_loss=norm_x_loss.item(),
                norm_z_loss=norm_z_loss.item(),
            )

            for monitor in self._monitors:
                monitor.update(trainer_data)

            if not signal_img_debug:
                del training_data

            print_str = self._state["print_str"]
            pbar.set_description(print_str)

            """START：保存扰动"""
            if signal_img_debug:
                save_dir = '/tmp/uap_debug'
            else:
                save_dir = '/tmp/{}_cls={}_ctr={}_reg={}_l2={}_lr={}'.format(optimize_mode, cls_weight, ctr_weight, reg_weight, l2_z_weight, lr)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            real_iter_num += 1
            if real_iter_num & (real_iter_num - 1) == 0:
                save_path_x = os.path.join(save_dir, 'x_{}'.format(real_iter_num))
                save_path_z = os.path.join(save_dir, 'z_{}'.format(real_iter_num))
                torch.save(patch_x, save_path_x)
                torch.save(uap_z, save_path_z)
                print(' save to: {} {}'.format(save_path_x, save_path_z))
            """END：保存扰动"""

        return patch_x, uap_z, real_iter_num


RegularTrainer.default_hyper_params = copy.deepcopy(
    RegularTrainer.default_hyper_params)
RegularTrainer.default_hyper_params.update(RegularTrainer.extra_hyper_params)
