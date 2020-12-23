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

from ..trainer_base import TRACK_TRAINERS, TrainerBase


def fgsm_attack(uap, data_grad, epsilon=0.5):
    sign_data_grad = data_grad.sign()
    new_uap = uap - epsilon*sign_data_grad
    return new_uap


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

    def train(self):
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


        """START：设置是否进行单图 debug"""
        signal_img_debug = False
        """END：设置是否进行单图 debug"""

        """START：声明通用扰动"""
        uap_x = None
        uap_z = None
        training_data = None
        """END：声明通用扰动"""

        time_dict = OrderedDict()
        for iteration, _ in enumerate(pbar):
            self._state["iteration"] = iteration
            with Timer(name="data", output_dict=time_dict):
                if not signal_img_debug or training_data is None:
                    training_data = next(self._dataloader)
            training_data = move_data_to_device(training_data,
                                                self._state["devices"][0])

            schedule_info = self._optimizer.schedule(epoch, iteration)
            self._optimizer.zero_grad()

            # forward propagation
            with Timer(name="fwd", output_dict=time_dict):

                """START：初始化通用扰动"""
                if uap_x is None:
                    uap_x = torch.zeros((1, 3, training_data['im_x'].shape[2], training_data['im_x'].shape[3])).to(self._state["devices"][0])
                    uap_z = torch.zeros((1, 3, training_data['im_z'].shape[2], training_data['im_z'].shape[3])).to(self._state["devices"][0])
                """END：初始化通用扰动"""

                """START：将扰动叠加至输入图像"""
                training_data['im_z'] = uap_z + training_data['im_z'].data
                training_data['im_x'] = uap_x + training_data['im_x'].data
                """END：将扰动叠加至输入图像"""

                """START：限制图像值在 0~255 之间"""
                training_data['im_z'] = torch.clamp(training_data['im_z'], 0, 255)
                training_data['im_x'] = torch.clamp(training_data['im_x'], 0, 255)
                """END：限制图像值在 0~255 之间"""

                """START：设置输入图像为可获取梯度"""
                training_data['im_z'].requires_grad = True
                training_data['im_x'].requires_grad = True
                """END：设置输入图像为可获取梯度"""

                """START：网络前向传播"""
                predict_data = self._model(training_data)
                """END：网络前向传播"""

                """START：计算损失"""
                training_losses, extras = OrderedDict(), OrderedDict()
                for loss_name, loss in self._losses.items():
                    training_losses[loss_name], extras[loss_name] = loss(
                        predict_data, training_data)
                total_loss = sum(training_losses.values())
                """END：计算损失"""

                """START：模型梯度清空"""
                self._model.zero_grad()
                """END：模型梯度清空"""

                """START：梯度反传"""
                total_loss.backward()
                """END：梯度反传"""

                """START：收集相对于输入图像的梯度"""
                im_z_grad = torch.mean(training_data['im_z'].grad.data, dim=0, keepdims=True)
                im_x_grad = torch.mean(training_data['im_x'].grad.data, dim=0, keepdims=True)
                """END：收集相对于输入图像的梯度"""

                """START：根据梯度得到新的扰动"""
                uap_x = fgsm_attack(uap_x, im_x_grad)
                uap_z = fgsm_attack(uap_z, im_z_grad)
                """END：根据梯度得到新的扰动"""

            trainer_data = dict(
                schedule_info=schedule_info,
                training_losses=training_losses,
                extras=extras,
                time_dict=time_dict,
            )

            for monitor in self._monitors:
                monitor.update(trainer_data)
            if not signal_img_debug:
                del training_data
            print_str = self._state["print_str"]
            pbar.set_description(print_str)

            """START：保存扰动"""
            save_dir = '/tmp/uap'
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            real_iter_num = iteration + 1
            if real_iter_num & (real_iter_num - 1) == 0:
                torch.save(uap_x, os.path.join(save_dir, 'x_{}'.format(real_iter_num)))
                torch.save(uap_z, os.path.join(save_dir, 'z_{}'.format(real_iter_num)))
            """END：保存扰动"""

RegularTrainer.default_hyper_params = copy.deepcopy(
    RegularTrainer.default_hyper_params)
RegularTrainer.default_hyper_params.update(RegularTrainer.extra_hyper_params)
