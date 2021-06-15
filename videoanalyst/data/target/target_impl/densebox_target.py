# -*- coding: utf-8 -*-
import math
import random
from typing import Dict

from ..target_base import TRACK_TARGETS, TargetBase
from .utils import make_densebox_target
from videoanalyst.pipeline.utils import cxywh2xyxy


@TRACK_TARGETS.register
class DenseboxTarget(TargetBase):
    r"""
    Tracking data filter

    Hyper-parameters
    ----------------
    """
    default_hyper_params = dict(
        z_size=127,
        x_size=303,
        score_size=17,
        score_offset=87,
        total_stride=8,
        num_conv3x3=3,
    )

    def __init__(self, patch_size=0, phase='OURS') -> None:
        super().__init__()
        self.patch_size = patch_size
        self.phase = phase

    def update_params(self):
        hps = self._hyper_params
        hps['score_size'] = (
            hps['x_size'] -
            hps['z_size']) // hps['total_stride'] + 1 - hps['num_conv3x3'] * 2
        hps['score_offset'] = (
            hps['x_size'] - 1 -
            (hps['score_size'] - 1) * hps['total_stride']) // 2
        self._hyper_params = hps
        return

    def generate_fake_gt_xyxy_in_search_img(self):
        """"""
        """END：设定小补丁尺寸"""
        fake_gt_w = self.patch_size
        fake_gt_h = self.patch_size
        """END：设定小补丁尺寸"""

        """START：随机补丁的位置"""
        # 希望补丁的中心点以搜索图像中心点为中心，均匀偏移±64像素。
        if self.phase != 'UAP':
            fake_gt_cx = self._hyper_params['x_size'] / 2 + random.uniform(-64, 64)
            fake_gt_cy = self._hyper_params['x_size'] / 2 + random.uniform(-64, 64)
        else:
            fake_gt_cx = self._hyper_params['x_size'] / 2 - 64
            fake_gt_cy = self._hyper_params['x_size'] / 2 - 64
        """END：随机补丁的位置"""

        fake_gt_cxywh_in_search_img = [fake_gt_cx, fake_gt_cy, fake_gt_w, fake_gt_h]
        return cxywh2xyxy(fake_gt_cxywh_in_search_img)

    def generate_fake_gt_xyxy_in_search_img_dummy(self):
        """"""
        print('ERROR')
        import numpy as np
        return np.array([38,73,114,125])

    def __call__(self, sampled_data: Dict) -> Dict:
        data_z = sampled_data["data1"]
        im_z, bbox_z = data_z["image"], data_z["anno"]  # xyxy

        data_x = sampled_data["data2"]
        im_x, bbox_x = data_x["image"], data_x["anno"]  # xyxy

        """START：生成 fake_gt_xyxy_in_search_img"""
        if self.phase != 'UAP':  # 若是UAP，则意味着background，则gt是真目标，fgsm取反
            fake_gt_xyxy_in_search_img = self.generate_fake_gt_xyxy_in_search_img()  # xyxy
            bbox_x = fake_gt_xyxy_in_search_img  # xyxy
        """END：生成 fake_gt_xyxy_in_search_img"""

        is_negative_pair = sampled_data["is_negative_pair"]

        # input tensor
        im_z = im_z.transpose(2, 0, 1)
        im_x = im_x.transpose(2, 0, 1)

        # training target
        cls_label, ctr_label, box_label = make_densebox_target(
            bbox_x.reshape(1, 4), self._hyper_params)
        if is_negative_pair:
            cls_label[cls_label == 0] = -1
            cls_label[cls_label == 1] = 0

        training_data = dict(
            im_z=im_z,
            im_x=im_x,
            bbox_z=bbox_z,  # xyxy
            bbox_x=bbox_x,  # xyxy
            cls_gt=cls_label,
            ctr_gt=ctr_label,
            box_gt=box_label,
            is_negative_pair=int(is_negative_pair),
        )
        #training_data = super().__call__(training_data)

        return training_data
