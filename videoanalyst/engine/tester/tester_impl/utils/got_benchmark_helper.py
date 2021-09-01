# -*- coding: utf-8 -*
import time
from typing import List
import os
# from PIL import Image
import cv2
import numpy as np

from videoanalyst.evaluation.got_benchmark.utils.viz import show_frame
from videoanalyst.pipeline.pipeline_base import PipelineBase
from videoanalyst.utils.visualize_inference import visualize_search_img, visualize_template_img, visualize_cls_map


class PipelineTracker(object):
    def __init__(self,
                 name: str,
                 backbone_name: str,
                 pipeline: PipelineBase,
                 is_deterministic: bool = True):
        """Helper tracker for comptability with 
        
        Parameters
        ----------
        name : str
            [description]
        pipeline : PipelineBase
            [description]
        is_deterministic : bool, optional
            [description], by default False
        """
        self.name = name
        self.backbone_name = backbone_name
        self.is_deterministic = is_deterministic
        self.pipeline = pipeline

    def init(self, image: np.array, box):
        """Initialize pipeline tracker
        
        Parameters
        ----------
        image : np.array
            image of the first frame
        box : np.array or List
            tracking bbox on the first frame
            format: (x, y, w, h)
        """
        self.pipeline.init(image, box)

    def update(self, image: np.array):
        """Perform tracking
        
        Parameters
        ----------
        image : np.array
            image of the current frame
        
        Returns
        -------
        np.array
            tracking bbox
            format: (x, y, w, h)
        """
        return self.pipeline.update(image)

    def track(self, img_files: List, box, visualize: bool = False, gts = None):
        """Perform tracking on a given video sequence
        
        Parameters
        ----------
        img_files : List
            list of image file paths of the sequence
        box : np.array or List
            box of the first frame
        visualize : bool, optional
            Visualize or not on each frame, by default False
        
        Returns
        -------
        [type]
            [description]
        """
        """获取dataset_name"""
        dataset_name = self.pipeline.dataset_name
        if dataset_name == 'OTB_2015':
            video_name = img_files[0].split('/')[-3]
        elif dataset_name == 'GOT-10k_Val':
            video_name = img_files[0].split('/')[-2]
        elif dataset_name == 'LaSOT':
            video_name = img_files[0].split('/')[-3]
        else:
            assert False, dataset_name
        """获取dataset_name"""

        """变量初始化"""
        patch_annos = gts
        frame_num = len(img_files)
        times = np.zeros(frame_num)
        """变量初始化"""
        
        """初始化用于保存跟踪结果的数组"""
        boxes = np.zeros((frame_num, 4))
        boxes[0] = box
        """初始化用于保存跟踪结果的数组"""

        """初始化用于保存FGT的数组"""
        boxes_fgt = np.zeros((frame_num, 4))
        boxes_fgt[0] = box
        """初始化用于保存FGT的数组"""

        for f, img_file in enumerate(img_files):
            self.pipeline._model.patch_gt_xywh_ori = patch_annos[f]
            image = cv2.imread(img_file, cv2.IMREAD_COLOR)
            start_time = time.time()

            """START：定义可视化文件夹"""
            visualize_flag = True
            if visualize_flag:
                vis_save_dir = os.path.join(self.pipeline.uap_root, 'visualization',
                                            str(self.pipeline.loop_num), dataset_name, video_name)
                if not os.path.exists(vis_save_dir):
                    os.makedirs(vis_save_dir)
            else:
                vis_save_dir = None
            """END：定义可视化文件夹"""

            if f == 0:

                """初始化跟踪器"""
                self.init(image, box)
                """初始化跟踪器"""

                """START：可视化模板图像"""
                if visualize_flag:
                    visualize_template_img(self.pipeline._state['z_crop'], vis_save_dir, f+1, 'clean_template_img')
                    visualize_template_img(self.pipeline._state['adv_template_img'], vis_save_dir, f+1, 'adv_template_img')
                """END：可视化模板图像"""

            else:
                
                """进行跟踪"""
                boxes[f, :], boxes_fgt[f, :] = self.update(image)
                """进行跟踪"""

                """START：可视化搜索图像"""
                if visualize_flag:
                    visualize_search_img(self.pipeline._state['x_crop'],
                                         self.pipeline._state['best_box_xyxy_in_search_img'],
                                         vis_save_dir, f + 1, 'clean_search_img')
                    visualize_search_img(self.pipeline._state['adv_search_img'],
                                         self.pipeline._state['best_box_xyxy_in_search_img'],
                                         vis_save_dir, f+1, 'adv_search_img_gt', self.pipeline._state['gt_xyxy'])
                    visualize_search_img(self.pipeline._state['adv_search_img'],
                                         self.pipeline._state['best_box_xyxy_in_search_img'],
                                         vis_save_dir, f+1, 'adv_search_img_fgt', self.pipeline._state['fgt_xyxy_search'])
                    visualize_search_img(self.pipeline._state['adv_search_img'],
                                         self.pipeline._state['best_box_xyxy_in_search_img'],
                                         vis_save_dir, f+1, 'adv_search_img_pred', self.pipeline._state['gt_xyxy'])
                    visualize_search_img(self.pipeline._state['adv_search_img'],
                                         self.pipeline._state['best_box_xyxy_in_search_img'],
                                         vis_save_dir, f+1, 'adv_search_img', self.pipeline._state['gt_xyxy'])
                    visualize_cls_map(self.pipeline._state['cls_pred'], 'cls_pred', vis_save_dir, f+1)
                    visualize_cls_map(self.pipeline._state['ctr_pred'], 'ctr_pred', vis_save_dir, f+1)
                """END：可视化搜索图像"""

            times[f] = time.time() - start_time

            if visualize:
                show_frame(image, boxes[f, :])

        return boxes, times, boxes_fgt
