import os
import cv2
import math
import numpy as np

search_img_size = 303


def to_2d_map(tensor):
    map_size = int(math.sqrt(tensor.shape[0]))
    arr = tensor.reshape(map_size, map_size)
    heatmap = ((arr - arr.min()) * (1 / (arr.max() - arr.min()) * 255)).astype('uint8')
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return heatmap


def visualize_search_img(adv_search_img, best_box_xyxy_in_search_img, save_root, idx, name):
    """
    adv_search_img: 未转为 unit8, ndarray, [h,w,3]
    """
    adv_search_img = np.clip(adv_search_img, 0, 255).astype(np.uint8)
    x1, y1, x2, y2 = [int(var) for var in best_box_xyxy_in_search_img]
    assert adv_search_img.flags['C_CONTIGUOUS']
    # adv_search_img = cv2.rectangle(adv_search_img, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
    save_path = os.path.join(save_root, '{}_{}.jpg'.format(idx, name))
    print(save_path)
    assert cv2.imwrite(save_path, adv_search_img)
    return


def visualize_template_img(adv_template_img, save_root, idx, name):
    adv_template_img = np.clip(adv_template_img, 0, 255).astype(np.uint8)
    save_path = os.path.join(save_root, '{}_{}.jpg'.format(idx, name))
    assert cv2.imwrite(save_path, adv_template_img)
    return


def visualize_cls_map(cls_pred, name, save_root, idx):
    """
    cls_pred: torch.tensor, [batch,361,1]  numpy.array, [361]
    """
    cls_map = to_2d_map(cls_pred)
    cls_img = cv2.resize(cls_map, (search_img_size, search_img_size))
    save_path = os.path.join(save_root, '{}_{}_pred.jpg'.format(idx, name))
    print(save_path)
    assert cv2.imwrite(save_path, cls_img)
    return
