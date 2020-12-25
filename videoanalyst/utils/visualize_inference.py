import cv2
import math


search_img_size = 303

def to_2d_map(tensor):
    map_size = int(math.sqrt(tensor.shape[0]))
    arr = tensor.reshape(map_size, map_size)
    map = ((arr - arr.min()) * (1 / (arr.max() - arr.min()) * 255)).astype('uint8')
    return map


def visualize_search_img(adv_search_img, best_box_xyxy_in_search_img):
    """
    adv_search_img: unit8, ndarray, [h,w,3]
    """
    x1, y1, x2, y2 = [int(var) for var in best_box_xyxy_in_search_img]
    assert adv_search_img.flags['C_CONTIGUOUS']
    adv_search_img = cv2.rectangle(adv_search_img, (x1, y1), (x2, y2), (0,0,255))
    save_path = '/tmp/adv_search_img.jpg'
    assert cv2.imwrite(save_path, adv_search_img)
    return


def visualize_template_img(adv_template_img):
    save_path = '/tmp/adv_template_img.jpg'
    assert cv2.imwrite(save_path, adv_template_img)
    return


def visualize_cls_map(cls_pred):
    """
    cls_pred: torch.tensor, [batch,361,1]  numpy.array, [361]
    """
    cls_map = to_2d_map(cls_pred)
    cls_img = cv2.resize(cls_map, (search_img_size, search_img_size))
    save_path = '/tmp/cls_pred.jpg'
    assert cv2.imwrite(save_path, cls_img)
    return