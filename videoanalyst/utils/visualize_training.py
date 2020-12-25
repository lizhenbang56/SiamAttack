import cv2
import math
import numpy as np


def to_2d_map(tensor):
    map_size = int(math.sqrt(tensor.shape[1]))
    tensor = tensor[0,:,0].cpu().data.numpy()
    arr = tensor.reshape(map_size, map_size)
    map = ((arr - arr.min()) * (1 / (arr.max() - arr.min()) * 255)).astype('uint8')
    return map


def visualize_training(training_data, predict_data):
    im_x = np.ascontiguousarray(training_data['im_x'][0].data.cpu().numpy().transpose(1,2,0).astype(np.uint8))
    im_x = cv2.cvtColor(im_x, cv2.COLOR_RGB2BGR)
    im_z = np.ascontiguousarray(training_data['im_z'][0].data.cpu().numpy().transpose(1, 2, 0).astype(np.uint8))
    im_z = cv2.cvtColor(im_z, cv2.COLOR_RGB2BGR)
    search_img_size = im_x.shape[0]
    fake_gt_xyxy_in_search_img = training_data['bbox_x'][0]
    x1, y1, x2, y2 = [int(var) for var in fake_gt_xyxy_in_search_img]
    im_x = cv2.rectangle(im_x, (x1, y1), (x2, y2), (0, 0, 255))
    cls_pred = predict_data['cls_pred']
    ctr_pred = predict_data['ctr_pred']
    box_pred = predict_data['box_pred']
    cls_map = to_2d_map(cls_pred)
    ctr_map = to_2d_map(ctr_pred)
    cls_img = cv2.resize(cls_map, (search_img_size, search_img_size))
    ctr_img = cv2.resize(ctr_map, (search_img_size, search_img_size))

    cv2.imwrite('/tmp/cls_map.jpg', cls_img)
    cv2.imwrite('/tmp/ctr_map.jpg', ctr_img)
    cv2.imwrite('/tmp/im_z.jpg', im_z)
    cv2.imwrite('/tmp/im_x.jpg', im_x)
    return


def visualize_patched_img(im, name):
    im_x = np.ascontiguousarray(im[0].data.cpu().numpy().transpose(1, 2, 0).astype(np.uint8))
    im_x = cv2.cvtColor(im_x, cv2.COLOR_RGB2BGR)
    save_path = '/tmp/{}.jpg'.format(name)
    assert cv2.imwrite(save_path, im_x)
    return
