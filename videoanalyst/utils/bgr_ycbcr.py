import cv2
import numpy as np


def rgb2ycbcr(im):
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:,:,[1,2]] += 128
    return np.uint8(ycbcr)


def ycbcr2rgb(im):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float)
    rgb[:,:,[1,2]] -= 128
    rgb = rgb.dot(xform.T)
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    return np.uint8(rgb)


def main():
    img = cv2.imread('/home/yyshi/zhbli/projects/Universal-Targeted-Attacks-for-Siamese-Visual-Tracking/snapshots_imperceptible_patch/64/visualization/256/GOT-10k_Val/GOT-10k_Val_000001/2_clean_search_img.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ycbcr = rgb2ycbcr(img).astype(np.float)
    h, w, _ = ycbcr.shape
    ycbcr[:,:,:2] += np.random.randint(-100, 100, size=(h,w,2), dtype=np.int).astype(np.float)
    ycbcr[ycbcr>255] = 255
    ycbcr[ycbcr<0] = 0
    img = ycbcr2rgb(ycbcr)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    assert cv2.imwrite('/tmp/YCb.jpg', img)


if __name__ == '__main__':
    main()