import cv2
import numpy as np


def bgr2ycbcr(im):
    im = im.astype(np.float64)
    cbcr = np.empty_like(im)
    b = im[:,:,0]
    g = im[:,:,1]
    r = im[:,:,2]
    # Y
    cbcr[:,:,0] = .299 * r + .587 * g + .114 * b
    # Cb
    cbcr[:,:,1] = 128 - .169 * r - .331 * g + .5 * b
    # Cr
    cbcr[:,:,2] = 128 + .5 * r - .419 * g - .081 * b
    return np.uint8(cbcr)

def ycbcr2bgr(im):
    im = im.astype(np.float64)
    rgb = np.empty_like(im)
    y   = im[:,:,0]
    cb  = im[:,:,1] - 128
    cr  = im[:,:,2] - 128
    # R
    rgb[:,:,2] = y + 1.402 * cr
    # G
    rgb[:,:,1] = y - .34414 * cb - .71414 * cr
    # B
    rgb[:,:,0] = y + 1.772 * cb
    rgb[rgb>255] = 255
    rgb[rgb<0] = 0
    return np.uint8(rgb)


def main():
    img = cv2.imread('/home/yyshi/zhbli/projects/Universal-Targeted-Attacks-for-Siamese-Visual-Tracking/snapshots_imperceptible_patch/64/visualization/256/GOT-10k_Val/GOT-10k_Val_000001/2_clean_search_img.jpg')
    ycbcr = bgr2ycbcr(img)
    img = ycbcr2bgr(ycbcr)
    assert cv2.imwrite('/tmp/YCb.jpg', img)


if __name__ == '__main__':
    main()