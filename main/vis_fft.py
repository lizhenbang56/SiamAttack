import numpy as np
import cv2


def generate_gaussian(size):
    nrows = ncols = size
    sigmax, sigmay = 20, 20  # 高斯函数的标准差，越小越模糊。已验证。
    cy, cx = nrows/2, ncols/2
    x = np.linspace(0, nrows, nrows)
    y = np.linspace(0, ncols, ncols)
    X, Y = np.meshgrid(x, y)
    gmask = np.exp(-(((X-cx)/sigmax)**2 + ((Y-cy)/sigmay)**2))
    return gmask


def filter(gray):
    # image size, square side length, number of squares
    ncols, nrows = 127, 127

    # Take the 2-dimensional DFT and centre the frequencies
    ftimage = np.fft.fft2(gray)
    ftimage = np.fft.fftshift(ftimage)

    """生成高斯mask"""
    gmask = generate_gaussian(127)
    """生成高斯mask"""

    """施加模板进行滤波"""
    ftimagep = ftimage * gmask  # 应该对“平移后”的频率进行mask。已验证。
    """施加模板进行滤波"""

    """逆变换"""
    imagep = np.fft.ifft2(ftimagep)
    imagep = np.abs(imagep)
    """逆变换"""
    
    return imagep


def filter_color():
    """读取rgb图像"""
    image = cv2.imread('/home/yyshi/zhbli/projects/Universal-Targeted-Attacks-for-Siamese-Visual-Tracking/snapshots_imperceptible_patch/FFT/visualization/1024/GOT-10k_Val/GOT-10k_Val_000001/1_clean_template_img.jpg')
    """读取rgb图像"""

    """分别处理每个通道"""
    for i in range(3):
        image[:,:,i] = filter(image[:,:,i])
    """分别处理每个通道"""

    """保存图像"""
    cv2.imwrite('/tmp/6_rgb.jpg', image)
    """保存图像"""

if __name__ == '__main__':
    filter_color()