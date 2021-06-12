import torch
import numpy as np
import cv2


def generate_gaussian1(size):
    x, y = np.meshgrid(np.linspace(-2,2,size), np.linspace(-2,2,size))
    dst = np.sqrt(x*x+y*y)
    
    # Intializing sigma and muu
    sigma = 1
    muu = 0.000
    
    # Calculating Gaussian array
    gauss = np.exp(-( (dst-muu)**2 / ( 2.0 * sigma**2 ) ) )
    
    return gauss


def generate_gaussian(size):
    nrows = ncols = size
    sigmax, sigmay = 20, 20  # 高斯函数的标准差，越小越模糊。已验证。
    cy, cx = nrows/2, ncols/2
    x = np.linspace(0, nrows, nrows)
    y = np.linspace(0, ncols, ncols)
    X, Y = np.meshgrid(x, y)
    gmask = np.exp(-(((X-cx)/sigmax)**2 + ((Y-cy)/sigmay)**2))
    return gmask

def torch_to_numpy(tensor):
    return tensor.numpy()[0].transpose((1,2,0))


def main():
    """获取图像的频谱图"""
    # # 方案1：使用网络学习的频谱图
    # fft_z = torch.load('/home/yyshi/zhbli/projects/Universal-Targeted-Attacks-for-Siamese-Visual-Tracking/snapshots_imperceptible_patch/FFT/z_1024', map_location='cpu')
    # # z = torch.fft.ifft2(fft_z).to(torch.float32)
    # # z = torch_to_numpy(z).astype(np.uint8)
    # fft_z_np = torch_to_numpy(fft_z)[:,:,0]  # 2d

    # 方案2：读取真实图像转化成频谱图
    z = cv2.imread('/home/yyshi/zhbli/projects/Universal-Targeted-Attacks-for-Siamese-Visual-Tracking/snapshots_imperceptible_patch/FFT/visualization/1024/GOT-10k_Val/GOT-10k_Val_000001/1_clean_template_img.jpg')[:,:,0]
    fft_z_np = np.fft.fft2(z)  # 对未经shift的频率进行ifft才能得到原始图像。已验证。
    """获取图像的频谱图"""

    """移位"""
    fshift = np.fft.fftshift(fft_z_np) #f是一个复数数组
    """移位"""

    """频域滤波"""
    # 滤波方案1：高斯滤波
    size = fft_z_np.shape[0]
    gaussian = generate_gaussian(size)
    masked_fshift = fshift * gaussian

    # 滤波方案2：直接变为0
    # masked_fshift = fshift.copy()
    # masked_fshift[50:70, 50:70, :] = 0
    
    shift_mask_shift = np.fft.fftshift(masked_fshift)
    """频域滤波"""

    """频域转图像"""
    shift_z = np.fft.ifft2(fshift).astype(np.uint8)  # 2d
    z = np.fft.ifft2(fft_z_np).astype(np.uint8)  # 复数直接转成实数是正确的，已验证。
    fft_img = np.log(np.abs(fshift)) * 10
    shift_mask_z = np.abs(np.fft.ifft2(masked_fshift))  # 2d
    shift_mask_shift_z = np.fft.ifft2(shift_mask_shift).astype(np.uint8)
    """频域转图像"""

    """保存图像"""
    cv2.imwrite('/tmp/0.jpg', fft_img)
    cv2.imwrite('/tmp/1_z.jpg', z)
    assert cv2.imwrite('/tmp/2_shift_mask_shift_z.jpg', shift_mask_shift_z)
    cv2.imwrite('/tmp/3_shift_z.jpg', shift_z)
    cv2.imwrite('/tmp/4_shift_mask_z.jpg', shift_mask_z)
    """保存图像"""


def filter():
    # image size, square side length, number of squares
    ncols, nrows = 127, 127

    # The image array (0=background, 1=square) and boolean array of allowed places
    # to add a square so that it doesn't touch another or the image sides
    image = cv2.imread('/home/yyshi/zhbli/projects/Universal-Targeted-Attacks-for-Siamese-Visual-Tracking/snapshots_imperceptible_patch/FFT/visualization/1024/GOT-10k_Val/GOT-10k_Val_000001/1_clean_template_img.jpg')[:,:,0]

    # Take the 2-dimensional DFT and centre the frequencies
    ftimage = np.fft.fft2(image)
    ftimage = np.fft.fftshift(ftimage)

    """生成高斯mask"""
    sigmax, sigmay = 20, 20  # 高斯函数的标准差，越小越模糊。已验证。
    cy, cx = nrows/2, ncols/2
    x = np.linspace(0, nrows, nrows)
    y = np.linspace(0, ncols, ncols)
    X, Y = np.meshgrid(x, y)
    gmask = np.exp(-(((X-cx)/sigmax)**2 + ((Y-cy)/sigmay)**2))
    """生成高斯mask"""

    """施加模板进行滤波"""
    ftimagep = ftimage * gmask  # 应该对对“平移后”的频率进行mask。已验证。
    """施加模板进行滤波"""

    """逆变换"""
    imagep = np.fft.ifft2(ftimagep)
    imagep = np.abs(imagep)
    """逆变换"""
    
    """保存图像"""
    cv2.imwrite('/tmp/5_np.jpg', imagep)
    """保存图像"""


if __name__ == '__main__':
    main()
    # filter()