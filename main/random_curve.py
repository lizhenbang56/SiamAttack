"""
生成随机曲线，用于模拟物体的运动轨迹。
步骤1：全图均匀采样 n 个点。
步骤2：对这些点进行插值。
"""


import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

image_width = 1024
image_height = 768
n = 6
random_x_list = sorted(random.sample(range(1, image_width), n))
random_y_list = random.sample(range(1, image_height), n)

points = [[x, y] for x, y in zip(random_x_list, random_y_list)]

# points = [[0, 1], [1, 1.1], [2, 1.2], [3, 3], [4, 2.9], [5, 2.8], [6, 2.7]]  # 应不应该对坐标进行排序？现在x是有序的y是无序的。

x = [point[0] for point in points]
y = [point[1] for point in points]

f = np.polyfit(x, y, deg=6)
p = np.poly1d(f)

ls = interpolate.interp1d(x, y, kind='linear')
qs = interpolate.interp1d(x, y, kind='quadratic')
cs = interpolate.interp1d(x, y, kind='cubic')

x_new = np.linspace(x[0], x[-1], 100)

plt.scatter(x, y, label='data')
# plt.plot(x_new, p(x_new), label='polynomial p6')
plt.plot(x_new, ls(x_new), label='linear spline')
plt.plot(x_new, qs(x_new), label='quadratic spline')
# plt.plot(x_new, cs(x_new), label='cubic spline')

plt.legend(loc='upper left')

save_path = '实验结果/curve.png'
plt.savefig(save_path)
print(save_path)