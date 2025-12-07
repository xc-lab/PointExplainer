import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义参数
theta = np.linspace(0, 8 * np.pi, 1000)  # 角度从0到8pi
z = np.linspace(0, 4, 1000)  # z 轴的高度从0到4
r = theta  # 半径等于角度，形成螺旋

# 转换为笛卡尔坐标
x = r * np.cos(theta)
y = r * np.sin(theta)

# 创建图形和三维坐标轴
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制螺旋线，并设置线宽
ax.plot(x, y, z, linewidth=5)

# 设置 x, y, z 轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 显示图形
plt.show()
