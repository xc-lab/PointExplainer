#  -*- coding: utf-8 -*-
'''
author: xuechao.wang@ugent.be
'''
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import os
from matplotlib.colors import to_rgba




def moving_average(data, window_size):
    '''
        Moving average is used to smooth the data between each segment to prevent sudden changes.
        '''
    pad_size = window_size // 2
    if window_size % 2 == 0:
        padded_data = np.pad(data, (pad_size, pad_size-1), mode='edge')
    else:
        padded_data = np.pad(data, (pad_size, pad_size), mode='edge')
    smoothed_data = np.convolve(padded_data, np.ones(window_size) / window_size, mode='valid')
    return smoothed_data



def saliency_map(mask):
    '''
        Map the weight or shap value to the corresponding position in the colorbar, ranging from 0 to 1
        Args:
            mask represents weight or shap value.
        '''
    weight_signal = [i[0] for i in mask]

    impact = moving_average(weight_signal, int((len(mask)*0.1)*0.5))

    # # 绘制原始组合信号和平滑后的信号
    # plt.figure(figsize=(10, 6))
    # plt.plot(range(len(impact)), weight_signal, label='Combined Signal', alpha=0.5)
    # plt.plot(range(len(impact)), impact, label='Smoothed Signal', linewidth=2)
    # plt.xlabel('Time')
    # plt.ylabel('Amplitude')
    # plt.title('Smoothing Combined Step Signals with Wavelet Transform')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    colors_map = []
    min_value = np.min(impact)
    max_value = np.max(impact)
    if max_value > 0 and min_value < 0:
        for i in np.arange(len(impact)):
            weight = impact[i]
            if weight >= 0:
                new_weight = 0.5 + (weight / max_value)*0.5
                colors_map.append(new_weight)
            else:
                new_weight = 0.5 - (weight / min_value)*0.5
                colors_map.append(new_weight)
    elif max_value <= 0 and min_value < 0:
        for i in np.arange(len(impact)):
            weight = impact[i]
            new_weight = 0.5 - (weight / min_value)*0.5
            colors_map.append(new_weight)
        colors_map[0] = 1
    elif min_value >= 0 and max_value > 0:
        for i in np.arange(len(impact)):
            weight = impact[i]
            new_weight = 0.5 + (weight / max_value)*0.5
            colors_map.append(new_weight)
        colors_map[0] = 0
    elif min_value==0 and max_value==0:
        for i in np.arange(len(impact)):
            new_weight = 0.5
            colors_map.append(new_weight)
        colors_map[0] = 1
        colors_map[1] = 0

    return colors_map




def plot_result(normalized_pc, colors_map, mask, out_path, file_name, data_pattern, if_save=False):
    '''
    Display and save the results, where the colors represent the weights
    '''
    x = normalized_pc[:, 0]
    y = normalized_pc[:, 1]
    z = normalized_pc[:, 2] + 0.3
    # Creating a Point Cloud
    points = np.vstack((x, y, z)).T

    # Using the coolwarm colormap
    cmap = cm.get_cmap('coolwarm')

    colors = cmap(colors_map)

    PD_colors_map = [0.5 if i < 0.5 else i for i in colors_map]
    PD_colors = cmap(PD_colors_map)

    HC_colors_map = [0.5 if i > 0.5 else i for i in colors_map]
    HC_colors = cmap(HC_colors_map)

    # Create a figure with GridSpec layout
    fig = plt.figure(figsize=(14, 7))
    plt.axis('off')

    gs = fig.add_gridspec(2, 3, width_ratios=[2, 1, 1])

    plt.title(file_name + '-' + str(data_pattern))

    # 3D Plot (Left side)
    ax3d = fig.add_subplot(gs[:, :2], projection='3d')
    sc = ax3d.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, marker='o', s=40)
    ax3d.scatter(points[:, 0], points[:, 1], np.zeros_like(points[:, 2]), c=colors, marker='o', s=5, alpha=0.5)
    ax3d.view_init(elev=15, azim=-45)
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z')

    # PD 2D Plot (Top-right)
    ax_pd = fig.add_subplot(gs[0, 2])
    ax_pd.scatter(points[:, 0], points[:, 1], c=PD_colors, marker='o', s=20)
    ax_pd.grid(True, color='gray', alpha=0.5)
    ax_pd.set_xlabel('X')
    ax_pd.set_ylabel('Y')
    ax_pd.set_aspect('equal', 'box')
    # ax_pd.set_xticklabels([])
    # ax_pd.set_yticklabels([])

    # HC 2D Plot (Bottom-right)
    ax_hc = fig.add_subplot(gs[1, 2])
    ax_hc.scatter(points[:, 0], points[:, 1], c=HC_colors, marker='o', s=20)
    ax_hc.grid(True, color='gray', alpha=0.5)
    ax_hc.set_xlabel('X')
    ax_hc.set_ylabel('Y')
    ax_hc.set_aspect('equal', 'box')
    # ax_hc.set_xticklabels([])
    # ax_hc.set_yticklabels([])
    if if_save:
        plt.savefig(os.path.join(out_path, 'img.jpg'), dpi=500)
        np.save(os.path.join(out_path, 'weight.npy'), mask)
    else:
        plt.show()
    plt.close()


def plot_fig1(normalized_pc, colors_map, mask, out_path, file_name, data_pattern, if_save=False):
    '''
        Display and save the results, where the colors represent the weights
        '''
    superpoint = normalized_pc[5150:5400, :]
    # 添加一个中心点（可以是点云的几何中心）
    center_x = np.mean(superpoint[:,0])
    center_y = np.mean(superpoint[:,1])
    center_z = np.mean(superpoint[:,2]) + 0.35


    normalized_pc = np.delete(normalized_pc, list(range(5150, 5400)), axis=0)

    x = normalized_pc[:, 0]
    y = normalized_pc[:, 1]
    z = normalized_pc[:, 2] + 0.35
    # Creating a Point Cloud
    points = np.vstack((x, y, z)).T

    # # Using the coolwarm colormap
    # cmap = cm.get_cmap('coolwarm')
    #
    # colors = cmap(colors_map)
    #
    # PD_colors_map = [0.5 if i < 0.55 else i for i in colors_map]
    # PD_colors = cmap(PD_colors_map)
    #
    # HC_colors_map = [0.5 if i > 0.45 else i for i in colors_map]
    # HC_colors = cmap(HC_colors_map)
    #
    # HC_PD_colors_map = [0.5 if i < 0.55 and i > 0.45 else i for i in colors_map]
    # HC_PD_colors = cmap(HC_PD_colors_map)

    # 图形设置
    for index in range(1):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')

        # 根据索引选择颜色
        if index == 0:
            colors_ = 'lightgray'
        # elif index == 1:
        #     colors_ = HC_colors
        # elif index == 2:
        #     colors_ = PD_colors

        # 绘制 3D 散点图
        sc = ax.scatter(x, y, z, c=colors_, marker='o', alpha=0.8)

        # 去掉背景面、网格线和坐标轴
        ax.grid(False)  # 关闭网格线
        ax.set_axis_off()  # 关闭坐标轴
        ax.set_xticks([])  # 去掉 x 轴刻度
        ax.set_yticks([])  # 去掉 y 轴刻度
        ax.set_zticks([])  # 去掉 z 轴刻度

        # 添加 xy 平面
        x_min, x_max = x.min() - 0.1, x.max() + 0.1
        y_min, y_max = y.min() - 0.1, y.max() + 0.1
        x_range = np.linspace(x_min, x_max, 100)
        y_range = np.linspace(y_min, y_max, 100)
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.zeros_like(X)  # xy 平面的 Z 值为 0

        # 绘制 xy 平面
        ax.plot_surface(X, Y, Z, color='white', alpha=0.1, rstride=100, cstride=100, edgecolor='none')

        # 在 xy 平面上绘制投影点
        ax.scatter(x, y, np.zeros_like(z), c=colors_, marker='o', alpha=0.8)

        # 添加 xy 平面的网格线
        grid_lines = 5  # 网格线数量
        x_grid = np.linspace(x_min, x_max, grid_lines)
        y_grid = np.linspace(y_min, y_max, grid_lines)
        for x_val in x_grid:
            ax.plot([x_val, x_val], [y_min, y_max], [0, 0], color='black', alpha=0.2)
        for y_val in y_grid:
            ax.plot([x_min, x_max], [y_val, y_val], [0, 0], color='black', alpha=0.2)

        # 保存每个子图

        save_path = os.path.join(r'D:\Project\Parkinson_Diagnosis\Papers\Parkinson\xai_ppt', f'{file_name}_{index}_fig1_1.png')
        # plt.savefig(save_path, dpi=1000)
        # plt.show()
        plt.close()

    # 图形设置
    for index in range(2):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')

        # 根据索引选择颜色
        if index == 0:
            colors_ = 'lightgray'
        # elif index == 1:
        #     colors_ = HC_PD_colors

        # 绘制 3D 散点图
        sc = ax.scatter(x, y, z, c=colors_, marker='o', alpha=0.8)

        # 去掉背景面、网格线和坐标轴
        ax.grid(False)  # 关闭网格线
        ax.set_axis_off()  # 关闭坐标轴
        ax.set_xticks([])  # 去掉 x 轴刻度
        ax.set_yticks([])  # 去掉 y 轴刻度
        ax.set_zticks([])  # 去掉 z 轴刻度

        # 添加 xy 平面
        x_min, x_max = x.min() - 0.1, x.max() + 0.1
        y_min, y_max = y.min() - 0.1, y.max() + 0.1
        x_range = np.linspace(x_min, x_max, 100)
        y_range = np.linspace(y_min, y_max, 100)
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.zeros_like(X)  # xy 平面的 Z 值为 0

        # 绘制 xy 平面
        ax.plot_surface(X, Y, Z, color='white', alpha=0.1, rstride=100, cstride=100, edgecolor='none')
        ax.scatter(center_x, center_y, center_z, c='lightgray', marker='o', alpha=0.8, s=5)

        # 在 xy 平面上绘制投影点
        ax.scatter(x, y, np.zeros_like(z), c=colors_, marker='o', alpha=0.8)

        # 添加 xy 平面的网格线
        grid_lines = 5  # 网格线数量
        x_grid = np.linspace(x_min, x_max, grid_lines)
        y_grid = np.linspace(y_min, y_max, grid_lines)
        for x_val in x_grid:
            ax.plot([x_val, x_val], [y_min, y_max], [0, 0], color='black', alpha=0.2)
        for y_val in y_grid:
            ax.plot([x_min, x_max], [y_val, y_val], [0, 0], color='black', alpha=0.2)

        # 保存每个子图

        save_path = os.path.join(r'D:\Project\Parkinson_Diagnosis\Papers\Parkinson\xai_ppt',
                                 f'{file_name}_{index}_fig1_2.png')
        # plt.savefig(save_path, dpi=1000)
        plt.show()
        plt.close()




def plot_fig2(normalized_pc, colors_map, mask, out_path, file_name, data_pattern, if_save=False):
    '''
        Display and save the results, where the colors represent the weights
        '''
    rows_to_delete = np.arange(0, 1001)  # 要删除的行索引
    normalized_pc = np.delete(normalized_pc, rows_to_delete, axis=0)

    x = normalized_pc[:, 0]
    y = normalized_pc[:, 1]
    z = normalized_pc[:, 2] + 0.35
    # Creating a Point Cloud
    points = np.vstack((x, y, z)).T

    begin = 5000
    end = 5200
    # 设置颜色，默认黑色，100到150索引的点为深红色
    colors_ = np.array([to_rgba('gray')] * len(x))  # 将所有点颜色设置为黑色的RGBA格式
    colors_[begin:end] = to_rgba('gray')  # 第100到150个点设置为深红色的RGBA格式

    # 图形设置
    for index in range(3):

        # 根据索引选择颜色
        if index == 0:
            fig = plt.figure(figsize=(6, 2))
            ax = fig.add_subplot(111)
            ax.plot(range(begin), x[:begin], color='gray', label='X curve', linewidth=5)
            ax.plot(range(begin, end + 1), x[begin:end + 1], color='black', linewidth=5)
            ax.plot(range(end + 1, len(x)), x[end + 1:], color='gray', linewidth=5)
            ax.axis('off')
            save_path = os.path.join(r'D:\Project\Parkinson_Diagnosis\Papers\Parkinson\xai_ppt',
                                     f'{file_name}_{index}_fig2.png')
            # plt.savefig(save_path, dpi=1000)
            # plt.show()
            plt.close()

        elif index == 1:
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111, projection='3d')
            # 去掉背景面、网格线和坐标轴
            ax.grid(False)  # 关闭网格线
            ax.set_axis_off()  # 关闭坐标轴
            ax.set_xticks([])  # 去掉 x 轴刻度
            ax.set_yticks([])  # 去掉 y 轴刻度
            ax.set_zticks([])  # 去掉 z 轴刻度

            # 添加 xy 平面
            x_min, x_max = x.min() - 0.1, x.max() + 0.1
            y_min, y_max = y.min() - 0.1, y.max() + 0.1
            x_range = np.linspace(x_min, x_max, 100)
            y_range = np.linspace(y_min, y_max, 100)
            X, Y = np.meshgrid(x_range, y_range)
            Z = np.zeros_like(X)  # xy 平面的 Z 值为 0
            # 在 xy 平面上绘制投影点，应用颜色
            ax.scatter(x, y, np.zeros_like(z), c=colors_, marker='o', alpha=0.8, s=1)
            # 去掉坐标轴、标签和标题
            ax.axis('off')  # 关闭坐标轴
            save_path = os.path.join(r'D:\Project\Parkinson_Diagnosis\Papers\Parkinson\xai_ppt',
                                     f'{file_name}_{index}_fig2.png')
            # plt.savefig(save_path, dpi=1000)
            # plt.show()
            plt.close()

            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111)
            # 绘制 xy 曲线
            ax.plot(x, y, color='black', linewidth=4)
            # 隐藏坐标轴
            ax.axis('off')
            # 保存图像
            save_path = os.path.join(r'D:\Project\Parkinson_Diagnosis\Papers\Parkinson\xai_ppt',
                                     f'{file_name}_{index}__fig2.png')
            plt.savefig(save_path, dpi=1000)
            # plt.show()
            plt.close()



        elif index == 2:
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111, projection='3d')


            # 绘制 3D 散点图
            sc = ax.scatter(x, y, z, c=colors_, marker='o', alpha=0.8, s=3)

            # 去掉背景面、网格线和坐标轴
            ax.grid(False)  # 关闭网格线
            ax.set_axis_off()  # 关闭坐标轴
            ax.set_xticks([])  # 去掉 x 轴刻度
            ax.set_yticks([])  # 去掉 y 轴刻度
            ax.set_zticks([])  # 去掉 z 轴刻度

            # 添加 xy 平面
            x_min, x_max = x.min() - 0.1, x.max() + 0.1
            y_min, y_max = y.min() - 0.1, y.max() + 0.1
            x_range = np.linspace(x_min, x_max, 100)
            y_range = np.linspace(y_min, y_max, 100)
            X, Y = np.meshgrid(x_range, y_range)
            Z = np.zeros_like(X)  # xy 平面的 Z 值为 0

            # 绘制 xy 平面
            ax.plot_surface(X, Y, Z, color='white', alpha=0.1, rstride=100, cstride=100, edgecolor='none')

            # 在 xy 平面上绘制投影点
            ax.scatter(x, y, np.zeros_like(z), c=colors_, marker='o', alpha=0.8, s=3)

            # 添加 xy 平面的网格线
            grid_lines = 5  # 网格线数量
            x_grid = np.linspace(x_min, x_max, grid_lines)
            y_grid = np.linspace(y_min, y_max, grid_lines)
            for x_val in x_grid:
                ax.plot([x_val, x_val], [y_min, y_max], [0, 0], color='black', alpha=0.2)
            for y_val in y_grid:
                ax.plot([x_min, x_max], [y_val, y_val], [0, 0], color='black', alpha=0.2)

            save_path = os.path.join(r'D:\Project\Parkinson_Diagnosis\Papers\Parkinson\xai_ppt',
                                     f'{file_name}_{index}_fig2.png')
            # plt.savefig(save_path, dpi=1000)
            plt.show()
            # plt.close()


def pc_normalize_all(data):
    '''
    Normalize the point cloud
    '''
    temp_data = data.copy()

    xy_point = data[:, :2]
    xy_point = (xy_point - np.mean(xy_point, axis=0)) / np.max(
        np.sqrt(np.sum(np.power((xy_point - np.mean(xy_point, axis=0)), 2), axis=1)))
    temp_data[:, :2] = xy_point

    feature_point = data[:, 2:]
    temp_data[:, 2:] = (feature_point - np.min(feature_point, axis=0)) / (np.max(feature_point, axis=0) - np.min(feature_point, axis=0))

    return temp_data


def plot_fig2_2(normalized_pc, colors_map, mask, out_path, file_name, data_pattern, if_save=False):
    '''
        Display and save the results, where the colors represent the weights
        '''
    window_size = 7000
    stride_size = 260

    segments = []
    num_points = normalized_pc.shape[0]  # 点云的总点数
    for start in range(0, num_points - window_size + 1, stride_size):
        segment = normalized_pc[start:start + window_size]
        nor_seg = pc_normalize_all(segment)
        segments.append(nor_seg)

    for i in range(len(segments)):
        x = segments[i][:, 0]
        y = segments[i][:, 1]
        z = segments[i][:, 2]
        # Creating a Point Cloud
        points = np.vstack((x, y, z)).T

        colors_ = np.array([to_rgba('gray')] * len(x))  # 将所有点颜色设置为黑色的RGBA格式

        # 图形设置
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制 3D 散点图
        sc = ax.scatter(x, y, z, c=colors_, marker='o', alpha=0.8, s=5)
        # ax.view_init(elev=30, azim=45)


        save_path = os.path.join(r'D:\Project\Parkinson_Diagnosis\Papers\Parkinson\xai_ppt',
                                 f'{file_name}_{i}_fig2_2.png')
        # plt.savefig(save_path, dpi=1000)
        plt.show()
        # plt.close()



def plot_fig2_3(normalized_pc, colors_map, mask, out_path, file_name, data_pattern, if_save=False):
    '''
        Display and save the results, where the colors represent the weights
    '''
    # normalized_pc = np.delete(normalized_pc, list(range(4300,5500)), axis=0)
    x = normalized_pc[:, 0]
    y = normalized_pc[:, 1]
    z = normalized_pc[:, 2] + 0.35
    # Creating a Point Cloud
    points = np.vstack((x, y, z)).T


    c = '#be002f' #f07b3f
    # 设置颜色，默认黑色，100到150索引的点为深红色
    colors_ = np.array([to_rgba('lightgray')] * len(x))  # 将所有点颜色设置为黑色的RGBA格式
    # colors_[5150:5350] = to_rgba(c)  # 第100到150个点设置为深红色的RGBA格式
    colors_[5150:5400] = to_rgba(c)
    # colors_[:400] = to_rgba(c)  # 第100到150个点设置为深红色的RGBA格式

    # 图形设置
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制 3D 散点图
    sc = ax.scatter(x, y, z, c=colors_, marker='o', alpha=0.8)

    # 去掉背景面、网格线和坐标轴
    ax.grid(False)  # 关闭网格线
    ax.set_axis_off()  # 关闭坐标轴
    ax.set_xticks([])  # 去掉 x 轴刻度
    ax.set_yticks([])  # 去掉 y 轴刻度
    ax.set_zticks([])  # 去掉 z 轴刻度

    # 添加 xy 平面
    x_min, x_max = x.min() - 0.1, x.max() + 0.1
    y_min, y_max = y.min() - 0.1, y.max() + 0.1
    x_range = np.linspace(x_min, x_max, 100)
    y_range = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)  # xy 平面的 Z 值为 0

    # 绘制 xy 平面
    ax.plot_surface(X, Y, Z, color='white', alpha=0.1, rstride=100, cstride=100, edgecolor='none')

    # 在 xy 平面上绘制投影点
    ax.scatter(x, y, np.zeros_like(z), c=colors_, marker='o', alpha=0.8)

    # 添加 xy 平面的网格线
    grid_lines = 5  # 网格线数量
    x_grid = np.linspace(x_min, x_max, grid_lines)
    y_grid = np.linspace(y_min, y_max, grid_lines)
    for x_val in x_grid:
        ax.plot([x_val, x_val], [y_min, y_max], [0, 0], color='black', alpha=0.2)
    for y_val in y_grid:
        ax.plot([x_min, x_max], [y_val, y_val], [0, 0], color='black', alpha=0.2)

    save_path = os.path.join(r'D:\Project\Parkinson_Diagnosis\Papers\Parkinson\xai_ppt',
                             f'{file_name}_fig2_3.png')
    # plt.savefig(save_path, dpi=1000)
    plt.show()
    # plt.close()



def plot_fig3(normalized_pc, colors_map, mask, out_path, file_name, data_pattern, if_save=False):
    '''
        Display and save the results, where the colors represent the weights
    '''
    normalized_pc = np.delete(normalized_pc, list(range(5100))+list(range(5450,7262)), axis=0)

    sample_size = int(len(normalized_pc) * 0.1)
    # 随机选择点的索引
    sample_indices = np.sort(np.random.choice(len(normalized_pc), sample_size, replace=False))
    # 根据采样后的索引选择点
    normalized_pc = normalized_pc[sample_indices]

    x = normalized_pc[:, 0]
    y = normalized_pc[:, 1]
    z = normalized_pc[:, 2]

    z_min = np.min(z)  # 获取z坐标的最小值
    z = z - z_min + 0.05 # 将z坐标平移，使最小值为0

    # 添加一个中心点（可以是点云的几何中心）
    center_x = np.mean(x)
    center_y = np.mean(y)
    center_z = np.mean(z)

    # Creating a Point Cloud
    points = np.vstack((x, y, z)).T


    c = '#be002f' #f07b3f
    # 设置颜色，默认黑色，100到150索引的点为深红色
    colors_ = np.array([to_rgba('gray')] * len(x))  # 将所有点颜色设置为黑色的RGBA格式
    colors_[5:-5] = to_rgba(c)  # 第100到150个点设置为深红色的RGBA格式

    # 图形设置
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制 3D 散点图
    sc = ax.scatter(x, y, z, c=colors_, marker='o', alpha=0.8, s=100)

    ax.scatter(center_x, center_y, center_z, c='black', marker='o', s=100)

    # 去掉背景面、网格线和坐标轴
    ax.grid(False)  # 关闭网格线
    ax.set_axis_off()  # 关闭坐标轴
    ax.set_xticks([])  # 去掉 x 轴刻度
    ax.set_yticks([])  # 去掉 y 轴刻度
    ax.set_zticks([])  # 去掉 z 轴刻度

    save_path = os.path.join(r'D:\Project\Parkinson_Diagnosis\Papers\Parkinson\xai_ppt',
                             f'{file_name}_fig2_4.png')
    # plt.savefig(save_path, dpi=1000)
    plt.show()
    # plt.close()


def plot_fig5(normalized_pc, colors_map, K_num, name, if_save=False):
    '''
    Display and save the results, where the colors represent the weights
    '''
    x = normalized_pc[:, 0]
    y = normalized_pc[:, 1]
    z = normalized_pc[:, 2] + 0.35
    # Creating a Point Cloud
    points = np.vstack((x, y, z)).T

    # Using the coolwarm colormap
    cmap = cm.get_cmap('coolwarm')
    colors = cmap(colors_map)
    PD_colors_map = [0.5 if i < 0.5 else i for i in colors_map]
    PD_colors = cmap(PD_colors_map)
    HC_colors_map = [0.5 if i > 0.5 else i for i in colors_map]
    HC_colors = cmap(HC_colors_map)

    # 图形设置
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 根据索引选择颜色
    colors_ = colors

    # 绘制 3D 散点图
    sc = ax.scatter(x, y, z, c=colors_, marker='o', alpha=0.8)

    # 去掉背景面、网格线和坐标轴
    ax.grid(False)  # 关闭网格线
    ax.set_axis_off()  # 关闭坐标轴
    ax.set_xticks([])  # 去掉 x 轴刻度
    ax.set_yticks([])  # 去掉 y 轴刻度
    ax.set_zticks([])  # 去掉 z 轴刻度

    # 添加 xy 平面
    x_min, x_max = x.min() - 0.1, x.max() + 0.1
    y_min, y_max = y.min() - 0.1, y.max() + 0.1
    x_range = np.linspace(x_min, x_max, 100)
    y_range = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)  # xy 平面的 Z 值为 0

    # 绘制 xy 平面
    ax.plot_surface(X, Y, Z, color='white', alpha=0.1, rstride=100, cstride=100, edgecolor='none')

    # 在 xy 平面上绘制投影点
    ax.scatter(x, y, np.zeros_like(z), c=colors_, marker='o', alpha=0.8)

    # 添加 xy 平面的网格线
    grid_lines = 5  # 网格线数量
    x_grid = np.linspace(x_min, x_max, grid_lines)
    y_grid = np.linspace(y_min, y_max, grid_lines)
    for x_val in x_grid:
        ax.plot([x_val, x_val], [y_min, y_max], [0, 0], color='black', alpha=0.2)
    for y_val in y_grid:
        ax.plot([x_min, x_max], [y_val, y_val], [0, 0], color='black', alpha=0.2)

    if if_save:
        save_path = os.path.join(r'data/ParkinsonHW', f'fig5_{K_num}.png')
        plt.savefig(save_path, dpi=1000)
        plt.close()
    else:
        plt.show()

    # 绘制二维散点图（PD_colors）
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, c=PD_colors, marker='o', alpha=0.8)
    x_grid = np.linspace(x.min(), x.max(), 5)
    y_grid = np.linspace(y.min(), y.max(), 5)
    for x_line in x_grid:
        plt.axvline(x=x_line, color='black', linestyle='--', alpha=0.5)
    for y_line in y_grid:
        plt.axhline(y=y_line, color='black', linestyle='--', alpha=0.5)
    plt.axis('off')  # 去掉坐标轴

    if if_save:
        save_path_pd = os.path.join(r'data/ParkinsonHW', f'fig5_PD_{K_num}.png')
        plt.savefig(save_path_pd, dpi=1000)
        plt.close()
    else:
        plt.show()

    # 绘制二维散点图（HC_colors）
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, c=HC_colors, marker='o', alpha=0.8)
    for x_line in x_grid:
        plt.axvline(x=x_line, color='black', linestyle='--', alpha=0.5)
    for y_line in y_grid:
        plt.axhline(y=y_line, color='black', linestyle='--', alpha=0.5)
    plt.axis('off')  # 去掉坐标轴

    if if_save:
        save_path_hc = os.path.join(r'data/ParkinsonHW', f'fig5_HC_{K_num}.png')
        plt.savefig(save_path_hc, dpi=1000)
        plt.close()
    else:
        plt.show()



def plot_fig7(normalized_pc, colors_map, out_file_path, model_name='LR', if_save=False):
    '''
    Display and save the results, where the colors represent the weights
    '''
    x = normalized_pc[:, 0]
    y = normalized_pc[:, 1]
    z = normalized_pc[:, 2] + 0.35
    # Creating a Point Cloud
    points = np.vstack((x, y, z)).T

    # Using the coolwarm colormap
    cmap = cm.get_cmap('coolwarm')
    colors = cmap(colors_map)
    PD_colors_map = [0.5 if i < 0.5 else i for i in colors_map]
    PD_colors = cmap(PD_colors_map)
    HC_colors_map = [0.5 if i > 0.5 else i for i in colors_map]
    HC_colors = cmap(HC_colors_map)

    # 图形设置
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 根据索引选择颜色
    colors_ = colors

    # 绘制 3D 散点图
    sc = ax.scatter(x, y, z, c=colors_, marker='o', alpha=0.8)

    # 去掉背景面、网格线和坐标轴
    ax.grid(False)  # 关闭网格线
    ax.set_axis_off()  # 关闭坐标轴
    ax.set_xticks([])  # 去掉 x 轴刻度
    ax.set_yticks([])  # 去掉 y 轴刻度
    ax.set_zticks([])  # 去掉 z 轴刻度

    # 添加 xy 平面
    x_min, x_max = x.min() - 0.1, x.max() + 0.1
    y_min, y_max = y.min() - 0.1, y.max() + 0.1
    x_range = np.linspace(x_min, x_max, 100)
    y_range = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)  # xy 平面的 Z 值为 0

    # 绘制 xy 平面
    ax.plot_surface(X, Y, Z, color='white', alpha=0.1, rstride=100, cstride=100, edgecolor='none')

    # 在 xy 平面上绘制投影点
    ax.scatter(x, y, np.zeros_like(z), c=colors_, marker='o', alpha=0.8)

    # 添加 xy 平面的网格线
    grid_lines = 5  # 网格线数量
    x_grid = np.linspace(x_min, x_max, grid_lines)
    y_grid = np.linspace(y_min, y_max, grid_lines)
    for x_val in x_grid:
        ax.plot([x_val, x_val], [y_min, y_max], [0, 0], color='black', alpha=0.2)
    for y_val in y_grid:
        ax.plot([x_min, x_max], [y_val, y_val], [0, 0], color='black', alpha=0.2)

    if if_save:
        save_path = os.path.join(out_file_path, f'{model_name}.png')
        plt.savefig(save_path, dpi=1000)
        plt.close()
    else:
        plt.show()

