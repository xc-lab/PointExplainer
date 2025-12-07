
import numpy as np
import torch
import os
from copy import deepcopy
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm

from model.pointnet import PointNet
from bag.utils.generic_utils import segment_fn
from utils import data_reading, get_testing_patches, pc_normalize_all
from dataset import MyDataset_test

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)






def batch_predict(pc, window_size, stride_size, model_name):
    pred_labels = []  # For each sequence, store the true category and model prediction category of the segmented patch
    patch_dataset = get_testing_patches(pc, window_size, stride_size)
    test_dataset = MyDataset_test(dataset=patch_dataset, name=label_type, transform=None)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0)
    # test_bar = tqdm(test_loader)
    for data in test_loader:
        inputs, labels = data
        inputs = inputs.to(device)
        with torch.no_grad():
            if model_name == 'PointNet':
                pred, trans_feat, _ = model(inputs)
            pred = torch.max(pred, dim=-1)[1]

            pred_data = pred.data.cpu().detach().numpy().flatten()
            for k in np.arange(len(inputs)):
                pred_labels.append(pred_data[k])

    result = len([value for value in pred_labels if value == 1]) / len(pred_labels) # Proportion of PD category patches
    prob = np.array([1-result, result], dtype=np.float32)
    return prob



def save_attributes(file_name, attributes, out_path, data_pattern):
    with open(os.path.join(out_path, 'attributes_results.txt'), 'a') as f:
        f.write(f'{file_name}-{str(data_pattern)}:\n')
        f.write(' '.join(map(str, attributes)) + '\n')  # 将列表元素转为字符串，使用空格连接
        f.write('\n')



def load_processed_files(results_path):
    if not os.path.exists(results_path):
        return set()
    processed_files = set()
    with open(results_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.strip().endswith(':'):
                processed_files.add(line.strip().rstrip(':'))
    return processed_files


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
    # 计算平滑后的影响值
    impact = moving_average(mask, max(1, int(len(mask) * 0.1 * 0.5)))  # 确保窗口大小至少为1
    min_value, max_value = np.min(impact), np.max(impact)
    abs_max_value = max(abs(min_value), abs(max_value))  # 绝对值最大值

    # 归一化映射
    if abs_max_value > 0:
        colors_map = 0.5 + (impact / abs_max_value) * 0.5
    else:
        colors_map = np.full_like(impact, 0.5)  # 如果全为0，初始化为0.5

    # 确保颜色范围有对比
    if len(colors_map) > 1:
        colors_map[0] = 0
        colors_map[1] = 1

    return colors_map.tolist()



def plot_pc(normalized_pc, colors_map, out_file_path, model_name='LR', if_save=False):
    '''
    Display and save the results, where the colors represent the weights
    '''
    normalized_pc = normalized_pc[2:, :]
    x = normalized_pc[:, 0]
    y = normalized_pc[:, 1]
    z = normalized_pc[:, 2] + 0.35
    # Creating a Point Cloud
    points = np.vstack((x, y, z)).T

    # Using the coolwarm colormap
    cmap = cm.get_cmap('coolwarm')
    colors = cmap(colors_map)
    colors = colors[2:, :]


    # 图形设置
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制 3D 散点图
    sc = ax.scatter(x, y, z, c=colors, marker='o', alpha=0.8)

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
    ax.scatter(x, y, np.zeros_like(z), c=colors, marker='o', alpha=0.8)

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


def perturbation_analysis(pc, mask, influ_name, segment_num, window_size, stride_size, model_name, batch_predict,
                          segment_fn):
    """
    进行扰动分析，并计算每个特征对模型结果的影响。
    返回：
    segment_attributes (list): 包含每个特征影响的属性结果
    """

    # 进行初始预测
    loc_pre = batch_predict(pc, window_size, stride_size, model_name)

    segments, fudged_data = segment_fn(pc, segment_num)  # 存储的是每个片段的均值，保持与原始数据相同的尺寸
    segment_attributes = []

    # 对每个片段进行扰动分析
    n_segments = np.unique(segments)
    for idx in n_segments:
        indices = np.where(segments == idx)[0]  # 获取所有等于该值的索引, 也就是该片段对应的索引

        weight = np.unique(mask[indices])
        assert len(weight) == 1, f"Error: Expected number of weights in one segment to be 1, but got {len(weight)}"


        reversed_data = deepcopy(pc)
        for y_id in influ_name:
            reversed_data[indices, y_id] = fudged_data[indices, y_id]

        # 进行扰动后的预测
        mask_pre = batch_predict(reversed_data, window_size, stride_size, model_name)
        # 计算扰动的影响度
        influence = (loc_pre[1] - mask_pre[1]) / (loc_pre[1] + 1e-20)
        # 将结果存入属性列表
        segment_attributes.append([weight[0], loc_pre[1], mask_pre[1], influence])

    return segment_attributes



def plot_perturbation_result(segment_attributes, out_file_path, if_save=False):
    '''
    画出扰动结果
    '''
    x_values = list(range(1, len(segment_attributes) + 1))  # 第几个 influence
    y_values = [item[3] * 100 for item in segment_attributes]
    val_values = [item[0] for item in segment_attributes]

    sorted_pairs = sorted(zip(val_values, y_values))  # 按 val_values 排序
    val_values, y_values = zip(*sorted_pairs)  # 拆分回两个列表
    val_values = list(val_values)
    y_values = list(y_values)

    colors = ['darkblue' if val < 0 else 'darkred' if val > 0 else 'gray' for val in val_values]

    plt.figure(figsize=(10, 8))
    bars = plt.bar(x_values, y_values, color=colors)
    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel('Superpoint Index', fontsize=25)
    plt.ylabel('Influence(in %)', fontsize=25)
    for i, bar in enumerate(bars):
        if colors[i] == 'darkred':  # 对于红色柱子
            va_position = 'bottom'
            color = 'black'
        else:  # 对于其他颜色的柱子（如蓝色）
            va_position = 'top'
            color = 'black'
        if bar.get_height() == 0:
            va_position = 'center'
            color = 'black'
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f'{val_values[i]:.4f}', ha='center', va=va_position, fontsize=13, color=color)
    legend_patches = [
        mpatches.Patch(color='darkred', label='attribution value > 0'),
        mpatches.Patch(color='darkblue', label='attribution value < 0'),
        mpatches.Patch(color='gray', label='attribution value = 0')
    ]
    plt.legend(handles=legend_patches, fontsize=15)
    plt.xticks(ticks=x_values, labels=x_values)
    plt.tick_params(labelsize=20)

    if if_save:
        save_path = os.path.join(out_file_path, 'infl.png')
        plt.savefig(save_path, dpi=1000)
        plt.close()
    else:
        plt.show()



if __name__=='__main__':

    neighbor_num = 200
    explainer_type = 'xgb'  # ['ridge', 'dt', 'lasso', 'rf', 'lr', 'elasticnet', 'xgb']

    for neighbor_num in [200]:

        influ_name = [0, 1, 2]  # See which features affect the model results

        segment_num = 10
        model_name = 'PointNet'
        dataset = 'ParkinsonHW'
        stride_size = 8

        out_path = os.path.join('./data', dataset, f'result_K_{neighbor_num}_{explainer_type}__')
        if not os.path.exists(out_path):
            os.mkdir(out_path)

        results_path = os.path.join(out_path, 'attributes_results.txt')
        processed_files = load_processed_files(results_path)

        for data_pattern in [0, 1]:
            window_size = 256 if data_pattern == 0 else 512
            for fold_d in ['fold_1', 'fold_2', 'fold_3']:
                if data_pattern == 0:
                    if fold_d == 'fold_1':
                        time_date = '2024_11_20_17_05_26'
                    elif fold_d == 'fold_2':
                        time_date = '2024_11_20_17_22_30'
                    elif fold_d == 'fold_3':
                        time_date = '2024_11_20_17_46_32'
                elif data_pattern == 1:
                    if fold_d == 'fold_1':
                        time_date = '2024_11_19_18_32_47'
                    elif fold_d == 'fold_2':
                        time_date = '2024_11_19_20_19_30'
                    elif fold_d == 'fold_3':
                        time_date = '2024_11_19_20_39_53'

                model_path = os.path.join('log_dir', time_date, 'checkpoints/best_model/PointNet_cls.pth')
                file_path = os.path.join('./data', dataset, fold_d, 'test_names.txt')

                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                print('device', device)

                if model_name == 'PointNet':
                    model = PointNet()
                model.to(device)
                model.load_state_dict(torch.load(model_path, map_location=device)['state_dict'])
                model.eval()

                with open(file_path, 'r') as f:
                    for line in f.readlines():

                        segment_attributes = []

                        label_type = line.strip().split('/')[0]
                        if label_type == 'KT':
                            label_tru = 0
                        elif label_type == 'PD':
                            label_tru = 1

                        file_name = line.strip().split('/')[1].split('.')[0]

                        if file_name+'-'+str(data_pattern) in processed_files:
                            print(f'Skipping already processed file: {str(data_pattern)}-{file_name}')
                            continue


                        print('Processing: ' + str(data_pattern) + ', ' + fold_d + ', file name:' + file_name)

                        out_file_path = os.path.join(out_path, file_name+'-'+str(data_pattern))
                        if not os.path.exists(out_file_path):
                            os.mkdir(out_file_path)

                        json_file_path = os.path.join('./data', dataset, 'raw_data', line.strip())
                        temp_data, L = data_reading(json_file_path, dataset, data_pattern)
                        # 0-x,1-y,2-z,3-p,4-g,5-t,6-v,7-acc,8-jerk,9-dx,10-dy,11-da,12-dl,13-dp,14-dt,15-radius,16-angle,17-curvature,18-idx
                        if L > window_size:
                            pc = temp_data[:, [0, 1, 3]] if data_pattern == 0 else temp_data[:, [0, 1, 15]]
                            mask = np.load(os.path.join(out_path, file_name + '-' + str(data_pattern), 'weight.npy')).reshape(-1)


                            # 任务一：归因图映射成颜色，且画出点云
                            colors_map = saliency_map(mask)
                            normalized_pc = pc_normalize_all(pc)
                            plot_pc(normalized_pc, colors_map, out_file_path, model_name=explainer_type, if_save=True)


                            # 任务二：进行扰动分析，且画图和保存扰动结果
                            segment_attributes = perturbation_analysis(
                                                                        pc=pc,
                                                                        mask=mask,
                                                                        influ_name=influ_name,
                                                                        segment_num=segment_num,
                                                                        window_size=window_size,
                                                                        stride_size=stride_size,
                                                                        model_name=model_name,
                                                                        batch_predict=batch_predict,
                                                                        segment_fn=segment_fn
                                                                    )
                            plot_perturbation_result(segment_attributes, out_file_path, if_save=True)

                            save_attributes(file_name, segment_attributes, out_path, data_pattern)


















