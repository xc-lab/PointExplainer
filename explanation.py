#  -*- coding: utf-8 -*-
'''
A dedicated interpreter was trained for each subject, and perturbation analysis was performed to verify the reliability of the interpretation results.

@author: xuechao.wang@ugent.be
'''

import shutil
import numpy as np
import torch
import os
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from torch.utils.data import DataLoader

from bag import xai_pointcloud
from model.pointnet import PointNet
from bag.utils.generic_utils import segment_fn
from utils import data_reading, get_testing_patches, pc_normalize_all
from dataset import MyDataset_test

import warnings
warnings.filterwarnings('ignore')


def batch_predict(pc, window_size, stride_size, model_name):
    pred_labels = []  # For each sequence, store the true category and model prediction category of the segmented patch
    patch_dataset = get_testing_patches(pc, window_size, stride_size)
    test_dataset = MyDataset_test(dataset=patch_dataset, name='PD', transform=None)
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


def save_metrics(file_name, metrics, out_path, data_pattern):
    with open(os.path.join(out_path, 'metrics_results.txt'), 'a') as f:
        f.write(f'{file_name}-{str(data_pattern)}:\n')
        for key, value in metrics.items():
            f.write(f'  {key}: {value}\n')
            # print(f'{file_name} - {key}: {value}')
        f.write('\n')

def save_attributes(file_name, attributes, out_path, data_pattern):
    with open(os.path.join(out_path, 'attributes_results.txt'), 'a') as f:
        f.write(f'{file_name}-{str(data_pattern)}:\n')
        f.write(' '.join(map(str, attributes)) + '\n')
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
    pad_size = window_size // 2
    if window_size % 2 == 0:
        padded_data = np.pad(data, (pad_size, pad_size-1), mode='edge')
    else:
        padded_data = np.pad(data, (pad_size, pad_size), mode='edge')
    smoothed_data = np.convolve(padded_data, np.ones(window_size) / window_size, mode='valid')
    return smoothed_data


def saliency_map(mask):
    impact = moving_average(mask, max(1, int(len(mask) * 0.1 * 0.5)))
    min_value, max_value = np.min(impact), np.max(impact)
    abs_max_value = max(abs(min_value), abs(max_value))
    if abs_max_value > 0:
        colors_map = 0.5 + (impact / abs_max_value) * 0.5
    else:
        colors_map = np.full_like(impact, 0.5)
    if len(colors_map) > 1:
        colors_map[0] = 0
        colors_map[1] = 1
    return colors_map.tolist()



def plot_pc(normalized_pc, colors_map, out_file_path, model_name='LR', if_save=False):
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

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(x, y, z, c=colors, marker='o', alpha=0.8)
    ax.grid(False)
    ax.set_axis_off()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    x_min, x_max = x.min() - 0.1, x.max() + 0.1
    y_min, y_max = y.min() - 0.1, y.max() + 0.1
    x_range = np.linspace(x_min, x_max, 100)
    y_range = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)

    ax.plot_surface(X, Y, Z, color='white', alpha=0.1, rstride=100, cstride=100, edgecolor='none')

    ax.scatter(x, y, np.zeros_like(z), c=colors, marker='o', alpha=0.8)
    grid_lines = 5
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


def perturbation_analysis(pc, mask, influ_name, segment_num, window_size, stride_size, model_name, batch_predict, segment_fn):
    loc_pre = batch_predict(pc, window_size, stride_size, model_name)
    segments, fudged_data = segment_fn(pc, segment_num)
    segment_attributes = []

    n_segments = np.unique(segments)
    for idx in n_segments:
        indices = np.where(segments == idx)[0]

        weight = np.unique(mask[indices])
        assert len(weight) == 1, f"Error: Expected number of weights in one segment to be 1, but got {len(weight)}"

        reversed_data = deepcopy(pc)
        for y_id in influ_name:
            reversed_data[indices, y_id] = fudged_data[indices, y_id]

        mask_pre = batch_predict(reversed_data, window_size, stride_size, model_name)
        influence = (loc_pre[1] - mask_pre[1]) / (loc_pre[1] + 1e-20)
        segment_attributes.append([weight[0], loc_pre[1], mask_pre[1], influence])

    return segment_attributes



def plot_perturbation_result(segment_attributes, out_file_path, if_save=False):
    x_values = list(range(1, len(segment_attributes) + 1))
    y_values = [item[3] * 100 for item in segment_attributes]
    val_values = [item[0] for item in segment_attributes]

    sorted_pairs = sorted(zip(val_values, y_values))
    val_values, y_values = zip(*sorted_pairs)
    val_values = list(val_values)
    y_values = list(y_values)

    colors = ['darkblue' if val < 0 else 'darkred' if val > 0 else 'gray' for val in val_values]

    plt.figure(figsize=(10, 8))
    bars = plt.bar(x_values, y_values, color=colors)
    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel('Superpoint Index', fontsize=25)
    plt.ylabel('Influence(in %)', fontsize=25)
    for i, bar in enumerate(bars):
        if colors[i] == 'darkred':
            va_position = 'bottom'
            color = 'black'
        else:
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

    file_name = 'H_P000-0001' # the subject ID
    data_pattern = 1 # 0 for SST, 1 for DST
    model_path = os.path.join('...') # here is the yourself checkpoint path


    out_path = os.path.join('./data', 'ParkinsonHW', f'explanation')
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device', device)
    model = PointNet()
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device)['state_dict'])
    model.eval()

    segment_attributes = []
    print('Processing: ' + 'data pattern:' + str(data_pattern) + ' | ' + ', file name:' + file_name)

    out_file_path = os.path.join(out_path, file_name+'-'+str(data_pattern))
    if not os.path.exists(out_file_path):
        os.mkdir(out_file_path)
    else:
        shutil.rmtree(out_file_path)
        os.mkdir(out_file_path)

    json_file_path = os.path.join('...') # here is the subject data path
    temp_data, L = data_reading(json_file_path, 'ParkinsonHW', data_pattern)
    pc = temp_data[:, [0, 1, 15]]

    explainer = xai_pointcloud.XaiPointcloudExplainer(random_state=2, verbose=False)
    explanation = explainer.explain_instance(pc,
                                             y_ids=[0, 1, 2],
                                             classifier_fn=batch_predict,  # the predicted function
                                             window_size=512,
                                             stride_size=8,
                                             model_name='PointNet',
                                             segment_fn=segment_fn,
                                             num_samples=200, # The number of neighborhood data <1024
                                             num_segments=10,
                                             model_regressor='xgb'
                                             )

    mask, metrics = explanation.get_weight_and_shap(1)  # 0-KT， 1-PD
    np.save(os.path.join(out_file_path, 'weight.npy'), mask)

    # Save the fitting results of the trained linear regression model for subject file.
    metrics_keys = ['True', 'Pre']
    file_metrics = {}
    for i, key in enumerate(metrics_keys):
        file_metrics[key] = metrics[i]

    save_metrics(file_name, file_metrics, out_path, data_pattern)

    # The attribution map is mapped to colors, and point clouds are drawn.
    mask = np.load(os.path.join(out_path, file_name + '-' + str(data_pattern), 'weight.npy')).reshape(-1)
    colors_map = saliency_map(mask)
    normalized_pc = pc_normalize_all(pc)
    plot_pc(normalized_pc, colors_map, out_file_path, model_name='xgb', if_save=True)

    # Perform perturbation analysis, and plot and save the perturbation results.
    segment_attributes = perturbation_analysis(
        pc=pc,
        mask=mask,
        influ_name=[0, 1, 2],
        segment_num=10,
        window_size=512,
        stride_size=8,
        model_name='PointNet',
        batch_predict=batch_predict,
        segment_fn=segment_fn
    )
    plot_perturbation_result(segment_attributes, out_file_path, if_save=True)
    save_attributes(file_name, segment_attributes, out_path, data_pattern)

















































