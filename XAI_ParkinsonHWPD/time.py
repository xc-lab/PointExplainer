#  -*- coding: utf-8 -*-
'''
@author: xuechao.wang@ugent.be
'''
import numpy as np
import torch
import os
import shutil
from torch.utils.data import DataLoader

from time import perf_counter  # ADDED: timing
import warnings
warnings.filterwarnings('ignore')

from model.pointnet import PointNet
from utils import data_reading, get_testing_patches
from dataset import MyDataset_test

from bag import xai_pointcloud
from bag.utils.generic_utils import segment_fn


def batch_predict(pc, window_size, stride_size, model_name):
    """
    黑箱预测函数（供 LIME 调用）。
    返回 [P(HC), P(PD)]，其中 P(PD) 用“PD 片段占比”估计。
    注意：依赖外部变量 model/device/label_type（保持与你原来一致）。
    """
    pred_labels = []  # For each sequence, store the predicted category of the segmented patch
    patch_dataset = get_testing_patches(pc, window_size, stride_size)
    test_dataset = MyDataset_test(dataset=patch_dataset, name=label_type, transform=None)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0)

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

    # Proportion of PD category patches
    result = len([value for value in pred_labels if value == 1]) / (len(pred_labels) if len(pred_labels) > 0 else 1)
    prob = np.array([1 - result, result], dtype=np.float32)
    return prob


def save_metrics(file_name, metrics, out_path, data_pattern):
    with open(os.path.join(out_path, 'metrics_results.txt'), 'a') as f:
        f.write(f'{file_name}-{str(data_pattern)}:\n')
        for key, value in metrics.items():
            f.write(f'  {key}: {value}\n')
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


if __name__=='__main__':

    explainer_type = 'xgb'  # ['ridge', 'dt', 'lasso', 'rf', 'lr', 'elasticnet', 'xgb']
    neighbor_num = 200      # [10, 20, 50, 100, 150, 200, 250, 300]

    # 全局计时累积（跨 data_pattern / fold / 文件）
    total_times = []  # ADDED: PointNet(单次) + LIME
    pn_times = []     # ADDED: 仅 PointNet(单次) 滑窗推理
    lime_times = []   # ADDED: 仅 LIME（含内部多次调用 batch_predict）

    for neighbor_num in [200]:

        influ_name = [0, 1, 2]   # 查看哪些特征对模型结果的影响
        segment_num = 10         # 分割的超点数

        model_name = 'PointNet'
        dataset = 'ParkinsonHW'
        stride_size = 8

        out_path = os.path.join('./data', dataset, f'result_K_{neighbor_num}_{explainer_type}__')
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        results_path = os.path.join(out_path, 'metrics_results.txt')
        processed_files = load_processed_files(results_path)

        for data_pattern in [0]:
            if data_pattern == 0:
                window_size = 256
            elif data_pattern == 1:
                window_size = 512

            for fold_d in ['fold_1','fold_2','fold_3']:
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

                model_path = os.path.join('./log_dir', time_date, 'checkpoints/best_model/PointNet_cls.pth')
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
                        label_type = line.strip().split('/')[0]
                        file_name = line.strip().split('/')[1].split('.')[0]

                        if file_name+'-'+str(data_pattern) in processed_files:
                            print(f'Skipping already processed file: {str(data_pattern)}-{file_name}')
                            continue

                        print('Processing: ' + 'data pattern:' + str(data_pattern) + ' | ' + fold_d + ', file name:' + file_name)

                        out_file_path = os.path.join(out_path, file_name+'-'+str(data_pattern))
                        if not os.path.exists(out_file_path):
                            os.mkdir(out_file_path)
                        else:
                            shutil.rmtree(out_file_path)
                            os.mkdir(out_file_path)

                        json_file_path = os.path.join('./data', dataset, 'raw_data', line.strip())
                        temp_data, L = data_reading(json_file_path, dataset, data_pattern)
                        # 0-x,1-y,2-z,3-p,4-g,5-t,6-v,7-acc,8-jerk,9-dx,10-dy,11-dz,12-dp,13-dg,14-dt,15-radius,16-angle,17-curvature

                        # 长度不足窗口则跳过
                        if L <= window_size:
                            print(f"[Skip] {file_name}-{data_pattern}: length {L} <= window {window_size}")
                            continue

                        # 构造用于解释的输入通道
                        if data_pattern == 0:
                            pc = temp_data[:, [0, 1, 3]]   # [x, y, p]
                        elif data_pattern == 1:
                            pc = temp_data[:, [0, 1, 15]]  # [x, y, radius]

                        # ========================
                        # 计时：PointNet 单次滑窗推理
                        # ========================
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        t_pn0 = perf_counter()

                        _ = batch_predict(pc=pc,
                                          window_size=window_size,
                                          stride_size=stride_size,
                                          model_name=model_name)

                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        t_pn1 = perf_counter()
                        sample_pointnet_time = t_pn1 - t_pn0

                        # ========================
                        # 计时：LIME（包含内部多次 batch_predict -> PointNet）
                        # ========================
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        t_lime0 = perf_counter()

                        explainer = xai_pointcloud.XaiPointcloudExplainer(random_state=2, verbose=False)
                        explanation = explainer.explain_instance(
                            pc,
                            y_ids=influ_name,                 # 控制查看哪些特征的影响
                            classifier_fn=batch_predict,      # 黑箱预测函数
                            window_size=window_size,
                            stride_size=stride_size,
                            model_name=model_name,
                            segment_fn=segment_fn,
                            num_samples=neighbor_num,         # 邻域样本数
                            num_segments=segment_num,         # 超点数
                            model_regressor=explainer_type
                        )

                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        t_lime1 = perf_counter()
                        sample_lime_time = t_lime1 - t_lime0

                        # 汇总
                        sample_total_time = sample_pointnet_time + sample_lime_time

                        # 逐样本打印
                        print(f"[Timing] {file_name}-{data_pattern} | "
                              f"PointNet(诊断): {sample_pointnet_time:.9f}s | "
                              f"LIME(诊断+解释): {sample_lime_time:.9f}s | "
                              f"Total(先诊断后解释): {sample_total_time:.9f}s")

                        # 累积到全局
                        pn_times.append(sample_pointnet_time)
                        lime_times.append(sample_lime_time)
                        total_times.append(sample_total_time)

                        # ====== 原有保存逻辑 ======
                        mask, metrics = explanation.get_weight_and_shap(1)  # 0-KT， 1-PD
                        np.save(os.path.join(out_file_path, 'weight.npy'), mask)

                        # 保存每个文件的 训练的线性回归模型 的拟合效果结果
                        metrics_keys = ['True', 'Pre']
                        file_metrics = {}
                        for i, key in enumerate(metrics_keys):
                            file_metrics[key] = metrics[i]

                        save_metrics(file_name, file_metrics, out_path, data_pattern)

    # ========================
    # 全部样本完成后，打印均值
    # ========================
    if len(total_times) > 0:
        mean_total = float(np.mean(total_times))
        mean_pn = float(np.mean(pn_times)) if len(pn_times) > 0 else 0.0
        mean_lime = float(np.mean(lime_times)) if len(lime_times) > 0 else 0.0
        print(f"[Timing][Mean] over {len(total_times)} samples | "
              f"PointNet: {mean_pn:.6f}s | LIME: {mean_lime:.6f}s | Total: {mean_total:.6f}s")
    else:
        print("[Timing] No samples were processed, no timing stats available.")
