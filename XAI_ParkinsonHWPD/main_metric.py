#  -*- coding: utf-8 -*-
'''
@author: xuechao.wang@ugent.be
'''
import numpy as np
import torch
import os
import shutil
from torch.utils.data import DataLoader

from model.pointnet import PointNet
from utils import data_reading, get_testing_patches
from dataset import MyDataset_test

from bag import xai_pointcloud
from bag.utils.generic_utils import segment_fn

import warnings
warnings.filterwarnings('ignore')


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



def save_metrics(file_name, metrics, out_path, data_pattern):
    with open(os.path.join(out_path, 'metrics_results.txt'), 'a') as f:
        f.write(f'{file_name}-{str(data_pattern)}:\n')
        for key, value in metrics.items():
            f.write(f'  {key}: {value}\n')
            # print(f'{file_name} - {key}: {value}')
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

    explainer_type = 'xgb' # ['ridge', 'dt', 'lasso', 'rf', 'lr', 'elasticnet', 'xgb']
    neighbor_num = 200 # [10, 20, 50, 100, 150, 200, 250, 300]


    for neighbor_num in [200]:

        influ_name = [0, 1, 2]  # 查看哪些特征对模型结果的影响
        segment_num = 10 # 分割的超点数

        model_name = 'PointNet'
        dataset = 'ParkinsonHW'
        stride_size = 8

        out_path = os.path.join('./data', dataset, f'result_K_{neighbor_num}_{explainer_type}__')
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        results_path = os.path.join(out_path, 'metrics_results.txt')
        processed_files = load_processed_files(results_path)

        for data_pattern in [0, 1]:
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

                        print('Processing: ' + 'data pattern:' + str( data_pattern) + ' | ' + fold_d + ', file name:' + file_name)

                        out_file_path = os.path.join(out_path, file_name+'-'+str(data_pattern))
                        if not os.path.exists(out_file_path):
                            os.mkdir(out_file_path)

                        else:
                            shutil.rmtree(out_file_path)
                            os.mkdir(out_file_path)


                        json_file_path = os.path.join('./data', dataset, 'raw_data', line.strip())
                        temp_data, L = data_reading(json_file_path, dataset, data_pattern)
                        # 0-x,1-y,2-z,3-p,4-g,5-t,6-v,7-acc,8-jerk,9-dx,10-dy,11-dz,12-dp,13-dg,14-dt,15-radius,16-angle,17-curvature
                        if L > window_size:
                            if data_pattern == 0:
                                pc = temp_data[:, [0, 1, 3]]
                            elif data_pattern == 1:
                                pc = temp_data[:, [0, 1, 15]]

                            explainer = xai_pointcloud.XaiPointcloudExplainer(random_state=2, verbose=False)
                            explanation = explainer.explain_instance(pc,
                                                                     y_ids=influ_name,  # 控制你要查看哪个特征的影响，具体的是将其局部取均值
                                                                     classifier_fn=batch_predict,  # the predicted function
                                                                     window_size=window_size,
                                                                     stride_size=stride_size,
                                                                     model_name=model_name,
                                                                     segment_fn=segment_fn,
                                                                     num_samples=neighbor_num, # The number of neighborhood data generated by LIME <1024
                                                                     num_segments=segment_num,  # 超点数 >1
                                                                     model_regressor=explainer_type
                                                                     )

                            mask, metrics = explanation.get_weight_and_shap(1)  # 0-KT， 1-PD。表示感兴趣的类别
                            np.save(os.path.join(out_file_path, 'weight.npy'), mask)

                            # 保存每个文件的 训练的线性回归模型 的拟合效果结果
                            metrics_keys = ['True', 'Pre']
                            file_metrics = {}
                            for i, key in enumerate(metrics_keys):
                                file_metrics[key] = metrics[i]

                            save_metrics(file_name, file_metrics, out_path, data_pattern)

















































