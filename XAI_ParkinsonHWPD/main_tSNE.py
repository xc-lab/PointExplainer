
import numpy as np
import torch
import os
from copy import deepcopy
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cosine
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import seaborn as sns
import umap.umap_ as umap
from sklearn.metrics import silhouette_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import calinski_harabasz_score


from bag.utils.generic_utils import segment_fn
from utils import data_reading

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)

def compute_velocity_cv(x, y):
    """
    计算给定 x, y 坐标信号的速度变异系数。
    """
    # 确保 x 和 y 的长度相同
    if len(x) != len(y):
        raise ValueError("x 和 y 必须具有相同的长度")
    # 计算相邻采样点的差值
    dx = np.diff(x)
    dy = np.diff(y)
    # 计算速度：假设采样时间间隔为 1
    speed = np.sqrt(dx ** 2 + dy ** 2)
    # 计算均值和标准差
    mean_speed = np.mean(speed)
    std_speed = np.std(speed)
    # 防止均值为 0 导致除零错误
    if mean_speed == 0:
        raise ValueError("存在均值速度为0")
    # 计算变异系数
    cv = std_speed / mean_speed
    return cv


def compute_acceleration_cv(x, y, tol=1e-6):
    """
    计算给定 x, y 坐标信号的加速度变异系数。
    """
    if len(x) != len(y):
        raise ValueError("x 和 y 必须具有相同的长度")

    # 计算相邻点之间的速度，假设采样时间间隔为1
    dx = np.diff(x)
    dy = np.diff(y)
    speed = np.sqrt(dx ** 2 + dy ** 2)

    # 计算加速度（速度的一阶差分），并取绝对值
    acceleration = np.diff(speed)
    abs_acc = np.abs(acceleration)

    # 计算均值和标准差
    mean_acc = np.mean(abs_acc)
    std_acc = np.std(abs_acc)

    # 如果均值加速度接近 0，则触发警报
    if np.isclose(mean_acc, 0, atol=tol):
        raise ValueError("加速度均值为0")

    acc_cv = std_acc / mean_acc
    return acc_cv



def extract_features(segments):
    """
    将变长的手绘片段转换为固定维度的特征向量
    """
    feature_list = []
    for seg in segments:
        if seg.shape[0] == 0:  # 避免空片段
            continue

        x, y, z = seg[:, 0], seg[:, 1], seg[:, 2]
        feature_vector = [
            compute_velocity_cv(x, y),  # 速度变异系数
            compute_acceleration_cv(x, y),  # 加速度变异系数


        ]
        feature_list.append(feature_vector)

    return np.array(feature_list)


def segment_extraction(pc, mask, influ_name, segment_num, segment_fn, weight_threshold):
    """
    提取片段
    """
    segments, fudged_data = segment_fn(pc, segment_num)  # 存储的是每个片段的均值，保持与原始数据相同的尺寸
    segment_HC_attributes = [] # 存储倾向于HC的片段
    segment_PD_attributes = [] # 存储倾向于PD的片段

    # 截取每个片段，根据weight值分为HC和PD两组
    n_segments = np.unique(segments)
    for idx in n_segments:
        indices = np.where(segments == idx)[0]  # 获取所有等于该值的索引, 也就是该片段对应的索引

        weight = np.unique(mask[indices])
        assert len(weight) == 1, f"Error: Expected number of weights in one segment to be 1, but got {len(weight)}"

        reversed_data = deepcopy(pc)
        segment_data = reversed_data[indices][:, influ_name]

        # 根据归因值将对应手绘片段存入对应属性片段列表
        if weight > weight_threshold and segment_data.shape[0] > 10:
            segment_PD_attributes.append(segment_data)
        elif weight < -weight_threshold and segment_data.shape[0] > 10:
            segment_HC_attributes.append(segment_data)

    return segment_HC_attributes, segment_PD_attributes







if __name__=='__main__':
    weight_thresholds = [0, 0.00001, 0.00005, 0.0001, 0.0005, 0.001]  # 归因阈值
    n_neighbors_list = [5, 20, 30, 50, 100, 105, 110, 120, 125, 130, 135, 140, 145, 150, 155, 160, 180, 200]  # UMAP 近邻数

    neighbor_num = 200
    explainer_type = 'xgb'
    influ_name = [0, 1, 2]  # See which features affect the model results
    segment_num = 10
    dataset = 'ParkinsonHW'
    stride_size = 8

    out_path = os.path.join('./data', dataset, f'result_K_{neighbor_num}_{explainer_type}')
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    for data_pattern in [0, 1]:
        best_sil_score = -1
        best_params = None
        window_size = 256 if data_pattern == 0 else 512

        for weight_threshold in weight_thresholds:
            segment_HC = []
            segment_PD = []

            for fold_d in ['fold_1', 'fold_2', 'fold_3']:
                file_path = os.path.join('./data', dataset, fold_d, 'test_names.txt')
                with open(file_path, 'r') as f:
                    for line in f.readlines():

                        file_name = line.strip().split('/')[1].split('.')[0]
                        json_file_path = os.path.join('./data', dataset, 'raw_data', line.strip())
                        temp_data, L = data_reading(json_file_path, dataset, data_pattern)
                        # 0-x,1-y,2-z,3-p,4-g,5-t,6-v,7-acc,8-jerk,9-dx,10-dy,11-da,12-dl,13-dp,14-dt,15-radius,16-angle,17-curvature,18-idx
                        if L > window_size:
                            pc = temp_data[:, [0, 1, 3]] if data_pattern == 0 else temp_data[:, [0, 1, 15]]
                            mask = np.load(os.path.join(out_path, file_name + '-' + str(data_pattern), 'weight.npy')).reshape(-1)

                            # 提取符合要求的片段
                            segment_HC_attributes, segment_PD_attributes = segment_extraction(
                                                                                                pc=pc,
                                                                                                mask=mask,
                                                                                                influ_name=influ_name,
                                                                                                segment_num=segment_num,
                                                                                                segment_fn=segment_fn,
                                                                                                weight_threshold=weight_threshold
                                                                                                )
                            segment_HC.extend(segment_HC_attributes)
                            segment_PD.extend(segment_PD_attributes)


            # 提取特征
            HC_features = extract_features(segment_HC)
            PD_features = extract_features(segment_PD)
            # 避免 np.vstack() 报错
            if HC_features.shape[0] == 0 or PD_features.shape[0] == 0:
                continue

            # 合并数据，添加标签
            data = np.vstack((HC_features, PD_features))
            labels = np.array([0] * len(HC_features) + [1] * len(PD_features))  # 0 = 健康, 1 = 帕金森

            for n_neighbors in n_neighbors_list:
                # 使用 UMAP 进行降维
                reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42, n_jobs=1, metric='cosine')
                embedding = reducer.fit_transform(data)  # 使用降维直接操作片段数据

                # 计算轮廓系数
                sil_score = silhouette_score(embedding, labels)
                print(f"##########---------- Silhouette Score (weight_threshold={weight_threshold}, n_neighbors={n_neighbors}): {sil_score:.4f}")
                # Calinski-Harabasz 指数
                ch_score = calinski_harabasz_score(embedding, labels)
                print(f"##########---------- Calinski-Harabasz Score: {ch_score:.4f}\n")

                # # 可视化
                # plt.figure(figsize=(8, 6))
                # plt.scatter(embedding[labels == 0, 0], embedding[labels == 0, 1], label="HC", alpha=0.7, s=5, color="blue")
                # plt.scatter(embedding[labels == 1, 0], embedding[labels == 1, 1], label="PD", alpha=0.7, s=5, color="red")
                # plt.legend()
                # plt.title("UMAP Visualization of Handwriting Segments (HC vs PD)")
                # plt.xlabel("UMAP Component 1")
                # plt.ylabel("UMAP Component 2")
                # plt.show()

                if sil_score > best_sil_score:
                    best_sil_score = sil_score
                    best_params = (weight_threshold, n_neighbors)

        print(f"Best Silhouette Score: {best_sil_score:.4f} with weight_threshold={best_params[0]} and n_neighbors={best_params[1]} \n")



















