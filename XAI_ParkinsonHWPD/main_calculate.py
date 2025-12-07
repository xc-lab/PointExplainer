
import os
import ast
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error, explained_variance_score
from scipy.stats import pearsonr
from scipy.stats import entropy




# def calculate_metrics_mean(results_path):
#     # Initialize separate metrics dictionaries for each data pattern
#     metrics_dict_0 = {
#         'True':[], 'Pre':[]
#     }
#
#     metrics_dict_1 = {
#         'True': [], 'Pre': []
#     }
#
#     if not os.path.exists(results_path):
#         print(f'No metrics results found at {results_path}')
#         return
#
#     with open(results_path, 'r') as f:
#         lines = f.readlines()
#         current_metrics = {}
#         current_file = None
#         current_data_pattern = None
#
#         for line in lines:
#             if line.strip().endswith(':'):
#                 if current_file:
#                     # Append the metrics to the appropriate dictionary based on data_pattern
#                     target_dict = metrics_dict_0 if current_data_pattern == 0 else metrics_dict_1
#                     for key, value in current_metrics.items():
#                         target_dict[key].append(float(value))
#                     current_metrics = {}
#
#                 current_file = line.strip().rstrip(':')
#                 current_data_pattern = int(current_file.split('-')[-1])  # Extract the data_pattern from the file name
#             else:
#                 parts = line.strip().split(': ', 1)
#                 if parts[0] in ['True', 'Pre']:
#                     key, value = parts
#                     current_metrics[key.strip()] = float(ast.literal_eval(value)[0])
#
#         # Add last file's metrics to the appropriate dictionary
#         if current_file:
#             target_dict = metrics_dict_0 if current_data_pattern == 0 else metrics_dict_1
#             for key, value in current_metrics.items():
#                 target_dict[key].append(float(value))
#
#     # 计算数据模式 0 的指标均值
#     print("Metrics means for data pattern 0:")
#
#     # 提取真实值和预测值
#     true_list = metrics_dict_0['True']
#     pre_list = metrics_dict_0['Pre']
#
#     # 计算各项指标
#     mae = mean_absolute_error(true_list, pre_list)
#     print(f"Mean Absolute Error (MAE): {mae:.4f}")
#
#     medae = median_absolute_error(true_list, pre_list)
#     print(f"Median Absolute Error (MedAE): {medae:.4f}")
#
#     mse = mean_squared_error(true_list, pre_list)
#     print(f"Mean Squared Error (MSE): {mse:.4f}")
#
#     rmse = np.sqrt(mse)
#     print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
#
#     n = len(true_list)  # 样本数量
#     p = 11  # 特征数量
#     rse = np.sqrt(mse * n / (n - p - 1))  # 计算残差标准差 (RSE)
#     print(f"Residual Standard Error (RSE): {rse:.4f}")
#
#     evs = explained_variance_score(true_list, pre_list)
#     print(f"Explained Variance Score (EVS): {evs:.4f}")
#
#     r2 = r2_score(true_list, pre_list)
#     print(f"R-squared (R²): {r2:.4f}")
#
#     adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)  # 调整后的 R²
#     print(f"Adjusted R-squared (Adjusted R²): {adjusted_r2:.4f}")
#
#     # 计算 AIC 和 BIC
#     rss = np.sum((np.array(true_list) - np.array(pre_list)) ** 2)  # 残差平方和
#     aic = n * np.log(rss / n) + 2 * (p + 1)
#     bic = n * np.log(rss / n) + (p + 1) * np.log(n)
#
#     print(f"Akaike Information Criterion (AIC): {aic:.4f}")
#     print(f"Bayesian Information Criterion (BIC): {bic:.4f}")
#
#
#     print("\nMetrics means for data pattern 1:")
#
#     # 提取真实值和预测值
#     true_list = metrics_dict_1['True']
#     pre_list = metrics_dict_1['Pre']
#
#     # 计算各项指标
#     mae = mean_absolute_error(true_list, pre_list)
#     print(f"Mean Absolute Error (MAE): {mae:.4f}")
#
#     medae = median_absolute_error(true_list, pre_list)
#     print(f"Median Absolute Error (MedAE): {medae:.4f}")
#
#     mse = mean_squared_error(true_list, pre_list)
#     print(f"Mean Squared Error (MSE): {mse:.4f}")
#
#     rmse = np.sqrt(mse)
#     print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
#
#     n = len(true_list)  # 样本数量
#     p = 11  # 特征数量
#     rse = np.sqrt(mse * n / (n - p - 1))  # 计算残差标准差 (RSE)
#     print(f"Residual Standard Error (RSE): {rse:.4f}")
#
#     evs = explained_variance_score(true_list, pre_list)
#     print(f"Explained Variance Score (EVS): {evs:.4f}")
#
#     r2 = r2_score(true_list, pre_list)
#     print(f"R-squared (R²): {r2:.4f}")
#
#     adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)  # 调整后的 R²
#     print(f"Adjusted R-squared (Adjusted R²): {adjusted_r2:.4f}")
#
#     # 计算 AIC 和 BIC
#     rss = np.sum((np.array(true_list) - np.array(pre_list)) ** 2)  # 残差平方和
#     aic = n * np.log(rss / n) + 2 * (p + 1)
#     bic = n * np.log(rss / n) + (p + 1) * np.log(n)
#
#     print(f"Akaike Information Criterion (AIC): {aic:.4f}")
#     print(f"Bayesian Information Criterion (BIC): {bic:.4f}")


def calculate_metrics_mean(results_path):
    # Initialize separate metrics dictionaries for each data pattern
    metrics_dict_0 = {'AE': [], 'RE': []}
    metrics_dict_1 = {'AE': [], 'RE': []}

    if not os.path.exists(results_path):
        print(f'No metrics results found at {results_path}')
        return

    with open(results_path, 'r') as f:
        lines = f.readlines()
        current_metrics = {}
        current_file = None
        current_data_pattern = None

        for line in lines:
            if line.strip().endswith(':'):
                if current_file:
                    # Append AE and RE to the appropriate dictionary
                    target_dict = metrics_dict_0 if current_data_pattern == 0 else metrics_dict_1
                    ae = abs(current_metrics['True'] - current_metrics['Pre'])
                    re = ae / abs(current_metrics['True']) if current_metrics['True'] != 0 else np.nan
                    target_dict['AE'].append(ae)
                    target_dict['RE'].append(re)
                    current_metrics = {}

                current_file = line.strip().rstrip(':')
                current_data_pattern = int(current_file.split('-')[-1])  # Extract the data_pattern from the file name
            else:
                parts = line.strip().split(': ', 1)
                if parts[0] in ['True', 'Pre']:
                    key, value = parts
                    current_metrics[key.strip()] = float(ast.literal_eval(value)[0])

        # Add last file's AE and RE to the appropriate dictionary
        if current_file:
            target_dict = metrics_dict_0 if current_data_pattern == 0 else metrics_dict_1
            ae = abs(current_metrics['True'] - current_metrics['Pre'])
            re = ae / abs(current_metrics['True']) if current_metrics['True'] != 0 else np.nan
            target_dict['AE'].append(ae)
            target_dict['RE'].append(re)

    # 计算数据模式 0 的 AE 和 RE 均值及标准差
    print("Metrics for data pattern 0:")
    ae_list_0 = metrics_dict_0['AE']
    re_list_0 = [x for x in metrics_dict_0['RE'] if not np.isnan(x)]  # 排除 NaN
    print(f"AE Mean: {np.mean(ae_list_0):.4f}")
    print(f"AE Std: {np.std(ae_list_0):.4f}")

    # 计算数据模式 1 的 AE 和 RE 均值及标准差
    print("\nMetrics for data pattern 1:")
    ae_list_1 = metrics_dict_1['AE']
    re_list_1 = [x for x in metrics_dict_1['RE'] if not np.isnan(x)]  # 排除 NaN
    print(f"AE Mean: {np.mean(ae_list_1):.4f}")
    print(f"AE Std: {np.std(ae_list_1):.4f}")



def calculate_metrics_mean_(results_path):
    # Initialize separate metrics dictionaries for each data pattern
    metrics_dict_0 = {'AE': [], 'RE': []}
    metrics_dict_1 = {'AE': [], 'RE': []}

    if not os.path.exists(results_path):
        print(f'No metrics results found at {results_path}')
        return

    with open(results_path, 'r') as f:
        lines = f.readlines()
        current_metrics = {}
        current_file = None
        current_data_pattern = None

        for line in lines:
            if line.strip().endswith(':'):
                if current_file:
                    # Append AE and RE to the appropriate dictionary
                    target_dict = metrics_dict_0 if current_data_pattern == 0 else metrics_dict_1
                    ae = 1 if (current_metrics['True'] > 0.5) == (current_metrics['Pre'] > 0.5) else 0
                    re = ae / abs(current_metrics['True']) if current_metrics['True'] != 0 else np.nan
                    target_dict['AE'].append(ae)
                    target_dict['RE'].append(re)
                    current_metrics = {}

                current_file = line.strip().rstrip(':')
                current_data_pattern = int(current_file.split('-')[-1])  # Extract the data_pattern from the file name
            else:
                parts = line.strip().split(': ', 1)
                if parts[0] in ['True', 'Pre']:
                    key, value = parts
                    current_metrics[key.strip()] = float(ast.literal_eval(value)[0])

        # Add last file's AE and RE to the appropriate dictionary
        if current_file:
            target_dict = metrics_dict_0 if current_data_pattern == 0 else metrics_dict_1
            ae = 1 if (current_metrics['True'] > 0.5) == (current_metrics['Pre'] > 0.5) else 0
            re = ae / abs(current_metrics['True']) if current_metrics['True'] != 0 else np.nan
            target_dict['AE'].append(ae)
            target_dict['RE'].append(re)

    # 计算数据模式 0 的 AE 和 RE 均值及标准差
    print("Metrics for data pattern 0:")
    ae_list_0 = metrics_dict_0['AE']
    re_list_0 = [x for x in metrics_dict_0['RE'] if not np.isnan(x)]  # 排除 NaN
    print(f"AE_ Mean: {np.mean(ae_list_0):.4f}")
    print(f"AE_ Std: {np.std(ae_list_0):.4f}")

    # 计算数据模式 1 的 AE 和 RE 均值及标准差
    print("\nMetrics for data pattern 1:")
    ae_list_1 = metrics_dict_1['AE']
    re_list_1 = [x for x in metrics_dict_1['RE'] if not np.isnan(x)]  # 排除 NaN
    print(f"AE_ Mean: {np.mean(ae_list_1):.4f}")
    print(f"AE_ Std: {np.std(ae_list_1):.4f}\n")


def read_and_group_attributes(file_path):
    grouped_data = {0: [], 1: []}

    with open(file_path, 'r') as f:
        lines = f.readlines()

    current_pattern = None
    current_file = None
    for line in lines:
        line = line.strip()
        if not line:
            continue

        if ':' in line:  # 解析文件名和 data_pattern
            split_index = line.rfind('-')  # 从右侧寻找最后一个'-'
            current_file = line[:split_index]
            pattern = line[split_index + 1:].rstrip(':')  # 去除冒号
            current_pattern = int(pattern)
        else:  # 解析数据
            attributes = [list(map(float, item.strip('[]').split(','))) for item in line.split('] [')]
            grouped_data[current_pattern].append({current_file: attributes})

    return grouped_data



def calculate_fidelity(grouped_data):
    """
    评估归因值绝对值大小和对应的模型预测值差值之间是否存在线性关系。
    """
    fidelity_scores = {}

    # 遍历每个模式及其对应的条目
    for pattern, entries in grouped_data.items():
        pearson_values = []

        # 遍历每个条目（entry）
        for entry in entries:
            # 获取条目的第一个键（文件名），并获取属性值
            file_name, attributes = list(entry.items())[0]
            attribution_value = []
            prediction_difference = []

            # 遍历每个属性的归因值（attribute[0]）和预测差异（attribute[1] - attribute[2]）
            for attribute in attributes:
                value = abs(attribute[0])  # 归因值的绝对值
                difference = abs(attribute[1] - attribute[2])  # 预测差异的绝对值

                attribution_value.append(value)
                prediction_difference.append(difference)

            # 如果有有效的归因值和预测差异，计算Spearman相关系数
            if attribution_value and prediction_difference:
                corr, _ = pearsonr(attribution_value, prediction_difference)
                # 过滤掉NaN值
                if not np.isnan(corr):
                    pearson_values.append(corr)

        # 计算pearson相关系数的均值和标准差
        fidelity_scores[pattern] = {
            "mean": np.mean(pearson_values),
            "std": np.std(pearson_values)
        }

    print(fidelity_scores)



def calculate_directional_consistency(grouped_data):
    """
    计算方向一致性（Directional Consistency, DC）：
    检查归因值的符号与预测变化是否一致。
    """
    directional_consistency_scores = {}

    for pattern, entries in grouped_data.items():
        consistency_values = []

        for entry in entries:
            attributes = list(entry.values())[0]  # 获取属性列表
            attribution_values = np.array([attr[0] for attr in attributes])  # 归因值
            prediction_differences = np.array([attr[1] - attr[2] for attr in attributes])  # 预测值变化

            # 计算方向一致性（归因值和预测变化方向相同的比例）
            consistency = np.mean((attribution_values >= 0) & (prediction_differences >= 0) |
                                  (attribution_values <= 0) & (prediction_differences <= 0))
            consistency_values.append(consistency)

        # 计算均值和标准差
        directional_consistency_scores[pattern] = {
            "mean": np.mean(consistency_values),
            "std": np.std(consistency_values)
        }

    print(directional_consistency_scores)



def calculate_entropy(grouped_data):
    """
    计算归因值的熵（Entropy），用于衡量解释的复杂度。
    """
    entropy_scores = {}
    eps = 1e-10  # 避免 log(0) 问题

    for pattern, entries in grouped_data.items():
        entropy_values = []

        for entry in entries:
            file_name, attributes = list(entry.items())[0]

            attributes = np.array([abs(a[0]) for a in attributes])  # 取绝对值并转换为 NumPy 数组
            attributes_sum = np.sum(attributes)

            if attributes_sum == 0:
                entropy_values.append(np.nan)  # 如果所有归因值都为 0，则返回 NaN
            else:
                attributes /= attributes_sum  # 归一化
                attributes = np.clip(attributes, eps, 1.0)  # 避免 log(0) 计算问题
                entropy_values.append(entropy(attributes))  # 计算熵

        # 计算熵的均值和标准差，忽略 NaN
        entropy_values = np.array(entropy_values)
        entropy_scores[pattern] = {
            "mean": np.nanmean(entropy_values),  # 忽略 NaN 计算均值
            "std": np.nanstd(entropy_values)    # 忽略 NaN 计算标准差
        }

    print(entropy_scores)





if __name__=='__main__':
    neighbor_num = 200
    explainer_type = 'xgb'  # ['ridge', 'dt', 'lasso', 'rf', 'lr', 'elasticnet']

    dataset = 'ParkinsonHW'
    result_path = os.path.join('./data', dataset, f'result_K_{neighbor_num}_{explainer_type}')

    # 1. Local Fidelity, LF
    result_metric_path = os.path.join(result_path, 'metrics_results.txt')
    calculate_metrics_mean(result_metric_path) # PC
    calculate_metrics_mean_(result_metric_path) # CA



    result_attribute_path = os.path.join(result_path, 'attributes_results.txt')
    grouped_attributes = read_and_group_attributes(result_attribute_path)

    # 2. Impact Alignment， AC
    calculate_fidelity(grouped_attributes)

    # 3. Directional Consistency, DA
    calculate_directional_consistency(grouped_attributes)
    #
    # # 4. Attribution Sparsity (AS)
    # calculate_entropy(grouped_attributes)





























