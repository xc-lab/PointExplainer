#  -*- coding: utf-8 -*-
'''
Necessary configuration functions

@author: xuechao.wang@ugent.be
'''

import random
import numpy as np
import torch
import math


def curvature(x, y):
    '''
    Computes the curvature of a 2D trajectory based on the coordinates x and y.

    Parameters:
    x : np.array
        Array of x-coordinates of the trajectory.
    y : np.array
        Array of y-coordinates of the trajectory.

    Returns:
    curvature : np.array
        Array representing the curvature at each point of the trajectory.
    '''
    dx_dt = np.gradient(x)
    dy_dt = np.gradient(y)
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)

    curvature = np.abs(dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / ((dx_dt ** 2 + dy_dt ** 2) ** (3 / 2)+(1e-6))
    return curvature


def derived_feature(data, dataset='ParkinsonHW'):
    '''
    Computes derived features including velocity, acceleration, and jerk (third derivative),
    appending them to the original data array for further analysis.

    Parameters:
    data : np.array
        A 2D array with input features representing trajectory points, including
        coordinates (x, y, z) and additional features (p, g, t).
    dataset : str, optional
        Specifies the dataset, default is 'ParkinsonHW'. Used to differentiate
        processing steps for specific datasets.

    Returns:
    data : np.array
        A 2D array combining the original features with derived ones, including
        velocity, acceleration, jerk, and other differential features.
    '''
    x_coord = data[:, 0] - np.mean(data[:, 0])
    y_coord = data[:, 1] - np.mean(data[:, 1])
    z_coord = data[:, 2] - np.mean(data[:, 2])
    p_coord = data[:, 3]
    g_coord = data[:, 4]
    time_coord = data[:, 5]

    x_coord_diff = np.diff(x_coord, n=1, axis=-1, append=0)
    y_coord_diff = np.diff(y_coord, n=1, axis=-1, append=0)
    z_coord_diff = np.diff(z_coord, n=1, axis=-1, append=0)
    p_coord_diff = np.diff(p_coord, n=1, axis=-1, append=0)
    g_coord_diff = np.diff(g_coord, n=1, axis=-1, append=0)
    time_coord_diff = np.diff(time_coord, n=1, axis=-1, append=0)

    radius_coord = np.array([pow(pow((x_coord[idx]), 2) + pow((y_coord[idx]), 2), 0.5) for idx in range(len(x_coord_diff))])
    angle_coord = np.array([math.atan(y_coord[idx] / x_coord[idx]) for idx in range(len(x_coord_diff))])
    curva_coord = curvature(x_coord, y_coord)

    if dataset == 'ParkinsonHW':
        location_diff_list = np.array([pow(pow((x_coord_diff[idx]), 2) + pow((y_coord_diff[idx]), 2), 0.5) for idx in range(len(x_coord_diff))])
        velocity_list = location_diff_list
        velocity_diff_list = np.array(np.diff(velocity_list, n=1, axis=-1, append=0))
        acceleration_list = velocity_diff_list
        acceleration_diff_list = np.array(np.diff(acceleration_list, n=1, axis=-1, append=0))
        jerk_list = acceleration_diff_list
    else:
        print('%s is a new dataste!'%(dataset))

    data = np.hstack((data,
                      velocity_list.reshape((len(jerk_list), -1)),
                      acceleration_list.reshape((len(jerk_list), -1)),
                      jerk_list.reshape((len(jerk_list), -1)),
                      x_coord_diff.reshape(len(jerk_list), -1),
                      y_coord_diff.reshape(len(jerk_list), -1),
                      z_coord_diff.reshape(len(jerk_list), -1),
                      p_coord_diff.reshape(len(jerk_list), -1),
                      g_coord_diff.reshape(len(jerk_list), -1),
                      time_coord_diff.reshape(len(jerk_list), -1),
                      radius_coord.reshape(len(jerk_list), -1),
                      angle_coord.reshape(len(jerk_list), -1),
                      curva_coord.reshape(len(jerk_list), -1))
                     )
    return data[50:-50, :] # 0-x,1-y,2-z,3-p,4-g,5-t,6-v,7-acc,8-jerk,9-dx,10-dy,11-dz,12-dp,13-dg,14-dt,15-radius,16-angle,17-curvature



def data_reading(path, dataset, idx):
    '''
    Calculates derived features (such as speed, acceleration, jerk, etc.) for each individual stroke
    and appends them to the original features (['x', 'y', 'z', 'p', 'g', 't', 'test_id']). The final column indicates the stroke index.

    :param path: File path to the data file.
    :param dataset: Name of the dataset being processed.
    :param idx: The stroke index to filter from the data.
    :return: A numpy array of processed data with derived features, and the number of rows in the array (L).

    https://archive.ics.uci.edu/dataset/395/parkinson+disease+spiral+drawings+using+digitized+graphics+tablet

    '''

    if dataset == 'ParkinsonHW':
        test_data = np.loadtxt(path, dtype=np.float32, delimiter=';')
        pattern_data = test_data[np.where(test_data[:, -1] == idx), :-1][0]
        L, _ = pattern_data.shape

        if L > 0:
            # Generate derived features such as speed, acceleration, and jerk, adding them to the dataset
            data = derived_feature(pattern_data, dataset)
            m, n = data.shape
        else:
            data = np.array([])
            m = 0
            print('%s do not include %d pattern data.'%(path, idx))

    return data, m


def pc_normalize(data):
    '''
        Normalize point cloud data by centering spatial coordinates and standardizing feature dimensions.

        Parameters:
        - data (np.ndarray): Input data array with spatial and feature dimensions.

        Returns:
        - temp_data (np.ndarray): Normalized data with centered x-y coordinates and standardized z-features.
    '''
    temp_data = data.copy()

    xy_point = data[:, :2]
    xy_point = (xy_point - np.mean(xy_point, axis=0)) / (np.max(np.sqrt(np.sum(np.power((xy_point - np.mean(xy_point, axis=0)), 2), axis=1))) + 1e-8)
    temp_data[:, :2] = xy_point

    feature_point = data[:, 2:]
    temp_data[:, 2:] = (feature_point - np.mean(feature_point, axis=0)) / (np.std(feature_point, axis=0) + 1e-8)

    return temp_data



def get_training_patches(data, window_size, stride_size):
    '''
        Generate patches for training by sampling fixed-size segments of the data with a random stride.

        Parameters:
        - data (np.ndarray): Input point cloud data array.
        - window_size (int): Number of data points in each patch.
        - stride_size (int): Step size between patches for random sampling.

        Returns:
        - patch_dataset (list): List of normalized patches for training.
        '''
    patch_dataset = []
    if data.shape[0] > window_size:
        # num = math.ceil((data.shape[0]-window_size)/stride_size)
        # idxs = np.arange(0, data.shape[0]-window_size).tolist()
        # if len(idxs) < num:
        #     data_idxs = [0]
        # else:
        #     data_idxs = random.sample(idxs, num) # ...
        data_idxs = np.arange(0, data.shape[0] - window_size, stride_size)
        for p in data_idxs:
            patch = data[p:p+window_size, :]
            nor_patch = pc_normalize(patch)  # important
            patch_dataset.append(nor_patch)
    return patch_dataset


def get_testing_patches(data, window_size, stride_size):
    '''
        Generate patches for testing by sampling fixed-size segments of the data sequentially.

        Parameters:
        - data (np.ndarray): Input point cloud data array.
        - window_size (int): Number of data points in each patch.
        - stride_size (int): Step size between patches for sequential sampling.

        Returns:
        - patch_dataset (list): List of normalized patches for testing.
    '''
    patch_dataset = []
    if data.shape[0] > window_size:
        data_idxs = np.arange(0, data.shape[0]-window_size, stride_size)
        for p in data_idxs:
            patch = data[p:p+window_size, :]
            nor_patch = pc_normalize(patch)  # important
            patch_dataset.append(nor_patch)
    return patch_dataset


def setup_seed(seed):
    '''
    To set a fixed seed
    '''
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)





