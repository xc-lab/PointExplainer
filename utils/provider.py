#  -*- coding: utf-8 -*-
'''
Data Augmentation Methods for point cloud data.

https://github.com/charlesq34/pointnet
'''
import numpy as np

def mirror_point_cloud(data, axis='bottom-up'):
    if axis == 'bottom-up':
        nx, ny, nz = 1, 0, 0
    elif axis == 'left-right':
        nx, ny, nz = 0, 1, 0
    mirror_matrix = np.array([[1-2*nx*nx, -2*nx*ny, -2*nx*nz],
                              [-2*nx*ny, 1-2*ny*ny, -2*ny*nz],
                              [-2*nx*nz, -2*ny*nz, 1-2*nz*nz]])
    temp_data = np.dot(data, mirror_matrix)
    return temp_data


def rotate_point_cloud_by_angle(data, angle):
    rotation_angle = angle*np.pi/180
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, -sinval, 0],
                                [sinval, cosval,  0],
                                [0     , 0,       1]])
    temp_data = np.dot(data, rotation_matrix)
    return temp_data


def jitter_point_cloud(data, sigma=0.0001):
    N, C = data.shape
    clip = 0.001
    jittered_data = np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    jittered_data += data
    return jittered_data


def random_scale_point_cloud(pc, scale_low=0.75, scale_high=1.25):
    scale = np.random.uniform(scale_low, scale_high, 1)
    pc *= scale
    return pc


def shift_point_cloud(pc, shift_range=0.001):
    N, C = pc.shape
    shifts = np.random.uniform(-shift_range, shift_range, (1, C))
    pc += shifts
    return pc


def random_point_sampling(data, max_ratio=1.25):
    if max_ratio > 1: # upsampling, only on stroke
        insert_data = []
        insert_idx = []
        m,n = data.shape
        up_ratio = np.random.random()*(max_ratio-1) # 0~0.25
        # up_ratio = max_ratio-1 # 0~0.25
        up_idx = np.where(np.random.random(m)<=up_ratio)[0].tolist()
        if len(up_idx)>0:
            for idx in up_idx:
                if idx != 0 and idx != (m-1) and len(set([data[idx-1,-1],data[idx,-1],data[idx+1,-1]])) == 1: # Determine whether the index is inside the stroke
                    inf = [(data[idx-1,j]+data[idx,j])/2 for j in range(n)]
                    insert_data.append(inf)
                    insert_idx.append(idx)
            insert_data = np.array(insert_data)
            new_data = np.insert(data, insert_idx, insert_data, axis=0)
        else:
            new_data = data

    elif max_ratio < 1: # downsampling
        down_ratio = np.random.random() * (1-max_ratio)  # 0~0.25
        # down_ratio = 1-max_ratio  # 0~0.25
        save_idx = np.where(np.random.random((data.shape[0])) >= down_ratio)[0]
        if len(save_idx) > 0:
            new_data = data[save_idx, :]
        else:
            new_data = data

    return new_data



