#  -*- coding: utf-8 -*-
'''
Data segmentation into patches for classification: KT-00, PD-01.

@author: xuechao.wang@ugent.be
'''

import os
import shutil
import numpy as np

from utils.utils import get_training_patches


def get_patches_dataset(path, dataset, fold_d, dimension, window_size, stride_size):
    '''
        Segment augmented data into smaller patches for training and evaluation.

        Parameters:
        - path (str): Base directory where the dataset is stored.
        - dataset (str): Name of the dataset to be processed.
        - fold_d (str): Data fold, e.g., 'fold_1', 'fold_2', for cross-validation or testing.
        - dimension (str): Data dimensionality, generally fixed as 'pointcloud'.
        - window_size (int): The window size for each patch.
        - stride_size (dict): Step size for moving the window, different per label ('KT' and 'PD').

        Returns:
        - None. Writes patch files directly to disk.
    '''
    if not os.path.exists(os.path.join(path, dataset, fold_d, dimension, 'patches')):
        os.mkdir(os.path.join(path, dataset, fold_d, dimension, 'patches'))
        os.mkdir(os.path.join(path, dataset, fold_d, dimension, 'patches', 'KT'))
        os.mkdir(os.path.join(path, dataset, fold_d, dimension, 'patches', 'PD'))
    else:
        shutil.rmtree(os.path.join(path, dataset, fold_d, dimension, 'patches'))
        os.mkdir(os.path.join(path, dataset, fold_d, dimension, 'patches'))
        os.mkdir(os.path.join(path, dataset, fold_d, dimension, 'patches', 'KT'))
        os.mkdir(os.path.join(path, dataset, fold_d, dimension, 'patches', 'PD'))


    num_pd = 0
    num_kt = 0

    data_path = os.path.join(path, dataset, fold_d, dimension, 'dataset')
    for l, label in enumerate(os.listdir(data_path)):
        txt_path = os.path.join(path, dataset, fold_d, dimension, 'patches', label)

        for f, file in enumerate(os.listdir(os.path.join(data_path, label))):
            file_path = os.path.join(data_path, label, file)
            file_name = file.split('.')[0]
            # print(file_path)

            temp_data = np.loadtxt(file_path, dtype=np.float32, delimiter=',')
            # 0-x,1-y,2-z,3-p,4-g,5-t,6-v,7-acc,8-jerk,9-dx,10-dy,11-dz,12-dp,13-dg,14-dt,15-radius,16-angle,17-curvature
            full_data_array = temp_data[:, [0, 1, 15]]   # important.....-to choose another feature as z.

            patches_data = get_training_patches(full_data_array, window_size, stride_size[label])
            if len(patches_data) == 0:
                print('    This data can not generate patch set :{}'.format(file_path))
            else:
                if label == 'KT':
                    num_kt += len(patches_data)
                elif label == 'PD':
                    num_pd += len(patches_data)
                for p, patch_data in enumerate(patches_data):
                    patch_id = '{0:04d}'.format(p)
                    patch_full_name = os.path.join(txt_path, file_name + '-patch-' + patch_id + '.txt')
                    np.savetxt(patch_full_name, patch_data, delimiter=',', fmt='%.6f')
    print('KT patch dataset:%d, PD patch dataset:%d ************************************************************' % (num_kt, num_pd))



if __name__ == '__main__':

    path = '../data'
    dimension = 'pointcloud' # do not change

    dataset = 'ParkinsonHW'  # do not change

    window_size = 256
    stride_size = {'KT':16, 'PD':64}

    for fold_d in ['fold_1', 'fold_2', 'fold_3']:
        print("******{}******".format(fold_d))
        get_patches_dataset(path, dataset, fold_d, dimension, window_size, stride_size)
