#  -*- coding: utf-8 -*-

'''
This script performs data augmentation on a dataset containing samples from two classes:
KT (healthy control) and PD (Parkinson's disease). Augmentation operations include
mirror transformation, rotation, and sampling adjustments, aimed at balancing the sample
length distributions between classes.

KT-00, PD-01

@author: xuechao.wang@ugent.be
'''

import shutil
import os
import numpy as np

from utils.utils import data_reading
from utils.combination import combination_method, aug_pc_dataset



aug_params_list = {'mirror': ['bottom-up'], # here just can use one mirror method
                   'rotation': [90, 180, 270],
                   'sampling': [0.75, 1.25],
                   'jitter': [0.0005] }



def get_aug_dataset(path, dataset, fold_d, dimension, augment, hw_pattern_list):
    '''
    Processes data with optional augmentation, generating point clouds for Parkinson's disease analysis.

    Parameters:
        path (str): The base path where data directories are located.
        dataset (str): The name of the dataset directory (e.g., 'ParkinsonHW').
        fold_d (str): The fold directory name (e.g., 'fold_1') for cross-validation splits.
        dimension (str): Data type (e.g., 'pointcloud') used in directory organization.
        augment (bool): Whether to apply augmentation on the dataset.
        hw_pattern_list (int): An integer specifying the test type (0: Static Spiral Test, 1: Dynamic Spiral Test).
    '''

    if not os.path.exists(os.path.join(path, dataset, fold_d, dimension)):
        os.mkdir(os.path.join(path, dataset, fold_d, dimension))
        os.mkdir(os.path.join(path, dataset, fold_d, dimension, 'dataset'))
        os.mkdir(os.path.join(path, dataset, fold_d, dimension, 'dataset', 'KT'))
        os.mkdir(os.path.join(path, dataset, fold_d, dimension, 'dataset', 'PD'))
    else:
        shutil.rmtree(os.path.join(path, dataset, fold_d, dimension))
        os.mkdir(os.path.join(path, dataset, fold_d, dimension))
        os.mkdir(os.path.join(path, dataset, fold_d, dimension, 'dataset'))
        os.mkdir(os.path.join(path, dataset, fold_d, dimension, 'dataset', 'KT'))
        os.mkdir(os.path.join(path, dataset, fold_d, dimension, 'dataset', 'PD'))

    num_pd = 0
    num_kt = 0
    # kt_length = []
    # pd_length = []

    with open(os.path.join(path, dataset, fold_d, 'train_names.txt'), 'r') as f:
        for line in f.readlines():

            label_path = line.strip().split('/')[0]
            file_id = line.strip().split('/')[1].split('.')[0]

            tuple_methods = combination_method(aug_params_list, dataset, label_path)

            aug_path = os.path.join(path, dataset, fold_d, dimension, 'dataset', label_path)  # '../data/DraWritePD/fold_1/pointcloud/dataset/KT'

            if label_path == 'KT':
                label_id = '00'
            elif label_path == 'PD':
                label_id = '01'
            else:
                print('There is no %s class.' % (label_path))

            # Read and preprocess the data, adding derived features and stroke information
            json_file_name = os.path.join(path, dataset, 'raw_data', line.strip())
            temp_data, L = data_reading(json_file_name, dataset, hw_pattern_list)
            if L > 0:
                # 0-x,1-y,2-z,3-p,4-g,5-t,6-v,7-acc,8-jerk,9-dx,10-dy,11-dz,12-dp,13-dg,14-dt,15-radius,16-angle,17-curvature
                pc = temp_data

                # Apply augmentation if specified; otherwise, use raw data
                if augment == True:
                    aug_dataset = aug_pc_dataset(pc, tuple_methods)
                else:
                    aug_dataset = [pc]

                if len(aug_dataset) == 0:
                    print('      This data can not generate aug set!: {}'.format(json_file_name))
                else:
                    new_aug_dataset = aug_dataset.copy()

                    # if label_path == 'KT':
                    #     for p, aug_data in enumerate(new_aug_dataset):
                    #         kt_length.append(aug_data.shape[0])
                    # else:
                    #     for p, aug_data in enumerate(new_aug_dataset):
                    #         pd_length.append(aug_data.shape[0])

                    if label_path == 'KT':
                        num_kt += len(new_aug_dataset)
                    elif label_path == 'PD':
                        num_pd += len(new_aug_dataset)

                    for p, aug_data in enumerate(new_aug_dataset):
                        aug_id = '{0:04d}'.format(p)
                        aug_full_name = label_path + '_label-' + label_id + '-file-' + file_id + '-aug-' + aug_id
                        np.savetxt(os.path.join(aug_path, aug_full_name+'.txt'), aug_data, delimiter=',', fmt='%.10f')

        print(f'      KT augmented dataset count: {num_kt}, PD augmented dataset count: {num_pd}')



if __name__ == '__main__':

    path = '../data'
    dimension = 'pointcloud' # Data type, do not change
    augment = False
    dataset = 'ParkinsonHW'  # not change
    hw_pattern_list = 0 # 0: Static Spiral Test ;  1: Dynamic Spiral Test

    for fold_d in ['fold_1', 'fold_2', 'fold_3']:
        print("******{}******".format(fold_d))
        get_aug_dataset(path, dataset, fold_d, dimension, augment, hw_pattern_list)


