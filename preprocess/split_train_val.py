#  -*- coding: utf-8 -*-
'''
This script is responsible for dividing the patches dataset into training and validation sets.

@author: xuechao.wang@ugent.be
'''
import os
import numpy as np


def get_train_val_dataset(path, dataset, fold_d):
    '''
    Splits the patch data into training and validation sets according to a specified proportion.

    :param path: The root directory where the dataset is located.
    :param dataset: The name of the dataset (e.g., 'ParkinsonHW').
    :param fold_d: The specific fold (e.g., 'fold_1', 'fold_2', etc.) for cross-validation.
    '''
    if os.path.exists(os.path.join(path, dataset, fold_d, dimension, 'patches', 'shape_names.txt')):
        os.remove(os.path.join(path, dataset, fold_d, dimension, 'patches', 'shape_names.txt'))
        os.remove(os.path.join(path, dataset, fold_d, dimension, 'patches', 'train.txt'))
        os.remove(os.path.join(path, dataset, fold_d, dimension, 'patches', 'val.txt'))


    for type_name in os.listdir(os.path.join(path, dataset, fold_d, dimension, 'patches')):
        if type_name == 'KT' or type_name == 'PD':
            with open(os.path.join(path, dataset, fold_d, dimension, 'patches', 'shape_names.txt'), "a") as f:
                f.write(type_name + "\n")
            files = os.listdir(os.path.join(path, dataset, fold_d, dimension, 'patches', type_name))
            N = len(files)
            idx = np.arange(N)
            np.random.shuffle(idx)
            for i in range(N):
                if i < (0.8 * N): # 80% of the data for training
                    with open(os.path.join(path, dataset, fold_d, dimension, 'patches', 'train.txt'), "a") as f:
                        f.write(files[idx[i]].split(".")[0] + "\n")
                else:
                    with open(os.path.join(path, dataset, fold_d, dimension, 'patches', 'val.txt'), "a") as f:
                        f.write(files[idx[i]].split(".")[0] + "\n")


if __name__ == '__main__':

    path = '../data'
    dimension = 'pointcloud' # do not change

    dataset = 'ParkinsonHW'

    for fold_d in ['fold_1', 'fold_2', 'fold_3']:
        print("******{}******".format(fold_d))
        get_train_val_dataset(path, dataset, fold_d)