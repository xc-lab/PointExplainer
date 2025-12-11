#  -*- coding: utf-8 -*-
'''
DATA_PREPROCESSING StepI:

This script performs a 3-fold stratified K-Fold split at the individual level.

@author: xuechao.wang@ugent.be
'''
import shutil
import os

from sklearn.model_selection import StratifiedKFold



def kfold_split(path, dataset):
    """
    Performs a 3-fold stratified K-Fold split on data at the individual level.

    Parameters:
    path (str): Path to the main directory containing dataset subdirectories.
    dataset (str): Name of the dataset.

    Function Details:
    - Each fold is stored in a new directory named `fold_X` (where X is the fold number).
    - For each fold, two files ('train_names.txt' and 'test_names.txt') are
      created, listing the file paths for the training and test sets respectively.
    """

    files = []
    labels = []
    for l, label in enumerate(os.listdir(os.path.join(path, dataset, 'raw_data'))):
        for t, file_path in enumerate(os.listdir(os.path.join(path, dataset, 'raw_data', label))):
            if dataset == 'ParkinsonHW':
                files.append(label + '/' + file_path)
                labels.append(label)

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=2)
    K = 0

    for train_index, test_index in skf.split(files, labels):
        K += 1

        if not os.path.exists(os.path.join(path, dataset, 'fold_{}'.format(K))):
            os.mkdir(os.path.join(path, dataset, 'fold_{}'.format(K)))
        else:
            shutil.rmtree(os.path.join(path, dataset, 'fold_{}'.format(K)))
            os.mkdir(os.path.join(path, dataset, 'fold_{}'.format(K)))

        for i in train_index:
            with open(os.path.join(path, dataset, 'fold_{}'.format(K), 'train_names.txt'), "a") as f:
                f.write(files[i] + "\n")
        for i in test_index:
            with open(os.path.join(path, dataset, 'fold_{}'.format(K), 'test_names.txt'), "a") as f:
                f.write(files[i] + "\n")


if __name__ == '__main__':
    path = '../data'
    dataset = 'ParkinsonHW'

    kfold_split(path, dataset) # be careful if run this line


