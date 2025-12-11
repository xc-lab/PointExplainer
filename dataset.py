#  -*- coding: utf-8 -*-
'''
@author: xuechao.wang@ugent.be
'''
import numpy as np
import os
from torch.utils.data import Dataset

from utils.provider import random_scale_point_cloud, shift_point_cloud, rotate_point_cloud_by_angle



class MyDataset(Dataset):
    def __init__(self, data_root, split, transform=None, augment=False):
        assert(split == 'train' or split == 'val')
        cls2name, name2cls = self.decode_classes(os.path.join(data_root, 'shape_names.txt'))
        train_list_path = os.path.join(data_root, 'train.txt')
        train_files_list = self.read_list_file(train_list_path, name2cls)
        val_list_path = os.path.join(data_root, 'val.txt')
        val_files_list = self.read_list_file(val_list_path, name2cls)
        self.files_list = train_files_list if split == 'train' else val_files_list

        self.split = split
        self.transform = transform
        self.augment = augment
        self.caches = {}

    def decode_classes(self, file_path):
        cls2name, name2cls = {}, {}
        with open(file_path, 'r') as f:
            for i, name in enumerate(f.readlines()):
                cls2name[i] = name.strip()
                name2cls[name.strip()] = i
        return cls2name, name2cls

    def read_list_file(self, file_path, name2cls):
        base = os.path.dirname(file_path)
        files_list = []
        with open(file_path, 'r') as f:
            for line in f.readlines():
                name = line.strip().split('_')[0]
                cur = os.path.join(base, name, '{}'.format(line.strip()))
                files_list.append([cur, name2cls[name]])
        return files_list

    def augment_pc(self, pc_normal):
        rotation_angle = np.random.uniform() * 360
        jittered_pc = rotate_point_cloud_by_angle(pc_normal[:, :3], rotation_angle)
        jittered_pc = random_scale_point_cloud(jittered_pc)
        jittered_pc = shift_point_cloud(jittered_pc)
        pc_normal[:, :3] = jittered_pc
        return pc_normal

    def __getitem__(self, index):
        if index in self.caches:
            return self.caches[index]
        file, label = self.files_list[index]
        data = np.loadtxt(file+'.txt', dtype=np.float32, delimiter=',')
        if self.augment:
            data = self.augment_pc(data)
        self.caches[index] = [data, label]

        return data, label

    def __len__(self):
        return len(self.files_list)



class MyDataset_test(Dataset):
    def __init__(self, data_root, dataset, name, transform=None):
        self.name = name
        self.data_root = data_root
        self.transform = transform

        cls2name, name2cls = self.decode_classes(os.path.join(data_root, 'patches', 'shape_names.txt'))
        test_files_list = self.read_list_file(dataset, name2cls)
        self.files_list = test_files_list
        self.caches = {}

    def decode_classes(self, file_path):
        cls2name, name2cls = {}, {}
        with open(file_path, 'r') as f:
            for i, name in enumerate(f.readlines()):
                cls2name[i] = name.strip()
                name2cls[name.strip()] = i
        return cls2name, name2cls

    def read_list_file(self, files, name2cls):
        files_list = []
        for patch_pc in files:
            name_type = self.name
            cur = patch_pc
            files_list.append([cur, name2cls[name_type]])
        return files_list


    def __getitem__(self, index):
        if index in self.caches:
            return self.caches[index]
        file, label = self.files_list[index]
        data = np.float32(file)
        self.caches[index] = [data, label]

        return data, label

    def __len__(self):
        return len(self.files_list)
