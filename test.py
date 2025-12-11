#  -*- coding: utf-8 -*-
'''
@author: xuechao.wang@ugent.be
'''
import torch
from tqdm import tqdm
import os
import shutil
import numpy as np
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


from dataset import MyDataset_test
from models.pointnet import PointNet
from utils.utils import setup_seed, data_reading, get_testing_patches
from estimation import Performance


def result_to_threshold(preds_targets_dict):

    acc_list = []
    f1_list = []
    recall_list = []
    precision_list = []
    specificity_list = []
    npv_list = []

    targets = preds_targets_dict['targets']
    target_labels = []
    for index in range(len(targets)):
        target = targets[index]
        target_labels.append(target[0])

    preds = preds_targets_dict['preds']

    threshold_list = np.arange(0, 1.0001, 0.01)
    for threshold in threshold_list:

        pred_labels = []
        for index in range(len(preds)):
            pred = preds[index].tolist()
            a = pred.count(1)/len(pred)
            if a > threshold:
                pred_labels.append(1)
            else:
                pred_labels.append(0)

        metric = Performance(target_labels, pred_labels)
        acc_list.append(metric.accuracy())
        f1_list.append(metric.f1_score())
        recall_list.append(metric.recall())
        # precision_list.append(metric.presision())
        specificity_list.append(metric.specificity())
        # npv_list.append(metric.npv())

    # Prepare file path
    file_path = os.path.join('data/vis_results', f'{args.fold}-{args.data_pattern}.txt')
    # if os.path.exists(file_path):
    #     os.remove(file_path)
    # Save the new result to the file
    array_result = np.array([acc_list, f1_list, recall_list, specificity_list])
    # np.savetxt(file_path, array_result, delimiter=',', fmt='%.10f')

    fig = plt.figure(figsize=(10, 8))
    ax = fig.subplots()
    # plt.ylim(-1, 101)
    ax.plot(threshold_list, acc_list, color='#266A2E', label='acc', lw=3)
    ax.plot(threshold_list, f1_list, color='#f07818', label='f1', lw=3)
    ax.plot(threshold_list, recall_list, color='b', label='recall', lw=3)
    # ax.plot(threshold_list, precision_list, color='r', label='precision', lw=3)
    ax.plot(threshold_list, specificity_list, color='k', label='specificity', lw=3)
    # ax.plot(threshold_list, npv_list, color='y', label='npv', lw=3)
    plt.grid()
    plt.tick_params(labelsize=20)
    plt.ylabel('Performance(in %)', fontsize=25)
    plt.xlabel('Threshold', fontsize=25)
    plt.legend()
    plt.show()


def evaluate_cls(args):

    setup_seed(222)
    print('Loading..')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preds_patches_dataset = []  # stores patch-level classification results
    targets_patches_dataset = []

    preds_sequence_dataset = []  # stores sequence-level classification results
    targets_sequence_dataset = []

    pres_targets_dataset = {'targets': [], 'preds': []}  # For each piece of data, the true and pre labels in the split patch

    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)
    else:
        shutil.rmtree(args.out_path)
        os.mkdir(args.out_path)

    if args.model == 'PointNet':
        model = PointNet(args.num_category, normal_channel=False) # if feature just x y z, the normal_channel is False; if feature is x y z R G B, the normal_channel is True
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(args.model))

    model = model.to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)['state_dict'])
    # model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    with open(os.path.join(args.data_root, args.dataset, args.fold, 'test_names.txt'), 'r') as f:

        for line in f.readlines():

            label_type = line.strip().split('/')[0]
            json_file_path = os.path.join(args.data_root, args.dataset, 'raw_data', line.strip())

            temp_data, L = data_reading(json_file_path, args.dataset, args.data_pattern)
            if L > 0:
                # 0-x,1-y,2-z,3-p,4-g,5-t,6-v,7-acc,8-jerk,9-dx,10-dy,11-dz,12-dp,13-dg,14-dt,15-radius,16-angle,17-curvature
                pc = temp_data[:, [0, 1, 15]]
                # pc = temp_data[:, [0, 1, 3, 2, 2, 2]]  # important! here control Z channel uses which han-drawn feature
                # pc[:, 3:] = 1

                patch_dataset = get_testing_patches(pc, args.window_size, args.stride_size) # segment into patch, do normalization, delate the stroke info


                if not patch_dataset:
                    print('    Empty data:')
                    print('        '+json_file_path)
                else:

                    pred_labels = []
                    target_labels = []

                    test_dataset = MyDataset_test(data_root=os.path.join(args.data_root, args.dataset, args.fold, args.data_type),
                                                  dataset=patch_dataset,
                                                  name=label_type,
                                                  transform=None)
                    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False,
                                            num_workers=1)

                    print()
                    print("----Dataset: {}, Evaluating..----".format(len(test_dataset)))

                    test_bar = tqdm(test_loader)
                    for i, data in enumerate(test_bar):
                        inputs, labels = data

                        labels = labels.to(device)
                        inputs = inputs.to(device)

                        with torch.no_grad():
                            if args.model == 'PointNet':
                                pred, trans_feat = model(inputs)
                            pred = torch.max(pred, dim=-1)[1]

                            pred_data = pred.data.cpu().detach().numpy().flatten()
                            target_data = labels.data.cpu().detach().numpy().flatten()
                            for k in np.arange(len(labels)):
                                pred_labels.append(pred_data[k])
                                target_labels.append(target_data[k])

                    target_labels = np.array(target_labels) # true label
                    target_ar, target_num = np.unique(target_labels, return_counts=True)
                    target_labels_dict = dict(zip(target_ar, target_num))

                    pred_labels = np.array(pred_labels) # pre label
                    pred_ar, pred_num = np.unique(pred_labels, return_counts=True)
                    pred_labels_dict = dict(zip(pred_ar, pred_num))

                    # the probability of each data being classified correctly
                    print('Classification: %s, data length: %d.' % (json_file_path, len(pc)))
                    print('              True label:%s, Predict label:%s, Acc:%f' % (
                        target_labels_dict, pred_labels_dict, np.sum(
                            np.where(target_labels - pred_labels, 0, 1)) / len(target_labels)))

                    targets_patches_dataset.extend(target_labels)  # for all patch
                    preds_patches_dataset.extend(pred_labels)  # for all patch

                    pres_targets_dataset['targets'].append(target_labels)
                    pres_targets_dataset['preds'].append(pred_labels)

                    threshold = 0.5
                    targets_sequence_dataset.append(target_labels[0])
                    a = pred_labels.tolist()
                    if a.count(1) / len(a) > threshold:
                        preds_sequence_dataset.append(1)
                    else:
                        preds_sequence_dataset.append(0)

    # result_to_threshold(pres_targets_dataset)

    targets_sequence_dataset = np.array(targets_sequence_dataset)
    preds_sequence_dataset = np.array(preds_sequence_dataset)

    print(targets_sequence_dataset)
    print(preds_sequence_dataset)
    print(targets_sequence_dataset - preds_sequence_dataset)

    metric = Performance(targets_sequence_dataset, preds_sequence_dataset)
    # metric.roc_plot()
    # metric.plot_matrix()
    acc_score = metric.accuracy()
    f1_score = metric.f1_score()
    recall_score = metric.recall()
    precision_score = metric.presision()
    specificity = metric.specificity()
    npv = metric.npv()
    mcc = metric.mcc()
    print(
        "Sequence: Accuracy(ACC) = {:f}, F1_score = {:f}, Recall(Sensitivity,TPR) = {:f}, and Precision(PPV) = {:f}, and NPV = {:f}, and Specificity(TNR) = {:f}, and Matthews correlation coefficient(MCC) = {:f}. \n".format(acc_score, f1_score, recall_score,
                                                                      precision_score, npv, specificity, mcc))

   


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_pattern', type=int, default=1, choices=[0, 1]) # 0: Static Spiral Test (SST Dataset) with pressure as height ;  1: Dynamic Spiral Test (DST Dataset) with radius as height
    parser.add_argument('--window_size', type=int, default=512, help='Sequence patch length')
    parser.add_argument('--fold', type=str, default='', choices=['fold_1', 'fold_2', 'fold_3'])
    parser.add_argument('--checkpoint', type=str, default='', help='Root to the best checkpoint')

    parser.add_argument('--data_root', type=str, default='data', help='Root to the test dataset')
    parser.add_argument('--dataset', type=str, default='ParkinsonHW', help='Dataset name')
    parser.add_argument('--data_type', type=str, default='pointcloud', choices=['pointcloud'])

    parser.add_argument('--num_category', default=2, type=int, choices=[2], help='training on spiral shape dataset')

    parser.add_argument('--model', type=str, default='PointNet', help='Model name')

    parser.add_argument('--stride_size', type=int, default=8, help='Degree of overlap of adjacent patches')

    parser.add_argument('--out_path', type=str, default='output', help='Root for saving the val results')

    args = parser.parse_args()

    evaluate_cls(args)
