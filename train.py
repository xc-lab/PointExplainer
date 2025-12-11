#  -*- coding: utf-8 -*-
'''
@author: xuechao.wang@ugent.be
'''
import argparse
import os
import copy
import time
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.pointnet import PointNet, PointNet_loss
from dataset import MyDataset


def train_one_epoch(train_loader, model, loss_func, optimizer, device, epoch):
    train_bar = tqdm(train_loader)
    running_results = {'total_data_size': 0, 'loss': 0, 'acc': 0}

    for i, data in enumerate(train_bar):
        inputs, labels = data
        labels = labels.to(device)
        inputs = inputs.to(device)

        if args.model == 'PointNet':
            pred, trans_feat = model(inputs)
            loss = loss_func(pred, labels, trans_feat)

        elif args.model == 'PointNet++':
            pred = model(inputs)
            loss = loss_func(pred, labels)


        pred = torch.max(pred, dim=-1)[1]

        if not args.large_batch_size:

            optimizer.zero_grad()  # Important
            loss.backward()
            optimizer.step()

            lr = optimizer.state_dict()['param_groups'][0]['lr']

            running_results['total_data_size'] += args.batch_size
            running_results['loss'] += loss.item() * args.batch_size
            running_results['acc'] += torch.sum(pred == labels)

            train_bar.set_description(
                desc='Epoch[%003d/%d]: Training, lr = %.8f, loss: %.4f, acc: %.4f' % (epoch, args.nepoches, lr,
                                                                                      running_results['loss'] /
                                                                                      running_results[
                                                                                          'total_data_size'],
                                                                                      running_results['acc'] /
                                                                                      running_results[
                                                                                          'total_data_size']))
        else:
            loss = loss / args.accumulation_steps  # Normalize our loss (if averaged)
            loss.backward()  # Backward pass
            if (i + 1) % args.accumulation_steps == 0:  # Wait for several backward steps
                optimizer.step()  # Now we can do an optimizer step
                model.zero_grad()  # Initialize gradient with all 0 for next step
                lr = optimizer.state_dict()['param_groups'][0]['lr']

                running_results['total_data_size'] += args.batch_size
                running_results['loss'] += loss.item() * args.batch_size
                running_results['acc'] += torch.sum(pred == labels)

                train_bar.set_description(
                    desc='Epoch[%003d/%d]: Training, lr = %.8f, loss: %.4f, acc: %.4f' % (epoch, args.nepoches, lr,
                                                                                          running_results['loss'] /
                                                                                          running_results[
                                                                                              'total_data_size'],
                                                                                          running_results['acc'] /
                                                                                          running_results[
                                                                                              'total_data_size']))

    return running_results['loss'] / running_results['total_data_size'], running_results['acc'] / running_results[
        'total_data_size']


def val_one_epoch(val_loader, model, loss_func, device):
    val_results = {'total_data_size': 0, 'acc': 0, 'loss': 0}
    val_bar = tqdm(val_loader)

    for i, data in enumerate(val_bar):
        inputs, labels = data

        labels = labels.to(device)
        inputs = inputs.to(device)

        with torch.no_grad():

            if args.model == 'PointNet':
                pred, trans_feat = model(inputs)
                loss = loss_func(pred, labels, trans_feat)

            elif args.model == 'PointNet++':
                pred = model(inputs)
                loss = loss_func(pred, labels)

            pred = torch.max(pred, dim=-1)[1]

            lr = optimizer.state_dict()['param_groups'][0]['lr']

            val_results['total_data_size'] += args.batch_size
            val_results['loss'] += loss.item() * args.batch_size
            val_results['acc'] += torch.sum(pred == labels)

            val_bar.set_description(desc='                Val, lr = %.8f, loss = %.4f, acc = %.4f' % (
            lr, val_results['loss'] / val_results['total_data_size'],
            val_results['acc'] / val_results['total_data_size']))

    return val_results['loss'] / val_results['total_data_size'], val_results['acc'] / val_results['total_data_size']



def train(train_loader, val_loader, model, loss_func, optimizer, scheduler, device, ngpus):

    current_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))

    # optionally resume from a checkpoint
    start_epoch = args.start_epoch

    if args.resume:
        assert os.path.isfile(os.path.join('./log_dir', args.resume, 'checkpoints', 'best_model',
                                           args.model + '_cls.pth')), 'Resume checkpoint file not exit!'
        current_time = args.resume
        checkpoint = torch.load(
            os.path.join('./log_dir', args.resume, 'checkpoints', 'best_model', args.model + '_cls.pth'))
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> resume training: loaded checkpoint '{}' (epoch {})"
              .format(os.path.join('log_dir', args.resume, 'checkpoints', 'best_model', args.model + '_cls.pth'),
                      checkpoint['epoch']))

    else:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
            os.makedirs(os.path.join(args.log_dir, current_time))

        if not os.path.exists(os.path.join(args.log_dir, current_time, 'checkpoints')):
            os.makedirs(os.path.join(args.log_dir, current_time, 'checkpoints'))
            os.makedirs(os.path.join(args.log_dir, current_time, 'checkpoints', 'best_model'))

        if not os.path.exists(os.path.join(args.log_dir, current_time, 'tensorboards')):
            os.makedirs(os.path.join(args.log_dir, current_time, 'tensorboards'))

    writer = SummaryWriter(os.path.join(args.log_dir, current_time, 'tensorboards'))

    best_acc = 0
    best_loss = 1.0
    for epoch in range(start_epoch, args.nepoches):

        model.train()
        loss, acc = train_one_epoch(train_loader, model, loss_func, optimizer, device, epoch)
        writer.add_scalar('train loss', loss, epoch)
        writer.add_scalar('train acc', acc, epoch)

        if epoch % args.checkpoint_interval == 0:
            # if ngpus > 1:
            #     torch.save(model.module.state_dict(),
            #                os.path.join(args.log_dir, current_time, 'checkpoints', args.model + "_cls_%d.pth" % epoch))
            # else:
            #     torch.save(model.state_dict(),
            #                os.path.join(args.log_dir, current_time, 'checkpoints', args.model + "_cls_%d.pth" % epoch))

            model.eval()
            loss, acc = val_one_epoch(val_loader, model, loss_func, device)

            if acc > best_acc and loss < best_loss:
                best_epoch = epoch + 1
                best_acc = acc
                best_loss = loss
                best_weights = copy.deepcopy(model.state_dict())
                # torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model', "SiT_cls.pth"))
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, os.path.join(args.log_dir, current_time, 'checkpoints', 'best_model', args.model + "_cls.pth"))

            writer.add_scalar('test loss', loss, epoch)
            writer.add_scalar('test acc', acc, epoch)

        scheduler.step()

    print('\n best epoch: {}, best_loss: {:.2f}, best_acc: {:.2f}'.format(best_epoch, best_loss, best_acc))



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data/ParkinsonHW/fold_1/pointcloud/patches',help='Root to the dataset')

    parser.add_argument('--model', type=str, default='PointNet', help='Model name',
                        choices=['SiT', 'PointNet', 'PointNet++'])

    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')

    parser.add_argument('--npoints', type=int, default=256, help='Number of the training points')

    parser.add_argument('--num_category', default=2, type=int, choices=[2], help='training on spiral shape dataset')

    parser.add_argument('--gpus', type=str, default='0', help='Cuda ids')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learing rate')

    parser.add_argument('--nepoches', type=int, default=300, help='Number of traing epoches')
    parser.add_argument('--start-epoch', metavar='N', type=int, default=0, help='manual epoch number (useful on restarts)')

    parser.add_argument('--opt', default='AdamW', choices=['Adam', 'SGD'],
                        help='loss: ' + ' | '.join(['Adam', 'SGD']) + ' (default: Adam)')

    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])

    parser.add_argument('--augment', type=bool, default=False, help='Augment the train data')

    parser.add_argument('--log_dir', type=str, default='log_dir', help='Train/val loss and accuracy logs')

    parser.add_argument('--checkpoint_interval', type=int, default=1, help='Checkpoint saved interval')

    # checkpoint and resume    #2023_10_26_21_27_25
    parser.add_argument('--resume', metavar='PATH', type=str, default='',
                        help='path to latest checkpoint (default: none)')

    parser.add_argument('--large-batch-size', metavar='N', type=bool, default=False,
                        help='using large batch size (default: False)')

    parser.add_argument('--accumulation_steps', metavar='N', type=int, default=10,
                        help='The accumulation steps of using large batch size (default: 10)')

    args = parser.parse_args()
    print(args)

    device_ids = list(map(int, args.gpus.strip().split(','))) if ',' in args.gpus else [int(args.gpus)]
    ngpus = len(device_ids)

    train_dataset = MyDataset(data_root=args.data_root, split='train', transform=None, augment=args.augment)
    val_dataset = MyDataset(data_root=args.data_root, split='val', transform=None)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size // ngpus, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size // ngpus, shuffle=False, num_workers=4)

    print('Train set: {}'.format(len(train_dataset)))
    print('Test set: {}'.format(len(val_dataset)))


    # Mutli-gpus
    device = torch.device("cuda:{}".format(device_ids[0]) if torch.cuda.is_available() else "cpu")
    # device = 'cpu'

    # MODEL LOADING
    if args.model == 'PointNet':
        model = PointNet(args.num_category, normal_channel=True) # if feature just x y z, the normal_channel is False; if feature is x y z R G B, the normal_channel is True
        cls_loss = PointNet_loss()
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(args.model))

    if ngpus > 1 and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=device_ids)
    model = model.to(device)
    loss = cls_loss.to(device)

    if args.opt == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
    elif args.opt == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-4)
    elif args.opt == 'Adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=1e-4)
    elif args.opt == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=1e-4)
    elif args.opt == 'Adamax':
        optimizer = torch.optim.Adamax(model.parameters(), lr=args.lr, weight_decay=1e-4)
    elif args.opt == 'Adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr, weight_decay=1e-4)
    elif args.opt == 'ASGD':
        optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, weight_decay=1e-4)
    elif args.opt == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    else:
        raise NotImplementedError


    if args.scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.nepoches, eta_min=1e-6)
    elif args.scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, verbose=1, min_lr=1e-6)
    elif args.scheduler == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30, 50], gamma=0.1)
    elif args.scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    else:
        raise NotImplementedError


    train(train_loader=train_loader,
          val_loader=val_loader,
          model=model,
          loss_func=loss,
          optimizer=optimizer,
          scheduler=scheduler,
          device=device,
          ngpus=ngpus,
          )
