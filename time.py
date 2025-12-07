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
from models.pointtransformer import PointTransformerCls, PiT_loss
from dataset import MyDataset


# ADDED: bytes -> GB 的小工具
def _bytes_to_gb(x):
    return float(x) / (1024 ** 3)


def train_one_epoch(train_loader, model, loss_func, optimizer, device, epoch):
    train_bar = tqdm(train_loader)
    running_results = {'total_data_size': 0, 'loss': 0, 'acc': 0}

    # ADDED: 重置本 epoch 的峰值显存统计
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

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
                desc='Epoch[%003d/%d]: Training, lr = %.8f, loss: %.4f, acc: %.4f' % (
                    epoch, args.nepoches, lr,
                    running_results['loss'] / running_results['total_data_size'],
                    running_results['acc'] / running_results['total_data_size'])
            )
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
                    desc='Epoch[%003d/%d]: Training, lr = %.8f, loss: %.4f, acc: %.4f' % (
                        epoch, args.nepoches, lr,
                        running_results['loss'] / running_results['total_data_size'],
                        running_results['acc'] / running_results['total_data_size'])
                )

    # ADDED: 读取本 epoch 的峰值显存（字节）
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        tr_peak_alloc = torch.cuda.max_memory_allocated(device)
        tr_peak_resv = torch.cuda.max_memory_reserved(device)
    else:
        tr_peak_alloc = tr_peak_resv = 0

    return (
        running_results['loss'] / running_results['total_data_size'],
        running_results['acc'] / running_results['total_data_size'],
        tr_peak_alloc,  # ADDED
        tr_peak_resv    # ADDED
    )


def val_one_epoch(val_loader, model, loss_func, device):
    val_results = {'total_data_size': 0, 'acc': 0, 'loss': 0}
    val_bar = tqdm(val_loader)

    # ADDED: 重置本 val epoch 的峰值显存统计
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

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

            # NOTE: 这里原代码使用了全局 optimizer 取 lr（保持不变）
            lr = optimizer.state_dict()['param_groups'][0]['lr']

            val_results['total_data_size'] += args.batch_size
            val_results['loss'] += loss.item() * args.batch_size
            val_results['acc'] += torch.sum(pred == labels)

            val_bar.set_description(desc='                Val, lr = %.8f, loss = %.4f, acc = %.4f' % (
                lr,
                val_results['loss'] / val_results['total_data_size'],
                val_results['acc'] / val_results['total_data_size'])
            )

    # ADDED: 读取本 val epoch 的峰值显存（字节）
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        va_peak_alloc = torch.cuda.max_memory_allocated(device)
        va_peak_resv = torch.cuda.max_memory_reserved(device)
    else:
        va_peak_alloc = va_peak_resv = 0

    return (
        val_results['loss'] / val_results['total_data_size'],
        val_results['acc'] / val_results['total_data_size'],
        va_peak_alloc,  # ADDED
        va_peak_resv    # ADDED
    )


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
        # CHANGED: 接收训练 epoch 的峰值显存
        tr_loss, tr_acc, tr_peak_alloc_b, tr_peak_resv_b = train_one_epoch(
            train_loader, model, loss_func, optimizer, device, epoch
        )
        writer.add_scalar('train loss', tr_loss, epoch)
        writer.add_scalar('train acc', tr_acc, epoch)
        # ADDED: 训练显存（GB）
        writer.add_scalar('mem/train_peak_alloc_GB', _bytes_to_gb(tr_peak_alloc_b), epoch)
        writer.add_scalar('mem/train_peak_reserved_GB', _bytes_to_gb(tr_peak_resv_b), epoch)

        if epoch % args.checkpoint_interval == 0:
            model.eval()
            # CHANGED: 接收验证 epoch 的峰值显存
            va_loss, va_acc, va_peak_alloc_b, va_peak_resv_b = val_one_epoch(
                val_loader, model, loss_func, device
            )

            # 保存最优模型（保持原逻辑）
            if va_acc > best_acc and va_loss < best_loss:
                best_epoch = epoch + 1
                best_acc = va_acc
                best_loss = va_loss
                best_weights = copy.deepcopy(model.state_dict())
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, os.path.join(args.log_dir, current_time, 'checkpoints', 'best_model', args.model + "_cls.pth"))

            writer.add_scalar('test loss', va_loss, epoch)
            writer.add_scalar('test acc', va_acc, epoch)
            # ADDED: 验证显存（GB）
            writer.add_scalar('mem/val_peak_alloc_GB', _bytes_to_gb(va_peak_alloc_b), epoch)
            writer.add_scalar('mem/val_peak_reserved_GB', _bytes_to_gb(va_peak_resv_b), epoch)

            # 控制台打印一份，便于观察
            print(f"[Mem][Epoch {epoch}] "
                  f"Train peak: alloc={_bytes_to_gb(tr_peak_alloc_b):.2f} GB, reserv={_bytes_to_gb(tr_peak_resv_b):.2f} GB | "
                  f"Val peak: alloc={_bytes_to_gb(va_peak_alloc_b):.2f} GB, reserv={_bytes_to_gb(va_peak_resv_b):.2f} GB")

        scheduler.step()

    print('\n best epoch: {}, best_loss: {:.2f}, best_acc: {:.2f}'.format(best_epoch, best_loss, best_acc))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data/ParkinsonHW/fold_1/pointcloud/patches', help='Root to the dataset')
    # DraWritePD

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

    parser.add_argument('--augment', type=bool, default=True, help='Augment the train data')

    parser.add_argument('--log_dir', type=str, default='log_dir', help='Train/val loss and accuracy logs')

    parser.add_argument('--checkpoint_interval', type=int, default=1, help='Checkpoint saved interval')

    # checkpoint and resume    #2023_10_26_21_27_25
    parser.add_argument('--resume', metavar='PATH', type=str, default='',
                        help='path to latest checkpoint (default: none)')

    # large batch size.当数据太大或者硬件有限时，累计多次之后再进行迭代，可以和更大batchsize性能相当
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
        # if feature just x y z, the normal_channel is False; if feature is x y z R G B, the normal_channel is True
        model = PointNet(args.num_category, normal_channel=True)
        cls_loss = PointNet_loss()
    elif args.model == 'PointNet++':
        model = PointTransformerCls(npoints=args.npoints)
        cls_loss = PiT_loss()
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
          ngpus=ngpus)

