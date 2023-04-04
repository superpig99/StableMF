import math
import os
import random
import shutil
import time
import warnings

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import models
from ops.config import parser
from training.schedule import lr_setter
from training.train import train
from training.validate import validate
from utilis.meters import AverageMeter
from utilis.saving import save_checkpoint

from dataset import MyDataset

best_loss1 = 0


def main():
    args = parser.parse_args()

    args.log_path = os.path.join(args.log_base, args.dataset, args.arch, 'mae_' + str(args.cv) +'_log.txt')

    if not os.path.exists(os.path.dirname(args.log_path)):
        os.makedirs(os.path.dirname(args.log_path))

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    main_worker(ngpus_per_node, args)


def main_worker(ngpus_per_node, args):
    global best_loss1

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    cv = args.cv
    trainpath = os.path.join(args.datapath, 'u%d.base' % cv)
    testpath = os.path.join(args.datapath, 'u%d.test' % cv)

    if args.dataset == 'DoubanMusic':
        train_dataset = MyDataset(trainpath)
        test_dataset = MyDataset(testpath)
    elif args.dataset == 'ml-100k':
        train_dataset = MyDataset(trainpath, timestamp=True)
        test_dataset = MyDataset(testpath, timestamp=True)

    if args.distributed:
        print("initializing distributed sampler")
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)


    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](train_loader, args=args, biased=args.biased)
    print("=> finished creating model '{}'".format(args.arch))

    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.L1Loss()
    criterion_train = nn.L1Loss(reduction='none').cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    log_dir = os.path.dirname(args.log_path)
    print('tensorboard dir {}'.format(log_dir))
    tensor_writer = SummaryWriter(log_dir)
    
    if args.evaluate:
        validate(test_loader, model, criterion, 0, True, args, tensor_writer)
        return

    front_loss = validate(test_loader, model, criterion, -1, True, args, tensor_writer)
    early_stop = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        lr_setter(optimizer, epoch, args)

        train(train_loader, model, criterion_train, optimizer, epoch, args, tensor_writer)

        loss1 = validate(test_loader, model, criterion, epoch, True, args, tensor_writer)

        if loss1 > front_loss:
            early_stop = early_stop + 1
        else:
            early_stop = 0

        if early_stop == args.early_stop:
            break
        else:
            front_loss = loss1

        is_best = loss1 < best_loss1
        best_loss1 = min(loss1, best_loss1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            pass
            # save_checkpoint({
            #     'epoch': epoch + 1,
            #     'arch': args.arch,
            #     'state_dict': model.state_dict(),
            #     'best_acc1': best_acc1,
            #     'optimizer' : optimizer.state_dict(),
            # }, is_best, args.log_path, epoch)


if __name__ == '__main__':
    main()
