import os
import random
import shutil
import time

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.autograd import Variable
from utilis.metrics import accuracy
from utilis.meters import AverageMeter, ProgressMeter

from training.reweighting import weight_learner


def train(train_loader, model, criterion, optimizer, epoch, args, tensor_writer=None):
    ''' TODO write a dict to save previous featrues  check vqvae,
        # the size of each feature is 512, so we need a tensor of 1024 * 512
        replace the last one every time
        and a weight with size of 1024,
        replace the last one every time
        TODO init the tensors
    '''

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    end = time.time()
    for i, (user_item, target) in enumerate(train_loader):
        if target.shape[0] != train_loader.batch_size:
            break

        data_time.update(time.time() - end)

        if args.gpu is not None:
            user_item = user_item.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        output, cfeatures = model(user_item)
        pre_features = model.pre_features
        pre_weight1 = model.pre_weight1

        print('\nLine 55 of train.py:')
        print('input shape:\n', '\tuser_item.size:',user_item.size(), '\ttarget.size:',target.size())
        print('\tcfeatures.size:',cfeatures.size(), '\tpre_features.size:',pre_features.size(), '\tpre_weight1.size:',pre_weight1.size())

        if epoch >= args.epochp:
            weight1, pre_features, pre_weight1 = weight_learner(cfeatures, pre_features, pre_weight1, args, epoch, i)

        else:
            weight1 = Variable(torch.ones(cfeatures.size()[0], 1).cuda())       # cfeatures.size()[0] maybe the batch size

        weight1 = weight1 * train_loader.batch_size
        model.pre_features.data.copy_(pre_features)
        model.pre_weight1.data.copy_(pre_weight1)

        loss = criterion(output, target).view(1, -1).mm(weight1).view(1)
        losses.update(loss.item(), user_item.size(0))                           # size(0) is the batch size

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        method_name = args.log_path.split('/')[-2]
        if i % args.print_freq == 0:
            progress.display(i, method_name)
            progress.write_log(i, args.log_path)

    tensor_writer.add_scalar('loss/train', losses.avg, epoch)
