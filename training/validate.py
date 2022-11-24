import time

import torch
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from utilis.metrics import accuracy
from utilis.meters import AverageMeter, ProgressMeter


def validate(val_loader, model, criterion, epoch=0, test=True, args=None, tensor_writer=None):
    if test:
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        progress = ProgressMeter(
            len(val_loader),
            [batch_time, losses],
            prefix='Test: ')
    else:
        batch_time = AverageMeter('val Time', ':6.3f')
        losses = AverageMeter('val Loss', ':.4e')
        progress = ProgressMeter(
            len(val_loader),
            [batch_time, losses],
            prefix='Val: ')

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (user_item, target) in enumerate(val_loader):
            if target.shape[0] != val_loader.batch_size:
                break
            if args.gpu is not None:
                user_item = user_item.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            output, cfeatures = model(user_item)
            loss = criterion(output, target)

            losses.update(loss.item(), user_item.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                method_name = args.log_path.split('/')[-2]
                progress.display(i, method_name)
                progress.write_log(i, args.log_path)

        if test:
            tensor_writer.add_scalar('loss/test', losses.avg, epoch)
        else:
            tensor_writer.add_scalar('loss/val', losses.avg, epoch)

    return losses.avg   # top1.avg
