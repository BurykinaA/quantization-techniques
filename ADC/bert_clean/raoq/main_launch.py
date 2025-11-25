import argparse
import os
import random
import time
from datetime import datetime
import logging
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
#import torchvision.models as models
import models
from torch.utils.tensorboard import SummaryWriter
from torch.distributed.optim import ZeroRedundancyOptimizer
from utils import datasets_loader
from utils.utils import *
from train.utils import AugScheduler
import numpy as np
import sys
from torch.utils.data import Subset
import pdb


parser = argparse.ArgumentParser(description='Quantized Neural Network for IMC SRAM Chip')

# data & model
parser.add_argument('--dataset', type=str, default='imagenet', help='dataset name or folder')
parser.add_argument('--data-backend', type=str, default='pytorch', help='approach to build dataset loader')
parser.add_argument('--num_classes', default=1000, type=int, help='number of classes')
parser.add_argument('--model', default='imagenet_quant', help='model architecture')
parser.add_argument('--workers', default=8, type=int, help='number of data loading workers')
parser.add_argument('--type', default='torch.cuda.FloatTensor', help='type of tensor - e.g torch.cuda.HalfTensor')

# training configurations
#parser.add_argument('--optimizer', default='SGD', type=str, help='optimizer function used')
parser.add_argument('--epochs', default=20, type=int, help='number of epochs')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=128, type=int, help='Batch size, this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--accumulate_steps', default=4, type=int, help='Steps for gradient accumulation')
parser.add_argument('--lr_policy', type=str, default='onecycle', help='type of the learning rate policy')
parser.add_argument('--lr_initial', type=float, default=5e-5, help='initial learning rate')
parser.add_argument('--lr_end', type=float, default=5e-5, help='final learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight decay ',)
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
parser.add_argument('--custom', dest='custom', action='store_true', help='whether pretrained checkpoint is from user end')
parser.add_argument('--model_dir', default='', help='directory of pretrained model')
parser.add_argument('--ref_dir', default='', help='directory of pretrained reference model')
parser.add_argument('--resume', type=str, default='', help='path to latest checkpoint')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--aug', dest='aug', action='store_true', help='whether perform augmented training')

# misc & distributed data parallel
parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str, help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--use_zero', dest='use_zero', action='store_true', help='Whether to use ZeroRedundancyOptimizer')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--ngpus', default=1, type=int, help='# of GPUs to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# quantization
parser.add_argument('--fp_mode', dest='fp_mode', action='store_true', help='Use floating point for computation')
parser.add_argument('--train_scale', dest='train_scale', action='store_true', help='Make scaling factors for quantization trainable')
parser.add_argument('--sen', dest='sen', action='store_true', help='Whether to quantize end layers')
parser.add_argument('--set_bits', dest='set_bits', action='store_true', help='Set bits to scaling factors')
parser.add_argument('--first_batch_fp', dest='first_batch_fp', action='store_true', help='Perform floating point computation for 1st batch')
parser.add_argument('--imc', dest='imc', action='store_true', help='Perform MVM based on bit-parallel/bit-serial (BP/BS) mode chip models')
parser.add_argument('--adc', dest='adc', action='store_true', help='Flag for including ADC quantization')
parser.add_argument('--quant_chip', dest='quant_chip', action='store_true', help='Perform quantization behavior of the chip')
parser.add_argument('--mode', type=str, default='and', help='The mode of CIMA computations')
parser.add_argument('--weight_bits', type=int, default=8, help='number of bits for weight quantization')
parser.add_argument('--input_bits', type=int, default=8, help='number of bits for input quantization')
parser.add_argument('--bias_bits', type=int, default=32, help='number of bits for bias quantization')
parser.add_argument('--adc_bits', type=int, default=8, help='number of bits for ADC')
parser.add_argument('--noise', dest='noise', action='store_true', help='number of bits for input quantization')

# interaction
parser.add_argument('--print-freq', default=50, type=int, help='print frequency')
parser.add_argument('--examine', dest='examine', action='store_true', help='Check data using tensorboard')
parser.add_argument('--save_dir', default='./results', help='directory to write the results')

best_acc1 = 0
bit_candidates = [7,9,10]

def main():
    args = parser.parse_args()
    save_path = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    writer = SummaryWriter('runs/experiment'+save_path) if args.examine else None

    args.save_dir = os.path.join(args.save_dir, save_path)
    if not os.path.exists(args.save_dir) and int(os.environ['LOCAL_RANK']) == 0:
        os.makedirs(args.save_dir)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')


    if args.dist_url == "env://":
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

#    ngpus_per_node = torch.cuda.device_count()
    ngpus_per_node = args.ngpus
    args.workers = args.workers * ngpus_per_node
    print('Total workers:', args.workers)
    if args.multiprocessing_distributed:
        os.environ['OMP_NUM_THREADS'] = str(mp.cpu_count() / args.world_size)
        print('threads:', os.environ['OMP_NUM_THREADS'])
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = 'DETAIL'
        args.rank = int(os.environ['LOCAL_RANK'])
        if args.dist_url == 'env://' and 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            args.world_size = int(os.environ['WORLD_SIZE'])
            args.gpu = int(os.environ['LOCAL_RANK'])
        else:
            args.gpu = args.rank
            print('Manually set rank and world size')
        print('rank:', args.rank)
        print('world_size:', args.world_size)
        print('gpu:', args.gpu)
        main_worker(args.gpu, args.world_size, args, writer)
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args, writer)


def main_worker(gpu, ngpus_per_node, args, writer):
    global best_acc1
    args.gpu = gpu

    log_flag = (gpu == 0) if args.distributed else True
    logger = setup_logging(os.path.join(args.save_dir, 'log.txt'), log_flag)
#    logger = utils.get_logger(args.save_dir, resume=False, is_rank0=log_flag)
    logger.info("saving to {}".format(args.save_dir))
    print('gpu: {}'.format(gpu))

    if args.gpu is not None:
        logger.info("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        env_dict = {
            key: os.environ[key]
            for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
        }
        print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
#        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
#                                world_size=args.world_size, rank=args.rank)
        dist.init_process_group(backend=args.dist_backend)
    # create model
    model = models.__dict__[args.model]
    config = {}
    config['bx'] = args.input_bits
    config['bw'] = args.weight_bits
    config['bbias'] = args.bias_bits
    config['mode'] = args.mode
    config['imc_mode'] = args.imc
    config['fp_mode'] = args.fp_mode
    config['train_scale'] = args.train_scale
    config['sen'] = args.sen
    config['run_1st_batch_fp'] = args.first_batch_fp
    config['x_range_type'] = 'dist'
    config['w_range_type'] = 'dist'
    config['bias_range_type'] = 'dist'
    config['set_range_once'] = True
    config['x_signed'] = False
    config['adc_quant'] = args.adc
    config['quant_flag'] = args.quant_chip
    config['b_adc'] = args.adc_bits
    config['noise'] = args.noise
    config['set_bits'] = args.set_bits
    
    model_config = {'num_classes':args.num_classes, 'general_config':config, 'model_dir':args.model_dir, 'device':args.gpu, 'custom':args.custom, 'writer':writer}
    if args.pretrained:
        model_config['pretrained'] = True
        logger.info("=> using pre-trained model '{}'".format(args.model))
    else:
        model_config['pretrained'] = False
        logger.info("=> creating model '{}'".format(args.model))
    model = model(**model_config)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            print('# workers per proc:', args.workers)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False, gradient_as_bucket_view=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        logger.info('Single GPU enabled!')
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # Data loading code
    train_loader, train_sampler, val_loader = datasets_loader.build_dataset(args.dataset, args.batch_size, args.workers, args.distributed)

    total_iter = len(train_loader) * (args.epochs-10)
    weight_scheduler = AugScheduler(total_iter, 1, 0)

    design_params = weight_decay_sep(model, args.weight_decay)
    if args.use_zero and args.distributed:
        optimizer = ZeroRedundancyOptimizer(model.parameters(), optimizer_class=torch.optim.SGD,
                                            lr=args.lr_initial, momentum=args.momentum, 
                                            weight_decay=args.weight_decay, 
                                            multiplier=1, gamma=0.9)
    else:
        optimizer = torch.optim.SGD(design_params, args.lr_initial,
                                    momentum=args.momentum)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    #torch.backends.cudnn.enabled=False

    if args.evaluate:
        validate(val_loader, model, criterion, args, logger)
        return

    freeze_scale = False
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        if epoch >= args.epochs - 10:
            weight_scheduler = None
        train(train_loader, model, criterion, optimizer, scheduler, epoch, args, logger, freeze_scale, weight_scheduler)

        scheduler.step()

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args, logger)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.model,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict(),
            }, is_best, args.save_dir)


def train(train_loader, model, criterion, optimizer, scheduler, epoch, args, logger, freeze_scale, weight_scheduler):
    batch_time = average_meter('Time', ':6.3f')
    data_time = average_meter('Data', ':6.3f')
    losses_ce = average_meter('Loss (CE)', ':.2e')
    losses_kt = average_meter('Loss (KT)', ':.2e')
    top1 = average_meter('Acc@1', ':6.2f')
    top5 = average_meter('Acc@5', ':6.2f')
    progress = progress_meter(
        len(train_loader),
        [batch_time, data_time, losses_ce, losses_kt, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    counter_start = time.perf_counter()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
       
        if weight_scheduler is not None:
            weight = weight_scheduler(epoch * len(train_loader) + i)
        else:
            weight = 0
        if args.aug and weight > 0:
            # base
            output, _ = model(images, args.adc_bits)
            loss_ce = criterion(output, target)
            loss = loss_ce 
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses_ce.update(loss_ce.detach(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            loss.backward()

            # sample ADC bit
            adc_bit = np.random.choice(bit_candidates)
            output, _ = model(images, adc_bit)
            loss_aug = weight * criterion(output, target)
            loss_aug.backward()
        else:
            output, loss_kt = model(images, args.adc_bits)
            #loss_kt = 0.0006 * loss_kt
            loss_ce = criterion(output, target)
            loss = loss_ce 
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses_ce.update(loss_ce.detach(), images.size(0))
            #losses_kt.update(loss_kt.detach(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            loss.backward()
            
        # compute gradient and do SGD step
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i, logger)

    counter_end = time.perf_counter()
    logger.info('training time: {}'.format(counter_end - counter_start))


def validate(val_loader, model, criterion, args, logger):
    batch_time = average_meter('Time', ':6.3f')
    losses = average_meter('Loss', ':.4e')
    top1 = average_meter('Acc@1', ':6.2f')
    top5 = average_meter('Acc@5', ':6.2f')
    progress = progress_meter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    counter_start = time.perf_counter()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output, _ = model(images, args.adc_bits)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.detach(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i, logger)
    if args.distributed:
        top1.all_reduce()
        top5.all_reduce()
    progress.display_summary(logger)
    counter_end = time.perf_counter()
    logger.info('Testing time: {}'.format(counter_end - counter_start))

    return top1.avg


if __name__ == '__main__':
    main()
