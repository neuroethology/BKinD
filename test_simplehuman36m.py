import argparse
import os
import random
import shutil
import time
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
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from dataloader import simple_human36m_dataset as h36m

from evaluation import human36m_skeleton
from evaluation.evaluate_simplehuman36m import compute_mean_distance, \
        mean_per_activity, format_results_per_activity, \
        format_results_per_activity2, y_human36m_path_to_activity

from model.unsupervised_model import Model as orgModel
from model.kpt_detector import Model

from regressor.regression import linearRegressor

import numpy as np

import torch.nn.functional as F
import pickle
import math

import cv2

from utils.model_utils import *


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Regression with Supervision')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--nkpts', default=10, type=int, metavar='N')
parser.add_argument('--schedule', default=30, type=int, metavar='N')


best_acc1 = 100000  # error


def main():
    args = parser.parse_args()

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

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    model = Model(args.nkpts)

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
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()
    # model = torch.nn.DataParallel(model).cuda()

    # optionally resume from a checkpoint
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    title = 'Landmark-discovery'
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = 0

            org_model = orgModel(args.nkpts)

            org_model.load_state_dict(checkpoint['state_dict'])
            org_model_dict = org_model.state_dict()

            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in org_model_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = h36m.SimpleH36MRegressionDataset(args.data, transforms.Compose([
                          transforms.Resize(128),
                          transforms.CenterCrop(128),
                          transforms.ToTensor(),
                          normalize,]), image_size=[128, 128], loader = h36m.default_loader)
    val_dataset = h36m.SimpleH36MRegressionDataset(args.data, transforms.Compose([
                          transforms.Resize(128),
                          transforms.CenterCrop(128),
                          transforms.ToTensor(),
                          normalize,]), image_size=[128, 128], loader = h36m.default_loader, evaluation=True)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    linear_regressor(train_loader, val_loader, model, 0, args)

    return



def normalize_points(points):
    return (points + 1) / 2.0


def linear_regressor(train_loader, val_loader, model, epoch, args):

    debug = True

    # switch to train mode
    model.eval()

    distances = []
    distances_min = []
    correct_flips = []
    paths = []

    num_test = 1000000
    num_save = 60

    target_correspondeces = human36m_skeleton.get_simple_lr_correspondences()
    correspondeces = human36m_skeleton.get_simple_lr_correspondences()

    n_batches = int(math.ceil(float(len(val_loader)) / args.batch_size))  # batch size was 64

    save_frq = int(math.ceil(float(min(n_batches, num_test)) / num_save))

    avg_time = []

    path_fn = y_human36m_path_to_activity
    used_points = None

    offline_prediction = None

    with torch.no_grad():
        for i, (images, target) in enumerate(train_loader):

            if args.gpu is not None:
                inputs, kpts = images[0].cuda(args.gpu, non_blocking=True), images[1].cuda(args.gpu, non_blocking=True)

            target = target.cuda(args.gpu, non_blocking=True)

            im_path = images[2]

            # compute output
            output = model(inputs)

            pred_kpts = torch.stack((output[0][0], output[0][1]), dim=2)
            pred_kpts = pred_kpts.data.cpu().numpy()

            if i == 0:
                preds = pred_kpts
                gts = kpts.float().data.cpu().numpy()
            else:
                preds = np.concatenate((preds, pred_kpts), axis=0)
                gts = np.concatenate((gts, kpts.float().data.cpu().numpy()), axis=0)

    print("Training data done")


    iter_start_time = time.time()
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):

            if args.gpu is not None:
                inputs, kpts = images[0].cuda(args.gpu, non_blocking=True), images[1].cuda(args.gpu, non_blocking=True)

            target = target.cuda(args.gpu, non_blocking=True)
            im_path = images[2]
            box = images[3]
            details = images[4]

            # compute output
            output = model(inputs) #, inputs)

            pred_kpts = torch.stack((output[0][0], output[0][1]), dim=2)
            pred_kpts = pred_kpts.data.cpu().numpy()

            if i == 0:
                test_pred = pred_kpts
                test_gts = kpts.float().data.cpu().numpy()
            else:
                test_pred = np.concatenate((test_pred, pred_kpts), axis=0)
                test_gts = np.concatenate((test_gts, kpts.float().data.cpu().numpy()), axis=0)

            paths.extend(im_path)

    print("Test data done")

    regressed = linearRegressor(preds, gts[:,:,:2], test_pred, n_kpts=args.nkpts, reg_pts=kpts.shape[1])

    regressed = torch.from_numpy(regressed)
    test_gts = torch.from_numpy(test_gts)

    regressed = normalize_points(regressed)
    test_gts = normalize_points(test_gts)

    dist, dist_min, correct_flip = compute_mean_distance(
        regressed, test_gts, correspondeces=correspondeces,
        target_correspondeces=target_correspondeces,
        used_points=used_points, offline_prediction=offline_prediction)

    distances = dist.cpu().numpy()
    distances_min = dist_min.cpu().numpy()
    correct_flips = correct_flip.cpu().numpy()

    mean_distances = mean_per_activity(distances, paths, path_fn)
    mean_distance = np.mean(list(mean_distances.values()))
    mean_min_distances = mean_per_activity(distances_min, paths, path_fn)
    mean_min_distance = np.mean(list(mean_min_distances.values()))
    mean_correct_flips = mean_per_activity(correct_flips, paths, path_fn)
    mean_correct_flip = np.mean(list(mean_correct_flips.values()))

    results_str = 'mean distance %.4f\n' % mean_distance
    results_str += 'mean min distance %.4f\n' % mean_min_distance
    results_str += 'mean correct flips %.4f\n' % mean_correct_flip
    results_str += '%s\n' % format_results_per_activity(mean_distances)
    results_str += '%s\n' % format_results_per_activity(mean_min_distances)
    results_str += '%s\n' % format_results_per_activity(mean_min_distances)
    results_str += '%s\n' % format_results_per_activity(mean_correct_flips)

    print(results_str)

    results_str = format_results_per_activity2(mean_distances)
    print(results_str[0])
    print(results_str[1])

    results_str = format_results_per_activity2(mean_min_distances)
    print(results_str[0])
    print(results_str[1])

    return



if __name__ == '__main__':
    main()
