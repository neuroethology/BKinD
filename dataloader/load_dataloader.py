from __future__ import print_function, absolute_import

import os
import torchvision.transforms as transforms

from dataloader import h36m_dataset as h36m
from dataloader import mouse_dataset as mouse
from dataloader import custom_dataset 

from dataloader import data_utils

def load_dataloader(args):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    loader = data_utils.box_loader if args.bounding_box else data_utils.default_loader

    if args.dataset == 'H36M':
        traindir = os.path.join(args.data)
        valdir = os.path.join(args.data)

        train_dataset = h36m.H36MDataset(traindir, transforms.Compose([
                              transforms.Resize(args.image_size),
                              transforms.CenterCrop(args.image_size),
                              transforms.ToTensor(),
                              normalize,]),
                              target_transform=transforms.Compose([
                              transforms.Resize(args.image_size),
                              transforms.CenterCrop(args.image_size),
                              transforms.ToTensor(),
                              normalize,]),
                              image_size=[args.image_size, args.image_size],
                              loader=loader, frame_gap=args.frame_gap, crop_box=args.bounding_box)

        val_dataset = h36m.H36MDataset(valdir, transforms.Compose([
                              transforms.Resize(args.image_size),
                              transforms.CenterCrop(args.image_size),
                              transforms.ToTensor(),
                              normalize,]),
                              target_transform=transforms.Compose([
                              transforms.Resize(args.image_size),
                              transforms.CenterCrop(args.image_size),
                              transforms.ToTensor(),
                              normalize,]),
                              image_size=[args.image_size, args.image_size],
                              loader=loader, frame_gap=args.frame_gap, crop_box=args.bounding_box)

    elif args.dataset == 'CalMS21':
        traindir = os.path.join(args.data)
        valdir = os.path.join(args.data)

        train_dataset = mouse.MouseDataset(traindir, transforms.Compose([
                              transforms.Resize(args.image_size),
                              transforms.CenterCrop(args.image_size),
                              transforms.ToTensor(),
                              normalize,]),
                              target_transform=transforms.Compose([
                              transforms.Resize(args.image_size),
                              transforms.CenterCrop(args.image_size),
                              transforms.ToTensor(),
                              normalize,]),
                              image_size=[args.image_size, args.image_size],
                              loader=loader)

        val_dataset = mouse.MouseDataset(valdir, transforms.Compose([
                              transforms.Resize(args.image_size),
                              transforms.CenterCrop(args.image_size),
                              transforms.ToTensor(),
                              normalize,]),
                              target_transform=transforms.Compose([
                              transforms.Resize(args.image_size),
                              transforms.CenterCrop(args.image_size),
                              transforms.ToTensor(),
                              normalize,]),
                              image_size=[args.image_size, args.image_size],
                              loader=loader)

    elif args.dataset == 'custom_dataset':
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')

        train_dataset = custom_dataset.CustomDataset(traindir, transforms.Compose([
                              transforms.Resize(args.image_size),
                              transforms.CenterCrop(args.image_size),
                              transforms.ToTensor(),
                              normalize,]),
                              target_transform=transforms.Compose([
                              transforms.Resize(args.image_size),
                              transforms.CenterCrop(args.image_size),
                              transforms.ToTensor(),
                              normalize,]),
                              image_size=[args.image_size, args.image_size],
                              loader=loader, frame_gap=args.frame_gap)

        val_dataset = custom_dataset.CustomDataset(valdir, transforms.Compose([
                              transforms.Resize(args.image_size),
                              transforms.CenterCrop(args.image_size),
                              transforms.ToTensor(),
                              normalize,]),
                              target_transform=transforms.Compose([
                              transforms.Resize(args.image_size),
                              transforms.CenterCrop(args.image_size),
                              transforms.ToTensor(),
                              normalize,]),
                              image_size=[args.image_size, args.image_size],
                              loader=loader, frame_gap=args.frame_gap)

    else:
        assert args.dataset in ["H36M", "CalMS21", "custom_dataset"], \
        "Please write your own dataloader using custom_dataset.py and add your dataset name here"

    return train_dataset, val_dataset
