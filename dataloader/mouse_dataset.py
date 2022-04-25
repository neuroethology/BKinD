from __future__ import print_function, absolute_import

import os
import os.path

import numpy as np
import math
import numbers

import torch
import torch.utils.data as data
import torch.nn as nn

from torchvision.datasets.folder import DatasetFolder
import torchvision

from PIL import Image
import cv2

import pickle

import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import pad

import pandas as pd

from torchvision import transforms

import h5py

from dataloader.data_utils import *


def generate_pair_images(root, gap=6, win_gap = 7):
    images = []

    root = os.path.expanduser(root)

    no_cable_list = ['mouse001', 'mouse002', 'mouse051', 'mouse052','mouse055',
                    'mouse056', 'mouse058', 'mouse053', 'mouse054', 'mouse057']

    for video in sorted(os.listdir(root)):

        if video in no_cable_list:
            img_root = os.path.join(root, video)
            img_list = sorted(os.listdir(img_root))
            for index in range(0, len(img_list) - gap, win_gap):
                im0 = os.path.join(img_root, img_list[index])
                im1 = os.path.join(img_root, img_list[index+gap])

                item = [im0, im1]

                images.append(item)

    return images


class MouseDataset(data.Dataset):
    """
    DataLoader for CalMS21 dataset
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, image_size=[128, 128], rotation=True,
                 simplified=False, crop_box=True):

        samples = generate_pair_images(root)

        self.root = root
        self.loader = loader

        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform
        self.crop_box = crop_box

        # Parameters for transformation
        self._image_size = image_size

        # Rotation loss
        self.rotation = rotation
        if rotation:
            self.transform_rot = RandomRotation(180)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        img_path0, img_path1 = self.samples[index]

        im0 = self.loader(img_path0)
        im1 = self.loader(img_path1)

        height, width = self._image_size[:2]

        image0 = torchvision.transforms.Resize((height, width))(im0)
        image1 = torchvision.transforms.Resize((height, width))(im1)

        # Create 3 rotations
        deg = 90
        rot_image1 = TF.rotate(image1, deg)
        rot_image1 = self.target_transform(rot_image1)

        deg = 180
        rot_image2 = TF.rotate(image1, deg)
        rot_image2 = self.target_transform(rot_image2)

        deg = -90
        rot_image3 = TF.rotate(image1, deg)
        rot_image3 = self.target_transform(rot_image3)

        if self.transform is not None:
            image0 = self.transform(image0)
        if self.target_transform is not None:
            image1 = self.target_transform(image1)

        mask = torch.ones((1, height, width))

        return (image0, image1, mask, mask, rot_image1, rot_image2, rot_image3, img_path0, img_path1)


    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
