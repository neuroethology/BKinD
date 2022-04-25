from __future__ import print_function, absolute_import

import os
import numpy as np
import json
import random
import math

import torch
import torch.utils.data as data
import torch.nn as nn

from torchvision.datasets.folder import DatasetFolder
import torchvision

from PIL import Image
import pickle

import torch.nn.functional as F

# import os
import os.path
import torchvision.transforms as transforms

import cv2
import skimage
import skimage.transform
import scipy

from torchvision.transforms.functional import pad
from torchvision import transforms
import numpy as np
import numbers

import torchvision.transforms.functional as TF

import h5py
from scipy.io import loadmat

from data_utils import *


def assign_classes(actions):
    classes = [x for x in actions]
    class_to_idx = {}
    cnt = 0
    for i in range(len(classes)):
        if classes[i] not in class_to_idx.keys():
            class_to_idx[classes[i]] = cnt
            cnt += 1

    return classes, class_to_idx


def generate_kp_images(root, subjects, actions, class_to_idx):
    images = []

    root = os.path.expanduser(root)

    for subject in subjects:
        subject_root = os.path.join(root, subject, 'BackgroudMask')
        im_root = os.path.join(root, subject, 'WithBackground')
        action_dirs = os.listdir(subject_root)

        for action_dir in action_dirs:
            if len(action_dir.split(' ')) == 2:
                action = action_dir.split(' ')[0]
            else:
                action = action_dir.split('.')[0]

            if action in actions:
                frames = os.listdir(os.path.join(subject_root, action_dir))
                frames = [int(os.path.splitext(x)[0]) for x in frames]
                frames = sorted(frames)

                # not used for now.. (2d keypoints)
                anno_dir = os.path.join(root, subject, 'Landmarks', action_dir)
                bbox0 = [0, 0, 128, 128]

                for i in range(0, len(frames)):
                    anno = os.path.join(anno_dir, str(frames[i])+'.mat')
                    kps = anno

                    im0 = os.path.join(im_root, action_dir, str(frames[i])+'.jpg')
                    mask0 = os.path.join(subject_root, action_dir, str(frames[i])+'.png')
                    action_id = class_to_idx[action]

                    item = [im0, mask0, bbox0, kps, action_id]

                    images.append(item)

    return images


class SimpleH36MRegressionDataset(data.Dataset):
    """Simplified Human 3.6M dataset for pose regression
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, image_size=[128, 128], simplified=True,
                 evaluation=False, masked=False, subset=True):

        if evaluation:
            subjects = ['S9', 'S11']
        else:
            subjects = ['S1', 'S5', 'S6', 'S7', 'S8']

        self.evaluation = evaluation
        actions = ['Waiting', 'Posing', 'Greeting', 'Directions', 'Discussion', 'Walking']

        classes, class_to_idx = assign_classes(actions)
        samples = generate_kp_images(root, subjects, actions, class_to_idx)
        self.subset = subset

        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader

        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform
        self.imgs = self.samples

        # Parameters for transformation
        self._image_size = image_size

        self.masked = masked
        self.transform_mask = transforms.Compose([
                          transforms.Resize(image_size[0]),
                          transforms.CenterCrop(image_size[0]),
                          transforms.ToTensor(),])


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        img_path0, mask_path0, bbox0, anno, target = self.samples[index]

        im0 = self.loader(img_path0)

        m0 = self.loader(mask_path0)

        details = [0, 0, im0.size[0], im0.size[1]]

        height, width = self._image_size[:2]

        anno = loadmat(anno)

        kps = anno['keypoints_2d']

        left_ids = [6,7,8,9,10,16,17,18,19,20,21,22,23]
        right_ids = [1,2,3,4,5,24,25,26,27,28,29,30,31]

        # compare counterpart
        x_left = kps[left_ids,1]
        x_right = kps[right_ids,1]

        if not self.evaluation:

            if sum(x_left > x_right) < 9:  # Flip annotation for back facing images
                flip_kps = kps[left_ids]
                kps[left_ids] = kps[right_ids]
                kps[right_ids] = flip_kps

        kps_im = kps

        if self.subset:
            kp_ids = [1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27]
            kps_im = kps_im[kp_ids,:]

        kps_im = (kps_im * 2.0) - 1.0
        kps = kps_im.copy()

        kps[:,0] = kps_im[:,1]
        kps[:,1] = kps_im[:,0]

        kps_im = kps

        image0 = torchvision.transforms.Resize((height, width))(im0)

        if self.transform is not None:
            image0 = self.transform(image0)

            if self.masked:
                m0 = self.transform_mask(m0)
                image0 = image0 * m0

        details = np.asarray(details)
        kps_im = np.asarray(kps_im)
        kps = np.asarray(kps)

        # Resize bounding box
        return (image0, kps, img_path0, bbox0, details, kps_im), target

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
