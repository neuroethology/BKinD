from __future__ import print_function, absolute_import

import os
import os.path

import numpy as np

import torch
import torch.utils.data as data
import torchvision

import torchvision.transforms.functional as TF

import h5py

from dataloader.data_utils import *


def generate_pair_images(root, subjects, actions, gap=20):
    images = []

    root = os.path.expanduser(root)

    for subject in subjects:
        subject_root = os.path.join(root, subject)

        for action in actions:
            action_root = os.path.join(subject_root, action)

            h5_file = os.path.join(action_root, 'annot.h5')

            img_root = os.path.join(action_root, 'imageSequence')

            annos = h5py.File(h5_file, 'r')

            # y_min, x_min, y_max, x_max
            y_min = np.min(annos['pose']['2d'][()][:, :, 1], axis = -1)
            x_min = np.min(annos['pose']['2d'][()][:, :, 0], axis = -1)
            y_max = np.max(annos['pose']['2d'][()][:, :, 1], axis = -1)
            x_max = np.max(annos['pose']['2d'][()][:, :, 0], axis = -1)

            for i in range(0, len(annos['frame'][()])-gap, gap):

                camera_0 = annos['camera'][()][i]
                camera_1 = annos['camera'][()][i+gap]
                if camera_0 == camera_1:
                    name_0 = 'img_{:06d}.jpg'.format(annos['frame'][()][i])
                    name_1 = 'img_{:06d}.jpg'.format(annos['frame'][()][i+gap])
                    im0 = os.path.join(img_root, str(camera_0), name_0)
                    im1 = os.path.join(img_root, str(camera_1), name_1)

                bbox0 = [y_min[i], x_min[i], y_max[i], x_max[i]]
                bbox1 = [y_min[i+gap], x_min[i+gap], y_max[i+gap], x_max[i+gap]]

                item = [im0, im1, bbox0, bbox1]

                images.append(item)

    return images


class H36MDataset(data.Dataset):
    """DataLoader for Human 3.6M dataset
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=box_loader, image_size=[128, 128],
                 simplified=False, crop_box=True, frame_gap=20):

        subjects = ['S1', 'S5', 'S6', 'S7', 'S8']

        if simplified:
            actions = ['Waiting-1', 'Waiting-2', 'Posing-1', 'Posing-2', 'Greeting-1', 'Greeting-2',
                       'Directions-1', 'Directions-2', 'Discussion-1', 'Discussion-2', 'Walking-1', 'Walking-2']
        else:
            actions = ['Directions-1', 'Eating-1', 'Phoning-1', 'Purchases-1',
                       'SittingDown-1', 'TakingPhoto-1', 'Walking-1', 'WalkingTogether-1',
                       'Directions-2', 'Eating-2', 'Phoning-2', 'Purchases-2',
                       'SittingDown-2', 'TakingPhoto-2', 'Walking-2', 'WalkingTogether-2',
                       'Discussion-1', 'Greeting-1', 'Posing-1', 'Sitting-1', 'Smoking-1',
                       'Waiting-1', 'WalkingDog-1', 'Discussion-2', 'Greeting-2', 'Posing-2',
                       'Sitting-2', 'Smoking-2', 'Waiting-2', 'WalkingDog-2']

        samples = generate_pair_images(root, subjects, actions, gap=frame_gap)

        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader

        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform
        self.crop_box = crop_box

        # Parameters for transformation
        self._image_size = image_size


    def __getitem__(self, index):
        
        img_path0, img_path1, bbox0, bbox1 = self.samples[index]

        if self.crop_box:
            im0 = self.loader(img_path0, bbox0)
            im1 = self.loader(img_path1, bbox0)
        else:
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
