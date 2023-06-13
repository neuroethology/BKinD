from __future__ import print_function, absolute_import

import os
import torch
import torch.utils.data as data
from torchvision import transforms
import torchvision.transforms.functional as TF

from dataloader.data_utils import *

def generate_pair_images(root, gap):
    images = []
    root = os.path.expanduser(root)

    for video in sorted(os.listdir(root)):
        img_root = os.path.join(root, video)
        img_list = sorted(os.listdir(img_root))
        for index in range(0, len(img_list) - gap, gap//10+1):
            im0 = os.path.join(img_root, img_list[index])
            im1 = os.path.join(img_root, img_list[index+gap])

            # TODO: BBox fill ins
            bbox0 = [0,0,0,0]
            bbox1 = [0,0,0,0]

            item = [im0, im1, bbox0, bbox1]

            images.append(item)
    
    return images

class CustomDataset(data.Dataset):
    """DataLoader for your own dataset:
    This dataloader generates a pair of images, mask, and rotated images
        - root: path to your own dataset (please modify generate_pair_images
                function based on your dataset structure)
        - transform, target_transform: resize and crop as a default transform
        - loader: if you want to use a cropped bounding box as an input to the network,
                  set loader=box_loader, otherwise it will load a full image.
        - image_size: input image size
        - frame_gap: sample images with N frame gap
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=box_loader, image_size=[128, 128], frame_gap=20, crop_box = False):

        samples = generate_pair_images(root, gap=frame_gap)

        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader

        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform

        # Parameters for transformation
        self._image_size = image_size

        self.crop_box = crop_box


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        img_path0, img_path1, bbox0, bbox1 = self.samples[index]

        if self.crop_box:
            im0 = self.loader(img_path0, bbox0)
            im1 = self.loader(img_path1, bbox1)
        else:
            im0 = self.loader(img_path0)
            im1 = self.loader(img_path1)

        height, width = self._image_size[:2]

        image0 = transforms.Resize((height, width))(im0)
        image1 = transforms.Resize((height, width))(im1)

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
