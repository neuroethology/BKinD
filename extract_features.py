import argparse
import datetime
import time
import cv2

import skimage
import skimage.transform
from skimage.metrics import structural_similarity as ssim

from scipy.ndimage.filters import gaussian_filter

import numpy as np

import os

import scipy
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
import torch.nn.functional as F

import torchvision

from scipy.io import savemat

from model.unsupervised_model import Model as orgModel
from model.kpt_detector import Model


from PIL import Image

from scipy.spatial import distance

import csv


# resume, checkpoint, num keypoints
def load_model(resume, output_dir, num_keypoints = 10):

    model = Model(num_keypoints)

    # Assume GPU 0
    torch.cuda.set_device(0)
    model.cuda(0)

    save_dir = os.path.join(output_dir, 'keypoints_confidence')

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
        os.mkdir(os.path.join(save_dir, 'train'))
        os.mkdir(os.path.join(save_dir, 'test'))

    # Map model to be loaded to specified single gpu.
    loc = 'cuda:{}'.format(0)
    checkpoint = torch.load(resume, map_location=loc)

    org_model = orgModel(num_keypoints)

    org_model.load_state_dict(checkpoint['state_dict'])
    org_model_dict = org_model.state_dict()

    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in org_model_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    print("=> loaded checkpoint '{}' (epoch {})"
          .format(resume, checkpoint['epoch']))

    model.eval()

    return model, save_dir


def get_image_tensor(frame, size = 256):

    current = Image.fromarray(frame)

    crop_percent = 1.0
    final_sz = size
    resize_sz = np.round(final_sz / crop_percent).astype(np.int32)

    current = torchvision.transforms.Resize((resize_sz, resize_sz))(current)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    to_tensor = transforms.Compose([
                          transforms.Resize(size),
                          transforms.CenterCrop(size),
                          transforms.ToTensor(),
                          normalize,])

    current_tensor = to_tensor(current)


    return current_tensor.unsqueeze(0)



def compute_keypoints(inputs, model, width = 1024, height = 570):

    # Assume GPU 0
    loc = 'cuda:{}'.format(0)
    inputs = inputs.to(loc)

    output = model(inputs)


    xy = torch.stack((output[0][0], output[0][1]), dim=2).detach().cpu().numpy()[0]+1

    scale_x = (width / 2.0)

    scale_y = (height / 2.0)

    to_plot = []


    confidence = output[5].detach().cpu().numpy()[0]

    covs = torch.stack((output[6][0], output[6][1], output[6][2]), dim=2).detach().cpu().numpy()[0]+1


    for i in range(0,xy.shape[0]):

        st_y = int(xy[i,1]*scale_y); st_x = int(xy[i,0]*scale_x)

        to_plot.append([st_y, st_x])

    return xy, to_plot, confidence, covs


# Parse input arguments.
ap = argparse.ArgumentParser()
ap.add_argument("--train_dir", help="Path to train directory with directory of images", type = str)
ap.add_argument("--test_dir", help="Path to test directory with directory of images", type = str)
ap.add_argument("--resume", help="Path to checkpoint to resume", type = str)
ap.add_argument("--output_dir", help="Output directory to store the keypoints", type = str)
ap.add_argument("--imsize", default=256, help="Training image size", type=int)

args = vars(ap.parse_args())


model, save_dir = load_model(args['resume'], args['output_dir'])

# input train & test directory
train_dir = args['train_dir']
test_dir = args['test_dir']


counter = 0


# Extract for train dir
for vid in sorted(os.listdir(train_dir)):
    print(counter, vid)

    counter = counter + 1

    keypoint_array =[]
    conf_array = []
    covs_array = []

    vid_name = vid

    current_directory = os.path.join(train_dir, vid)

    for images in sorted(os.listdir(current_directory)):

        draw_frame = cv2.cvtColor(cv2.imread(os.path.join(current_directory, images),
             cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        
        height, width, _ = draw_frame.shape
        input_tensor = get_image_tensor(draw_frame)

        _, plot_keypoints, confidence, covs = compute_keypoints(input_tensor, model, width=width, height=height)


        colors = [[255,0,0], [0,255,0], [0,0,255], [255,255,0], [0,255,255], [255,0,255],  # yellow, cyan, pink
                  [255,255,255], [0,0,0], [125,125,0], [125,0,125], [0,125,125], [74,29,0],  #w b, gray-green, violet, turquiose
                  [23,87,120], [59,100,3], [130,150,10], [40,29,100], [54,12,90], [53,123,40],
                  [50,189,34], [230,142,10]]

        image = draw_frame

        ## For visualization
        # for c, j in enumerate(range(len(plot_keypoints))):
        # 
        #     item = plot_keypoints[j]
        # 
        #     image = cv2.circle(image, (item[1], item[0]),
        #         radius=2, color=colors[c], thickness = 2)
        #
        # cv2.imshow("Keypoints", image)

        conf_array.append(confidence)
        keypoint_array.append(plot_keypoints)
        covs_array.append(covs)

        # result.write(image)

        # key = cv2.waitKey(1) & 0xFF
        ## If q is pressed, then quit the loop.
        # if key == ord("q"):
        #     break

    print(np.array(keypoint_array).shape, np.array(conf_array).shape, np.array(covs_array).shape)


    np.savez(os.path.join(save_dir, 'train', vid_name), keypoints = keypoint_array, confidence = conf_array,
            covs = covs_array)


# Extract for test dir
for vid in sorted(os.listdir(test_dir)):
    print(counter, vid)

    counter = counter + 1

    keypoint_array =[]
    conf_array = []
    covs_array = []

    vid_name = vid

    current_directory = os.path.join(test_dir, vid)

    for images in sorted(os.listdir(current_directory)):

        draw_frame = cv2.cvtColor(cv2.imread(os.path.join(current_directory, images), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        height, width, _ = draw_frame.shape
        
        input_tensor = get_image_tensor(draw_frame)

        _, plot_keypoints, confidence, covs = compute_keypoints(input_tensor, model, width=width, height=height)

        conf_array.append(confidence)
        keypoint_array.append(plot_keypoints)
        covs_array.append(covs)


    print(np.array(keypoint_array).shape, np.array(conf_array).shape, np.array(covs_array).shape)

    np.savez(os.path.join(save_dir, 'test', vid_name), keypoints = keypoint_array, confidence = conf_array,
            covs = covs_array)
