import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from .misc import *   

import seaborn as sns

from PIL import Image, ImageDraw
import cv2

__all__ = ['show_heatmaps', 'show_img_with_heatmap', 'visualize_with_circles', 'save_images']


def save_images(image, output, epoch, args, curr_epoch):
    mean=[0.485, 0.456, 0.406]
    _mean = np.asarray(mean).reshape((3,1,1))
    std=[0.229, 0.224, 0.225]
    _std = np.asarray(std).reshape((3,1,1))
        
    # Image with keypoints
    im_dir = os.path.join(args.checkpoint, 'samples/epoch_' + str(curr_epoch)) #, str(sample_id)+'.png')
    
    if not os.path.isdir(im_dir):
        os.makedirs(im_dir)
        
    sample_ids = np.random.permutation(len(output[0]))
    sample_ids = sample_ids[:min(5, len(output[0]))]
    
    im = image.data.cpu().numpy()
    
    kps = output[1]
    recon = output[0]
    if epoch < args.curriculum:
        heatmap = output[2]
        confidence = output[5]
    else:
        heatmap = output[2][0]
        confidence = output[6]
        
    # keypoints
    xy = torch.stack((kps[0], kps[1]), dim=2)
    
    for i, ix in enumerate(sample_ids):
        # Visualize keypoints
        im_with_pts = visualize_with_circles(im[ix], xy[ix].data.cpu().numpy()+1, confidence[ix],
                                             mean=mean, std=std)
        im_with_pts = im_with_pts.astype('uint8')
        im_with_pts = cv2.cvtColor(im_with_pts, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(im_dir, 'image_'+str(i)+'.png'), im_with_pts)

        # Heatmap
        heatmaps = show_heatmaps(heatmap[ix])
        heatmaps = (heatmaps.data.cpu().numpy() * 255).astype('uint8')
        heatmaps = heatmaps.transpose((1,2,0))
        cv2.imwrite(os.path.join(im_dir, 'heatmaps_'+str(i)+'.png'), heatmaps)

        # Reconstruction
        recon_im = recon[ix].data.cpu().numpy()
        recon_im = (recon_im * _std + _mean) * 255
        recon_im = recon_im.astype('uint8')
        recon_im = recon_im.transpose((1,2,0))
        cv2.imwrite(os.path.join(im_dir, 'recon_'+str(i)+'.png'), recon_im)
    

# functions to show an image
def make_image(img, mean=(0,0,0), std=(1,1,1)):
    for i in range(0, 3):
        img[i] = img[i] * std[i] + mean[i]    # unnormalize
    npimg = img.numpy()
    return np.transpose(npimg, (1, 2, 0))

def gauss(x,a,b,c):
    return torch.exp(-torch.pow(torch.add(x,-b),2).div(2*c*c)).mul(a)

def colorize(x):
    ''' Converts a one-channel grayscale image to a color heatmap image '''
    if x.dim() == 2:
        torch.unsqueeze(x, 0, out=x)
    if x.dim() == 3:
        cl = torch.zeros([3, x.size(1), x.size(2)])
        cl[0] = gauss(x,.5,.6,.2) + gauss(x,1,.8,.3)
        cl[1] = gauss(x,1,.5,.3)
        cl[2] = gauss(x,1,.2,.3)
        cl[cl.gt(1)] = 1
    elif x.dim() == 4:
        cl = torch.zeros([x.size(0), 3, x.size(2), x.size(3)])
        cl[:,0,:,:] = gauss(x,.5,.6,.2) + gauss(x,1,.8,.3)
        cl[:,1,:,:] = gauss(x,1,.5,.3)
        cl[:,2,:,:] = gauss(x,1,.2,.3)
    return cl

def show_batch(images, Mean=(2, 2, 2), Std=(0.5,0.5,0.5)):
    images = make_image(torchvision.utils.make_grid(images), Mean, Std)
    plt.imshow(images)
    plt.show()
    
def show_heatmaps(images, Mean=(2, 2, 2), Std=(0.5,0.5,0.5)):
    upsampling = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    images = images.unsqueeze(1)
    images = torchvision.utils.make_grid(images, nrow=5, normalize=True, padding=3, pad_value=1.0)
    images = images.unsqueeze(0)
    images = upsampling(images)
    
    return images[0]  

        
def visualize_with_circles(image, pts, confidence=None, scale=None, mean=[0.5,0.5,0.5], std=[0.2,0.2,0.2]):
    _std = np.asarray(std).reshape((3,1,1)); _mean = np.asarray(mean).reshape((3,1,1))
    image = (image * _std + _mean) * 255
    image = image.transpose((1,2,0))
    image = image.astype('uint8')
    image = np.ascontiguousarray(image)
    
    # im = np.array(image)
    if scale is not None:
        new_width, new_height = int(scale * image.shape[1]), int(scale * image.shape[0])
        image = cv2.resize(image, (new_width, new_height))    
    
    values = sns.color_palette("husl", pts.shape[0])
    
    colors = []
    for i in range(len(values)):
        color = [int(values[i][0]*255), int(values[i][1]*255), int(values[i][2]*255)]
        colors.append(color)
        
    scale_x = (image.shape[1] / 2.0)
    scale_y = (image.shape[0] / 2.0)
    circle_size = int(round(image.shape[1] / 35))
    
    for i in range(0,pts.shape[0]):
        pt_y = int(pts[i,1]*scale_y)
        pt_x = int(pts[i,0]*scale_x)
        
        if (confidence is not None) and confidence[i] < 0.01:
            # background threshold for visualization is set to 0.01
            image = cv2.circle(image, (pt_x, pt_y), int(circle_size*0.5), tuple(colors[i]), -1)
        else:
            image = cv2.circle(image, (pt_x, pt_y), circle_size, tuple(colors[i]), -1)
        
    return image



def show_img_with_heatmap(image, heatmap, mean=[0.5,0.5,0.5], std=[0.2,0.2,0.2]):
    _std = np.asarray(std).reshape((3,1,1)); _mean = np.asarray(mean).reshape((3,1,1))
    image = (image * _std + _mean) * 255
    image = image.transpose((1,2,0))
    image = image.astype('uint8')
    
    _heatmap = (heatmap / heatmap.max()) * 255
    _heatmap = 255 - _heatmap
    
    _heatmap = cv2.resize(_heatmap, (image.shape[0], image.shape[1]))
    _heatmap = np.uint8(_heatmap)
    
    heatmap_img = cv2.applyColorMap(_heatmap, cv2.COLORMAP_JET)
    fin = cv2.addWeighted(heatmap_img, 0.6, image, 0.4, 0)
    
    # font
    font = cv2.FONT_HERSHEY_COMPLEX

    # fontScale
    fontScale = 0.5

    # Blue color in BGR
    color = (255, 255, 255)

    # Line thickness of 2 px
    thickness = 1
    
    score = "confidence: " + "%.2f"%heatmap.max()
        
    # get boundary of this text
    textsize = cv2.getTextSize(score, font, fontScale, thickness)[0]

    # get coords based on boundary
    textX = (fin.shape[1] - textsize[0]) / 2
    textX = int(textX)
    
    textY = 30
    
    # org
    org = (textX, textY)

    fin = cv2.putText(fin, score, org, font, 
            fontScale, color, thickness, cv2.LINE_AA)
    
    return fin
