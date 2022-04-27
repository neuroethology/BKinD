import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models

from .resnet_updated import conv3x3, conv1x1
from .resnet_updated import resnetbank50all as resnetbank50
from .globalNet import globalNet

import numpy as np
import math
import itertools

    
class Model(nn.Module):
    def __init__(self, n_kpts=10, pretrained=True, output_shape=(64, 64)):
        super(Model, self).__init__()
        self.K = n_kpts
        
        channel_settings = [2048, 1024, 512, 256]
        output_shape = output_shape
        self.kptNet = globalNet(channel_settings, output_shape, n_kpts)
        self.ch_softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        self.encoder = resnetbank50(pretrained=pretrained)
        
        
    def _lateral(self, input_size, output_shape):
        out_dim = 256
        
        layers = []
        layers.append(nn.Conv2d(input_size, out_dim,
            kernel_size=1, stride=1, bias=False))
        layers.append(nn.BatchNorm2d(out_dim))
        layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.Conv2d(out_dim, out_dim,
            kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.Upsample(size=output_shape, mode='bilinear', align_corners=True))
        layers.append(nn.BatchNorm2d(out_dim))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)
        
    def forward(self, x):
                
        x_res = self.encoder(x)
        
        kpt_feat, kpt_out = self.kptNet(x_res)
        
        # Classification module
        heatmap = kpt_out[-1].view(-1, self.K, kpt_out[-1].size(2) * kpt_out[-1].size(3))
        heatmap = self.ch_softmax(heatmap) 
        heatmap = heatmap.view(-1, self.K, kpt_out[-1].size(2), kpt_out[-1].size(3))
        
        confidence = heatmap.max(dim=-1)[0].max(dim=-1)[0]
            
        u_x, u_y, covs = self._mapTokpt(heatmap)
        
        kpts = []
        for ii in range(1, len(kpt_out)):
            _heatmap = kpt_out[ii-1].view(-1, self.K, kpt_out[ii-1].size(2) * kpt_out[ii-1].size(3))
            _heatmap = self.ch_softmax(_heatmap) 
            _heatmap = _heatmap.view(-1, self.K, kpt_out[ii-1].size(2), kpt_out[ii-1].size(3))

            _u_x, _u_y, _covs = self._mapTokpt(_heatmap)
            kpts.append((u_x, u_y))
        
        return (u_x, u_y), kpt_out[-1], heatmap, kpts, kpt_out, confidence, covs
        
    
    def _mapTokpt(self, heatmap):
        # heatmap: (N, K, H, W)    
            
        H = heatmap.size(2)
        W = heatmap.size(3)
        
        s_y = heatmap.sum(3)  # (N, K, H)
        s_x = heatmap.sum(2)  # (N, K, W)
        
        y = torch.linspace(-1.0, 1.0, H).cuda()
        x = torch.linspace(-1.0, 1.0, W).cuda()
        
        # u_y = (self.H_tensor * s_y).sum(2) / s_y.sum(2)  # (N, K)
        # u_x = (self.W_tensor * s_x).sum(2) / s_x.sum(2)
        u_y = (y * s_y).sum(2) / s_y.sum(2)  # (N, K)
        u_x = (x * s_x).sum(2) / s_x.sum(2)
        
        y = torch.reshape(y, (1, 1, H, 1))
        x = torch.reshape(x, (1, 1, 1, W))
        
        # Covariance
        var_y = ((heatmap * y.pow(2)).sum(2).sum(2) - u_y.pow(2)).clamp(min=1e-6)
        var_x = ((heatmap * x.pow(2)).sum(2).sum(2) - u_x.pow(2)).clamp(min=1e-6)
        
        cov = ((heatmap * (x - u_x.view(-1, self.K, 1, 1)) * (y - u_y.view(-1, self.K, 1, 1))).sum(2).sum(2)) #.clamp(min=1e-6)
                
        return u_x, u_y, (var_x, var_y, cov)
    