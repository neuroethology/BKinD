import torch
import torch.nn as nn

from .resnet_updated import conv3x3
from .resnet_updated import resnetbank50all as resnetbank50
from .globalNet import globalNet

import math


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        
        if self.upsample is not None:
            x = self.upsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out
    

class Decoder(nn.Module):
    def __init__(self, in_planes=256, wh=14, n_kps=10, ratio=1.0):
        super(Decoder, self).__init__()
        
        self.K = n_kps
        
        w, h = wh, wh
        if ratio != 1.0:
            w = wh
            h = ratio * wh
            
        self.layer1 = BasicBlock(int(in_planes)+self.K, int(in_planes/2),
                                 upsample=nn.Upsample((int(h*2), int(w*2)), mode='bilinear')); in_planes /= 2
        self.layer2 = BasicBlock(int(in_planes)+self.K, int(in_planes/2), 
                                 upsample=nn.Upsample((int(h*4), int(w*4)), mode='bilinear')); in_planes /= 2
        self.layer3 = BasicBlock(int(in_planes)+self.K, int(in_planes/2), 
                                 upsample=nn.Upsample((int(h*8), int(w*8)), mode='bilinear')); in_planes /= 2
        self.layer4 = BasicBlock(int(in_planes)+self.K, int(in_planes/2),
                                 upsample=nn.Upsample((int(h*16), int(w*16)), mode='bilinear')); in_planes /= 2
        self.layer5 = BasicBlock(int(in_planes)+self.K, max(int(in_planes/2), 32),
                                 upsample=nn.Upsample((int(h*32), int(w*32)), mode='bilinear'))
        in_planes = max(int(in_planes/2), 32)
        
        self.conv_final = nn.Conv2d(int(in_planes), 3, kernel_size=1, stride=1)
        
    def forward(self, x, heatmap):
        
        x = torch.cat((x[0], heatmap[0]), dim=1)
        x = self.layer1(x)
        x = torch.cat((x, heatmap[1]), dim=1)
        x = self.layer2(x)
        x = torch.cat((x, heatmap[2]), dim=1)
        x = self.layer3(x)
        x = torch.cat((x, heatmap[3]), dim=1)
        x = self.layer4(x)
        
        x = torch.cat((x, heatmap[4]), dim=1)
        x = self.layer5(x)
        x = self.conv_final(x)
        
        return x
    
    
class Model(nn.Module):
    def __init__(self, n_kps=10, output_dim=200, pretrained=True, output_shape=(64, 64)):
        
        super(Model, self).__init__()
        self.K = n_kps
        
        channel_settings = [2048, 1024, 512, 256]
        self.output_shape = output_shape
        self.kptNet = globalNet(channel_settings, output_shape, n_kps)
        self.ch_softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # width == height for now
        self.decoder = Decoder(in_planes=2048, wh=int(output_shape[0]/8), n_kps=self.K*2, ratio=1.0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        self.encoder = resnetbank50(pretrained=pretrained)

    def get_keypoints(self, x):
        x_res = self.encoder(x)

        # Get keypoints of x
        kpt_feat, kpt_out = self.kptNet(x_res)  # keypoint for reconstruction
        
        # Reconstruction module
        heatmap = kpt_out[-1].view(-1, self.K, kpt_out[-1].size(2) * kpt_out[-1].size(3))
        heatmap = self.ch_softmax(heatmap)
        heatmap = heatmap.view(-1, self.K, kpt_out[-1].size(2), kpt_out[-1].size(3))
                
        u_x, u_y, covs = self._mapTokpt(heatmap)        

        return (u_x, u_y)        

    def get_all_keypoints(self, x_list):

        u_x_list = []
        u_y_list = []
        for item in x_list:
            u_x, u_y = self.get_keypoints(item)
            u_x_list.append(u_x)
            u_y_list.append(u_y)
            

        return (u_x_list, u_y_list)

    def forward(self, x, tr_x=None, gmtr_x1 = None, gmtr_x2 = None, gmtr_x3 = None):
        
        x_res = self.encoder(x)
        
        if tr_x is not None:
            tr_x_res = self.encoder(tr_x)
            tr_kpt_feat, tr_kpt_out = self.kptNet(tr_x_res)  # keypoint for reconstruction

            # Reconstruction module
            tr_heatmap = tr_kpt_out[-1].view(-1, self.K, tr_kpt_out[-1].size(2) * tr_kpt_out[-1].size(3))
            tr_heatmap = self.ch_softmax(tr_heatmap)
            tr_heatmap = tr_heatmap.view(-1, self.K, tr_kpt_out[-1].size(2), tr_kpt_out[-1].size(3))

            tr_confidence = tr_heatmap.max(dim=-1)[0].max(dim=-1)[0]

            tr_u_x, tr_u_y, tr_covs = self._mapTokpt(tr_heatmap)

        # Get keypoints of x
        kpt_feat, kpt_out = self.kptNet(x_res)  # keypoint for reconstruction
        
        # Reconstruction module
        heatmap = kpt_out[-1].view(-1, self.K, kpt_out[-1].size(2) * kpt_out[-1].size(3))
        heatmap = self.ch_softmax(heatmap)
        heatmap = heatmap.view(-1, self.K, kpt_out[-1].size(2), kpt_out[-1].size(3))
                
        u_x, u_y, covs = self._mapTokpt(heatmap)   
        
        if tr_x is None:
            return (u_x, u_y), kpt_out[-1]

        tr_kpt_conds = []
        
        prev_w, prev_h = int(self.output_shape[0]/16), int(self.output_shape[1]/16)
        std_in = [0.1, 0.1, 0.01, 0.01, 0.001]
        
        for i in range(0, 5):
            prev_h *= 2;  prev_w *= 2
            
            # _We can concatenate both keypoint representation
            hmaps = self._kptTomap(tr_u_x, tr_u_y, H=prev_h, W=prev_w, inv_std=std_in[i], normalize=False)

            hmaps_2 = self._kptTomap(u_x, u_y, H=prev_h, W=prev_w, inv_std=std_in[i], normalize=False)

            hmaps = torch.cat([hmaps, hmaps_2], dim = 1)

            tr_kpt_conds.append(hmaps)
            
        recon = self.decoder(x_res, tr_kpt_conds)
        
        if gmtr_x1 is not None:  # Rotation loss
            out_h, out_w = int(self.output_shape[0]*2), int(self.output_shape[1]*2)
            
            gmtr_x_res = self.encoder(gmtr_x1)
            gmtr_kpt_feat, gmtr_kpt_out = self.kptNet(gmtr_x_res)
            
            gmtr_heatmap = gmtr_kpt_out[-1].view(-1, self.K, gmtr_kpt_out[-1].size(2) * gmtr_kpt_out[-1].size(3))
            gmtr_heatmap = self.ch_softmax(gmtr_heatmap)
            gmtr_heatmap = gmtr_heatmap.view(-1, self.K, gmtr_kpt_out[-1].size(2), gmtr_kpt_out[-1].size(3))
            
            gmtr_u_x, gmtr_u_y, gmtr_covs = self._mapTokpt(gmtr_heatmap)

            gmtr_kpt_conds_1 = self._kptTomap(gmtr_u_x, gmtr_u_y, H=out_h, W=out_w, inv_std=0.001, normalize=False)

            #################################################
            gmtr_x_res = self.encoder(gmtr_x2)
            gmtr_kpt_feat, gmtr_kpt_out = self.kptNet(gmtr_x_res)
            
            gmtr_heatmap = gmtr_kpt_out[-1].view(-1, self.K, gmtr_kpt_out[-1].size(2) * gmtr_kpt_out[-1].size(3))
            gmtr_heatmap = self.ch_softmax(gmtr_heatmap)
            gmtr_heatmap = gmtr_heatmap.view(-1, self.K, gmtr_kpt_out[-1].size(2), gmtr_kpt_out[-1].size(3))
            
            gmtr_u_x_2, gmtr_u_y_2, gmtr_covs = self._mapTokpt(gmtr_heatmap)

            gmtr_kpt_conds_2 = self._kptTomap(gmtr_u_x_2, gmtr_u_y_2, H=out_h, W=out_w, inv_std=0.001, normalize=False)

            ###########################################
            gmtr_x_res = self.encoder(gmtr_x3)
            gmtr_kpt_feat, gmtr_kpt_out = self.kptNet(gmtr_x_res)
            
            gmtr_heatmap = gmtr_kpt_out[-1].view(-1, self.K, gmtr_kpt_out[-1].size(2) * gmtr_kpt_out[-1].size(3))
            gmtr_heatmap = self.ch_softmax(gmtr_heatmap)
            gmtr_heatmap = gmtr_heatmap.view(-1, self.K, gmtr_kpt_out[-1].size(2), gmtr_kpt_out[-1].size(3))
            
            gmtr_u_x_3, gmtr_u_y_3, gmtr_covs = self._mapTokpt(gmtr_heatmap)

            gmtr_kpt_conds_3 = self._kptTomap(gmtr_u_x_3, gmtr_u_y_3, H=out_h, W=out_w, inv_std=0.001, normalize=False)

            return (recon, (tr_u_x, tr_u_y), (tr_kpt_conds[-1], gmtr_kpt_conds_1, gmtr_kpt_conds_2, gmtr_kpt_conds_3),
             (tr_kpt_out[-1], gmtr_kpt_out[-1]), (u_x, u_y), (gmtr_u_x, gmtr_u_y, gmtr_u_x_2, gmtr_u_y_2, gmtr_u_x_3, gmtr_u_y_3),
             tr_confidence)
        
        
        return recon, (tr_u_x, tr_u_y), tr_kpt_conds[-1], tr_kpt_out[-1], (u_x, u_y), tr_confidence
    
        
    def _mapTokpt(self, heatmap):
        # heatmap: (N, K, H, W)    
            
        H = heatmap.size(2)
        W = heatmap.size(3)
        
        s_y = heatmap.sum(3)  # (N, K, H)
        s_x = heatmap.sum(2)  # (N, K, W)
        
        y = torch.linspace(-1.0, 1.0, H).cuda()
        x = torch.linspace(-1.0, 1.0, W).cuda()
        
        u_y = (y * s_y).sum(2) / s_y.sum(2)  # (N, K)
        u_x = (x * s_x).sum(2) / s_x.sum(2)
        
        y = torch.reshape(y, (1, 1, H, 1))
        x = torch.reshape(x, (1, 1, 1, W))
        
        # Covariance
        var_y = ((heatmap * y.pow(2)).sum(2).sum(2) - u_y.pow(2)).clamp(min=1e-6)
        var_x = ((heatmap * x.pow(2)).sum(2).sum(2) - u_x.pow(2)).clamp(min=1e-6)
        
        cov = ((heatmap * (x - u_x.view(-1, self.K, 1, 1)) * (y - u_y.view(-1, self.K, 1, 1))).sum(2).sum(2)) #.clamp(min=1e-6)
                
        return u_x, u_y, (var_x, var_y, cov)
    
    
    def _kptTomap(self, u_x, u_y, inv_std=1.0/0.1, H=16, W=16, normalize=False):        
        mu_x = u_x.unsqueeze(2).unsqueeze(3)  # (N, K, 1, 1)
        mu_y = u_y.unsqueeze(2) 
        
        y = torch.linspace(-1.0, 1.0, H).cuda()
        x = torch.linspace(-1.0, 1.0, W).cuda()
        y = torch.reshape(y, (1, H))
        x = torch.reshape(x, (1, 1, W))
        
        g_y = (mu_y - y).pow(2)
        g_x = (mu_x - x).pow(2)
        
        g_y = g_y.unsqueeze(3)
        g_yx = g_y + g_x
        
        g_yx = torch.exp(- g_yx / (2 * inv_std) ) * 1 / math.sqrt(2 * math.pi * inv_std)
        
        if normalize:
            g_yx = g_yx / g_yx.sum(2, keepdim=True).sum(3, keepdim=True)
        
        return g_yx
