import torch
import torch.nn as nn
import torch.nn.functional as F

import math


def rot_img(x, rot_mat):
    grid = F.affine_grid(rot_mat, x.size())
    x = F.grid_sample(x, grid)
    return x

class rotation_loss(nn.Module):
    def __init__(self):
        super(rotation_loss, self).__init__()
        self.loss = nn.MSELoss()
        
    def forward(self, heatmap_1, heatmap_2, deg):        
        # Use rotation matrix
        # degree to matrix
        # move x,y coordinate
        # import pdb; pdb.set_trace()
        angle_rad =  math.pi * deg / 180.0
        c, s = torch.cos(angle_rad), torch.sin(angle_rad)
        rot_vec = torch.stack([c, s], dim=1)  # rot_matrix = [c, -s], [s, c]
        rot_vec2 = torch.stack([-s, c], dim=1)
        rot_matrix = torch.stack((rot_vec, rot_vec2, torch.zeros(rot_vec2.size()).to(rot_vec2.device)), dim=2)
        rot_matrix = rot_matrix.type(torch.float32)
        
        # crr_xy = torch.stack((pt[0], pt[1]), dim=1)
        # crr_xy = torch.matmul(rot_matrix, crr_xy)
        
        # tr_xy = torch.stack((pt_tr[0], pt_tr[1]), dim=1)

        # 10x10x64x64

        # heatmap_1 = heatmap_1.view(-1, 64, 64)
        # heatmap_2 = heatmap_2.view(-1, 64, 64)
        rotated_heatmap = rot_img(heatmap_1, rot_matrix).clone().detach()

        #print(heatmap_1.size(), rot_matrix.size(), rotated_heatmap.size())
        loss = self.loss(rotated_heatmap, heatmap_2)
        
        return loss, rotated_heatmap # for visualization
        
