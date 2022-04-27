import torch
import torch.nn as nn

class separation_loss(nn.Module):
    """ Separation constraint
    """
    
    def __init__(self, n_kpts=8, weight=0.01):
        super(separation_loss, self).__init__()
        self.weight = weight
        self.K = n_kpts
        
    def forward(self, input):

        input_1 = torch.stack([input[0], input[1]], dim = -1)
        target = torch.stack([input[0], input[1]], dim = -1)    
        
        dist_mat = torch.cdist(input_1, target)**2

        mask_matrix = torch.ones((self.K, self.K)).fill_diagonal_(0).unsqueeze(0).to(input_1.device)

        dist_mat = torch.exp(-1*dist_mat/(2*0.08**2))

        return (dist_mat*mask_matrix).mean()*0.05
    
