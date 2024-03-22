import torch
import torch.nn as nn

class SqueezeExciteBlock(nn.Module):
    def __init__(self, num_channels, HW):
        super(SqueezeExciteBlock, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # As per the paper, it's just a MLP w/ one layer
        # input_dim == output_dim
        self.fc = nn.Sequential(
            nn.Linear(num_channels, num_channels // HW, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels // HW, num_channels, bias=True),
            nn.Sigmoid()
        )
        
    def forward(self, U):
        """
        Takes an input U where U is the feature map
        """
        B, C, H, W = U.size()
        
        q = self.avg_pool(U).view(1,1)
        q = self.fc(y).view(B, C, 1, 1)     
        return U * q.expand_as(U)
    