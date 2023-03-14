import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from torch.utils.data import DataLoader
from utils import param_count #weight_scaling_init
import argparse

class TheroNet(nn.Module):
    
    def __init__(self,
                 NF,
                 NL,
                 KS,
                 stride, 
                 NC=6):
        super(TheroNet, self).__init__()

        self.conv_layers = nn.ModuleList()
        
        for i in range(NL):
            if i == 0:
                CH = 1
            else:
                CH = NF
                NF *= 2 
            
            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(CH, NF, kernel_size=KS, stride=stride, padding=1),
                nn.BatchNorm2d(NF),
                nn.ReLU(inplace = True),
                nn.MaxPool2d(kernel_size=2, stride=2)))
 
        self.fc_layers = nn.Sequential(
            nn.Linear(NF * 4 * 4, NF//2),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(NF//2, NC),
        )


    def forward(self, x):      
        for layer in self.conv_layers:
            x = layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


if __name__ == '__main__':
    net = TheroNet(NF=16,
                   NL = 5,
                   KS=3,
                   stride=1)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
    net.to(device)
    
    pixels = 128    #image shape of (128, 128)
    n_chans = 1     #gray scale image
    n_classes = 6
    batch_size = 8
    
    #test input
    x = torch.ones([batch_size, 
                    n_chans, 
                    pixels, 
                    pixels])
    
    #forward propagations
    y = net(x)
    
    print(net)
    print(f"Number of TheröNet Parameters: {param_count(net)}")
    assert y.shape == (batch_size, n_classes)
    print(f"Network Output Shape: {y.shape}")
