import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn import init


from utils import param_count #weight_scaling_init
import argparse
import json

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
            nn.LogSoftmax(dim=1)
        )


    def forward(self, x):      
        for layer in self.conv_layers:
            x = layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='.config/theronet.json')
    args = parser.parse_args()

    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    network_config = config['network']

    model = TheroNet(**network_config)
    
    #test input
    x = torch.ones([8, 
                    1, 
                    128, 
                    128])
    
    #forward propagations
    y = model(x)
    
    #logs
    print(model)
    print(f"Number of TheröNet Parameters: {param_count(model)}")
    assert y.shape == (8, 6)
    print(f"Network Intput Shape: {x.shape}")
    print(f"Network Output Shape: {y.shape}")
