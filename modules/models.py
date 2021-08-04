import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models

class resnet_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet50(pretrained=True)
        # self.backbone.avgpool = nn.Identity()
        self.backbone.fc = nn.Identity()

        # self.conv_1 = nn.Sequential(nn.Dropout(0.4),
        #                             nn.Conv2d(2048,64,kernel_size=(1,1)),
        #                             nn.BatchNorm2d(64),
        #                             nn.ReLU(),
        #                             nn.Dropout(0.3))
        # num_features = 64*7*7
        num_features = 2048
        self.head = nn.Sequential(nn.Dropout(0.5),
                                nn.Linear(num_features,400),
                                nn.BatchNorm1d(400),
                                nn.LeakyReLU(),
                                nn.Dropout(0.4),
                                nn.Linear(400,102),
                                nn.LogSoftmax(dim=1))
    def forward(self,x):
        # x = self.backbone(x).reshape(-1,2048,7,7)
        # x = self.conv_1(x).reshape(-1,64*7*7)
        x = self.backbone(x)
        out = self.head(x)
        return out
    
    def freeze(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze(self, num_layers=-1):
        if num_layers==-1:
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            for chld in list(self.backbone.children())[::-1][:num_layers]:
                for p in chld.parameters():
                    p.requires_grad = True