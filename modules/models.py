import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import cv2

class resnet_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet50(pretrained=True)
        self.backbone.avgpool = nn.Identity()
        self.backbone.fc = nn.Identity()

        num_features = 2048
        self.pool = nn.AvgPool2d(7)
        self.head = nn.Sequential(nn.Dropout(0.5),
                                nn.Linear(num_features,400),
                                nn.LeakyReLU(),
                                nn.Dropout(0.4),
                                nn.Linear(400,102),
                                nn.LogSoftmax(dim=1))
        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self,x, hook=False):
        x = self.backbone(x).reshape(-1,2048,7,7)
        if hook:
            x.register_hook(self.activations_hook)

        x = self.pool(x).reshape(x.shape[0],2048)
        out = self.head(x)
        return out
    
    def get_heatmap(self, x, class_id):
        pred = self.forward(x, hook=True)
        pred[:,class_id.type(torch.LongTensor)].backward()

        grad = torch.mean(self.gradients, dim=[2,3]).reshape(x.shape[0],2048,1,1)
        activations = self.backbone(x).reshape(-1,2048,7,7)

        heatmaps = torch.mean(grad*activations,dim=1).reshape(x.shape[0],7,7)
        heatmaps = F.relu(heatmaps)

        heatmaps = heatmaps.detach().cpu().numpy()

        all_heatmaps = []
        for heatmap in heatmaps:
          heatmap /= np.max(heatmap)
          heatmap = cv2.resize(heatmap, (x.shape[3], x.shape[2]))
          heatmap = np.uint8(255 * heatmap)
          all_heatmaps.append(heatmap)
        
        return np.array(all_heatmaps)

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