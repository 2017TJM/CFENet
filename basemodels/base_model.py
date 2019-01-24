import torch
import numpy as np 
import torch.nn as nn 
import os
from torchvision import models 

class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16,self).__init__()
        vgg=models.vgg16(pretarined=True)
        for param in vgg.parameters():
            param.requires_grad_(False)
        self.vgg16=list(vgg.children())[0]
        self.feture_layers=[]
    
    def get_features(self):
        pass
    
    def forward(self,x):
        pass



class ResNet(nn.Module):

    def __init__(self):
        super(ResNet,self).__init__()
        pass

    def forward(self):
        pass








