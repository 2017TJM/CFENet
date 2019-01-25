import torch
import numpy as np 
import torch.nn as nn 
import os
from torchvision import models 

class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16,self).__init__()
        b_name=1
        l_name=1
        vgg=models.vgg16(pretarined=True)
        for param in vgg.parameters():
            param.requires_grad_(False)
        self.vgg16=list(vgg.children())[0]
        self.backend=nn.ModuleDict()
        self.outputs={}
        for module in vgg:
            if isinstance(module,nn.Conv2d):
                self.backend.add_module('Conv_{}_{}'.format(b_name,l_name),
                                        module)
                l_name+=1 
            elif isinstance(module,nn.ReLU):
                self.backend.add_module('Relu_{}_{}'.format(b_name,l_name),
                                        module)
                l_name+=1 
            elif isinstance(module,nn.MaxPool2d):
                self.backend.add_module('Maxpool_{}_{}'.format(b_name,l_name),
                                        module)
                b_name+=1
                l_name=1
                
    def forward(self,x):
        for key,module in self.backend:
            self.outputs[key]=module(x)
        return self.outputs