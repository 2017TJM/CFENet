import torch
import torch.nn as nn
import torch.nn.functional as F 

'''
build CFE and FFB modules respectively
'''
class ConvBlock(nn.Module):
    def __init__(self,input_c,output_c,
                 kernel,strides=1,groups=1):
        super(ConvBlock,self).__init__()
        self.input_c=input_c
        self.output_c=output_c
        self.kernel=kernel
        self.strides=strides
        self.groups=groups
        
        self.net=nn.Sequential(nn.Conv2d(self.input_c,self.output_c,
                                         self.kernel,stride=self.strides,
                                         groups=self.groups),
                               nn.ReLU(inplace=True),
                               nn.BatchNorm2d(self.output_c))
    
    def forward(self,x):
        return self.net(x)


class Cfe(nn.Module):
    def __init__(self,p,output_c,k,groups):
        super(Cfe,self).__init__()
        self.p=p
        self.output_c=output_c
        self.k=k
        self.groups=groups

        self.initial=ConvBlock(self.p,self.output,1)
        self.mid=nn.Sequential(ConvBlock(self.p,self.output_c,1),
                               ConvBlock(self.output_c,self.output_c,(1,self.k),groups=self.groups),
                               ConvBlock(self.output_c,self.output_c,(self.k,1),groups=self.groups),
                               ConvBlock(self.output_c,self.output_c,1))
        self.right=nn.Sequential(ConvBlock(self.p,self.output_c,1),
                                 ConvBlock(self.output_c,self.output_c,(self.k,1),groups=self.groups),
                                 ConvBlock(self.output_c,self.output_c,(1,self.k),groups=self.groups),
                                 ConvBlock(self.output_c,self.output_c,1))
        
    def forward(self,x):
        y=x
        x2=self.initial(x)
        x3=self.mid(x2)
        x4=self.right(x2)
        x=torch.cat([x3,x4],dim=1)
        return y+x


class Ffb(nn.Module):
    def __init__(self,input_c,output_c):
        super(Ffb,self).__init__()
        self.input_c=input_c
        self.output_c=output_c
        
        self.left=ConvBlock(self.input_c,self.output_c,1)
        self.right=ConvBlock(self.input_c,self.output_c,1)

    def forward(self,x,y):
        y=x

        x=self.left(x)

        y=self.right(y)
        y=F.upsample(y,size=(x.size(2),x.size(3)))
        return y+x