from abc import ABCMeta, abstractmethod


import torch
import torch.nn as nn
import math
# from ...core import top_k_accuracy
# from ..builder import build_loss
from mmcv.cnn import NonLocal1d
from mmcv.cnn import ConvModule
from mmaction.models.backbones.resnet import ResNet

class TemporalConsensus(nn.Module):
    def __init__(self, n_seg, scale_rate,dim=1) -> None:
        super().__init__()
        
        # self.avg_pool = nn.AdaptiveAvgPool1d((1))
        self.n_seg = n_seg
        self.scale_rate = scale_rate
        self.fc1 = nn.Linear(n_seg, int(n_seg * scale_rate), bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(int(n_seg * scale_rate), n_seg, bias=False)
        self.softmax = nn.Softmax(1)
    
    def forward(self, x):
        b, n_seg, _, _, _ = x.shape
        y = torch.mean(x, dim=(2, 3, 4)).detach()
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.softmax(y).view(b, n_seg, 1, 1, 1)
        x = x * y.expand_as(x)
        # x +=y
        # x = 
        return x.sum(1, keepdim=True)


def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


class TemporalSimGc(nn.Module):
    def __init__(self, in_channels, n_seg, dim=1) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.n_seg = n_seg
        self.phi = nn.Conv1d(self.in_channels, n_seg, kernel_size=1, bias=False)
        # self.theta = nn.Conv1d(self.in_channels, self.reduction,kernel_size=1)
        self.softmax = nn.Softmax(1)
        # self.nlblock = NonLocal1d(in_channels,use_scale = True, reduction=reduction,std=0.001)
        self.pos_embed = positionalencoding1d(in_channels, n_seg)
        self.bn2 = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(self.in_channels, self.in_channels, kernel_size=1, bias=False)
        
    def forward(self, x):
        b, n_seg, c, _, _ = x.shape
        x_trans = (x.squeeze() + self.pos_embed.unsqueeze(0).to(x.device)).permute(0, 2, 1)
        x_phi = self.phi(x_trans)
        y = torch.bmm(x_trans, self.softmax(x_phi))
        y = self.bn2(y)
        y = self.relu(y)
        y = self.conv2(y)
        
        y = y.permute(0, 2, 1).contiguous().reshape(*x.size())
        out = y + x 
        return out.mean(dim=1, keepdim = True)
        
class MaxConsensus(nn.Module):
    """Average consensus module.

    Args:
        dim (int): Decide which dim consensus function to apply.
            Default: 1.
    """

    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """Defines the computation performed at every call."""
        return x.max(dim=self.dim, keepdim=True)[0]

class LinearConsensus(nn.Module):
    """Average consensus module.

    Args:
        dim (int): Decide which dim consensus function to apply.
            Default: 1.
    """

    def __init__(self, n_seg, dim=1):
        super().__init__()
        self.dim = dim
        self.n_seg = n_seg
        self.linear = nn.Linear(self.n_seg,1 ,bias=False)
    def forward(self, x):
        """Defines the computation performed at every call."""
        if x.size()[1]==1:
            y=x.squeeze().unsqueeze(-1)
            print("only TPN")
        else:
            y = x.squeeze().permute(0, 2, 1)
        y = self.linear(y)
        y = y.permute(0, 2, 1)[...,None,None]
        return y



    
class AttentionConsensus(nn.Module):

    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
        # self.n_seg = n_seg
        # self.linear = nn.Linear(self.n_seg,1 ,bias=False)
        self.omega_att = nn.Parameter(torch.ones((1)))
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        """Defines the computation performed at every call."""
        # y = x.squeeze().permute(0, 2, 1)
        e = self.omega_att * x # 10, 32, 512, 1, 1
        A = self.softmax(e)
        # y = self.linear(y)
        y = x * A
        return y.sum(dim=self.dim, keepdim=True) # 10, 1, 512, 1, 1
        # y = y.permute(0, 2, 1)
        # return y

        


