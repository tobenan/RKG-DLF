from distutils.command.build import build
from turtle import forward
# from pip import main
import torch.nn as nn
from mmcv.cnn import ConvModule, constant_init, kaiming_init
from mmcv.runner import _load_checkpoint, load_checkpoint
from mmcv.utils import _BatchNorm
from torch.utils import checkpoint as cp

from ...utils import get_root_logger
from ..builder import BACKBONES,build_loss
import torch
import mmcv
class FuseBlock(nn.Module):
    def __init__(self,
                inplanes,
                planes,
                conv_cfg=dict(type='Conv'),
                norm_cfg=dict(type='BN', requires_grad=True),
                act_cfg=dict(type='ReLU', inplace=True),) -> None:
        super().__init__()
        self.conv = ConvModule(
            inplanes,
            planes,
            kernel_size=1,
            stride=1,
            padding=0,
            # dilation=0,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        
    def forward(self, x):
        # x_out = x_ceus - x_b
        x_out = self.conv(x)
        return x_out

@BACKBONES.register_module()
class FuseNet(nn.Module):
    arch_settings = {
        18: (64, 128, 256, 512),
        34: (64, 128, 256, 512)
    }
    def __init__(self, depth) -> None:
        super().__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        self.channels = self.arch_settings[depth]
        self.conv1 = FuseBlock(self.channels[0], self.channels[0])
        self.conv2 = FuseBlock(self.channels[0] + self.channels[1], self.channels[1])
        self.conv3 = FuseBlock(self.channels[1] + self.channels[2], self.channels[2])
        self.conv4 = FuseBlock(self.channels[2] + self.channels[3], self.channels[3])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
        # self.loss_1 = build_loss(dict(type='BCELossWithLogits', loss_weight=1/2**3))
        # self.loss_2 = build_loss(dict(type='BCELossWithLogits', loss_weight=1/2**2))
        # self.loss_3 = build_loss(dict(type='BCELossWithLogits', loss_weight=1/2**1))
        # self.loss_4 = build_loss(dict(type='BCELossWithLogits', loss_weight=1))
        # self.loss_1 = nn.MSELoss()
        # self.loss_2 = nn.MSELoss()
        # self.loss_3 = nn.MSELoss()
        # self.loss_4 = nn.MSELoss()
        
    def forward(self, x_ceus, x_b):
        outputs = []
        for i in range(len(x_ceus)):
            assert x_ceus[i].shape == x_b[i].shape
        # s1
        x_output = self.conv1(x_ceus[0] - x_b[0]) # 32, 64, 56, 56
        outputs.append(x_output)

        # s2
        x_in = self.pool(x_output) # 32, 64, 28, 28
        x_in = torch.cat([x_in, x_ceus[1] - x_b[1]], dim=1) # 32, 192, 28, 28
        x_output = self.conv2(x_in) # 32, 128, 28, 28
        outputs.append(x_output)
        # torch
        # s3 
        x_in = self.pool(x_output) # 32, 128, 14, 14
        x_in = torch.cat([x_in, x_ceus[2] - x_b[2]], dim=1) # 32, 384, 14, 14
        x_output = self.conv3(x_in) # 32, 256, 14, 14
        outputs.append(x_output)
        
        # s4 
        x_in = self.pool(x_output) # 32, 256, 7, 7
        x_in = torch.cat([x_in, x_ceus[3] - x_b[3]], dim=1) # 32, 768, 7, 7
        x_output = self.conv4(x_in) # 32, 512, 7, 7
        outputs.append(x_output)
        
        return tuple(outputs)
    
    
    def loss(self, x_fuse, num_segs, coarse_seg):
        # coarse_seg b, h, w 
        losses = dict()
        # x_fuse = x_fuse[3].unsqueeze(0)# different stage
        for i, feat in enumerate(x_fuse):
            x = feat.reshape((-1, num_segs) + feat.shape[1:]) # b, t, c, h, w
            b, t, c, h, w = x.shape
            mask = nn.functional.interpolate(coarse_seg.unsqueeze(1).to(torch.uint8), size=(h, w), mode='nearest').unsqueeze(1)
            mask = mask.expand_as(x)
            # x_mask = ~mask.bool() * x
            # x_mask = x.mean(dim=(3, 4)) # b, t, c
            # loss_guide = getat
            # tr(self, 'loss_' + str(i+1))(x_mask, torch.ones_like(x_mask) * .5)
            x_mask = x[~mask.bool()]
            # print(x_mask.shape)
            loss_guide = getattr(self, 'loss_' + str(i+1))(x_mask, torch.zeros_like(x_mask) * .5) * (1/2**(3-i))
            losses.update({'loss_' + str(i+1) : loss_guide})
        return losses
            
        
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)

    
    

