import torch
# import mmcls
from mmcv.cnn import (ConvModule, build_activation_layer, build_conv_layer,
                      build_norm_layer, constant_init)
import torch.nn as nn
from torch import nn
from mmcls.models.backbones.resnet import ResNet,BasicBlock,ResLayer
from einops import rearrange
from einops.layers.torch import Rearrange
# from .base_backbone import BaseBackbone
from ..builder import BACKBONES
from mmcv.cnn import ConvModule, constant_init, kaiming_init

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(patches, temperature = 10000, dtype = torch.float32):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    return pe.type(dtype)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


# BasicBlock
# model = BasicBlock(3,6)
# model = ResNet(18)

# model = ResLayer(BasicBlock,1,3,6,stride=2)
model = ResLayer(BasicBlock,1,3,6,stride=2)

# x = torch.randn((1,3,224, 224))
# print(model)
@BACKBONES.register_module()
class RES(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = build_conv_layer(
            # self.conv_cfg,
            None,
            3, #6 输入通道数，单模态3，双模态6
            32,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False)
        _, self.norm1 = build_norm_layer(
            dict(type='BN'), 32, postfix=1)
        # self.add_module(self.norm1, norm1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.reslayer1 = ResLayer(BasicBlock,1,32,64,stride=2)
        self.reslayer2 = ResLayer(BasicBlock,1,64,128,stride=2)
        # TransFormer
        self.trans_dim = 256
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c h w -> b h w c'),
            nn.LayerNorm(128),
            nn.Linear(128, self.trans_dim),
            nn.LayerNorm(self.trans_dim),
        )
        self.transformer = Transformer(self.trans_dim, 1, heads=4, dim_head=128,mlp_dim=512)
        self.ln = nn.LayerNorm(self.trans_dim)
    def forward(self, x):
        x = self.conv1(x)  #[320,3,224,224]
        x = self.norm1(x)  #[320,32,112,112]
        x = self.relu(x)   #[320,32,112,112]
        x = self.maxpool(x)  #[320,32,112,112]
        x = self.reslayer1(x) #[320,32,56,56]
        x = self.reslayer2(x) #[320,64,28,28]
        #[320,128,14,14]
        # trans_x = self.to_patch_embedding(x)
        # pe = posemb_sincos_2d(trans_x) #[320,14,14,256]
        # trans_x = rearrange(trans_x, 'b ... d -> b (...) d') + pe  #pe[196,256]
        # trans_x = self.transformer(trans_x) #[320, 196, 256]
        # trans_x = self.ln(trans_x) #[320, 196, 256]
        # trans_x = rearrange(trans_x, 'b h w  -> b w h 1') #[320, 196, 256]
        # return trans_x #[320,196,256,1]
        return x
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)