from turtle import forward
import torch.nn as nn
from mmcv.cnn import ConvModule, constant_init, kaiming_init
from mmcv.runner import _load_checkpoint, load_checkpoint
from mmcv.utils import _BatchNorm
from torch.utils import checkpoint as cp
import torch
from ..builder import BACKBONES
import numpy as np
from mmaction.models.backbones import resnet
from ...utils import get_root_logger
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
import math

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

class ConvBasicBlock(nn.Module):
    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                #  downsample=None,
                #  style='pytorch',
                 conv_cfg=dict(type='Conv'),
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True),):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = ConvModule(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv2 = ConvModule(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.pool(out)
        return out
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
        

class ChannelTimeAttention(nn.Module):
    def __init__(self,d_t, d_scaled, dropout=.1):
        super().__init__()
        self.scaled_pool = nn.AdaptiveAvgPool3d((None, d_scaled, d_scaled))
        self.fc_q = nn.Linear(d_scaled*d_scaled, d_t * 2)
        self.fc_k = nn.Linear(d_scaled*d_scaled, d_t * 2)
        self.fc_v = nn.Identity()
        self.dropout=nn.Dropout(dropout)
    
    def forward(self, x:torch.Tensor):
        """_summary_

        Args:
            x (_type_): x_size: [b , t, c, h, w]
        """
        b, t, c, h, w = x.shape
        shift_x = x.permute(0, 2, 1, 3, 4)
        scaled_x = self.scaled_pool(shift_x) # [b, c, t, scale, sclae]
        q = self.fc_q(scaled_x.view(b*c, t, -1)) # [b x c, t, t*2]
        k = self.fc_k(scaled_x.view(b*c, t, -1)).permute(0, 2, 1)
        
        v = self.fc_v(shift_x.reshape(b*c, t, h*w))
        
        att = torch.matmul(q, k) / np.sqrt(t)
        att = torch.softmax(att, -1)
        att=self.dropout(att)
        
        out = torch.matmul(att, v).contiguous().view(b, c, t, h, w).permute(0, 2, 1, 3, 4)
        
        return out
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.normal_(m.bias, 0)
        

@BACKBONES.register_module()
class CTANet(nn.Module):
    def __init__(self,
                 num_segs,
                 scaled,
                 ):
        super().__init__()
        self.conv1 = ConvBasicBlock(3, 8)
        # self.att1 = ChannelTimeAttention(num_segs, scaled)
        
        self.conv2 = ConvBasicBlock(8, 16)
        # self.att2 = ChannelTimeAttention(num_segs, scaled)
        
        self.conv3 = ConvBasicBlock(16, 32)
        # self.att3 = ChannelTimeAttention(num_segs, scaled)

        self.spp = [nn.AdaptiveAvgPool2d((1,1)), 
                    nn.AdaptiveAvgPool2d((2,2)),
                    nn.AdaptiveAvgPool2d((4,4))]
    
    def forward(self, x: torch.Tensor):
        """_summary_

        Args:
            x (torch.Tensor): [b , t, c, h, w]
        """
        batches = x.shape[0]
        x = x.reshape((-1, ) + x.shape[2:])
        num_segs = x.shape[0] // batches
        out = self.conv1(x)
        # out = self.att1(out.reshape((batches, num_segs,)+ out.shape[1:])).reshape((-1, ) + out.shape[1:])
        
        out = self.conv2(out)
        # out = self.att2(out.reshape((batches, num_segs,)+ out.shape[1:])).reshape((-1, ) + out.shape[1:])

        out = self.conv3(out)
        # out = self.att3(out.view(batches, num_segs,)+ out.shape[1:])
        out = [spp(out).reshape((batches * num_segs , ) + (-1, 1, 1)) for spp in self.spp]
        out = torch.cat(out, dim=1)
        
        return out
    
    def init_weights(self):
        self.conv1.init_weights()
        self.conv2.init_weights()
        self.conv3.init_weights()
        # self.att1.init_weights()
        # self.att2.init_weights()


@BACKBONES.register_module()
class ResNet_cta(nn.Module):
    """ResNet backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        pretrained (str | None): Name of pretrained model. Default: None.
        in_channels (int): Channel num of input features. Default: 3.
        num_stages (int): Resnet stages. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        out_indices (Sequence[int]): Indices of output feature. Default: (3, ).
        dilations (Sequence[int]): Dilation of each stage.
        style (str): ``pytorch`` or ``caffe``. If set to "pytorch", the
            stride-two layer is the 3x3 conv layer, otherwise the stride-two
            layer is the first 1x1 conv layer. Default: ``pytorch``.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters. Default: -1.
        conv_cfg (dict): Config for norm layers. Default: dict(type='Conv').
        norm_cfg (dict):
            Config for norm layers. required keys are `type` and
            `requires_grad`. Default: dict(type='BN2d', requires_grad=True).
        act_cfg (dict): Config for activate layers.
            Default: dict(type='ReLU', inplace=True).
        norm_eval (bool): Whether to set BN layers to eval mode, namely, freeze
            running stats (mean and var). Default: False.
        partial_bn (bool): Whether to use partial bn. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                #  depth,
                 pretrained=None,
                 torchvision_pretrain=True,
                 in_channels=3,
                 num_stages=4,
                 out_indices=(3, ),
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 style='pytorch',
                 frozen_stages=-1,
                 conv_cfg=dict(type='Conv'),
                 norm_cfg=dict(type='BN2d', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_eval=False,
                 partial_bn=False,
                 with_cp=False,
                 trans_dim = 7,
                 dim_head = 32,
                 dropout = 0.2,
                 num_segs = 16
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.pretrained = pretrained
        self.torchvision_pretrain = torchvision_pretrain
        self.num_stages = num_stages
        assert 1 <= num_stages <= 4
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.style = style
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.partial_bn = partial_bn
        self.with_cp = with_cp

        # self.block, stage_blocks = self.arch_settings[depth]
        self.block = resnet.BasicBlock
        stage_blocks = (1, 1, 1, 1)
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = 16

        self._make_stem_layer()

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            planes = 16 * 2**i
            res_layer = resnet.make_res_layer(
                self.block,
                self.inplanes,
                planes,
                num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                with_cp=with_cp)
            self.inplanes = planes * self.block.expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self.feat_dim = self.block.expansion * 16 * 2**(
            len(self.stage_blocks) - 1)
        
        self.trans_layers = nn.ModuleList([])
        # self.pos_embedding = nn.Parameter(torch.randn(1, num_segs, trans_dim * trans_dim),)
        self.pos_embedding = nn.Parameter(positionalencoding1d(trans_dim * trans_dim, num_segs,).unsqueeze(0),requires_grad=False)
        for _ in self.stage_blocks:
            self.trans_layers.append(
                CTAModule(trans_dim, dim_head = dim_head, dropout = dropout)
                )
        
        
        
        self.spp = [nn.AdaptiveAvgPool2d((1,1)), 
                    nn.AdaptiveAvgPool2d((2,2)),]
                    # nn.AdaptiveAvgPool2d((4,4))]

    def _make_stem_layer(self):
        """Construct the stem layers consists of a conv+norm+act module and a
        pooling layer."""
        self.conv1 = ConvModule(
            self.in_channels,
            self.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    @staticmethod
    def _load_conv_params(conv, state_dict_tv, module_name_tv,
                          loaded_param_names):
        """Load the conv parameters of resnet from torchvision.

        Args:
            conv (nn.Module): The destination conv module.
            state_dict_tv (OrderedDict): The state dict of pretrained
                torchvision model.
            module_name_tv (str): The name of corresponding conv module in the
                torchvision model.
            loaded_param_names (list[str]): List of parameters that have been
                loaded.
        """

        weight_tv_name = module_name_tv + '.weight'
        if conv.weight.data.shape == state_dict_tv[weight_tv_name].shape:
            conv.weight.data.copy_(state_dict_tv[weight_tv_name])
            loaded_param_names.append(weight_tv_name)

        if getattr(conv, 'bias') is not None:
            bias_tv_name = module_name_tv + '.bias'
            if conv.bias.data.shape == state_dict_tv[bias_tv_name].shape:
                conv.bias.data.copy_(state_dict_tv[bias_tv_name])
                loaded_param_names.append(bias_tv_name)

    @staticmethod
    def _load_bn_params(bn, state_dict_tv, module_name_tv, loaded_param_names):
        """Load the bn parameters of resnet from torchvision.

        Args:
            bn (nn.Module): The destination bn module.
            state_dict_tv (OrderedDict): The state dict of pretrained
                torchvision model.
            module_name_tv (str): The name of corresponding bn module in the
                torchvision model.
            loaded_param_names (list[str]): List of parameters that have been
                loaded.
        """

        for param_name, param in bn.named_parameters():
            param_tv_name = f'{module_name_tv}.{param_name}'
            param_tv = state_dict_tv[param_tv_name]
            if param.data.shape == param_tv.shape:
                param.data.copy_(param_tv)
                loaded_param_names.append(param_tv_name)

        for param_name, param in bn.named_buffers():
            param_tv_name = f'{module_name_tv}.{param_name}'
            # some buffers like num_batches_tracked may not exist
            if param_tv_name in state_dict_tv:
                param_tv = state_dict_tv[param_tv_name]
                if param.data.shape == param_tv.shape:
                    param.data.copy_(param_tv)
                    loaded_param_names.append(param_tv_name)


    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            if self.torchvision_pretrain:
                # torchvision's
                self._load_torchvision_checkpoint(logger)
            else:
                # ours
                load_checkpoint(
                    self, self.pretrained, strict=False, logger=logger)
        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.001)
                    if m.bias is not None:
                        nn.init.normal_(m.bias, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The feature of the input samples extracted
            by the backbone.
        """
        b, t, c, h, w = x.shape
        x = x.reshape((-1,) + x.shape[2:])
        x = self.conv1(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            trans_layer = self.trans_layers[i]
            x = rearrange(x, '(b t) c h w -> b t c h w', t=t)
            x = trans_layer(x, self.pos_embedding)
            # x = rearrange(x, '(b t) c (h w) -> b t c h w', t=t, )
            x = rearrange(x, 'b t c h w ->(b t) c h w')
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            out = outs[0]
            out = [spp(out).reshape((b * t , ) + (-1, 1, 1)) for spp in self.spp]
            out = torch.cat(out, dim=1)
            return out

        return tuple(outs)


    def train(self, mode=True):
        """Set the optimization status when training."""
        super().train(mode)
        # self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()



class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class CTAModule(nn.Module):
    def __init__(self, dim, dim_head = 64, dropout = 0.2):
        super().__init__()
        inner_dim = dim_head
        # project_out = not (heads == 1 and dim_head == dim)
        self.scaled_pool = nn.AdaptiveAvgPool3d((None, dim, dim))
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qk = nn.Linear(dim * dim, inner_dim * 2, bias = False)

        self.to_out = nn.Identity()
        self.norm = nn.LayerNorm(dim * dim)
    def forward(self, x, pos_embedding):
        # x size b t c h w
        # return size: (b c) t (h w)  
        
        b, t, c, h, w = x.shape
        scaled_x = self.scaled_pool(x)
        scaled_x = rearrange(scaled_x, 'b t c h w -> (b c) t (h w)')
        scaled_x += pos_embedding
        scaled_x = self.norm(scaled_x)
        q, k = self.to_qk(scaled_x).chunk(2, dim = -1) # (b c) t inner_dim
        # q, k = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qk)
        
        v = rearrange(x, 'b t c h w ->  (b c) t (h w)')
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        # out = rearrange(out, 'b h n d -> b n (h d)')
        out = rearrange(out, '(b c) t (h w) -> b t c h w', c=c,h=h)
        return self.to_out(out + x) 

class Transformer(nn.Module):
    def __init__(self, dim, depth, dim_head, dropout = 0.2):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                PreNorm(dim * dim , CTAModule(dim, dim_head = dim_head, dropout = dropout))
                )
