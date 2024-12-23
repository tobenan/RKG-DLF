# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import normal_init, constant_init, kaiming_init
import torch
# from ..builder import HEADS
from mmaction.models.builder import HEADS
# from .base import AvgConsensus, BaseHead
from mmaction.models.heads.base import AvgConsensus, BaseHead
# from .temporal_consensus import TemporalConsensus, TemporalSimGc, MaxConsensus,LinearConsensus,AttentionConsensus,CTAttention
from mmaction.models.heads.temporal_consensus import TemporalConsensus, TemporalSimGc, MaxConsensus,LinearConsensus,AttentionConsensus,CTAttention
from einops import rearrange
from torch.nn.utils import weight_norm

@HEADS.register_module()
class TSN_FuseHead_conv_mask_fusion_qual_relate11(BaseHead):
    """Class head for TSN.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss').
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        consensus (dict): Consensus config dict.
        dropout_ratio (float): Probability of dropout layer. Default: 0.4.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 fuse_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 spatial_type='avg',
                 consensus=dict(type='AvgConsensus', dim=1),
                 dropout_ratio=0.4,
                 init_std=0.01,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls=loss_cls, **kwargs)

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        self.fuse_channels = fuse_channels
        
        consensus_ = consensus.copy()

        consensus_type = consensus_.pop('type')
        if consensus_type == 'AvgConsensus':
            self.consensus = AvgConsensus(**consensus_)
        elif consensus_type == "TemporalConsensus":
            self.consensus = TemporalConsensus(**consensus_)
        elif consensus_type == "TemporalSimGc":
            self.consensus = TemporalSimGc(**consensus_)
        elif consensus_type == "MaxConsensus":
            self.consensus = MaxConsensus(**consensus_)
        elif consensus_type == "LinearConsensus":
            self.consensus = LinearConsensus(**consensus_)
        elif consensus_type == "AttentionConsensus":
            self.consensus = AttentionConsensus(**consensus_) 
        elif consensus_type == "CTAttention":
            self.consensus = CTAttention(**consensus_)    
        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool2d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.avg_pool = None

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
            
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(self.in_channels, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
        self.tabmlp = nn.Sequential(
            nn.Linear(self.fuse_channels, int(self.fuse_channels//2)),# int(self.fuse_channels//2)
            nn.ReLU(),
            nn.Linear(int(self.fuse_channels//2) , int(self.fuse_channels * 2)),
            # nn.Dropout(p =self.dropout_ratio),
        )
        
        # self.affine_tem = Affine(32)
        # self.affine_fuse = Affine2d(256)
        # self.MLP = nn.Sequential(
        #     nn.Linear(288, 128),
        #     nn.ReLU(),
        #     nn.Dropout(p=self.dropout_ratio),
        # )
        # self.bn = nn.BatchNorm1d(256)
        # self.sigmoid = nn.Sigmoid()
        ############ spa
        kernel_size = 3 # 7
        self.spaconv = nn.Sequential(
            nn.Conv2d(256+int(self.fuse_channels * 2), 256, kernel_size=kernel_size, stride=1, 
            padding=(kernel_size - 1) // 2, dilation=1, groups=1, bias=False),
            # nn.BatchNorm2d(1, eps=1e-5, momentum=0.01, affine=True),
            nn.ReLU(),
        )
        
        ############## tem
        self.tcn = TemporalConvNet(256+int(self.fuse_channels * 2), [256], kernel_size=7, dropout=self.dropout_ratio)
        
        ############## 
        # self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()
        # self.fc_cls = nn.Sequential(
        #     nn.Linear(256+int(self.fuse_channels * 2), self.num_classes),
        # )
        self.mlp_cha = nn.Sequential(
            nn.Linear(256+self.fuse_channels*2, 128),
            nn.ReLU(),
            # nn.Dropout(p=self.dropout_ratio),
            nn.Linear(128, 256),
            nn.Dropout(p=self.dropout_ratio),
        )
        
        # self.fc_cls = nn.Sequential(
        #     nn.Linear(512, self.num_classes),
        # )        
        self.fc_cls = nn.Sequential(
            nn.Linear(512+128*(self.num_classes), 3),
        )
        # learnable label embedding
        self.label_emb = nn.Parameter(torch.rand(num_classes, 128))
        # self.query_embed = nn.Embedding(num_queries, hidden_size)
        self.linear = nn.Linear(512,128)
        decoder_layer = nn.TransformerDecoderLayer(d_model=128, nhead=4, batch_first=True)
        decoder_norm = nn.LayerNorm(128, eps= 1e-5, )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=1, norm=decoder_norm)
        self.mlc_cls = nn.Sequential(
            nn.Linear(128*(self.num_classes), self.num_classes),
        )
        # self.transformer = nn.Transformer(
        #     d_model=256, nhead=4, num_encoder_layers=0, num_decoder_layers=1, batch_first=True)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, nn.BatchNorm1d):
                constant_init(m, 1)
                # normal_init(m, std =self.init_std)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
                # normal_init(m, std =self.init_std)
            elif isinstance(m, nn.Linear):
                kaiming_init(m)
            elif isinstance(m, nn.MultiheadAttention):
                # normal_init(m, std =self.init_std)
                kaiming_init(m)
            elif isinstance(m, nn.Parameter):
                kaiming_init(m)
                
    def forward(self, x, f, num_segs):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.
            num_segs (int): Number of segments into which a video
                is divided.
        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        
        ########## INPUT ############
        x =self.conv1x1(x) 
        x_256 = x
        f = self.tabmlp(f) 
        ########### global fusion ############
        if self.avg_pool is not None:
            if isinstance(x, tuple):
                shapes = [y.shape for y in x]
                assert 1 == 0, f'x is tuple {shapes}'
            x = self.avg_pool(x) 
        x = x.reshape((-1, num_segs) + x.shape[1:])
        x = self.consensus(x)
        x = x.squeeze(1)
        x = x.view(x.size(0), -1)
        ####################
        f_scale = torch.unsqueeze(f, -1).unsqueeze(-1) 
        f_spa = f_scale.repeat(num_segs, 1, x_256.shape[-2], x_256.shape[-1]) 
        spa = torch.concat([x_256, f_spa], dim=1) 
        spa = self.spaconv(spa) 
        final_spa = x_256 + spa #
        final_spa = self.avg_pool(final_spa)  
        final_spa = final_spa.squeeze(-1).squeeze(-1) 
        final_tem = final_spa.reshape((-1, num_segs) + final_spa.shape[1:]) 
        final_tem = final_tem.transpose(-2,-1) 
        f_scale1 = f.unsqueeze(-1)
        f_tem = f_scale1.repeat(1, 1, final_tem.shape[2]) 
        tem = torch.concat([final_tem, f_tem], dim=1) 
        tem = self.tcn(tem) 
        final_tem = tem + final_tem  
        final_cha = final_tem.mean(dim=-1)
        final_cha = torch.cat((final_cha, f), dim=1) 
        final_cha = self.mlp_cha(final_cha)
        #############
        Final = torch.cat((x, final_cha), dim=1) 
        downFinal = self.linear(Final).unsqueeze(1)
        ########## qual 
        label_emb = self.label_emb.repeat(Final.shape[0], 1, 1) 
        qualfuse = self.decoder(label_emb, downFinal ) 
        qualfuse = qualfuse.reshape(Final.shape[0],-1)
        mlc_score = self.mlc_cls(qualfuse)
        finalfuse = torch.cat((Final, qualfuse), dim=1) 
        cls_score = self.fc_cls(finalfuse)
        cls_score = torch.cat((cls_score,mlc_score),dim=-1)
        return cls_score
    
    def forward_feat(self, x, f, num_segs):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.
            num_segs (int): Number of segments into which a video
                is divided.
        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        if self.avg_pool is not None:
            if isinstance(x, tuple):
                shapes = [y.shape for y in x]
                assert 1 == 0, f'x is tuple {shapes}'
            x = self.avg_pool(x)
        x = x.reshape((-1, num_segs) + x.shape[1:])
        x = self.consensus(x)
        x = x.squeeze(1)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        return x
    
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()
    
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        # num_levels = num_channels
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
