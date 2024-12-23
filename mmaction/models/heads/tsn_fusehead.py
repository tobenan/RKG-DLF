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

@HEADS.register_module()
class TSN_FuseHead(BaseHead):
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
        
        self.fc_cls = nn.Linear(256+int(self.fuse_channels * 2), self.num_classes)

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
                normal_init(m, std =self.init_std)
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
        #[N * num_segs, in_channels, 7, 7]
        x =self.conv1x1(x) #! [N * num_segs, 256, 7, 7]
        x_256 = x # 
        f = self.tabmlp(f) # [8, 2d]
        
        if self.avg_pool is not None:
            if isinstance(x, tuple):
                shapes = [y.shape for y in x]
                assert 1 == 0, f'x is tuple {shapes}'
            x = self.avg_pool(x) 
            # [N * num_segs, in_channels, 1, 1]
        x = x.reshape((-1, num_segs) + x.shape[1:])
        # [N, num_segs, in_channels, 1, 1]
        x = self.consensus(x)
        # [N, 1, in_channels, 1, 1]
        x = x.squeeze(1)
        # [N, in_channels, 1, 1]
        
        
        if self.dropout is not None:
            x = self.dropout(x)
            # [N, in_channels, 1, 1]
        x = x.view(x.size(0), -1)
        # [N, in_channels]
        
        
        # 融合特征
        F = torch.cat((x, f), dim=1)  # [batch_size, 512, 7, 7]

        
        cls_score = self.fc_cls(F)
        # [N, num_classes]
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
        # [N * num_segs, in_channels, 7, 7]
        if self.avg_pool is not None:
            if isinstance(x, tuple):
                shapes = [y.shape for y in x]
                assert 1 == 0, f'x is tuple {shapes}'
            x = self.avg_pool(x)
            # [N * num_segs, in_channels, 1, 1]
        x = x.reshape((-1, num_segs) + x.shape[1:])
        # [N, num_segs, in_channels, 1, 1]
        x = self.consensus(x)
        # [N, 1, in_channels, 1, 1]
        x = x.squeeze(1)
        # [N, in_channels, 1, 1]
        if self.dropout is not None:
            x = self.dropout(x)
            # [N, in_channels, 1, 1]
        x = x.view(x.size(0), -1)
        # [N, in_channels]
        return x
