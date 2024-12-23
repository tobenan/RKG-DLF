# Copyright (c) OpenMMLab. All rights reserved.
from .audio_tsn_head import AudioTSNHead
from .base import BaseHead
from .bbox_head import BBoxHeadAVA
from .fbo_head import FBOHead
from .i3d_head import I3DHead
from .lfb_infer_head import LFBInferHead
from .misc_head import ACRNHead
from .roi_head import AVARoIHead
from .slowfast_head import SlowFastHead
from .ssn_head import SSNHead
from .stgcn_head import STGCNHead
from .timesformer_head import TimeSformerHead
from .tpn_head import TPNHead
from .trn_head import TRNHead
from .tsm_head import TSMHead
from .tsn_head import TSNHead
from .x3d_head import X3DHead
from .tsn_fusehead import TSN_FuseHead
from .tsn_fusehead_DAFT import TSN_FuseHead_DAFT
from .tsn_fusehead_my import TSN_FuseHead_MY
from .tsn_fusehead_my_MHSA import TSN_FuseHead_MY_MHSA
from .tsn_fusehead_my_cross import TSN_FuseHead_MY_CROSS
from .tsn_fusehead_my_mlp import TSN_FuseHead_MY_MLP
from .tsn_fusehead_my_tabattention import TSN_FuseHead_MY_Attention
from .tsn_fusehead_STatt import TSN_FuseHead_STatt
from .tsn_fusehead_STatt_spatt import TSN_FuseHead_STatt_spatt
from .tsn_fusehead_STatt_conv_mask import TSN_FuseHead_STatt_conv_mask
from .tsn_fusehead_conv_mask_fusion import TSN_FuseHead_conv_mask_fusion
from .tsn_fusehead_conv_mask_fusion_qual import TSN_FuseHead_conv_mask_fusion_qual
from .tsn_fusehead_conv_mask_fusion_qual_relation import TSN_FuseHead_conv_mask_fusion_qual_relation
from .tsn_fusehead_conv_mask_fusion_qual_relate1 import TSN_FuseHead_conv_mask_fusion_qual_relate1
from .tsn_fusehead_conv_mask_fusion_qual_relate11 import TSN_FuseHead_conv_mask_fusion_qual_relate11
from .tsn_fusehead_conv_mask_fusion_qual_relate2 import TSN_FuseHead_conv_mask_fusion_qual_relate2
from .tsn_fusehead_conv_mask_fusion_qual_relate3 import TSN_FuseHead_conv_mask_fusion_qual_relate3
__all__ = [
    'TSNHead', 'I3DHead', 'BaseHead', 'TSMHead', 'SlowFastHead', 'SSNHead',
    'TPNHead', 'AudioTSNHead', 'X3DHead', 'BBoxHeadAVA', 'AVARoIHead',
    'FBOHead', 'LFBInferHead', 'TRNHead', 'TimeSformerHead', 'ACRNHead',
    'STGCNHead', 'TSN_FuseHead','TSN_FuseHead_DAFT','TSN_FuseHead_MY',
    'TSN_FuseHead_MY_MHSA','TSN_FuseHead_MY_CROSS','TSN_FuseHead_MY_MLP',
    'TSN_FuseHead_MY_Attention', 'TSN_FuseHead_STatt','TSN_FuseHead_STatt_spatt',
    'TSN_FuseHead_STatt_conv_mask', 'TSN_FuseHead_conv_mask_fusion',
    'TSN_FuseHead_conv_mask_fusion_qual','TSN_FuseHead_conv_mask_fusion_qual_relation',
    'TSN_FuseHead_conv_mask_fusion_qual_relate1', 'TSN_FuseHead_conv_mask_fusion_qual_relate2',
    'TSN_FuseHead_conv_mask_fusion_qual_relate3','TSN_FuseHead_conv_mask_fusion_qual_relate11',
]
