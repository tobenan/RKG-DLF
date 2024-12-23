import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

# import torch
import torch.distributed as dist
# import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16

from .. import builder
import torch
from torch import nn

from ..builder import RECOGNIZERS
from .base import BaseRecognizer

@RECOGNIZERS.register_module()
class RecoginizerTwoStream_DAFT_QUAL_5infer(nn.Module):
    def __init__(self, 
                 mlc_loss,
                 backbone, 
                 fuse_cfg,
                 cls_head_CEUS,
                 cls_head_B,
                 train_cfg=None,
                 test_cfg=None) -> None:
        super().__init__()
        self.backbone_CEUS = builder.build_backbone(backbone)
        self.backbone_B = builder.build_backbone(backbone)
        self.fuseNet = builder.build_backbone(fuse_cfg)
        
        
        self.cls_head_CUES = builder.build_head(cls_head_CEUS)
        self.cls_head_B = builder.build_head(cls_head_B)
        self.cls_head_fuse = builder.build_head({
        'type': 'TSNHead', 'num_classes': 3, 
        'in_channels': 512, 'spatial_type': 'avg', 
        'consensus': {'type': 'AvgConsensus', 'dim': 1}, 
        'dropout_ratio': 0.8, 'init_std': 0.001})
        
        
        self.backbone_CEUS.init_weights()
        self.backbone_B.init_weights()
        self.fuseNet.init_weights()
        
        
        self.cls_head_CUES.init_weights()
        self.cls_head_B.init_weights()  
        self.cls_head_fuse.init_weights()

        if mlc_loss == 'ml_softmax':
            self.loss = MultiLabelSoftmax(gamma_pos= 1.0, gamma_neg= 1.0)
        elif mlc_loss == 'asl':
            self.loss = AsymmetricLossOptimized(gamma_neg= 2.0, gamma_pos= 1.0, clip= 0.05)
        elif mlc_loss == 'calibrated_hinge':
            self.loss = HingeCalibratedRanking()
        else: 
            self.loss = nn.MultiLabelSoftMarginLoss()
            
        self.blending = None
        
    def train_step(self, data_batch, optimizer, **kwargs):
        if 'img_ceus' not in data_batch:
            data_batch["img_ceus"]=data_batch["img_b"]
        img_ceus = data_batch['img_ceus']
        img_b = data_batch['img_b']
        coarse_seg = data_batch['coarse_seg']
        label = data_batch['label']
        
        ceus_quan = data_batch['ceus_quan'] 
        bus_quan = data_batch['bus_quan']
        qual = data_batch['qual'] 
        aux_info = {}

        
        losses = self(img_ceus, img_b, coarse_seg, label, ceus_quan, bus_quan,qual,\
            return_loss=True, **aux_info)
        loss, log_vars = BaseRecognizer._parse_losses(losses)
        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))))

        return outputs
    
    def val_step(self, data_batch, optimizer, **kwargs):
        return self.train_step(data_batch, optimizer, **kwargs)
    
    
    def forward(self, img_ceus, img_b, coarse_seg=None, label=None, \
        ceus_quan=None, bus_quan=None, qual=None,\
        return_loss=True, **kwargs):
        """Define the computation performed at every call."""
        if kwargs.get('gradcam', False):
            del kwargs['gradcam']
            return self.forward_gradcam(img_ceus, img_b, **kwargs)
        if return_loss:
            if label is None:
                raise ValueError('Label should not be None.')
            if self.blending is not None:
                imgs, label = self.blending(imgs, label)
            return self.forward_train(img_ceus, img_b, coarse_seg, label, \
                ceus_quan, bus_quan, qual, **kwargs)

        return self.forward_test(img_ceus, img_b, ceus_quan, bus_quan, qual, **kwargs)

    def forward_train(self, img_ceus, img_b, coarse_seg, label, \
        ceus_quan, bus_quan, qual, **kwargs):
        assert img_ceus.shape == img_b.shape
        batches = img_ceus.shape[0]
        img_ceus = img_ceus.reshape((-1, ) + img_ceus.shape[2:])
        img_b = img_b.reshape((-1, ) + img_b.shape[2:])
        num_segs = img_ceus.shape[0] // batches
        gt_labels = label.squeeze()
        losses = dict()
        gt_one_hot = F.one_hot(gt_labels, num_classes=3)
        qual_ceus = torch.cat((gt_one_hot, qual[:,:4]),dim=-1)
        qual_b = torch.cat((gt_one_hot, qual[:,4:]),dim=-1)
        
        x_ceus = self.backbone_CEUS(img_ceus)
        cls_score_cues = self.cls_head_CUES(x_ceus[-1], ceus_quan, num_segs)
        cls_cues = cls_score_cues[:,:3]
        mlc_cues = cls_score_cues[:,3:]
        loss_cls_ceus = self.cls_head_CUES.loss(cls_cues, gt_labels)['loss_cls']
        if self.loss == HingeCalibratedRanking():
            mlc_cues = torch.tensor(mlc_cues)
        loss_mlc_ceus = self.loss(mlc_cues, qual_ceus)
        
        x_b = self.backbone_B(img_b)
        cls_score_b = self.cls_head_B(x_b[-1], bus_quan, num_segs)
        cls_b = cls_score_b[:,:3] 
        mlc_b = cls_score_b[:,3:]
        loss_cls_b = self.cls_head_CUES.loss(cls_b, gt_labels)['loss_cls']
        if self.loss == HingeCalibratedRanking():
            mlc_b = torch.tensor(mlc_b)
        loss_mlc_b = self.loss(mlc_b, qual_b)
        
        x_fuse = self.fuseNet(x_ceus, x_b)
        cls_socre_fuse = self.cls_head_fuse(x_fuse[-1], num_segs) 
        loss_cls_fuse = self.cls_head_fuse.loss(cls_socre_fuse, gt_labels)['loss_cls']
        
        
        
        losses.update({'loss_cls_fuse': loss_cls_fuse,
                       'loss_cls_ceus': loss_cls_ceus,
                       'loss_mlc_ceus': loss_mlc_ceus,
                       'loss_cls_b': loss_cls_b,
                       'loss_mlc_b': loss_mlc_b,
                       })
        
        return losses
    
    def forward_test(self, img_ceus, img_b, ceus_quan, bus_quan, qual,):
        assert img_ceus.shape == img_b.shape
        batches = img_ceus.shape[0]
        img_ceus = img_ceus.reshape((-1, ) + img_ceus.shape[2:])
        img_b = img_b.reshape((-1, ) + img_b.shape[2:])
        num_segs = img_ceus.shape[0] // batches

        x_ceus = self.backbone_CEUS(img_ceus)
        cls_score_cues = self.cls_head_CUES(x_ceus[-1], ceus_quan, num_segs)
        cls_cues = cls_score_cues[:,:3]
        mlc_cls_cues = cls_score_cues[:,3:6]
        mlc_cues = cls_score_cues[:,6:]
        
        x_b = self.backbone_B(img_b)
        cls_score_b = self.cls_head_B(x_b[-1], bus_quan, num_segs)
        cls_b = cls_score_b[:,:3] # benign malign inflam
        mlc_cls_b = cls_score_b[:,3:6]
        mlc_b = cls_score_b[:,6:]
        
        x_fuse = self.fuseNet(x_ceus, x_b)
        cls_score_fuse = self.cls_head_fuse(x_fuse[-1], num_segs)
        
        softmax_mlc_ceus = torch.softmax(mlc_cls_cues,dim=1)
        softmax_mlc_b = torch.softmax(mlc_cls_b,dim=1)
        softmax_ceus = torch.softmax(cls_cues, dim=1)
        softmax_b = torch.softmax(cls_b, dim=1)
        softmax_fuse = torch.softmax(cls_score_fuse, dim=1)
        
        cls_score = ( (softmax_b + softmax_ceus + softmax_fuse) + (softmax_mlc_b +softmax_mlc_ceus) ) / 5

        mlc = torch.cat((mlc_cues, mlc_b), dim=-1)
        mlc_sigmoid = torch.sigmoid(mlc) 
        total = torch.cat((cls_score,mlc_sigmoid),dim=-1)
        return total.cpu().numpy()




    def forward_gradcam(self, img_ceus, img_b):
        assert img_ceus.shape == img_b.shape
        batches = img_ceus.shape[0]
        img_ceus = img_ceus.reshape((-1, ) + img_ceus.shape[2:])
        img_b = img_b.reshape((-1, ) + img_b.shape[2:])
        num_segs = img_ceus.shape[0] // batches

        x_ceus = self.backbone_CEUS(img_ceus)
        cls_score_cues = self.cls_head_CUES(x_ceus[-1], num_segs)
        x_b = self.backbone_B(img_b)
        cls_score_b = self.cls_head_B(x_b[-1], num_segs)
        
        x_fuse = self.fuseNet(x_ceus, x_b)
        cls_score_fuse = self.cls_head_fuse(x_fuse[-1], num_segs)
        
        cls_score = (cls_score_b + cls_score_cues + cls_score_fuse) / 3
        return cls_score
        

class MultiLabelSoftmax(nn.Module):
    def __init__(self, gamma_pos=1., gamma_neg=1.):
        super(MultiLabelSoftmax, self).__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg

    def forward(self, outputs, targets):
        targets = targets.float()
        outputs = (1 - 2 * targets) * outputs
        y_pred_neg = outputs - targets * 1e15
        y_pred_pos = outputs - (1 - targets) * 1e15
        zeros = torch.zeros_like(outputs[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)

        neg_loss = (1 / self.gamma_neg) * torch.log(torch.sum(torch.exp(self.gamma_neg * y_pred_neg), dim=-1))
        pos_loss = (1 / self.gamma_pos) * torch.log(torch.sum(torch.exp(self.gamma_pos * y_pred_pos), dim=-1))

        loss = torch.mean(neg_loss + pos_loss)
        return loss



class BinaryCEL(nn.Module):
    def __init__(self):
        super(BinaryCEL, self).__init__()

    def forward(self, outputs, targets):
        return F.binary_cross_entropy_with_logits(
            outputs, targets.float() )



class HingeCalibratedRanking(nn.Module):
    def __init__(self):
        super(HingeCalibratedRanking, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, outputs, targets, reduce=True):
        loss_op = []
        for i in range(outputs.size(0)):
            # positive = torch.masked_select(outputs[i], targets[i].byte())
            positive = torch.masked_select(outputs[i], targets[i].bool())
            negative = torch.masked_select(outputs[i], (1-targets[i]).bool())

            if negative.size(0) != 0:
                neg_calib = F.relu(1 + negative).mean()
            else:
                neg_calib, negative = torch.tensor(0.,device=outputs.device), torch.tensor(0.,device=outputs.device)
            if positive.size(0) != 0:
                pos_calib = F.relu(1 - positive).mean()
            else:
                pos_calib, positive = torch.tensor(0.,device=outputs.device), torch.tensor(0.,device=outputs.device)
            hinge = 1. + negative.unsqueeze(-1) - positive
            l_hinge = F.relu(hinge).mean()
            loss_op.append(l_hinge + neg_calib + pos_calib)

        loss_op = torch.stack(loss_op, dim=0)
        return loss_op.mean()


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """
        
        y = y.float()
        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()

from torch.autograd import Variable

class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        y = y.float()
        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        unit_count = Variable(y.sum() + (1 - y).sum(), requires_grad=False)
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)
            unit_count = Variable(y.sum() + (self.xs_neg <= 1).float().sum(), requires_grad=False)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))
        # self.loss = self.targets * torch.where(x >= 0, F.softplus(x, -1, 50), x - F.softplus(x, 1, 50))
        # self.loss.add_(self.anti_targets * torch.where(x >= 0, -x + F.softplus(x, -1, 50), -F.softplus(x, 1, 50)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                    self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        # return self.loss.neg().mean()
        return self.loss.neg().sum() / unit_count
 