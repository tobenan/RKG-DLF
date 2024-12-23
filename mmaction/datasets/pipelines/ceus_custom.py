import copy as cp
import io
import os
import os.path as osp
from re import L
import shutil
from turtle import down
from unittest import result
import warnings
import cv2
import mmcv
import numpy as np
import torch
from mmcv.fileio import FileClient
from torch.nn.modules.utils import _pair

# from ...utils import get_random_string, get_shm_dir, get_thread_id
from ..builder import PIPELINES
from .loading import SampleFrames
from . import Normalize
import json
# from statistics import NormalDist
from .augmentations import RandomResizedCrop, CenterCrop

# import scipy.stats as st
from scipy.stats import norm


@PIPELINES.register_module()
class SelectROI():
    def __init__(self, key='rec_c') -> None:
        self.key = key
    
    def __call__(self, results):
        imgs = results['imgs']
        roi = results[self.key]
        img_roi = [img_cur[roi[1]:roi[3], roi[0]:roi[2]]
                    for img_cur in imgs]
        results['imgs'] = img_roi
        # results['original_shape'] = img_roi[0].shape[:2]
        results['img_shape'] = img_roi[0].shape[:2]
        
        return results

@PIPELINES.register_module()
class SelectTumor:
    # def __init__(self, roi_cfg) -> None:
    #     self.roi_cfg = roi_cfg
    
    def __call__(self, results):
        imgs = results['imgs']
        a, b, c, d = results['tumor_bbox']
        
        img_roi = [img_cur[b:b+d,a:a+c]
                    for img_cur in imgs]
        results['imgs'] = img_roi
        # results['original_shape'] = img_roi[0].shape[:2]
        results['img_shape'] = img_roi[0].shape[:2]
        
        return results


@PIPELINES.register_module()
class GetCoarseSeg:
    def __init__(self) -> None:
        pass
    
    def __call__(self, results):
        if 'coarse_seg_dir' not in results:#(方便使用cam)
            list = results['frame_dir'].split('/')
            coarse_seg_dir = os.path.join(list[0], list[1], 'coarse_seg', list[2], list[3]+'.jpg')#+'.jpg' .npy
            results['coarse_seg_dir'] = coarse_seg_dir
        if results['coarse_seg_dir'].endswith('npy'):
            coarse_seg = np.load(results['coarse_seg_dir'])
        else :
            coarse_seg = cv2.imread(results['coarse_seg_dir'], -1)//255
        results['coarse_seg'] = coarse_seg
        return results


@PIPELINES.register_module()
class SelectROI2Mode:
    # def __init__(self) -> None:
    #     # self.roi_ceus_cfg = roi_ceus_cfg
    #     # self.roi_b_cfg = roi_b_cfg
    #     pass
    
    def __call__(self, results):
        imgs = results['imgs']
        # roi_ceus = self.roi_ceus_cfg[imgs[0].shape]
        roi_ceus = results['rec_c']
        img_ceus = [img_cur[roi_ceus[1]:roi_ceus[3], roi_ceus[0]:roi_ceus[2]]
                    for img_cur in imgs]
        results['img_ceus'] = img_ceus
        
        # results['original_shape'] = img_roi[0].shape[:2]
        results['img_ceus_shape'] = img_ceus[0].shape[:2]
        
        # roi_b = self.roi_b_cfg[imgs[0].shape]
        roi_b = results['rec_b']
        img_b = [img_cur[roi_b[1]:roi_b[3], roi_b[0]:roi_b[2]]
                    for img_cur in imgs]
        results['img_b'] = img_b
        # results['original_shape'] = img_roi[0].shape[:2]
        results['img_b_shape'] = img_b[0].shape[:2]
        assert results['img_b_shape'] == results['img_ceus_shape']
        results['img_shape'] = results['img_b_shape']
        # 处理 coarse seg
        if 'coarse_seg' in results.keys():
            coarse_seg = results['coarse_seg']
            results['coarse_seg'] = coarse_seg[roi_b[1]:roi_b[3], roi_b[0]:roi_b[2]]
        return results


@PIPELINES.register_module()
class ResizeCEUS:
    def __init__(self, scale, keep_ratio = True):
        max_long_edge = max(scale)
        max_short_edge = min(scale)
        if max_short_edge == -1:
        # assign np.inf to long edge for rescaling short edge later.
            scale = (np.inf, max_long_edge)
        self.scale = scale
        self.keep_ratio = keep_ratio
    def _resize_imgs(self, imgs, new_w, new_h) :
        return [
            mmcv.imresize(
                img, (new_w, new_h), interpolation='bilinear')
            for img in imgs
        ]

    def __call__(self, results):
        # assert results['img_b_shape'] == results['img_ceus_shape']
        img_h, img_w = results['img_shape']
        if self.keep_ratio:
            new_w, new_h = mmcv.rescale_size((img_w, img_h), self.scale)

        else:
            new_w, new_h = self.scale
        results['img_b'] = self._resize_imgs(results['img_b'], new_w, new_h)
        results['img_ceus'] = self._resize_imgs(results['img_ceus'], new_w, new_h)
        results['coarse_seg'] = mmcv.imresize(results['coarse_seg'],(new_w, new_h),interpolation='nearest')
        # results['img_b_shape'] = new_h, new_w
        results['img_shape'] = new_h, new_w
        return results

@PIPELINES.register_module()
class RandomResizedCropCEUS(RandomResizedCrop):
    def __init__(self, area_range=(0.6, 1.0), aspect_ratio_range=(3 / 4, 4 / 3), lazy=False):
        super().__init__(area_range, aspect_ratio_range, lazy)
    

    def __call__(self, results):
        img_h, img_w = results['img_shape']
        left, top, right, bottom = self.get_crop_bbox(
            (img_h, img_w), self.area_range, self.aspect_ratio_range)
        new_h, new_w = bottom - top, right - left
        crop_bbox = np.array([left, top, right, bottom])
        results['crop_bbox'] = crop_bbox
        results['img_shape'] = (new_h, new_w)

        results['img_b'] = self._crop_imgs(results['img_b'], crop_bbox)
        results['img_ceus'] = self._crop_imgs(results['img_ceus'], crop_bbox)

        results['coarse_seg'] = results['coarse_seg'][top:bottom, left:right]

        return results



@PIPELINES.register_module()
class FlipCEUS:
    def __init__(self, flip_ratio=0.5, direction='horizontal') -> None:
        self.flip_ratio = flip_ratio
        self.direction = direction    

    def __call__(self, results):
        flip = np.random.rand() < self.flip_ratio
        if flip:
            # results['img_b'] = [mmcv.imflip(img, self.direction) for img in results['img_b']]
            # results['img_ceus'] = [mmcv.imflip(img, self.direction) for img in results['img_ceus']]
            results['img_ceus'] = [mmcv.imflip(img, self.direction) for img in results['imgs']]
            results['coarse_seg'] = mmcv.imflip(results['coarse_seg'], self.direction).copy()
        
        return results

@PIPELINES.register_module()
class NormalizeCEUS:
    def __init__(self, mean, std) -> None:
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)    

    def __call__(self, results):
        results['img_b'] = [mmcv.imnormalize(img, self.mean, self.std,) for img in results['img_b']]
        results['img_ceus'] = [mmcv.imnormalize(img, self.mean, self.std,) for img in results['img_ceus']]
        return results

@PIPELINES.register_module()
class FormatShapeCEUS:
    def __init__(self, input_format,):
        self.input_format = input_format
        if self.input_format not in ['NCTHW','NCHW']:
            raise ValueError(
                f'The input format {self.input_format} is invalid.')
        
    
    def __call__(self, results):
        num_clips = results['num_clips']
        clip_len = results['clip_len']
        results['img_b'] = self.getFormat(results['img_b'], num_clips, clip_len)
        results['img_ceus'] = self.getFormat(results['img_ceus'], num_clips, clip_len)
        if not isinstance(results['coarse_seg'], np.ndarray):
            results['coarse_seg'] = np.array(results['coarse_seg'])
        return results

    def getFormat(self, imgs, num_clips, clip_len):
        if not isinstance(imgs, np.ndarray):
            imgs = np.array(imgs)
        if self.input_format == 'NCTHW':
            # num_clips = results['num_clips']
            # clip_len = results['clip_len']
            imgs = imgs.reshape((-1, num_clips, clip_len) + imgs.shape[1:])
            imgs = np.transpose(imgs, (0, 1, 5, 2, 3, 4))
            # N_crops x N_clips x C x L x H x W
            imgs = imgs.reshape((-1, ) + imgs.shape[2:])
            # M' x C x L x H x W
            # M' = N_crops x N_clips
        elif self.input_format == 'NCHW':
            imgs = np.transpose(imgs, (0, 3, 1, 2))
        return imgs



@PIPELINES.register_module()
class TopCropCEUS(CenterCrop):
    def __init__(self, crop_size, lazy=False):
        super().__init__(crop_size, lazy)
    
    def __call__(self, results):
        img_h, img_w = results['img_shape']
        crop_w, crop_h = self.crop_size
        left = (img_w - crop_w) // 2
        top = 0
        right = left + crop_w
        bottom = top + crop_h
        new_h, new_w = bottom - top, right - left
        results['img_shape'] = (new_h, new_w)
        crop_bbox = np.array([left, top, right, bottom])
        results['img_b'] = self._crop_imgs(results['img_b'], crop_bbox)
        results['img_ceus'] = self._crop_imgs(results['img_ceus'], crop_bbox)
        # results['img_b'] = self._crop_imgs(results['img_b'], crop_bbox)
        results['coarse_seg'] = results['coarse_seg'][top:bottom, left:right]
        return results


@PIPELINES.register_module()
class CenterCropCEUS(CenterCrop):
    def __init__(self, crop_size, lazy=False):
        super().__init__(crop_size, lazy)
    
    def __call__(self, results):
        img_h, img_w = results['img_shape']
        crop_w, crop_h = self.crop_size
        left = (img_w - crop_w) // 2
        top = (img_h - crop_h) // 2
        right = left + crop_w
        bottom = top + crop_h
        new_h, new_w = bottom - top, right - left
        results['img_shape'] = (new_h, new_w)
        crop_bbox = np.array([left, top, right, bottom])
        results['img_b'] = self._crop_imgs(results['img_b'], crop_bbox)
        results['img_ceus'] = self._crop_imgs(results['img_ceus'], crop_bbox)
        # results['img_b'] = self._crop_imgs(results['img_b'], crop_bbox)
        results['coarse_seg'] = results['coarse_seg'][top:bottom, left:right]
        return results


@PIPELINES.register_module()
class VideoNorm(Normalize):
    def __init__(self):
        super().__init__([0, 0, 0], [1, 1, 1])
        self.norm_cfg = dict()
    def __call__(self, results):
        # img_path = results['frame_dir']
        # if img_path in self.norm_cfg.keys():
        #     self.mean = self.norm_cfg[img_path]['mean']
        #     self.std = self.norm_cfg[img_path]['std']
        # else:
        #     self.calstatic(results['imgs'])
        #     norm_cfg = dict(
        #         mean=self.mean,
        #         std = self.std
        #     )
        #     self.norm_cfg.update({img_path:norm_cfg})
        if 'norm_cfg' not in results:
            norm_json = 'data/preprocessing5/video_norms_cut_2mode.json'
            with open(norm_json, 'r') as f:
                norms = json.load(f)
                results['norm_cfg'] = norms[results['frame_dir']]
        self.mean = np.array(results['norm_cfg']['mean'], dtype=np.float32)
        self.std = np.array(results['norm_cfg']['std'], dtype=np.float32)
        return super().__call__(results)
    
    def calstatic(self, imgs):
        imgs = np.array(imgs) # t h w c
        self.mean = imgs.mean(axis=(0, 1, 2), keepdims=True)
        self.std = imgs.std(axis=(0, 1, 2), keepdims=True)
        
        
    

@PIPELINES.register_module()
class ToFloat32:
    def __call__(self, results):
        n = len(results['imgs'])
        h, w, c = results['imgs'][0].shape
        imgs = np.empty((n, h, w, c), dtype=np.float32)
        for i, img in enumerate(results['imgs']):
            imgs[i] = img
        results['imgs'] = imgs
        return results
    
@PIPELINES.register_module()
class Concat2Mode:
    def __call__(self, results:dict):
        img_b = results['img_b']
        img_ceus = results['img_ceus']
        assert len(img_b) == len(img_ceus)
        img = [np.concatenate((img_ceus[i],img_b[i]), axis=2) for i in range(len(img_ceus))]
        results['imgs'] = img
        del results['img_b']
        del results['img_ceus']
        return results
    
    
@PIPELINES.register_module()
class AddDiffIndx:
    def __init__(self, diff) -> None:
        self.diff = diff
        
    
    def __call__(self, results:dict):
        frame_inds = results['frame_inds']
        total_frames = results['total_frames']
        diff_inds =  (frame_inds + self.diff).clip(0, total_frames)
        all_inds = np.concatenate((frame_inds, diff_inds), axis= 0)
        results['frame_inds'] = all_inds
        return results

@PIPELINES.register_module()
class DiffImgs:
    def __call__(self, results:dict):
        num_clips = len(results['frame_inds']) // 2
        imgs = [results['imgs'][i] - results['imgs'][i + num_clips] for i in range(num_clips)]
        results['imgs'] = imgs
        return results 
        

@PIPELINES.register_module()
class SelectModeCut:
    def __init__(self, key = 'ceus') -> None:
        self.key = key
    def __call__(self, results):
        imgs = results['imgs']
        h, w, _ = imgs[0].shape
        if self.key == 'ceus':
            img_roi = [img_cur[:,:,:3] for img_cur in imgs]
        else:
            img_roi = [img_cur[:,:,3:] for img_cur in imgs]
        results['imgs'] =img_roi
        results['img_shape'] = img_roi[0].shape
        return results


@PIPELINES.register_module()
class ConcatModeCut:
    def __call__(self, results):
        imgs = results['imgs']
        h, w, _ = imgs[0].shape
        # if self.key == 'ceus':
        # img_c = [img_cur[:,0:w//2,:] for img_cur in imgs]
        # img_b = [img_cur[:,w//2:,:] for img_cur in imgs]
        imgs_concat = [
            np.concatenate((img_cur[:,0:w//2,:], img_cur[:,w//2:,:]),axis=2) for img_cur in imgs
        ] #  (32, 224, 224, 6)
        results['imgs'] =imgs_concat
        results['img_shape'] = imgs_concat[0].shape[:2]
        return results


@PIPELINES.register_module()
class SplitCEUS:
    def __call__(self, results):
        imgs = results['imgs']# (32, 224, 224, 6)
        results['img_ceus'] = [i[...,0:3] for i in imgs]
        results['img_b'] = [i[...,3:] for i in imgs]
        del results['imgs']
        return results
