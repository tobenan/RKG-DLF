import copy
import os.path as osp

import torch
import math
from collections import defaultdict
import numpy as np 
from mmaction.datasets.pipelines import Resize
from .base import BaseDataset
from .builder import DATASETS
from . import RawframeDataset
import os
import json
from mmaction.datasets import build_dataset
import random
import pandas


@DATASETS.register_module()
class CEUSDatsaset(RawframeDataset):
    def __init__(self, ann_file, pipeline, data_prefix=None, coarse_seg=None, bus_quan=None, ceus_quan=None, qual =None, 
                 tabquannorm = False, tabquan_enc = None, n_bins=2,
                 rec_json=None, norm_json = None,test_mode=False, trimmed=None,filename_tmpl='img_{:05}.jpg', with_offset=False, 
                 multi_class=False, num_classes=None, start_index=1, modality='RGB', tumor_bbox=None, sample_by_class=False, power=0,
                 dynamic_length=False, patient_text_info=None, **kwargs):
        self.coarse_seg = coarse_seg
        self.rec_json = rec_json
        # with open(no)
        self.norm_json = norm_json
        self.tumor_bbox = tumor_bbox
        # self.mix_up = mix_up
        self.trimmed = trimmed 
        self.patient_text_info = patient_text_info
        
        self.bus_quan= bus_quan
        self.ceus_quan= ceus_quan
        self.qual= qual
        
        ##### encode
        self.tabquannrom = tabquannorm
        self.tabquan_enc = tabquan_enc
        self.n_bins = n_bins
        super().__init__(ann_file, pipeline, data_prefix, test_mode, filename_tmpl, with_offset, multi_class, num_classes, start_index, modality, sample_by_class, power, dynamic_length, **kwargs)
        
    
    
    def load_annotations(self):
        video_infos = []
        if self.trimmed is not None:
            trimmed_df = pandas.read_excel(self.trimmed)
        if self.patient_text_info is not None:
            text_info_df = pandas.read_csv(self.patient_text_info)
        if self.ceus_quan is not None:
            ceus_quan = pandas.read_excel(self.ceus_quan)
            bus_quan = pandas.read_excel(self.bus_quan)
            qual = pandas.read_excel(self.qual)
 
            
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                line_split = line.strip().split()
                video_info = {}
                video_info['frame_dir'] = line_split[0]
                video_info['total_frames'] = int(line_split[1])
                video_info['label'] = int(line_split[2])
                # video_info['rate'] = float(line_split[3])
                perfix, idx = os.path.split(video_info['frame_dir'])
                _, cls_ceus = os.path.split(perfix)
                if self.coarse_seg is not None:
                    coarse_seg_dir = os.path.join(self.coarse_seg, cls_ceus, idx+'.jpg')
                    video_info['coarse_seg_dir'] = coarse_seg_dir
                if self.rec_json is not None:
                    rec_json_path = os.path.join(self.rec_json, cls_ceus, idx+'.json')
                    rec_c, rec_b = self.getRec(rec_json_path)
                    video_info['rec_c'] = rec_c
                    video_info['rec_b'] = rec_b
                if self.norm_json is not None:
                    with open(self.norm_json, 'r') as f:
                        norms = json.load(f)
                    video_info['norm_cfg'] = norms[video_info['frame_dir']]
                    # label_me_json = json.load(f)
                if self.tumor_bbox is not None:
                    with open(self.tumor_bbox, 'r') as f:
                        bbox = json.load(f)
                    video_info['tumor_bbox'] = bbox[video_info['frame_dir']]
                if self.trimmed is not None:
                    ans = trimmed_df[(trimmed_df['label'] == cls_ceus) & (trimmed_df['index'] == int(idx))]
                    # print(ans)
                    video_info['offset'] = int(ans['start'])
                    video_info['total_frames'] = int(ans['end']) - int(ans['start'])
                    video_info['peak_frame'] = int(ans['peak_frame'])#
                    video_info['contrast_start'] = int(ans['start'])#
                    video_info['contrast_end'] = int(ans['end'])    #
                if self.patient_text_info is not None:
                    # dataframe1.loc[(dataframe1['cls']=='benign') &  (dataframe1['idx']==1)]
                    text_info_quer = text_info_df[(text_info_df['cls'] == cls_ceus) & (text_info_df['idx'] == int(idx))]
                    
                if self.ceus_quan is not None:
                    video_info['ceus_quan'] = ceus_quan[(ceus_quan['cls'] == cls_ceus) & (ceus_quan['idx'] == int(idx))].values[0,2:].astype(np.float32)
                    video_info['bus_quan'] = bus_quan[(bus_quan['cls'] == cls_ceus) & (bus_quan['idx'] == int(idx))].values[0,2:].astype(np.float32)
                    video_info['qual'] = qual[(qual['cls'] == cls_ceus) & (qual['idx'] == int(idx))].values[0,2:].astype(np.float32)
                
                video_infos.append(video_info)
                
        video_infos = video_infos        
        if self.tabquannrom is True: 
            ceus_quan_values = [sample['ceus_quan'] for sample in video_infos]
            bus_quan_values = [sample['bus_quan'] for sample in video_infos]
            ceus_matrix = np.array(ceus_quan_values)   
            bus_matrix = np.array(bus_quan_values)   
            
            scaler = sklearn.preprocessing.StandardScaler()
            ceus_normalized = scaler.fit_transform(ceus_matrix)
            bus_normalized = scaler.fit_transform(bus_matrix)
            for i, sample in enumerate(video_infos):
                sample['ceus_quan'] = ceus_normalized[i].tolist()
                sample['bus_quan'] = bus_normalized[i].tolist()
                # sample['ceus_quan_normalized'] = data_normalized[i].tolist()
            
            
        if self.tabquan_enc is not None:
            ceus_quan_values = [sample['ceus_quan'] for sample in video_infos]
            bus_quan_values = [sample['bus_quan'] for sample in video_infos]
            ceus_matrix = torch.tensor(np.array(ceus_quan_values))  
            bus_matrix = torch.tensor(np.array(bus_quan_values))   
            
            y = np.array([sample['label'] for sample in video_infos]) 
            
            ceus_enc, _ = num_enc_process(ceus_matrix, self.tabquan_enc, n_bins = self.n_bins, y_train=y)
            bus_enc, _ = num_enc_process(bus_matrix, self.tabquan_enc, n_bins=self.n_bins, y_train=y) 
             
            for i, sample in enumerate(video_infos):
                
                sample['ceus_quan'] = ceus_enc[i].tolist()
                sample['bus_quan'] = bus_enc[i].tolist()
      
   
        return video_infos 
    
    def getRec(self, rec_json_path):
        with open(rec_json_path) as f:
            label_me_json = json.load(f)
        for shape in label_me_json['shapes']:
            if shape['label'] == "ceus":
                ceus_rec = shape['points']
            elif shape['label'] == "b":
                b_point = shape['points']
        (left_c, top_c), (right_c, down_c) = ceus_rec
        left_b, top_b = b_point[0]
        h = int(down_c) - int(top_c)
        w = int(right_c) - int(left_c)
        right_b = left_b + w
        down_b = top_b + h
        return [int(left_c), int(top_c), int(right_c), int(down_c)], [int(left_b), int(top_b), int(right_b), int(down_b)]

    
    def get_cat_ids(self, idx):
        return [self.video_infos[idx]['label']]

    def __getitem__(self, idx):
        # prob_mix = random.random()
        # if prob_mix < self.mix_up    and not self.test_mode:
        #     lam = random.random()
        #     mix_idx = random.randint(0, len(self)-1)
        #     a = super().__getitem__(idx)
        #     b = super().__getitem__(mix_idx)
        #     a['imgs'] = lam * a['imgs'] + (1 - lam) * b['imgs']
        #     a['label'] = (lam * a['label'] + (1 - lam) * b['label'])
        #     print("label", a['label'])
        #     return a
        return super().__getitem__(idx)
    
import sklearn.preprocessing
from .num_embeddings import compute_bins,PiecewiseLinearEncoding,UnaryEncoding,JohnsonEncoding,BinsEncoding
    
def num_enc_process(N_data,num_policy,n_bins=2,y_train=None,is_regression=False,encoder=None):
    """
    Process the numerical features in the dataset.

    :param N_data: ArrayDict
    :param num_policy: str
    :param n_bins: int
    :param y_train: Optional[np.ndarray]
    :param is_regression: bool
    :param encoder: Optional[PiecewiseLinearEncoding]
    :return: Tuple[ArrayDict, Optional[PiecewiseLinearEncoding]]
    """
    if N_data is not None:
        if num_policy == 'none':
            return N_data,None
        
        elif num_policy == 'Q_PLE':
            if encoder is None:
                bins = compute_bins(N_data, n_bins = n_bins,tree_kwargs = None,y=None,regression=None)
                encoder = PiecewiseLinearEncoding(bins)
            N_data = encoder(N_data).cpu().numpy()

        elif num_policy == 'T_PLE':
            if encoder is None:
                tree_kwargs = {'min_samples_leaf': 64, 'min_impurity_decrease': 1e-4}
                bins = compute_bins(N_data, n_bins = n_bins, tree_kwargs = tree_kwargs,y=torch.from_numpy(y_train),regression=is_regression)
                encoder = PiecewiseLinearEncoding(bins)
            N_data = encoder(N_data).cpu().numpy()
            
        elif num_policy == 'Q_Unary':
            if encoder is None:
                bins = compute_bins(N_data, n_bins = n_bins,tree_kwargs = None,y=None,regression=None)
                encoder = UnaryEncoding(bins)
            N_data = encoder(N_data).cpu().numpy()
            
        elif num_policy == 'T_Unary':
            if encoder is None:
                tree_kwargs = {'min_samples_leaf': 64, 'min_impurity_decrease': 1e-4}
                bins = compute_bins(N_data, n_bins = n_bins,tree_kwargs = tree_kwargs,y=torch.from_numpy(y_train),regression=is_regression)
                encoder = UnaryEncoding(bins)
            N_data = encoder(N_data).cpu().numpy()    
            
        elif num_policy == 'Q_bins':
            if encoder is None:
                bins = compute_bins(N_data, n_bins = n_bins,tree_kwargs = None,y=None,regression=None)
                encoder = BinsEncoding(bins)
            N_data = encoder(N_data).cpu().numpy()
            
        elif num_policy == 'T_bins':
            if encoder is None:
                tree_kwargs = {'min_samples_leaf': 64, 'min_impurity_decrease': 1e-4}
                bins = compute_bins(N_data, n_bins = n_bins,tree_kwargs = tree_kwargs,y=torch.from_numpy(y_train),regression=is_regression)
                encoder = BinsEncoding(bins)
            N_data = encoder(N_data).cpu().numpy()  
            
        elif num_policy == 'Q_Johnson':
            if encoder is None:
                bins = compute_bins(N_data, n_bins = n_bins,tree_kwargs = None,y=None,regression=None)
                encoder = JohnsonEncoding(bins)
            N_data = encoder(N_data).cpu().numpy()
            
        elif num_policy == 'T_Johnson':
            if encoder is None:
                tree_kwargs = {'min_samples_leaf': 64, 'min_impurity_decrease': 1e-4}
                bins = compute_bins(N_data, n_bins = n_bins,tree_kwargs = tree_kwargs,y=torch.from_numpy(y_train),regression=is_regression)
                encoder = JohnsonEncoding(bins)
            N_data = encoder(N_data).cpu().numpy()            
        
        return N_data,encoder
    else:
        return N_data,None

@DATASETS.register_module()
class ClassBalancedDataset(object):
    r"""A wrapper of repeated dataset with repeat factor.

    Suitable for training on class imbalanced datasets like LVIS. Following the
    sampling strategy in `this paper`_, in each epoch, an image may appear
    multiple times based on its "repeat factor".

    .. _this paper: https://arxiv.org/pdf/1908.03195.pdf

    The repeat factor for an image is a function of the frequency the rarest
    category labeled in that image. The "frequency of category c" in [0, 1]
    is defined by the fraction of images in the training set (without repeats)
    in which category c appears.

    The dataset needs to implement :func:`self.get_cat_ids` to support
    ClassBalancedDataset.

    The repeat factor is computed as followed.

    1. For each category c, compute the fraction :math:`f(c)` of images that
       contain it.
    2. For each category c, compute the category-level repeat factor

        .. math::
            r(c) = \max(1, \sqrt{\frac{t}{f(c)}})

    3. For each image I and its labels :math:`L(I)`, compute the image-level
       repeat factor

        .. math::
            r(I) = \max_{c \in L(I)} r(c)

    Args:
        dataset (:obj:`BaseDataset`): The dataset to be repeated.
        oversample_thr (float): frequency threshold below which data is
            repeated. For categories with ``f_c`` >= ``oversample_thr``, there
            is no oversampling. For categories with ``f_c`` <
            ``oversample_thr``, the degree of oversampling following the
            square-root inverse frequency heuristic above.
    """

    def __init__(self, dataset, oversample_thr):
        self.dataset = build_dataset(dataset)
        self.oversample_thr = oversample_thr
        # self.CLASSES = dataset.CLASSES
        

        repeat_factors = self._get_repeat_factors(self.dataset, oversample_thr)
        repeat_indices = []
        for dataset_index, repeat_factor in enumerate(repeat_factors):
            repeat_indices.extend([dataset_index] * math.ceil(repeat_factor))
        self.repeat_indices = repeat_indices

        flags = []
        if hasattr(self.dataset, 'flag'):
            for flag, repeat_factor in zip(self.dataset.flag, repeat_factors):
                flags.extend([flag] * int(math.ceil(repeat_factor)))
            assert len(flags) == len(repeat_indices)
        self.flag = np.asarray(flags, dtype=np.uint8)

    def _get_repeat_factors(self, dataset, repeat_thr):
        # 1. For each category c, compute the fraction # of images
        #   that contain it: f(c)
        category_freq = defaultdict(int)
        num_images = len(dataset)
        for idx in range(num_images):
            cat_ids = set(self.dataset.get_cat_ids(idx))
            for cat_id in cat_ids:
                category_freq[cat_id] += 1
        for k, v in category_freq.items():
            assert v > 0, f'caterogy {k} does not contain any images'
            category_freq[k] = v / num_images

        # 2. For each category c, compute the category-level repeat factor:
        #    r(c) = max(1, sqrt(t/f(c)))
        category_repeat = {
            cat_id: max(1.0, math.sqrt(repeat_thr / cat_freq))
            for cat_id, cat_freq in category_freq.items()
        }

        # 3. For each image I and its labels L(I), compute the image-level
        # repeat factor:
        #    r(I) = max_{c in L(I)} r(c)
        repeat_factors = []
        for idx in range(num_images):
            cat_ids = set(self.dataset.get_cat_ids(idx))
            repeat_factor = max(
                {category_repeat[cat_id]
                 for cat_id in cat_ids})
            repeat_factors.append(repeat_factor)

        return repeat_factors

    def __getitem__(self, idx):
        ori_index = self.repeat_indices[idx]
        return self.dataset[ori_index]

    def __len__(self):
        return len(self.repeat_indices)

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError(
            'evaluate results on a class-balanced dataset is weird. '
            'Please inference and evaluate on the original dataset.')

    def __repr__(self):
        """Print the number of instance number."""
        dataset_type = 'Test' if self.test_mode else 'Train'
        result = (
            f'\n{self.__class__.__name__} ({self.dataset.__class__.__name__}) '
            f'{dataset_type} dataset with total number of samples {len(self)}.'
        )
        return result