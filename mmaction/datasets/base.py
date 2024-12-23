# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict, defaultdict

import mmcv
import numpy as np
import torch
from mmcv.utils import print_log
from torch.utils.data import Dataset

from ..core import (mean_average_precision, mean_class_accuracy,
                    mmit_mean_average_precision, top_k_accuracy, confusion_matrix)
from .pipelines import Compose


class BaseDataset(Dataset, metaclass=ABCMeta):
    """Base class for datasets.

    All datasets to process video should subclass it.
    All subclasses should overwrite:

    - Methods:`load_annotations`, supporting to load information from an
    annotation file.
    - Methods:`prepare_train_frames`, providing train data.
    - Methods:`prepare_test_frames`, providing test data.

    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        data_prefix (str | None): Path to a directory where videos are held.
            Default: None.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        multi_class (bool): Determines whether the dataset is a multi-class
            dataset. Default: False.
        num_classes (int | None): Number of classes of the dataset, used in
            multi-class datasets. Default: None.
        start_index (int): Specify a start index for frames in consideration of
            different filename format. However, when taking videos as input,
            it should be set to 0, since frames loaded from videos count
            from 0. Default: 1.
        modality (str): Modality of data. Support 'RGB', 'Flow', 'Audio'.
            Default: 'RGB'.
        sample_by_class (bool): Sampling by class, should be set `True` when
            performing inter-class data balancing. Only compatible with
            `multi_class == False`. Only applies for training. Default: False.
        power (float): We support sampling data with the probability
            proportional to the power of its label frequency (freq ^ power)
            when sampling data. `power == 1` indicates uniformly sampling all
            data; `power == 0` indicates uniformly sampling all classes.
            Default: 0.
        dynamic_length (bool): If the dataset length is dynamic (used by
            ClassSpecificDistributedSampler). Default: False.
    """

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_prefix=None,
                 test_mode=False,
                 multi_class=False,
                 num_classes=None,
                 start_index=1,
                 modality='RGB',
                 sample_by_class=False,
                 power=0,
                 dynamic_length=False):
        super().__init__()

        self.ann_file = ann_file
        self.data_prefix = osp.realpath(
            data_prefix) if data_prefix is not None and osp.isdir(
                data_prefix) else data_prefix
        self.test_mode = test_mode
        self.multi_class = multi_class
        self.num_classes = num_classes
        self.start_index = start_index
        self.modality = modality
        self.sample_by_class = sample_by_class
        self.power = power
        self.dynamic_length = dynamic_length

        assert not (self.multi_class and self.sample_by_class)

        self.pipeline = Compose(pipeline)
        self.video_infos = self.load_annotations()
        if self.sample_by_class:
            self.video_infos_by_class = self.parse_by_class()

            class_prob = []
            for _, samples in self.video_infos_by_class.items():
                class_prob.append(len(samples) / len(self.video_infos))
            class_prob = [x**self.power for x in class_prob]

            summ = sum(class_prob)
            class_prob = [x / summ for x in class_prob]

            self.class_prob = dict(zip(self.video_infos_by_class, class_prob))

    @abstractmethod
    def load_annotations(self):
        """Load the annotation according to ann_file into video_infos."""

    # json annotations already looks like video_infos, so for each dataset,
    # this func should be the same
    def load_json_annotations(self):
        """Load json annotation file to get video information."""
        video_infos = mmcv.load(self.ann_file)
        num_videos = len(video_infos)
        path_key = 'frame_dir' if 'frame_dir' in video_infos[0] else 'filename'
        for i in range(num_videos):
            path_value = video_infos[i][path_key]
            if self.data_prefix is not None:
                path_value = osp.join(self.data_prefix, path_value)
            video_infos[i][path_key] = path_value
            if self.multi_class:
                assert self.num_classes is not None
            else:
                assert len(video_infos[i]['label']) == 1
                video_infos[i]['label'] = video_infos[i]['label'][0]
        return video_infos

    def parse_by_class(self):
        video_infos_by_class = defaultdict(list)
        for item in self.video_infos:
            label = item['label']
            video_infos_by_class[label].append(item)
        return video_infos_by_class

    @staticmethod
    def label2array(num, label):
        arr = np.zeros(num, dtype=np.float32)
        arr[label] = 1.
        return arr

    def evaluate(self,
                 results,
                 metrics='top_k_accuracy',
                 metric_options=dict(top_k_accuracy=dict(topk=(1, 5))),
                 logger=None,
                 **deprecated_kwargs):
        """Perform evaluation for common datasets.

        Args:
            results (list): Output results.
            metrics (str | sequence[str]): Metrics to be performed.
                Defaults: 'top_k_accuracy'.
            metric_options (dict): Dict for metric options. Options are
                ``topk`` for ``top_k_accuracy``.
                Default: ``dict(top_k_accuracy=dict(topk=(1, 5)))``.
            logger (logging.Logger | None): Logger for recording.
                Default: None.
            deprecated_kwargs (dict): Used for containing deprecated arguments.
                See 'https://github.com/open-mmlab/mmaction2/pull/286'.

        Returns:
            dict: Evaluation results dict.
        """
        # Protect ``metric_options`` since it uses mutable value as default
        metric_options = copy.deepcopy(metric_options)

        if deprecated_kwargs != {}:
            warnings.warn(
                'Option arguments for metrics has been changed to '
                "`metric_options`, See 'https://github.com/open-mmlab/mmaction2/pull/286' "  # noqa: E501
                'for more details')
            metric_options['top_k_accuracy'] = dict(
                metric_options['top_k_accuracy'], **deprecated_kwargs)

        if not isinstance(results, list):
            raise TypeError(f'results must be a list, but got {type(results)}')
        assert len(results) == len(self), (
            f'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self)}')

        metrics = metrics if isinstance(metrics, (list, tuple)) else [metrics]
        allowed_metrics = [
            'top_k_accuracy', 'mean_class_accuracy', 'mean_average_precision',
            'mmit_mean_average_precision', 'confusion_matrix', 'ceus_metric'
        ]

        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        eval_results = OrderedDict()
        gt_labels = [ann['label'] for ann in self.video_infos]

        ########## prepare input
        if len(results[0])==9: # 
            qual = [ann['qual'] for ann in self.video_infos]
            main_labels = np.array(gt_labels)       
            main_labels = np.eye(3)[main_labels] 
            side_labels = np.array(qual)  

            qual_labels = np.concatenate([main_labels, side_labels], axis=1)  
            
            qual_pred = results 
            # qual_pred = [array[3:] for array in results] #
            results = [array[:3] for array in results] # 

        for metric in metrics:
            msg = f'Evaluating {metric} ...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'top_k_accuracy':
                topk = metric_options.setdefault('top_k_accuracy',
                                                 {}).setdefault(
                                                     'topk', (1, 5))
                if not isinstance(topk, (int, tuple)):
                    raise TypeError('topk must be int or tuple of int, '
                                    f'but got {type(topk)}')
                if isinstance(topk, int):
                    topk = (topk, )

                top_k_acc = top_k_accuracy(results, gt_labels, topk)
                log_msg = []
                for k, acc in zip(topk, top_k_acc):
                    eval_results[f'top{k}_acc'] = acc
                    log_msg.append(f'\ntop{k}_acc\t{acc:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            if metric == 'mean_class_accuracy':
                mean_acc = mean_class_accuracy(results, gt_labels)
                eval_results['mean_class_accuracy'] = mean_acc
                log_msg = f'\nmean_acc\t{mean_acc:.4f}'
                print_log(log_msg, logger=logger)
                continue
            

            if metric in [
                    'mean_average_precision', 'mmit_mean_average_precision'
            ]:
                gt_labels_arrays = [
                    self.label2array(self.num_classes, label)
                    for label in gt_labels
                ]
                if metric == 'mean_average_precision':
                    mAP = mean_average_precision(results, gt_labels_arrays)
                    eval_results['mean_average_precision'] = mAP
                    log_msg = f'\nmean_average_precision\t{mAP:.4f}'
                elif metric == 'mmit_mean_average_precision':
                    mAP = mmit_mean_average_precision(results,
                                                      gt_labels_arrays)
                    eval_results['mmit_mean_average_precision'] = mAP
                    log_msg = f'\nmmit_mean_average_precision\t{mAP:.4f}'
                print_log(log_msg, logger=logger)
                continue
            
            if metric == 'confusion_matrix':
                pred = np.argmax(results, axis=1)
                cf_mat = confusion_matrix(pred, gt_labels).astype(float)
                print_log(cf_mat, logger=logger)
                continue
            
            if metric == 'ceus_ROC_plot':
                import pandas as pd
                import matplotlib.pyplot as plt
                specific_class = "malignant"
                mal_score = results[:, 1]
                mal_test =gt_labels(gt_labels==1)
                from sklearn.metrics import roc_curve, auc
                fpr, tpr, threshold = roc_curve(mal_test, mal_score)
                print("fpr:", fpr)
                print("\n")
                print("tpr:", tpr)
                plt.figure(figsize=(12, 8))
                plt.plot(fpr, tpr, linewidth=5, label=specific_class)
                # plt.plot([0, 1], [0, 1],ls="--", c='.3', linewidth=3, label='随机模型')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.0])
                plt.rcParams['font.size'] = 22
                plt.title('{} ROC曲线  AUC:{:.3f}'.format(specific_class, auc(fpr, tpr)))
                plt.xlabel('False Positive Rate (1 - Specificity)')
                plt.ylabel('True Positive Rate (Sensitivity)')
                plt.legend()
                plt.grid(True)

                plt.savefig('{}-ROC曲线.pdf'.format(specific_class), dpi=120, bbox_inches='tight')
                plt.show()
                
                mal_auc = auc(fpr, tpr)
                log_msg = f'\n AUC \t{mal_auc:.4f}'
                print_log(log_msg, logger=logger)
                continue
                

                
            if metric == 'ceus_metric':
                
                if qual is not None: # 
                    from .mlc_metric import multi_label_metrics

                    mlc_results, mlc_auc = [], []
                    ml_res, aucs = multi_label_metrics(qual_pred, qual_labels) 
                    mlc_auc.append(aucs)
                    mlc_results = np.asarray(mlc_results) * 100
                    all_auc = np.asarray(mlc_auc) * 100 
                    out_str = ' ~~~~~~~~~~~~~~ Final results! ~~~~~~~~~~~~~~~\n'
                    np.set_printoptions(precision=4, suppress=True)
                    out_str +='Multi-label: subacc, hloss, rloss, avgprec, rec, prec, f1, auc\n'
                    out_str +='~~~~| {}\n'.format(mlc_results)
                    out_str +='AUC results: \n'
                    out_str +='~~~~| {}\n'.format(mlc_auc)
                    print_log(out_str, logger=logger)
                
                
                pred = np.argmax(results, axis=1)
                cf_mat = confusion_matrix(pred, gt_labels).astype(float)
                # bin_cf_mat = np.zeros((2, 2))
                # bin_cf_mat[0, 0] = 
                acc = (cf_mat[0,0]+cf_mat[1,1]+cf_mat[2,2])/cf_mat.sum()
                sensitivity = cf_mat[1, 1] / cf_mat[1].sum()
                specificity = (cf_mat[0, 0] + cf_mat[2, 2] + cf_mat[0, 2] + cf_mat[2, 0])/(cf_mat[0] + cf_mat[2]).sum()
                pre_b = cf_mat[0,0]/cf_mat[:,0].sum()
                pre_mal = cf_mat[1,1]/cf_mat[:,1].sum()
                pre_mas = cf_mat[2,2]/cf_mat[:,2].sum()
                rec_b = cf_mat[0,0]/cf_mat[0].sum()
                rec_mal = cf_mat[1,1]/cf_mat[1].sum()
                rec_mas = cf_mat[2,2]/cf_mat[2].sum()
                f1_b=2*pre_b*rec_b / (pre_b+rec_b)
                f1_mal=2*pre_mal*rec_mal / (pre_mal+rec_mal)
                f1_mas=2*pre_mas*rec_mas / (pre_mas+rec_mas)
                f1_marco =(f1_b+f1_mal+f1_mas)/3
                
                #####################
                import pandas as pd
                from scipy import interp
                from sklearn.metrics import roc_curve, auc, roc_auc_score
                fpr=dict()
                tpr=dict()
                roc_auc=dict()
                # import matplotlib.pyplot as plt
                onehot_label= np.eye(3)[gt_labels]
                for i in range(3):
                    fpr[i], tpr[i], _ = roc_curve(onehot_label[:,i], np.array(results)[:,i])
                    #roc_auc[i] = auc(fpr[i], tpr[i])
                # 首先收集所有的假正率
                all_fpr = np.unique(np.concatenate([fpr[i] for i in range(3)]))
                # 然后在此点内插所有ROC曲线
                mean_tpr = np.zeros_like(all_fpr)
                for i in range(3):
                    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
                mean_tpr/= 3
                fpr["macro"] = all_fpr
                tpr["macro"] = mean_tpr
                roc_auc["macro"] = auc(fpr["macro"], tpr["macro"]) 
                roc_auc_macro = roc_auc_score(onehot_label, np.array(results), average ="macro")
                
                #########mirco
                fpr['micro'] ,tpr['micro'], _ = roc_curve(onehot_label.ravel(), np.array(results).ravel())
                roc_auc["micro"] = auc( fpr["micro"],tpr["micro"])
                roc_auc_micro = roc_auc_score(onehot_label, np.array(results), average ="micro") #multi_class="ovo"
                # mal_score = np.array(results)[:, 1]
                # mal_test = np.array(gt_labels)==1
                # from sklearn.metrics import roc_curve, auc
                # fpr, tpr, threshold = roc_curve(mal_test, mal_score)
                # mal_auc = auc(fpr, tpr)
                
                log_msg = f'{acc:.4f},{sensitivity:.4f},{specificity:.4f},{roc_auc["macro"]:.5f},{roc_auc_macro:.5f},{roc_auc["micro"]:.5f},{roc_auc_micro:.5f},{f1_marco:.4f},{f1_b:.4f},{f1_mal:.4f},{f1_mas:.4f},{pre_b:.4f},{pre_mal:.4f},{pre_mas:.4f},{rec_b:.4f},{rec_mal:.4f},{rec_mas:.4f}'
                    
                log_msg += '\nfpr:'
                for x in fpr['macro']:
                    log_msg += f'{x:.4f},' 
                log_msg += '\ntpr:'
                for x in tpr['macro']:
                    log_msg += f'{x:.4f},' 
                # log_msg = f'\nacc\t{acc:.4f},sensitivity\t{sensitivity:.4f},\tspecificity\t{specificity:.4f},\
                #             \nf1_marco\t{f1_marco:.4f},f1_b\t{f1_b:.4f},f1_mal\t{f1_mal:.4f},f1_mas\t{f1_mas:.4f},\
                #             \npre_b\t{pre_b:.4f},pre_mal\t{pre_mal:.4f},pre_mas\t{pre_mas:.4f},\
                #             \nrec_b\t{rec_b:.4f},rec_mal\t{rec_mal:.4f},rec_mas\t{rec_mas:.4f} '
                #             # print_log(cf_mat, logger=logger)
                print_log(log_msg, logger=logger)
                eval_results['sensitivity'] = sensitivity
                eval_results['auc'] = roc_auc['macro']
                continue
                

        return eval_results

    @staticmethod
    def dump_results(results, out):
        """Dump data to json/yaml/pickle strings or files."""
        return mmcv.dump(results, out)

    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        results['modality'] = self.modality
        results['start_index'] = self.start_index

        # prepare tensor in getitem
        # If HVU, type(results['label']) is dict
        if self.multi_class and isinstance(results['label'], list):
            onehot = torch.zeros(self.num_classes)
            onehot[results['label']] = 1.
            results['label'] = onehot

        return self.pipeline(results)

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        results['modality'] = self.modality
        results['start_index'] = self.start_index

        # prepare tensor in getitem
        # If HVU, type(results['label']) is dict
        if self.multi_class and isinstance(results['label'], list):
            onehot = torch.zeros(self.num_classes)
            onehot[results['label']] = 1.
            results['label'] = onehot

        return self.pipeline(results)

    def __len__(self):
        """Get the size of the dataset."""
        return len(self.video_infos)

    def __getitem__(self, idx):
        """Get the sample for either training or testing given index."""
        if self.test_mode:
            return self.prepare_test_frames(idx)

        return self.prepare_train_frames(idx)
