import numpy as np
import torch

from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize
from sklearn.metrics import hamming_loss, zero_one_loss, coverage_error
from sklearn.metrics import average_precision_score, label_ranking_loss
from sklearn.metrics import multilabel_confusion_matrix

def compute_multilabel_confusion_matrix(targs, preds):
    """
    计算多标签分类的混淆矩阵。

    参数：
    - targs: 真实标签，形状为 (n_samples, n_classes)
    - preds: 预测标签，形状为 (n_samples, n_classes)

    返回：
    - conf_mat: 混淆矩阵，形状为 (n_classes, 2, 2)
    """
    conf_mat = multilabel_confusion_matrix(targs, preds)
    return conf_mat

def multi_label_metrics(probs, targs):
    # probs = probs.cpu().data.numpy()
    # targs = targs.cpu().data.numpy()
    probs = np.array(probs)
    targs = np.array(targs)
    
    if probs.shape[1]==9:
        probs[:,:3] = to_one_hot(probs[:,:3])

    preds = (probs > 0.5).astype(int) 
    cf_mat = compute_multilabel_confusion_matrix(targs,preds).astype(float)
    print(cf_mat)
    n_tasks = targs.shape[1]
    label = np.arange(2)

    ham_l = hamming_loss(targs, preds)
    zero_one = zero_one_loss(targs, preds)
    sub_acc = 1. - zero_one
    # cover = coverage_error(targs, probs)
    rank_l = label_ranking_loss(targs, probs)
    avgprec = average_precision_score(targs, probs)

    results = []
    overall = []

    for i in range(n_tasks):
        targ = targs[:, i]
        prob = probs[:, i]
        pred = preds[:, i]

        cm = confusion_matrix(targ, pred, labels=label).ravel()
        overall.append(cm)
        tn, fp, fn, tp = cm
        try:
            reca = tp / (tp + fn) if tp+fn != 0. else 0.
            prec = tp / (tp + fp) if tp+fp != 0. else 0.
            spec = tn / (tn + fp) if tn+fp != 0. else 0.
            f1 = 2*reca*prec/(reca+prec) if reca+prec != 0. else 0.
        except:
            import pdb; pdb.set_trace()

        acc = (tn + tp) / (tn + tp + fn + fp)
        auc = roc_auc_score(y_true=targ, y_score=prob)
        
        current = [acc, reca, prec, spec, f1, auc]
        results.append(current)
    results = np.asarray(results)
    aucs = results[..., -1]
    mean = np.mean(results, axis=0)
    pc_recall = mean[1]
    pc_precision = mean[2]

    mean_acc = mean[0]
    macro_f1 = mean[4]
    mean_auc = mean[5]

    return np.asarray([sub_acc, ham_l, rank_l, avgprec, 
            pc_recall, pc_precision, macro_f1, mean_auc]), aucs



def to_one_hot(predictions):
    """
    将预测概率转换为独热编码。
    
    参数:
    - predictions: NumPy数组,形状为 (n_samples, n_classes)
    
    返回:
    - one_hot: NumPy数组,形状为 (n_samples, n_classes)，独热编码形式
    """
    one_hot = np.zeros_like(predictions)
    # 对每个样本，找到概率最大的类别索引
    max_indices = np.argmax(predictions, axis=1)
    # 将对应位置设为1
    one_hot[np.arange(len(predictions)), max_indices] = 1
    return one_hot

