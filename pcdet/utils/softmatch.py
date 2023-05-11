
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.stats import norm

from torchmetrics import Metric
from pcdet.config import cfg

class AdaptiveThresholding(Metric):
    full_state_update: bool = False

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.percent = kwargs.get('PERCENT', 0.1)
        self.pre_filter_thresh = kwargs.get('PRE_FILTERING_THRESH', 0.25)
        self.tag = kwargs.get('tag', None)
        self.dataset = kwargs.get('dataset', None)
        self.quantile= kwargs.get('quantile', False)
        self.momentum= kwargs.get('momentum', 0.9)
        self.enable_clipping = kwargs.get('enable_clipping', True)
        self.metrics_name = ['batchwise_mean','batchwise_variance','ema_mean','ema_variance']
        self.config = kwargs['config']
        self.bg_thresh = self.config.ROI_HEAD.TARGET_CONFIG.CLS_BG_THRESH
        self.reset_state_interval = self.config.ROI_HEAD.ADAPTIVE_THRESH_CONFIG.RESET_STATE_INTERVAL
        if self.dataset is not None:
            self.class_names  = self.dataset.class_names
            if isinstance(self.class_names, list):
                self.class_names = {i: self.dataset.class_names[i] for i in range(len(self.dataset.class_names))}
            self.num_classes = len(self.dataset.class_names)
        else:
            self.class_names = {0: 'Car', 1: 'Pedestrian', 2: 'Cyclist'}
            self.num_classes = 3

        self.add_state("iou_scores", default=[], dist_reduce_fx='cat')
        self.add_state("labels", default=[], dist_reduce_fx='cat')
        self.st_mean = torch.ones(self.num_classes) / self.num_classes     
        self.st_var = torch.ones(self.num_classes)

        self.batch_mean = torch.zeros(self.num_classes) 
        self.batch_var = torch.ones(self.num_classes)

    def update(self, roi_labels: torch.Tensor, iou_wrt_pl: torch.Tensor) -> None:
        if roi_labels.ndim == 1: # Unsqueeze for DDP
            roi_labels=roi_labels.unsqueeze(dim=0)
        if iou_wrt_pl.ndim == 1: # Unsqueeze for DDP
            iou_wrt_pl=iou_wrt_pl.unsqueeze(dim=0)            
        self.iou_scores.append(iou_wrt_pl)
        self.labels.append(roi_labels)    


    def compute(self):
        results = {}
        classwise_metrics = {}
        num_classes = len(self.dataset.class_names)
        # if not cfg.MODEL.ROI_HEAD.ADAPTIVE_THRESH_CONFIG.get('ENABLE', False): return
        if len(self.iou_scores) >= self.reset_state_interval:
        # cls_wise_ious = get_cls_wise_ious(self.iou_scores, self.labels, fill_value=0.0, num_classes=self.num_classes)
            cls_wise_iou_mean_ = []
            cls_wise_iou_var_ = []
            cls_wise_thresholded = []
            all_iou = [i.detach().cpu() for i in self.iou_scores]
            all_label = [i.detach().cpu() for i in self.labels]
            ious = torch.cat(all_iou, dim=0)
            labels = torch.cat(all_label, dim=0)
            valid_mask = ious != 0
            if not valid_mask.any():
                return None
            ious = ious[valid_mask]
            labels = labels[valid_mask]
            labels -= 1
            cls_wise_ious = [ious[labels == cind] for cind in range(self.num_classes)]
            cls_wise_thresholded = [cls_wise_ious[cind][cls_wise_ious[cind] >= self.bg_thresh] for cind in range(self.num_classes)]            
            for i in  range(len(cls_wise_ious)):
                    cls_wise_iou_mean_.append(cls_wise_thresholded[i].mean().clone())
                    cls_wise_iou_var_.append(cls_wise_thresholded[i].var(unbiased=True).clone())
            #NOTE: mean of empty tensor is nan,common among tail classes
            self.batch_mean = torch.stack(cls_wise_iou_mean_).nan_to_num(nan=0.0).clone()
            self.batch_var = torch.stack(cls_wise_iou_var_).clone()
            for cind in range(num_classes):
                self.batch_var[cind] = self.batch_var[cind].nan_to_num(nan=self.st_var[cind])

            self.st_mean = self.momentum*(self.st_mean) + (1-self.momentum)*self.batch_mean
            self.st_var = self.momentum*(self.st_var) + (1-self.momentum)*self.batch_var
            self.st_mean = torch.clamp(self.st_mean, min=0.25,max=0.90).clone()
            self.st_var = torch.clamp(self.st_var,min=0.0).clone()
            classwise_metrics={}
            for metric_name in self.metrics_name:
                classwise_metrics[metric_name] = all_iou[0].new_zeros(self.num_classes).fill_(float('nan'))
            for cind in range(num_classes):
                classwise_metrics['batchwise_mean'][cind] = self.batch_mean[cind].item()
                classwise_metrics['batchwise_variance'][cind] = self.batch_var[cind].item()
                classwise_metrics['ema_mean'][cind] = self.st_mean[cind].item()
                classwise_metrics['ema_variance'][cind] = self.st_var[cind].item()
            self.reset()

        else:
            classwise_metrics = {}
            for metric_name in self.metrics_name:
               classwise_metrics[metric_name] = self.iou_scores[0].new_zeros(num_classes).fill_(float('nan'))

        classwise_results = {}
        for key in self.metrics_name: 
            for cind,cls in enumerate(self.dataset.class_names):
                #if all values are nan, then return a list with nan values(gets filtered in train_utils)
                if torch.all(classwise_metrics[key].isnan() == True):
                    results[key] = self.iou_scores[0].new_zeros(1).fill_(float('nan'))
                else:
                    classwise_results[cls] = classwise_metrics[key][cind]
                    results[key] = classwise_results
        return results




def compute_softmatch_weights(max_ious, mu_t, var_t, n_sigma=2):
    """
    Compute SoftMatch weights based on maximum IoU values and mean/variance parameters.
    Args:
        max_ious (torch.Tensor): Maximum IoU values, shape (N,)
        mu_t (torch.Tensor): Mean parameter, shape (N,)
        var_t (torch.Tensor): Variance parameter, shape (N,)
        n_sigma (float): Scaling factor for variance, default is 2.
    Returns:
        weights (torch.Tensor): SoftMatch weights, shape (N,)
    """
    diff = torch.clamp(max_ious - mu_t, max=0.0) ** 2
    scaled_var_t = var_t / (n_sigma ** 2)
    weights = torch.exp(-diff / (2 * scaled_var_t))

    return weights

def nanmean(v: torch.Tensor, *args, allnan=np.nan, **kwargs) -> torch.Tensor:
    """
    :param v: tensor to take mean
    :param dim: dimension(s) over which to take the mean
    :param allnan: value to use in case all values averaged are NaN.
        Defaults to np.nan, consistent with np.nanmean.
    :return: mean.
    """
    def isnan(v):
        if v.dtype is torch.long:
            return v == torch.tensor(np.nan).long()
        else:
            return torch.isnan(v)
    v = v.clone()
    is_nan = isnan(v)
    v[is_nan] = 0

    if np.isnan(allnan):
        return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)
    else:
        sum_nonnan = v.sum(*args, **kwargs)
        n_nonnan = float(~is_nan).sum(*args, **kwargs)
        mean_nonnan = torch.zeros_like(sum_nonnan) + allnan
        any_nonnan = n_nonnan > 1
        mean_nonnan[any_nonnan] = (
                sum_nonnan[any_nonnan] / n_nonnan[any_nonnan])
        return mean_nonnan