
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.stats import norm

from torchmetrics import Metric
from pcdet.config import cfg
from .AdaptiveThresholdingGMM import adaptive_thresholding_gmm

class SimplifiedAdaptiveThreshMetrics(Metric):
    full_state_update: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reset_state_interval = kwargs.get('RESET_STATE_INTERVAL', 64)
        self.percent = kwargs.get('PERCENT', 0.1)
        self.pre_filter_thresh = kwargs.get('PRE_FILTERING_THRESH', 0.25)

        # TODO(farzad) To be cautious with dist mode I impl. it simple. Refactor it later to be more general.
        self.add_state("roi_ious_car", default=[], dist_reduce_fx='cat')
        self.add_state("roi_ious_ped", default=[], dist_reduce_fx='cat')
        self.add_state("roi_ious_cyc", default=[], dist_reduce_fx='cat')
        self.add_state("global_sample_count", default=torch.tensor(0, dtype=torch.int), dist_reduce_fx="sum")

    def update(self, batch_roi_labels: torch.Tensor, batch_iou_wrt_pl: torch.Tensor) -> None:

        assert batch_roi_labels.ndim == 2

        self.roi_ious_car.append(batch_iou_wrt_pl[batch_roi_labels == 1].view(-1))
        self.roi_ious_ped.append(batch_iou_wrt_pl[batch_roi_labels == 2].view(-1))
        self.roi_ious_cyc.append(batch_iou_wrt_pl[batch_roi_labels == 3].view(-1))
        self.global_sample_count += batch_roi_labels.shape[0]

    def compute(self):
        results = {}
        threshs = []
        if self.global_sample_count >= self.reset_state_interval:
            fig, axs = plt.subplots(1, 3, figsize=(10, 3), gridspec_kw={'wspace': 0.5})
            for i, mstate in enumerate([self.roi_ious_car, self.roi_ious_ped, self.roi_ious_cyc]):
                if isinstance(mstate, torch.Tensor):
                    mstate = [mstate]
                roi_ious_pl = torch.cat(mstate, dim=0)
                filter_mask = roi_ious_pl > self.pre_filter_thresh
                roi_ious_pl = roi_ious_pl[filter_mask]
                if roi_ious_pl.shape[0] == 0:
                    threshs.append(0.0)
                    continue
                # Fit a Gaussian and draw densities
                mu, sigma = norm.fit(roi_ious_pl.cpu().numpy())
                thresh = norm.ppf(self.percent, loc=mu, scale=sigma)
                threshs.append(thresh)
                axs[i].hist(roi_ious_pl.cpu().numpy(), density=True, alpha=0.6, color='g')
                x = np.linspace(0, 1, 100)
                p = norm.pdf(x, mu, sigma)
                axs[i].plot(x, p, 'k', alpha=0.5)

            results['adapt_threshs_fig'] = fig.get_figure()
            plt.close()
            results['pre_adaptive_thresh'] = threshs

            self.reset()
        return results

## TODO (danish) here we have 
# batch_roi_labels = self.forward_ret_dict['roi_labels'][unlabeled_inds].detach().clone() # (Proposals) ROI info
# batch_roi_ious = self.forward_ret_dict['gt_iou_of_rois'][unlabeled_inds].detach().clone()
# metric_inputs = {'batch_roi_labels': batch_roi_labels, 'batch_iou_wrt_pl': batch_roi_ious}
# We could also collect labeled ious and correspondings labels(GT)
# We can use DIST-ALIGNMENT (softmatch) in order to treat ious as probs. 
# Moreover could also use for initialisation of unlabeled-thresholds
class CombinedAdaptiveThreshold(Metric):
    full_state_update: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reset_state_interval = kwargs.get('RESET_STATE_INTERVAL', 64)
        self.percent = kwargs.get('PERCENT', 0.1)
        self.pre_filter_thresh = kwargs.get('PRE_FILTERING_THRESH', 0.25)
        self.tag = kwargs.get('tag', None)
        self.dataset = kwargs.get('dataset', None)
        self.quantile= kwargs.get('quantile', False)
        self.momentum= kwargs.get('momentum', 0.99)
        self.enable_clipping = kwargs.get('enable_clipping', True)
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
        default_p_g = torch.tensor(1.0 / self.num_classes)
        default_p_l = torch.ones((self.num_classes)) / self.num_classes

        # adamatch
        self.tau= kwargs.get('tau', 0.9)
        self.adamatch_relative_threshold=default_p_g
        self.adamatch_relative_threshold_ema=default_p_g

        # freematch
        self.freematch_global_thresh_ema =  default_p_g
        self.freematch_mu_t_ema = default_p_l
        self.freematch_local_thresh = default_p_l
        self.freematch_local_thresh_ema = default_p_l

        # softmatch: Gaussian function for sample weighting
        self.n_sigma = kwargs.get('n_sigma', 2)
        self.lambdax = kwargs.get('lambdax', 0.7)
        self.softmatch_max_mu_t_ema = default_p_g
        self.softmatch_max_var_t_ema =  torch.tensor(1.0)
        self.softmatch_mu_t_ema = default_p_l
        self.softmatch_var_t_ema =  torch.ones((self.num_classes))
        self.softmatch_global_thresh = default_p_g
        self.softmatch_local_thresh = default_p_l
        self.softmatch_global_thresh_ema = default_p_g
        self.softmatch_local_thresh_ema = default_p_l
        # consistent-teacher GMM Policy parameters
        self.cons_teacher_local_thresh_ema = default_p_l
        self.cons_teacher_local_thresh_ema = default_p_l
        self.topK = kwargs.get('topK', 128)
        self.gmm_policy=kwargs.get('gmm_policy','high')


    def update(self, batch_roi_labels: torch.Tensor, batch_iou_wrt_pl: torch.Tensor) -> None:
        if batch_roi_labels.ndim == 1: # Unsqueeze for DDP
            batch_roi_labels=batch_roi_labels.unsqueeze(dim=0)
        if batch_iou_wrt_pl.ndim == 1: # Unsqueeze for DDP
            batch_iou_wrt_pl=batch_iou_wrt_pl.unsqueeze(dim=0)
        self.iou_scores.append(batch_iou_wrt_pl)
        self.labels.append(batch_roi_labels)


    def compute(self):
        results = {}
        if not cfg.MODEL.ROI_HEAD.ADAPTIVE_THRESH_CONFIG.get('ENABLE', False): return
        cls_wise_ious = get_cls_wise_ious(self.iou_scores[-1], self.labels[-1], fill_value=0.0, num_classes=self.num_classes)
        if cls_wise_ious is None:
            return
        max_ious, _ = torch.max(cls_wise_ious, dim=-1)
        max_mu_t = torch.quantile(max_ious, 0.8) if self.quantile else torch.mean(max_ious)
        max_var_t = torch.var(max_ious, unbiased=True)

        # Global Thresholds
        if not (max_mu_t==0 or torch.isnan(max_mu_t)):
            self.adamatch_relative_threshold = max_mu_t * self.tau # same as cls_wise_ious.max(1)[0].mean(0)*self.tau 
            self.adamatch_relative_threshold_ema =  self.momentum * self.adamatch_relative_threshold_ema + (1 - self.momentum)  * max_mu_t * self.tau  
            self.freematch_global_thresh_ema = self.momentum * self.freematch_global_thresh_ema  + (1 - self.momentum) * max_mu_t
            self.softmatch_max_mu_t_ema = self.momentum * self.softmatch_max_mu_t_ema + (1 - self.momentum) * max_mu_t
            self.softmatch_max_var_t_ema = self.momentum * self.softmatch_max_var_t_ema + (1 - self.momentum) * max_var_t


        self.softmatch_global_thresh = self.lambdax * compute_softmatch_weights(max_ious, self.softmatch_max_mu_t_ema, self.softmatch_max_var_t_ema, n_sigma=self.n_sigma)
        self.softmatch_global_thresh_ema =  self.momentum * self.softmatch_global_thresh_ema + (1 - self.momentum) * self.softmatch_global_thresh
        # Local Thresholds

        """
        Freematch
        Tt eq: 5: estimates a global threshold as the EMA of the confidence from the model
        ptilda_c eq: 6: expectation of the model's predictions on each class c to estimate the class-specific learning status
        Tt_c = maxNorm(ptilda_c) * Tt
        Softmatch
        truncated Gaussian distribution of mean µt and variance alphat-t
        within the range [0, λmax], on the confidence max(p)
        Consistent-Teacher
        high: the class scores that belong to +ve component (index 1) and 
        greater than or equal to the class score at the maximum index are identified. 
        The minimum of these class scores is used  (see: adaptive_thresholding_gmm)
        Note update with default (discard all thresh < default) will move threshold leftwords
        """
        pos_thr = np.ones((self.num_classes), dtype=np.float32) / self.num_classes  

        for i in range(self.num_classes):
            class_scores = cls_wise_ious[i, :]
            class_scores = class_scores[class_scores>0]

            if class_scores.shape[0] > 0:
                mu = torch.mean(class_scores)
                var = torch.var(class_scores, unbiased=True)
                self.softmatch_var_t_ema[i] =  self.momentum * self.softmatch_var_t_ema[i] + (1 - self.momentum) * (var if not torch.isnan(var) else 0.0)
                self.freematch_mu_t_ema[i] = self.momentum * self.freematch_mu_t_ema[i] + (1 - self.momentum) * (mu if not torch.isnan(mu) else 0.0)
                self.softmatch_mu_t_ema[i] =  self.momentum * self.softmatch_mu_t_ema[i] + (1 -  self.momentum) * (mu if not torch.isnan(mu) else 0.0)
                adaptive_thresholding_gmm(class_scores, pos_thr[i:i+1], topK = self.topK, gmm_policy=self.gmm_policy) # inplace function

        self.freematch_local_thresh =  self.freematch_global_thresh_ema * self.freematch_mu_t_ema / torch.max(self.freematch_mu_t_ema)
        self.freematch_local_thresh_ema =  self.momentum  * self.freematch_local_thresh_ema + (1 - self.momentum) * self.freematch_local_thresh 

        self.softmatch_local_thresh =  self.lambdax * compute_softmatch_weights(max_ious, 
                                                                self.softmatch_mu_t_ema, 
                                                                self.softmatch_var_t_ema, 
                                                                n_sigma=self.n_sigma)
        self.softmatch_local_thresh_ema =  self.momentum  * self.softmatch_local_thresh_ema + (1 - self.momentum) * self.softmatch_local_thresh

        self.cons_teacher_local_thresh_ema = self.momentum  * self.cons_teacher_local_thresh_ema + (1 - self.momentum) * pos_thr


        if self.enable_clipping:
            self.adamatch_relative_threshold_ema = torch.clip(self.adamatch_relative_threshold_ema, 0.1, 0.9)
            self.freematch_global_thresh_ema = torch.clip(self.freematch_global_thresh_ema, 0.1, 0.9)
            self.softmatch_global_thresh_ema = torch.clip(self.softmatch_global_thresh_ema, 0.1, 0.9)
            self.freematch_local_thresh_ema = torch.clip(self.freematch_local_thresh_ema, 0.1, 0.9)
            self.softmatch_local_thresh_ema = torch.clip(self.softmatch_local_thresh_ema, 0.1, 0.9)
            self.cons_teacher_local_thresh_ema = torch.clip(self.cons_teacher_local_thresh_ema, 0.1, 0.9)


        results = {
            'baseline_global': {cls_name: self.pre_filter_thresh for i, cls_name in self.class_names.items()},
            'adamatch_global': {cls_name: self.adamatch_relative_threshold_ema.item() for i, cls_name in self.class_names.items()},
            'freematch_global': {cls_name: self.freematch_global_thresh_ema.item() for i, cls_name in self.class_names.items()},
            'softmatch_global': {cls_name: self.softmatch_global_thresh_ema[i].item() for i, cls_name in self.class_names.items()},
            'freematch_local': {cls_name: self.freematch_local_thresh_ema[i].item() for i, cls_name in self.class_names.items()},
            'softmatch_local': {cls_name: self.softmatch_local_thresh_ema[i].item() for i, cls_name in self.class_names.items()},
            'cons_teacher_local': {cls_name: self.cons_teacher_local_thresh_ema[i].item() for i, cls_name in self.class_names.items()}
            }


        if len(self.iou_scores) >= self.reset_state_interval:

            cls_wise_ious = get_cls_wise_ious(self.iou_scores, self.labels, fill_value=0.0, num_classes=self.num_classes)
            if cls_wise_ious is None:
                return
            adaptive_thresh = {}
            for key in results.keys():
                adaptive_thresh[key + '_adaptive_thresh'] =  {cls_name: 0.0 for i, cls_name in self.class_names.items()}

            fig, axs = plt.subplots(nrows=len(adaptive_thresh), ncols=self.num_classes,  figsize=(3*len(results), 2*len(results)), gridspec_kw={'hspace': 0.35, 'wspace': 0.35}, sharex=True, sharey=False)
            x = np.linspace(0, 1, 100)
            colors = list('bgrcmyk')
            for i, key in enumerate(results.keys()):
                for j in range(self.num_classes):
                    cls_name=self.class_names[j]
                    axs[i][j].set_title(f"{cls_name}:{key}", fontsize='x-small')
                    cls_roi_ious_pl = cls_wise_ious[j]
                    roi_ious_pl = cls_roi_ious_pl[cls_roi_ious_pl > results[key][cls_name]].numpy()
                    if roi_ious_pl.shape[0] > 0:
                        mu, sigma = norm.fit(roi_ious_pl)
                        adaptive_thresh[key + '_adaptive_thresh'][cls_name] = norm.ppf(self.percent, loc=mu, scale=sigma)
                        p = norm.pdf(x, mu, sigma)
                        axs[i][j].hist(roi_ious_pl, density=True, alpha=0.4, color=colors[i])
                        axs[i][j].plot(x, p, color=colors[i], alpha=0.6, label=f'{key}-gauss')

                    axs[i][j].axvline(x= results[key][cls_name], color=colors[i], linestyle='--', label=f'{key}')
                    axs[i][j].legend(loc='upper right', framealpha=0.4, fontsize='x-small')
            # fig, axs = plt.subplots(3,1, figsize=(11, 11), gridspec_kw={'hspace': 0.25}, sharex=True)
            # x = np.linspace(0, 1, 100)
            # colors = list('bgrcmyk')
            # for i, cls_name in self.class_names.items():
            #     axs[i].set_title(f"{cls_name}", fontsize='x-small')
            #     for j, key in enumerate(results.keys()):
            #         cls_roi_ious_pl = cls_wise_ious[i]
            #         roi_ious_pl = cls_roi_ious_pl[cls_roi_ious_pl > results[key][cls_name]].numpy()
            #         if roi_ious_pl.shape[0] > 0:
            #             # Fit a Gaussian and draw densities
            #             mu, sigma = norm.fit(roi_ious_pl)
            #             adaptive_thresh[key + '_adaptive_thresh'][cls_name] = norm.ppf(self.percent, loc=mu, scale=sigma)
            #             p = norm.pdf(x, mu, sigma)
            #             axs[i].hist(roi_ious_pl, density=True, alpha=0.4, color=colors[j])
            #             axs[i].plot(x, p, color=colors[j], alpha=0.6, label=f'{key}-gauss')

            #         axs[i].axvline(x= results[key][cls_name], color=colors[j], linestyle='--', label=f'{key}')
            #         axs[i].legend(loc='upper right', framealpha=0.4, fontsize='x-small')
            self.reset()               
            results.update(**adaptive_thresh)
            results.update({'densities':fig.get_figure()})
            plt.close()
        return results



def get_cls_wise_ious(ious_, labels_, fill_value=0.0, num_classes=3):
    """
    Args:
        ious_ (List[torch.Tensor]): List of IoU score tensors.
        labels_ (List[torch.Tensor]): List of label tensors.
        fill_value (float): Value used for padding. Default is 0.0.
        num_classes (int): Number of classes. Default is 3.
    Returns:
        torch.Tensor: Class-wise IoU tensor of shape (num_classes, max_len), where max_len is the maximum
        length of class-wise IoU scores after padding.
    """
    if isinstance(ious_, torch.Tensor):
        ious_ = [ious_]
    if isinstance(labels_, torch.Tensor):
        labels_ = [labels_]

    all_iou = [i.detach().cpu() for i in ious_]
    all_label = [i.detach().cpu() for i in labels_]
    ious = torch.cat(all_iou, dim=0)
    labels = torch.cat(all_label, dim=0)
    valid_mask = ious != 0
    if not valid_mask.any():
        return None
    ious = ious[valid_mask]
    labels = labels[valid_mask]
    labels -= 1
    cls_wise_ious = [ious[labels == cind] for cind in range(num_classes)]
    max_len = max([t.shape[0] for t in cls_wise_ious])
    cls_wise_ious = [torch.cat([t, torch.full((max_len - t.shape[0],), fill_value)]) for t in cls_wise_ious]
    return torch.stack(cls_wise_ious)




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