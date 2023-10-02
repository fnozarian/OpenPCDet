import torch
from torchmetrics import Metric
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.mixture import GaussianMixture
palettes = dict(zip(['fp', 'tn', 'tp', 'fn'], sns.color_palette("hls", 4)))
import pandas as pd
import warnings

# TODO: can we delay the teacher/student weight updates until all the reset_state_interval number of samples are obtained?
# TODO: Test the effect of the reset_state_interval on the performance
# TODO: Set the FG of lbl data based on GTs

def _lbl(tensor, mask=None):
    return _lbl(tensor)[_lbl(mask)] if mask is not None else tensor.chunk(2)[0]

def _ulb(tensor, mask=None):
    return _ulb(tensor)[_ulb(mask)] if mask is not None else tensor.chunk(2)[1]


class AdaMatch(Metric):
    """
        Adamatch based relative Thresholding
        mean conf. of the top-1 prediction on the weakly aug source data multiplied by a user provided threshold

        Adamatch based Dist. Alignment
        Rectify the target unlabeled pseudo-labels by multiplying them by the ratio of the expected
        value of the weakly aug source labels E[YcapSL;w] to the expected
        value of the target labels E[YcapTU;w], obtaining the final pseudo-labels YtildaTU;w

        REF: UPS FRAMEWORK DA
        probs_x_ulb_w = accumulated_metrics['pl_scores_wa_unlab'].view(-1)
        probs_x_lb_s = accumulated_metrics['pl_scores_wa_lab'].view(-1)
        self.p_model = self.momentum  * self.p_model + (1 - self.momentum) * torch.mean(probs_x_ulb_w)
        self.p_target = self.momentum  * self.p_target + (1 - self.momentum) * torch.mean(probs_x_lb_s)
        probs_x_ulb_aligned = probs_x_ulb_w * (self.p_target + 1e-6) / (self.p_model + 1e-6)
    """
    full_state_update: bool = False

    def __init__(self, **configs):
        super().__init__(**configs)

        self.reset_state_interval = configs.get('RESET_STATE_INTERVAL', 32)
        self.pre_filtering_thresh = configs.get('PRE_FILTERING_THRESH', 0.33)
        self.enable_plots = configs.get('ENABLE_PLOTS', False)
        self.relative_val = configs.get('RELATIVE_VAL', 0.9)
        self.momentum = configs.get('MOMENTUM', 0.9)
        self.temperature = configs.get('TEMPERATURE', 1.0)
        self.ulb_ratio = configs.get('ULB_RATIO', 0.5)
        self.states_name = ['sem_scores_wa', 'sem_scores_pre_gt_wa']
        self.class_names = ['Car', 'Pedestrian', 'Cyclist']
        self.iteration_count = 0

        # States are of shape (N, M, P) where N is # samples, M is # RoIs and P = 4 is the Car, Ped, Cyc, FG scores
        for name in self.states_name:
            self.add_state(name, default=[], dist_reduce_fx='cat')

        self.p_target = {s_name: None for s_name in self.states_name}
        self.p_model = {s_name: None for s_name in self.states_name}
        # P(max(y)|x, argmax(y)) on WA version of *entire* labeled data
        self.p_max_model_lbl = {s_name: None for s_name in self.states_name}
        self.ratio = {'default': None}

        # Two fixed targets dists
        self.p_target['uniform'] = torch.ones(len(self.class_names)) / len(self.class_names)
        self.p_target['gt'] = torch.tensor([0.85, 0.1, 0.05]).cuda()

        # GMM
        self.gmm_policy=configs.get('GMM_POLICY', 'high')
        self.mu1=configs.get('MU1', 0.1)
        self.mu2=configs.get('MU2', 0.9)
        self.gmm = GaussianMixture(
            n_components=2,
            weights_init=[0.5, 0.5],
            means_init=[[self.mu1], [self.mu2]],
            precisions_init=[[[1.0]], [[1.0]]],
            init_params='k-means++',
            tol=1e-9,
            max_iter=1000
        )

    def update(self, **kwargs):
        for state_name in self.states_name:
            value = kwargs.get(state_name)
            if value is not None:
                getattr(self, state_name).append(value)

    def _accumulate_metrics(self):
        bs = len(self.sem_scores_wa[0])  # TODO: Refactor
        accumulated_metrics = {}
        for mname in self.states_name:
            mstate = getattr(self, mname)
            if isinstance(mstate, list):
                mstate = torch.cat(mstate, dim=0)
            splits = torch.split(mstate, int(self.ulb_ratio * bs), dim=0)
            lbl = torch.cat(splits[::2], dim=0)
            ulb = torch.cat(splits[1::2], dim=0)
            mstate = torch.cat([lbl, ulb], dim=0)
            mstate = mstate.view(-1, mstate.shape[-1])
            accumulated_metrics[mname] = mstate

        return accumulated_metrics

    def compute(self):
        results = {}

        if len(self.sem_scores_pre_gt_wa) < self.reset_state_interval:
            return

        self.iteration_count += 1
        accumulated_metrics = self._accumulate_metrics()
        for sname in self.states_name:
            sem_scores_wa = accumulated_metrics[sname]

            max_scores, labels = torch.max(sem_scores_wa, dim=-1)
            fg_mask = max_scores > 0.33  # TODO: Make it dynamic. Also not the same for both labeled and unlabeled data

            fg_max_scores_lbl = _lbl(max_scores, fg_mask)
            fg_labels_lbl = _lbl(labels, fg_mask)

            p_max_model_lbl = labels.new_zeros(3, dtype=max_scores.dtype).scatter_add_(0, fg_labels_lbl, fg_max_scores_lbl)
            fg_labels_hist_lbl = torch.bincount(fg_labels_lbl, minlength=3)
            p_max_model_lbl /= (fg_labels_hist_lbl + 1e-6)
            self._update_ema('p_max_model_lbl', p_max_model_lbl, sname)
            results[f'p_max_model_lbl/{sname}'] = self._arr2dict(self.p_max_model_lbl[sname])

            fg_max_scores_ulb = _ulb(max_scores, fg_mask)
            fg_labels_ulb = _ulb(labels, fg_mask)

            # For debugging
            fg_cls_prob_hist_lbl = self._get_cls_hist_probs(fg_labels_lbl)
            fg_cls_prob_hist_ulb = self._get_cls_hist_probs(fg_labels_ulb)
            # Bincount/histogram approach (fg_cls_prob_hist_lbl) is the sharpened
            # or one-hot version of the mean approach (fg_cls_prob_mean_lbl)
            results[f'fg_cls_prob_hist_lbl_sharpened_p_target)/{sname}'] = self._arr2dict(fg_cls_prob_hist_lbl)
            results[f'fg_cls_prob_hist_ulb_sharpened_p_model)/{sname}'] = self._arr2dict(fg_cls_prob_hist_ulb)

            fg_cls_prob_mean_lbl = _lbl(sem_scores_wa, fg_mask).mean(dim=0)
            self._update_ema('p_target', fg_cls_prob_mean_lbl, sname)
            results[f'p_target/{sname}'] = self._arr2dict(self.p_target[sname])

            fg_cls_prob_mean_ulb = _ulb(sem_scores_wa, fg_mask).mean(dim=0)
            self._update_ema('p_model', fg_cls_prob_mean_ulb, sname)
            results[f'p_model/{sname}'] = self._arr2dict(self.p_model[sname])

            if self.enable_plots:
                fig = self.draw_dist_plots(fg_max_scores_lbl, fg_labels_lbl, fg_max_scores_ulb, fg_labels_ulb, sname)
                results[f'dist_plots_{sname}'] = fig
                plt.close()

        ratio =  self.p_target['sem_scores_pre_gt_wa'] / (self.p_model['sem_scores_wa'] + 1e-6)
        self._update_ema('ratio', ratio, 'default')
        results['ratio/pre_gt_wa_over_wa'] = self._arr2dict(self.ratio['default'])

        results['threshold/pre_gt_wa'] = self._get_threshold(tag='sem_scores_pre_gt_wa')

        self.reset()
        return results

    def draw_dist_plots(self, max_scores_lbl, labels_lbl, max_scores_ulb, labels_ulb, tag, meta_info=''):
        BS = len(self.sem_scores_wa[0])
        WS = self.reset_state_interval * BS
        info = (f"Iter: {self.iteration_count}    Interval: {self.reset_state_interval}    " +
                f"BS: {BS}    W: {(self.iteration_count - 1) * WS} - {self.iteration_count * WS}    M: {meta_info}")
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex='col', sharey='row', layout="compressed")
        plt.suptitle(info, fontsize='small')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fg_scores_labels_lbl_df = pd.DataFrame(
                torch.cat([max_scores_lbl.view(-1, 1), labels_lbl.view(-1, 1)], dim=-1).cpu().numpy(),
                columns=['scores', 'labels'])
            fg_scores_labels_ulb_df = pd.DataFrame(
                torch.cat([max_scores_ulb.view(-1, 1), labels_ulb.view(-1, 1)], dim=-1).cpu().numpy(),
                columns=['scores', 'labels'])
            sns.histplot(data=fg_scores_labels_lbl_df, ax=axes[0], x='scores', hue='labels', kde=True).set(
                title=f"Dist of FG max-scores on WA LBL input {tag} ")
            sns.histplot(data=fg_scores_labels_ulb_df, ax=axes[1], x='scores', hue='labels', kde=True).set(
                title=f"Dist of FG max-scores on WA ULB input {tag}")
            plt.tight_layout()
        # plt.show()

        return fig.get_figure()

    def _get_cls_hist_probs(self, fg_labels):
        c_fg_labels = torch.bincount(fg_labels, minlength=len(self.class_names))
        return c_fg_labels / c_fg_labels.sum()

    def rectify_sem_scores(self, sem_scores_ulb):

        if self.iteration_count == 0:
            print("Skipping rectification as iteration count is 0")
            return

        max_scores, labels = torch.max(sem_scores_ulb, dim=-1)
        fg_mask = max_scores > 0.33

        rect_scores = sem_scores_ulb * self.ratio['default']
        rect_scores /= rect_scores.sum(dim=-1, keepdims=True)
        sem_scores_ulb[fg_mask] = rect_scores[fg_mask]  # Only rectify FG rois

        return sem_scores_ulb

    def _update_ema(self, p_name, probs, tag):
        prob = getattr(self, p_name)
        prob[tag] = probs if prob[tag] is None else self.momentum * prob[tag] + (1 - self.momentum) * probs

    def get_mask(self, rect_scores):
        max_rect_scores, labels = torch.max(rect_scores, dim=-1)
        fg_mask = max_rect_scores > 0.33
        thresh_mask = max_rect_scores > self._get_threshold()
        return thresh_mask & fg_mask

    def _get_threshold(self, sem_scores_wa_lbl=None, tag='sem_scores_pre_gt_wa'):
        if sem_scores_wa_lbl is None:
            return self.p_max_model_lbl[tag].mean() * self.relative_val
        max_scores, labels = torch.max(sem_scores_wa_lbl, dim=-1)
        fg_mask = max_scores > 0.33
        fg_max_scores_lbl = max_scores[fg_mask]
        return fg_max_scores_lbl.mean() * self.relative_val

    def _arr2dict(self, array):
        return {cls: array[cind] for cind, cls in enumerate(self.class_names)}