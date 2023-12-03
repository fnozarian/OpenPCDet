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
    return _lbl(tensor)[_lbl(mask)] if mask is not None else tensor.chunk(2)[0].squeeze(0)

def _ulb(tensor, mask=None):
    return _ulb(tensor)[_ulb(mask)] if mask is not None else tensor.chunk(2)[1].squeeze(0)

def _get_cls_dist(labels):
    cls_counts = labels.int().bincount(minlength=4)[1:]
    return cls_counts / cls_counts.sum()

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
        self.thresh_method = configs.get('THRESH_METHOD', 'AdaMatch')
        self.target_to_align = configs.get('TARGET_TO_ALIGN', 'gt_labels_pre_gt_wa')
        self.lab_thresh_tag = configs.get('LBL_THRESH_TAG','sem_scores_wa')
        self.prior_sem_fg_thresh = configs.get('SEM_FG_THRESH', 0.5)
        self.enable_plots = configs.get('ENABLE_PLOTS', False)
        self.fixed_thresh = configs.get('FIXED_THRESH', 0.9)
        self.momentum = configs.get('MOMENTUM', 0.9)
        self.temperature = configs.get('TEMPERATURE', 1)  # TODO: Both temperatures should be tuned
        self.temperature_sa = configs.get('TEMPERATURE_SA', 1)
        self.ulb_ratio = configs.get('ULB_RATIO', 0.5)
        self.states_name = ['sem_scores_wa', 'conf_scores_wa', 'roi_ious_wa', 'gt_labels_wa',
                            'sem_scores_sa', 'conf_scores_sa', 'box_cls_labels_sa', 'gt_labels_sa',
                            'sem_scores_pre_gt_wa', 'conf_scores_pre_gt_wa', 'roi_ious_pre_gt_wa', 'gt_labels_pre_gt_wa']
        self.class_names = ['Car', 'Pedestrian', 'Cyclist']
        self.iteration_count = 0

        # States are of shape (N, M, P) where N is # samples, M is # RoIs and P = 4 is the Car, Ped, Cyc, FG scores
        for name in self.states_name:
            self.add_state(name, default=[], dist_reduce_fx='cat')

        # mean_p_model aka p_target (_lbl(mean_p_model)) and p_model (_ulb(mean_p_model))
        self.mean_p_model = {s_name: None for s_name in self.states_name}
        self.mean_p_max_model = {s_name: None for s_name in self.states_name}
        self.labels_hist = {s_name: None for s_name in self.states_name}
        self.ratio = {'AdaMatch': None}

        # Two fixed targets dists
        self.mean_p_model['uniform'] = torch.ones(len(self.class_names)) / len(self.class_names)
        self.mean_p_model['gt'] = torch.tensor([0.85, 0.1, 0.05]).cuda()

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
            if not len(mstate): continue
            if isinstance(mstate, list):
                try:
                    mstate = torch.cat(mstate, dim=0)
                except RuntimeError:
                    # pad the second dim of each element of the list with zeros to make them of the same size
                    max_len = max(m.shape[1] for m in mstate)
                    mstate = [torch.cat([m, m.new_zeros((m.shape[0], max_len - m.shape[1], *m.shape[2:]))], dim=1) for m in mstate]
                    mstate = torch.cat(mstate, dim=0)
            splits = torch.split(mstate, int(self.ulb_ratio * bs), dim=0)
            #  lbl and ulb might be empty lists
            lbl = torch.cat(splits[::2], dim=0) if splits[::2] else torch.tensor([], dtype=mstate.dtype, device=mstate.device)
            ulb = torch.cat(splits[1::2], dim=0) if splits[1::2] else torch.tensor([], dtype=mstate.dtype, device=mstate.device)
            mstate = torch.cat([lbl, ulb], dim=0)
            mstate = mstate.view(-1, mstate.shape[-1])
            accumulated_metrics[mname] = mstate

        return accumulated_metrics

    def _get_mean_p_max_model_and_label_hist(self, max_scores, labels, fg_mask, hist_minlength=3, split=None, type='micro'):
        if split is None:
            split = ['lbl', 'ulb']
        if isinstance(split, str) and split in ['lbl', 'ulb']:
            _split = _lbl if split == 'lbl' else _ulb
            fg_max_scores = _split(max_scores, fg_mask)
            fg_labels = _split(labels, fg_mask)
            p_max_model = fg_labels.new_zeros(hist_minlength, dtype=fg_max_scores.dtype).scatter_add_(0, fg_labels, fg_max_scores)
            fg_labels_hist = torch.bincount(fg_labels, minlength=hist_minlength)

            if type == 'micro':
                p_max_model = p_max_model.sum() / fg_labels_hist.sum()
            elif type == 'macro':
                p_max_model /= (fg_labels_hist + 1e-6)
                p_max_model = p_max_model.mean()

            fg_labels_hist = fg_labels_hist / fg_labels_hist.sum()
            return p_max_model, fg_labels_hist
        elif isinstance(split, list) and len(split) == 2:
            p_s0, h_s0 = self._get_mean_p_max_model_and_label_hist(max_scores, labels, fg_mask, hist_minlength=hist_minlength, split=split[0], type=type)
            p_s1, h_s1 = self._get_mean_p_max_model_and_label_hist(max_scores, labels, fg_mask, hist_minlength=hist_minlength, split=split[1], type=type)
            return torch.vstack([p_s0, p_s1]).squeeze(), torch.vstack([h_s0, h_s1])
        else:
            raise ValueError(f"Invalid split type: {split}")

    def compute(self):
        results = {}

        if len(self.sem_scores_wa) < self.reset_state_interval:
            return

        self.iteration_count += 1
        acc_metrics = self._accumulate_metrics()
        for sname in ['sem_scores_wa', 'sem_scores_sa', 'sem_scores_pre_gt_wa']:
            sem_scores = acc_metrics[sname]
            conf_scores = acc_metrics[sname.replace('sem', 'conf')]

            # if sname == 'sem_scores_sa':
            #     fg_mask = (acc_metrics['box_cls_labels_sa'] > 0).squeeze()
            #     sem_scores = torch.softmax(sem_scores / self.temperature_sa, dim=-1)
            # elif sname in ['sem_scores_wa']:
            #     # TODO: Currently, we are using rois immediately produced by RPN.
            #     #  Thus, the min of RPN's matched_threshold (0.5) is used as the FG threshold.
            #     #  Note that the classwise RPN's thresholds are 0.6, 0.5, 0.5 for Car, Ped, Cyc respectively.
            #     fg_mask = (acc_metrics['roi_ious_wa'] > self.prior_sem_fg_thresh).squeeze()
            #     sem_scores = torch.softmax(sem_scores / self.temperature, dim=-1)
            # elif sname in ['sem_scores_pre_gt_wa']:
            #     fg_mask = (acc_metrics['roi_ious_pre_gt_wa'] > self.prior_sem_fg_thresh).squeeze()
            #     sem_scores = torch.softmax(sem_scores / self.temperature, dim=-1)
            
            fg_mask = (conf_scores > 0).squeeze() # remove padded 0 with fg_mask
            padding_mask = torch.logical_not(torch.all(sem_scores == 0, dim=-1))
            assert torch.equal(padding_mask, fg_mask), f"Padding mask and fg_mask are not equal for {sname}"

            sem_scores = torch.softmax(sem_scores / (self.temperature_sa if '_sa' in sname else self.temperature), dim=-1)

            max_scores, labels = torch.max(sem_scores, dim=-1)

            hist_minlength = sem_scores.shape[-1]
            mean_p_max_model, labels_hist = self._get_mean_p_max_model_and_label_hist(max_scores, labels, fg_mask, hist_minlength)
            mean_p_model_lbl = _lbl(sem_scores, fg_mask).mean(dim=0)
            mean_p_model_ulb = _ulb(sem_scores, fg_mask).mean(dim=0)
            mean_p_model = torch.vstack([mean_p_model_lbl, mean_p_model_ulb])

            self._update_ema('mean_p_max_model', mean_p_max_model, sname)
            self._update_ema('labels_hist', labels_hist, sname)
            self._update_ema('mean_p_model', mean_p_model, sname)

            self.log_results(results, sname=sname)
            if self.enable_plots:
                fig = self.draw_dist_plots(max_scores, labels, fg_mask, sname)
                results[f'dist_plots_{sname}'] = fig
                plt.close()

        # ratio =  _lbl(self.mean_p_model['sem_scores_pre_gt_wa']) / (_ulb(self.mean_p_model['sem_scores_wa']) + 1e-6)
        ratio =  _get_cls_dist(_lbl(acc_metrics[self.target_to_align].view(-1))) / (_ulb(self.mean_p_model['sem_scores_wa']) + 1e-6)
        self._update_ema('ratio', ratio, 'AdaMatch')
        results[f'ratio/lbl_{self.target_to_align}_over_ulb_wa'] = self._arr2dict(self.ratio['AdaMatch'])

        results['labels_hist_lbl/gts_wa'] = self._arr2dict(_get_cls_dist(_lbl(acc_metrics['gt_labels_wa'].view(-1))))
        results['labels_hist_lbl/gts_sa'] = self._arr2dict(_get_cls_dist(_lbl(acc_metrics['gt_labels_sa'].view(-1))))
        results['labels_hist_lbl/gts_pre_gt_wa'] = self._arr2dict(_get_cls_dist(_lbl(acc_metrics['gt_labels_pre_gt_wa'].view(-1))))
        results['labels_hist_ulb/gts_wa'] = self._arr2dict(_get_cls_dist(_ulb(acc_metrics['gt_labels_wa'].view(-1))))
        results['labels_hist_ulb/gts_sa'] = self._arr2dict(_get_cls_dist(_ulb(acc_metrics['gt_labels_sa'].view(-1))))
        results['labels_hist_ulb/gts_pre_gt_wa'] = self._arr2dict(_get_cls_dist(_ulb(acc_metrics['gt_labels_pre_gt_wa'].view(-1))))
        results[f'threshold/AdaMatch_{self.lab_thresh_tag}'] = self._get_threshold(tag=self.lab_thresh_tag, thresh_alg='AdaMatch') # NOTE also keep tag='sem_scores_pre_gt_wa' in all branches
        results['threshold/FreeMatch'] = self._arr2dict(self._get_threshold(thresh_alg='FreeMatch'))
        self.reset()
        return results

    def log_results(self, results, sname):
        results[f'mean_p_max_model_lbl/{sname}'] = _lbl(self.mean_p_max_model[sname])
        results[f'mean_p_max_model_ulb/{sname}'] = _ulb(self.mean_p_max_model[sname])
        results[f'labels_hist_lbl/{sname}'] = self._arr2dict(_lbl(self.labels_hist[sname]))
        results[f'labels_hist_ulb/{sname}'] = self._arr2dict(_ulb(self.labels_hist[sname]))

        unbiased_p_model = self.mean_p_model[sname] / self.labels_hist[sname]
        unbiased_p_model = unbiased_p_model / unbiased_p_model.sum(dim=-1, keepdim=True)
        results[f'unbiased_p_model_lbl/{sname}'] = self._arr2dict(_lbl(unbiased_p_model))
        results[f'unbiased_p_model_ulb/{sname}'] = self._arr2dict(_ulb(unbiased_p_model))
        results[f'mean_p_model_lbl/{sname}'] = self._arr2dict(_lbl(self.mean_p_model[sname]))
        results[f'mean_p_model_ulb/{sname}'] = self._arr2dict(_ulb(self.mean_p_model[sname]))


    def draw_dist_plots(self, max_scores, labels, fg_mask, tag, meta_info=''):

        max_scores_lbl = _lbl(max_scores, fg_mask)
        labels_lbl = _lbl(labels, fg_mask)

        max_scores_ulb = _ulb(max_scores, fg_mask)
        labels_ulb = _ulb(labels, fg_mask)

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

    def rectify_sem_scores(self, sem_scores_ulb):

        if self.iteration_count == 0:
            print("Skipping rectification as iteration count is 0")
            return
        sem_scores_ulb = sem_scores_ulb * self.ratio['AdaMatch']
        sem_scores_ulb /= sem_scores_ulb.sum(dim=-1, keepdims=True)
        return sem_scores_ulb

    def _update_ema(self, p_name, probs, tag):
        prob = getattr(self, p_name)
        prob[tag] = probs if prob[tag] is None else self.momentum * prob[tag] + (1 - self.momentum) * probs

    def get_mask(self, logits, ret_rectified=False):
        assert self.thresh_method in ['AdaMatch', 'FreeMatch'], f'{self.thresh_method} not in list [AdaMatch, FreeMatch, SoftMatch]'

        scores = torch.softmax(logits / self.temperature, dim=-1)
        if self.thresh_method == 'AdaMatch':
            scores = self.rectify_sem_scores(scores)
            max_scores, labels = torch.max(scores, dim=-1)
            if ret_rectified:
                return max_scores > self._get_threshold(tag=self.lab_thresh_tag, thresh_alg=self.thresh_method), scores
            return max_scores > self._get_threshold(tag=self.lab_thresh_tag, thresh_alg=self.thresh_method)

        elif self.thresh_method == 'FreeMatch':
            max_scores, labels = torch.max(scores, dim=-1)
            thresh = self._get_threshold(tag='sem_scores_wa', thresh_alg=self.thresh_method)
            thresh = thresh.repeat(max_scores.size(0), 1).gather(dim=1, index=labels.unsqueeze(-1)).squeeze()
            return max_scores > thresh

    def _get_threshold(self, sem_scores_wa_lbl=None, tag='sem_scores_wa', thresh_alg='AdaMatch'):
        if thresh_alg == 'AdaMatch':
            if sem_scores_wa_lbl is None:
                return _lbl(self.mean_p_max_model[tag]) * self.fixed_thresh
            max_scores, labels = torch.max(sem_scores_wa_lbl, dim=-1)
            fg_mask = max_scores > self.prior_sem_fg_thresh
            fg_max_scores_lbl = max_scores[fg_mask]
            return fg_max_scores_lbl.mean() * self.fixed_thresh

        elif thresh_alg == 'FreeMatch':
            normalized_p_model = torch.div(_ulb(self.mean_p_model[tag]), _ulb(self.mean_p_model[tag]).max())
            return normalized_p_model * _ulb(self.mean_p_max_model[tag])

    def _arr2dict(self, array):
        if array.shape[-1] == 2:
            return {cls: array[cind] for cind, cls in enumerate(['Bg', 'Fg'])}
        elif array.shape[-1] == len(self.class_names):
            return {cls: array[cind] for cind, cls in enumerate(self.class_names)}
        else:
            raise ValueError(f"Invalid array shape: {array.shape}")