import torch
from torchmetrics import Metric
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
palettes = {t: c for t, c in zip(['fp', 'tn', 'tp', 'fn'], sns.color_palette("hls", 4))}
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
class AdaMatchThreshold(Metric):
    full_state_update: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.thresh_tag = kwargs.get('tag', None)
        self.config = kwargs.get('config', None)
        self.reset_state_interval = self.config.get('RESET_STATE_INTERVAL', 32)
        self.pre_filtering_thresh = self.config.get('PRE_FILTERING_THRESH', 0.1)
        self.enable_plots = self.config.get('ENABLE_PLOTS', False)
        self.enable_clipping = self.config.get('ENABLE_CLIPPING', False)
        self.relative_val = self.config.get('RELATIVE_VAL', 0.9)
        self.momentum = self.config.get('MOMENTUM', 0.9)
        self.num_classes = 3
        self.class_names = ['Car', 'Pedestrian', 'Cyclist']
        self.iteration_count = 0
        self.states_name = kwargs.get('states', None)
        if self.states_name is None:
            self.states_name=[]
            for batch_type in ['lab', 'unlab']:
                self.states_name.append(f'pl_scores_wa_{batch_type}')
                self.states_name.append(f'pl_scores_sa_{batch_type}')
                self.states_name.append(f'rect_pl_scores_wa_{batch_type}')
                self.states_name.append(f'pl_scores_pre_gt_{batch_type}')
                self.states_name.append(f'roi_scores_wa_{batch_type}')
                self.states_name.append(f'rect_roi_scores_wa_{batch_type}')
                self.states_name.append(f'roi_scores_sa_{batch_type}')
                self.states_name.append(f'roi_scores_pre_gt_{batch_type}')

        for name in self.states_name:
            self.add_state(name, default=[], dist_reduce_fx='cat')

        self.relative_mu_threshold = {}
        self.relative_ema_threshold = {}
        self.means = {s_name: torch.ones((self.num_classes)) * self.pre_filtering_thresh for s_name in self.states_name}
        self.emas = {s_name: torch.ones((self.num_classes)) * self.pre_filtering_thresh for s_name in self.states_name}

    def update(self, **kwargs):
        for state_name in self.states_name:
            value = kwargs.get(state_name)
            if value is not None:
                getattr(self, state_name).append(value)

    def _accumulate_metrics(self):
        accumulated_metrics = {}
        for mname in self.states_name:
            mstate = getattr(self, mname)
            if isinstance(mstate, torch.Tensor):
                mstate = [mstate]
            mstate = torch.cat(mstate, dim=0)
            accumulated_metrics[mname] = mstate

        return accumulated_metrics
    
    def normalize_(self, data_in, prefilter=False):
        min_val = data_in.min()
        max_val = data_in.max()
        data_out = (data_in - min_val) / (max_val - min_val)
        if prefilter:
            data_out = data_out[data_out>self.pre_filtering_thresh]
        return data_out

    def init_figure(self):
        
        BS = len(self.pl_scores_wa_lab[0])
        WS = self.reset_state_interval * BS
        info = f"Iter: {self.iteration_count}    Interval: {self.reset_state_interval}    BS: {BS}    W: {(self.iteration_count - 1) * WS} - {self.iteration_count * WS}"
        fig, axs = plt.subplots(1, 3, figsize=(12, 6), sharex='col', sharey='row', layout="compressed")
        plt.suptitle(info, fontsize='x-small')
        return fig, axs
    
    def compute(self):
        results = {}

        if len(self.pl_scores_pre_gt_lab) >= self.reset_state_interval:
            self.iteration_count += 1
            accumulated_metrics = self._accumulate_metrics()
            roi_states = [sname for sname in self.states_name if 'roi' in sname]
            for roi_state in roi_states:
                if self.enable_plots:
                    if not f'{self.thresh_tag}_plots' in results:
                        results[f'{self.thresh_tag}_plots']={}
                    fig, axs = self.init_figure()
                
                pl_state = roi_state.replace('roi', 'pl')
                roi_scores = accumulated_metrics[roi_state]
                pl_scores = accumulated_metrics[pl_state]
                cur_roi_scores, cur_roi_labels = torch.max(roi_scores, dim=-1)
                for cind, cls in enumerate(self.class_names):
                    lab_mask = cur_roi_labels == cind
                    
                    cls_roi_scores = cur_roi_scores[lab_mask]
                    cls_roi_scores = cls_roi_scores[cls_roi_scores > self.pre_filtering_thresh]
                    if cls_roi_scores.shape[0]: # self.means[roi_state] keeps previous mean otherwise
                        self.means[roi_state][cind] = cls_roi_scores.mean().item()
                    
                    cls_pl_scores = pl_scores[lab_mask] if 'sa' not in pl_state else pl_scores # sa 512 rois =/ pls 
                    cls_pl_scores = cls_pl_scores[cls_pl_scores > self.pre_filtering_thresh]
                    if cls_pl_scores.shape[0]: # self.means[pl_state] keeps previous mean otherwise
                        self.means[pl_state][cind] = cls_pl_scores.mean().item()
                    
                    if self.enable_plots:
                        axs[cind].hist(cls_pl_scores.view(-1).cpu().numpy(), bins=30, alpha=0.7, label=pl_state, edgecolor='black', color=palettes['tp'])
                        axs[cind].hist(cls_roi_scores.view(-1).cpu().numpy(), bins=30, alpha=0.7, label=roi_state, edgecolor='black', color=palettes['fp'])
                        axs[cind].axvline(self.means[pl_state][cind], linestyle='--', label='pl-mu', color=palettes['tp'], alpha=0.9)
                        axs[cind].axvline(self.means[roi_state][cind], linestyle='--', label='roi-mu', color=palettes['fp'], alpha=0.9)
                        axs[cind].axvline(self.emas[pl_state][cind], linestyle='-', label='pl-ema', color=palettes['tp'], alpha=0.9, linewidth=1)
                        axs[cind].axvline(self.emas[roi_state][cind], linestyle='-', label='roi-ema', color=palettes['fp'], alpha=0.9, linewidth=1)
                        if 'rect' in pl_state and 'pl_scores_pre_gt_lab' in self.relative_ema_threshold:
                            axs[cind].axvline(self.relative_ema_threshold['pl_scores_pre_gt_lab'][cind], linestyle=':', label='pl-rt', color=palettes['tp'], alpha=0.9, linewidth=3)
                            axs[cind].axvline(self.relative_ema_threshold['roi_scores_pre_gt_lab'][cind], linestyle=':', label='roi-rt', color=palettes['fp'], alpha=0.9, linewidth=3)
                        axs[cind].set_title(cls,fontsize='x-small')
                        axs[cind].legend(loc='upper right', fontsize='x-small')
                        axs[cind].set_xlabel('score', fontsize='x-small')
                        axs[cind].set_ylabel('count', fontsize='x-small')
                        axs[cind].set_ylim(0, 800)
                
                if 'pre_gt_lab' in pl_state:
                    self._update_relative_thresholds(tag='pl_scores_pre_gt_lab')
                    self._update_relative_thresholds(tag='roi_scores_pre_gt_lab')

                self._update_ema(roi_state)
                self._update_ema(pl_state)
                if self.enable_plots:
                    results[f'{self.thresh_tag}_plots'][roi_state] = fig.get_figure()

            results.update(**self._get_results_dict())
            self.reset()

        return results

    def _update_ema(self, tag):
        if self.emas.get(tag) is None:
            self.emas[tag] = self.means[tag]
        else:
            self.emas[tag] = self.momentum * self.emas[tag] + (1 - self.momentum) * self.means[tag]

    def _update_relative_thresholds(self, tag='pl_scores_pre_gt_lab'):
        self.relative_mu_threshold[tag] = self.relative_val * self.means[tag]
        self.relative_ema_threshold[tag] = self.relative_val * self.emas[tag]

        if self.enable_clipping:
            self.relative_mu_threshold[tag] = torch.clip(self.relative_mu_threshold[tag], 0.0, 1.0)
            self.relative_ema_threshold[tag] = torch.clip(self.relative_ema_threshold[tag], 0.0, 1.0)

        print(f'\nUpdated RT tag: {tag}:')
        for cind, cls in enumerate(self.class_names):
            mu_value = self.relative_mu_threshold[tag][cind].item()
            ema_value = self.relative_ema_threshold[tag][cind].item()
            print(f'{cls: <12}: mu: {mu_value:.3f}\tema: {ema_value:.3f}')


    def _get_results_dict(self):
        results = {}

        for rec in ['pl', 'roi']:
            for cind, cls in enumerate(self.class_names):
                results[f'{self.thresh_tag}_{rec}_summary_{cls}'] = {'rect': self.emas[f'rect_{rec}_scores_wa_unlab'][cind].item(),
                                                                    'org': self.emas[f'{rec}_scores_wa_unlab'][cind].item(),
                                                                    'rt': self.relative_ema_threshold[f'{rec}_scores_pre_gt_lab'][cind].item()}
            results[f'{self.thresh_tag}_ratio/{rec}_mu'] = {cls: (self.means[f'{rec}_scores_pre_gt_lab'][cind] / self.means[f'{rec}_scores_wa_unlab'][cind]).item() for cind, cls in enumerate(self.class_names)} 
            results[f'{self.thresh_tag}_ratio/{rec}_ema'] = {cls: (self.emas[f'{rec}_scores_pre_gt_lab'][cind] / self.emas[f'{rec}_scores_wa_unlab'][cind]).item() for cind, cls in enumerate(self.class_names)} 
            for sname in self.states_name:
                if rec in sname:
                    results[f'{self.thresh_tag}_{rec}_ema/{sname}'] = {cls: self.emas[sname][cind].item() for cind, cls in enumerate(self.class_names)}
                    results[f'{self.thresh_tag}_{rec}_mu/{sname}'] = {cls: self.means[sname][cind].item() for cind, cls in enumerate(self.class_names)}
            
        for key, val in self.relative_mu_threshold.items():
            results[f'{self.thresh_tag}_rt_mu/{key}'] = {cls_name: val[i].item() for i, cls_name in enumerate(self.class_names)}
        for key, val in self.relative_ema_threshold.items():
            results[f'{self.thresh_tag}_rt_ema/{key}'] = {cls_name: val[i].item() for i, cls_name in enumerate(self.class_names)}


        return results