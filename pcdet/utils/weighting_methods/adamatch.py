import torch
from torchmetrics import Metric
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.mixture import GaussianMixture
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
        self.means = {s_name: None  for s_name in self.states_name}
        self.emas = {s_name: None  for s_name in self.states_name}
        self.p_model = {s_name: torch.ones((self.num_classes)) / self.num_classes  for s_name in self.states_name}
        self.p_model_ema = {s_name: torch.ones((self.num_classes)) / self.num_classes  for s_name in self.states_name}
        
        # GMM
        self.gmm_policy=self.config.get('GMM_POLICY','high')
        self.mu1=self.config.get('MU1',0.1)
        self.mu2=self.config.get('MU2',0.9)
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

    def init_figure(self, meta_info=''):
        
        BS = len(self.pl_scores_wa_lab[0])
        WS = self.reset_state_interval * BS
        info = f"Iter: {self.iteration_count}    Interval: {self.reset_state_interval}    BS: {BS}    W: {(self.iteration_count - 1) * WS} - {self.iteration_count * WS}    M: {meta_info}"
        fig, axs = plt.subplots(1, 3, figsize=(12, 6), sharex='col', sharey='row', layout="compressed")
        plt.suptitle(info, fontsize='small')
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
                    fig, axs = self.init_figure(meta_info = roi_state)
                
                
                roi_scores_multiclass = accumulated_metrics[roi_state].view(-1, 3)
                max_scores, labels = torch.max(roi_scores_multiclass, dim=-1)
                roi_fg_mask = max_scores > 0.6
                self.means[roi_state] = max_scores[roi_fg_mask].mean().item()

                # pl_state = roi_state.replace('roi', 'pl')
                # pl_scores = accumulated_metrics[pl_state].view(-1) # conf score
                # pl_fg_mask = pl_scores > 0.6 
                
                for cind, cls in enumerate(self.class_names):
                    cls_mask = (labels == cind)
                    cls_roi_fg_mask = cls_mask[roi_fg_mask]
                    p_model = (cls_roi_fg_mask.sum()/roi_fg_mask.sum())
                    if p_model: # keeps previous if invalid
                        self.p_model[roi_state][cind] = p_model.item()
                                        
                    # cls_pl_fg_mask = pl_scores[cls_mask] > 0.6
                    # p_model = (cls_pl_fg_mask.sum()/pl_fg_mask.sum()).item()
                    # if p_model: # keeps previous if invalid
                    #     self.p_model[pl_state][cind] = p_model

                    # classwise scores mean for thresholding
                    cls_roi = max_scores[cls_roi_fg_mask]
                    # cls_pl = pl_scores[cls_mask][cls_pl_fg_mask] if 'sa' not in pl_state else pl_scores # sa 512 rois =/ pls 
                    
                    # if cls_roi.shape[0]: # keeps previous mean otherwise
                    #     self.means[roi_state][cind] = cls_roi.mean().item()
                    
                    # if cls_pl.shape[0]: #  keeps previous mean otherwise
                    #     self.means[pl_state][cind] = cls_pl.mean().item()


                    if self.enable_plots:
                        axs[cind].hist(cls_roi.cpu().numpy(), bins=10, alpha=0.6, edgecolor='black', color=palettes['tp'], label='fg-sem-score')
                        # axs[cind].hist(cls_pl.cpu().numpy(), bins=10, alpha=0.6, edgecolor='black', color=palettes['fp'], label='fg-conf-score')
                        axs[cind].axvline(x=self.p_model_ema[roi_state][cind].item(), color=palettes['fp'], linestyle='-', label='p-ema (sem)',linewidth=3)
                        if self.p_model[roi_state] is not None:
                            axs[cind].axvline(x=self.p_model[roi_state][cind].item(), color=palettes['fp'], linestyle='--', label='p (sem)',linewidth=1)
                        # axs[cind].axvline(x=self.p_model_ema[pl_state][cind].item(), color=palettes['fp'], linestyle='-', label='p-ema (conf)',linewidth=3)
                        # axs[cind].axvline(x=self.p_model[pl_state][cind].item(), color=palettes['fp'], linestyle='--', label='p (conf)',linewidth=1)
                        axs[cind].set_title(cls,fontsize='x-small')
                        axs[cind].legend(loc='upper right', fontsize='x-small')
                        axs[cind].set_xlabel('score', fontsize='x-small')
                        axs[cind].set_ylabel('count', fontsize='x-small')
                        axs[cind].set_ylim(0, 400)
                        axs[cind].set_xlim(0, 1)
                        results[f'{self.thresh_tag}_plots'][roi_state] = fig.get_figure()
                        plt.close()
                
                self._update_ema(roi_state)
                self._update_p_model_ema(roi_state)

                if self.enable_plots:
                    results[f'{self.thresh_tag}_plots'][roi_state] = fig.get_figure()
            self._update_relative_thresholds(tag='roi_scores_pre_gt_lab')
            results.update(**self._get_results_dict())
            self.reset()
            return results



    def _update_ema(self, tag):
        if self.emas.get(tag) is None:
            self.emas[tag] = self.means[tag]
        else:
            self.emas[tag] = self.momentum * self.emas[tag] + (1 - self.momentum) * self.means[tag]
    
    def _update_p_model_ema(self, tag):
        if self.p_model_ema.get(tag) is None:
            self.p_model_ema[tag] = self.p_model[tag]
        else:
            self.p_model_ema[tag] = self.momentum * self.p_model_ema[tag] + (1 - self.momentum) * self.p_model[tag]

    def _update_relative_thresholds(self, tag='pl_scores_pre_gt_lab'):
        self.relative_mu_threshold[tag] = self.relative_val * self.means[tag]
        self.relative_ema_threshold[tag] = self.relative_val * self.emas[tag]

        if self.enable_clipping:
            self.relative_mu_threshold[tag] = torch.clip(self.relative_mu_threshold[tag], 0.0, 1.0)
            self.relative_ema_threshold[tag] = torch.clip(self.relative_ema_threshold[tag], 0.0, 1.0)

        print(f'\nUpdated RT tag: {tag}:')
        if isinstance(self.relative_mu_threshold[tag], float):
            mu_value = self.relative_mu_threshold[tag]
            ema_value = self.relative_ema_threshold[tag]
            print(f'mu: {mu_value:.3f}\tema: {ema_value:.3f}')
        else:
            for cind, cls in enumerate(self.class_names):
                mu_value = self.relative_mu_threshold[tag][cind].item()
                ema_value = self.relative_ema_threshold[tag][cind].item()
                print(f'{cls: <12}: mu: {mu_value:.3f}\tema: {ema_value:.3f}')


    def _get_results_dict(self):
        results = {}
        for rec in ['roi']:
            if isinstance(self.emas[f'rect_{rec}_scores_wa_unlab'], float):
                results[f'{self.thresh_tag}_{rec}_summary'] = \
                    {'rect': self.emas[f'rect_{rec}_scores_wa_unlab'],
                    'org': self.emas[f'{rec}_scores_wa_unlab'],
                    'rt': self.relative_ema_threshold[f'{rec}_scores_pre_gt_lab']}
            else:
                for cind, cls in enumerate(self.class_names):
                    results[f'{self.thresh_tag}_{rec}_summary_{cls}'] = \
                        {'rect': self.emas[f'rect_{rec}_scores_wa_unlab'][cind].item(),
                        'org': self.emas[f'{rec}_scores_wa_unlab'][cind].item(),
                        'rt': self.relative_ema_threshold[f'{rec}_scores_pre_gt_lab'][cind].item()}
            results[f'{self.thresh_tag}_ratio/{rec}_mu'] = \
                {cls: (self.p_model[f'{rec}_scores_pre_gt_lab'][cind] / self.p_model[f'{rec}_scores_wa_unlab'][cind]).item() \
                 for cind, cls in enumerate(self.class_names)} 
            results[f'{self.thresh_tag}_ratio/{rec}_ema'] = \
                {cls: (self.p_model_ema[f'{rec}_scores_pre_gt_lab'][cind] / self.p_model_ema[f'{rec}_scores_wa_unlab'][cind]).item() \
                 for cind, cls in enumerate(self.class_names)} 
            for sname in self.states_name:
                if rec in sname:
                    if isinstance(self.emas[sname], float):
                        results[f'{self.thresh_tag}_{rec}_ema/{sname}'] = self.emas[sname]
                        results[f'{self.thresh_tag}_{rec}_mu/{sname}'] = self.means[sname]
                    else: 
                        results[f'{self.thresh_tag}_{rec}_ema/{sname}'] = \
                            {cls: self.emas[sname][cind].item() for cind, cls in enumerate(self.class_names)}
                        results[f'{self.thresh_tag}_{rec}_mu/{sname}'] = \
                            {cls: self.means[sname][cind].item() for cind, cls in enumerate(self.class_names)}
                    results[f'{self.thresh_tag}_{rec}_p_model/{sname}'] = \
                        {cls: self.p_model[sname][cind].item() for cind, cls in enumerate(self.class_names)}
                    results[f'{self.thresh_tag}_{rec}_p_model_ema/{sname}'] = \
                        {cls: self.p_model_ema[sname][cind].item() for cind, cls in enumerate(self.class_names)}
        
        for key, val in self.relative_mu_threshold.items():
            if isinstance(val, float):
                results[f'{self.thresh_tag}_rt_mu/{key}'] = val
            else:
                results[f'{self.thresh_tag}_rt_mu/{key}'] = \
                    {cls_name: val[i].item() for i, cls_name in enumerate(self.class_names)}
        for key, val in self.relative_ema_threshold.items():
            if isinstance(val, float):
                results[f'{self.thresh_tag}_rt_ema/{key}'] = val
            else:
                results[f'{self.thresh_tag}_rt_ema/{key}'] = \
                    {cls_name: val[i].item() for i, cls_name in enumerate(self.class_names)}


        return results