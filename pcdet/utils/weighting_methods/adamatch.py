import torch
from torchmetrics import Metric
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

"""
Adamatch based relative Thresholding
mean conf. of the top-1 prediction on the weakly aug source data multiplied by a user provided threshold

Adamatch based Dist. Alignment 
Rectify the target unlabeled pseudo-labels by multiplying them by the ratio of the expected 
value of the weakly aug source labels E[YcapSL;w] to the expected
value of the target labels E[YcapTU;w], obtaining the final pseudo-labels YtildaTU;w

REF: UPS FRAMEWORK DA
probs_x_ulb_w = accumulated_metrics['pred_weak_aug_unlab_before_nms'].view(-1)
probs_x_lb_s = accumulated_metrics['pred_weak_aug_lab_before_nms'].view(-1)
self.p_model = self.momentum  * self.p_model + (1 - self.momentum) * torch.mean(probs_x_ulb_w)
self.p_target = self.momentum  * self.p_target + (1 - self.momentum) * torch.mean(probs_x_lb_s)
probs_x_ulb_aligned = probs_x_ulb_w * (self.p_target + 1e-6) / (self.p_model + 1e-6)
"""
class AdaMatchThreshold(Metric):
    full_state_update: bool = False
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.config = kwargs.get('config', None)
        self.reset_state_interval = self.config.ROI_HEAD.ADAPTIVE_THRESH_CONFIG.get('RESET_STATE_INTERVAL', 32)
        self.pre_filtering_thresh=self.config.ROI_HEAD.ADAPTIVE_THRESH_CONFIG.get('PRE_FILTERING_THRESH', 0.1)
        self.enable_plots=self.config.ROI_HEAD.ADAPTIVE_THRESH_CONFIG.get('ENABLE_PLOTS', False)
        self.enable_clipping = self.config.ROI_HEAD.ADAPTIVE_THRESH_CONFIG.get('ENABLE_CLIPPING', False)
        self.relative_val= self.config.ROI_HEAD.ADAPTIVE_THRESH_CONFIG.get('RELATIVE_VAL', 0.95)
        self.momentum= self.config.ROI_HEAD.ADAPTIVE_THRESH_CONFIG.get('MOMENTUM', 0.99)
        self.num_classes = 3
        self.iteration_count=0
        self.states_name = ['pred_weak_aug_unlab_before_nms', 'pred_weak_aug_unlab_after_nms', 
                            'pred_weak_aug_lab_before_nms', 'pred_weak_aug_lab_after_nms',
                            'pred_strong_aug_unlab_before_nms', 'pred_strong_aug_unlab_after_nms', 
                            'pred_strong_aug_lab_before_nms', 'pred_strong_aug_lab_after_nms']

        for name in self.states_name:
            self.add_state(name, default=[], dist_reduce_fx='cat')

        self.relative_threshold = None
        self.relative_ema_threshold = None

        self.ema_pred_weak_aug_unlab_before_nms = None
        self.ema_pred_weak_aug_lab_before_nms = None
        
        self.ema_pred_strong_aug_unlab_before_nms = None
        self.ema_pred_strong_aug_lab_before_nms = None


    def update(self, **kwargs):
        for state_name in self.states_name:
            value = kwargs.get(state_name)
            if value is not None:
                getattr(self, state_name).append(value)


    def _accumulate_metrics(self, prefilter=False):
        accumulated_metrics = {}
        for mname in self.states_name:
            mstate = getattr(self, mname)
            if isinstance(mstate, torch.Tensor):
                mstate = [mstate]
            if isinstance(mstate[0], list):  # Check if mstate is a list of pairs
                mstate = [torch.cat(pair, dim=0) for pair in mstate]
            mstate = torch.cat(mstate, dim=0)
            if prefilter:
                mstate = mstate[mstate>self.pre_filtering_thresh]
            accumulated_metrics[mname]=mstate
        
        return accumulated_metrics
        
    
    def normalize_(self, data_in, prefilter=False):
        min_val = data_in.min()
        max_val = data_in.max()
        data_out = (data_in - min_val) / (max_val - min_val)
        if prefilter:
            data_out = data_out[data_out>self.pre_filtering_thresh]
        return data_out
    
    
    def compute(self):
        results = {}

        if  len(self.pred_weak_aug_lab_before_nms) >= self.reset_state_interval:
            self.iteration_count+=1
            
            accumulated_metrics = self._accumulate_metrics(prefilter=True)  # shape (N, 1)
            
            pred_weak_aug_unlab_before_nms   = accumulated_metrics['pred_weak_aug_unlab_before_nms'].view(-1)
            pred_weak_aug_lab_before_nms     = accumulated_metrics['pred_weak_aug_lab_before_nms'].view(-1)
            pred_strong_aug_unlab_before_nms = accumulated_metrics['pred_strong_aug_unlab_before_nms'].view(-1)
            pred_strong_aug_lab_before_nms   = accumulated_metrics['pred_strong_aug_lab_before_nms'].view(-1)


            mu_pred_weak_aug_unlab_before_nms   = pred_weak_aug_unlab_before_nms.mean()
            mu_pred_weak_aug_lab_before_nms     = pred_weak_aug_lab_before_nms.mean()
            mu_pred_strong_aug_unlab_before_nms = pred_strong_aug_unlab_before_nms.mean()
            mu_pred_strong_aug_lab_before_nms   = pred_strong_aug_lab_before_nms.mean()

            if self.ema_pred_weak_aug_unlab_before_nms is None:
                self.ema_pred_weak_aug_unlab_before_nms =  mu_pred_weak_aug_unlab_before_nms
            else:    
                self.ema_pred_weak_aug_unlab_before_nms = self.momentum  * \
                    self.ema_pred_weak_aug_unlab_before_nms + (1 - self.momentum) * mu_pred_weak_aug_unlab_before_nms
            
            if self.ema_pred_weak_aug_lab_before_nms is None:
                self.ema_pred_weak_aug_lab_before_nms = mu_pred_weak_aug_lab_before_nms
            else:
                self.ema_pred_weak_aug_lab_before_nms = self.momentum  * \
                self.ema_pred_weak_aug_lab_before_nms + (1 - self.momentum) * mu_pred_weak_aug_lab_before_nms

            if self.ema_pred_strong_aug_unlab_before_nms is None:
                self.ema_pred_strong_aug_unlab_before_nms = mu_pred_strong_aug_unlab_before_nms
            else:
                self.ema_pred_strong_aug_unlab_before_nms = self.momentum  * \
                self.ema_pred_strong_aug_unlab_before_nms + (1 - self.momentum) * mu_pred_strong_aug_unlab_before_nms
            
            if self.ema_pred_strong_aug_lab_before_nms is None:
                self.ema_pred_strong_aug_lab_before_nms =  mu_pred_strong_aug_lab_before_nms
            else:
                self.ema_pred_strong_aug_lab_before_nms = self.momentum  * \
                self.ema_pred_strong_aug_lab_before_nms + (1 - self.momentum) * mu_pred_strong_aug_lab_before_nms

            # 1. relative threshold using pred_weak_aug_lab_before_nms
            self.relative_threshold = self.relative_val * mu_pred_weak_aug_lab_before_nms
            self.relative_ema_threshold = self.relative_val * self.ema_pred_weak_aug_lab_before_nms
            
            # 2. DA of weak-augmnted-unlabeled data using target as weak-augmnted-labled (using Teacher predictions)
            pred_weak_aug_unlab_before_nms_aligned = pred_weak_aug_unlab_before_nms * \
                (self.ema_pred_weak_aug_lab_before_nms + 1e-6) / (self.ema_pred_weak_aug_unlab_before_nms + 1e-6)
            pred_weak_aug_unlab_before_nms_aligned = self.normalize_(pred_weak_aug_unlab_before_nms_aligned, prefilter=True)
            
            # 3. DA of strong-augmnted-unlabeled data using target as strong-augmnted-labled (using Student predictions)
            pred_strong_aug_unlab_before_nms_aligned = pred_strong_aug_unlab_before_nms * \
                (self.ema_pred_strong_aug_lab_before_nms + 1e-6) / (self.ema_pred_strong_aug_unlab_before_nms + 1e-6)
            pred_strong_aug_unlab_before_nms_aligned = self.normalize_(pred_strong_aug_unlab_before_nms_aligned, prefilter=True)
            

            if self.enable_clipping:
                self.relative_threshold = torch.clip(self.relative_threshold, 0.1, 0.9)
                self.relative_ema_threshold = torch.clip(self.relative_ema_threshold, 0.1, 0.9)

            results['adamatch_mu_weak_unlab']= mu_pred_weak_aug_unlab_before_nms.item()
            results['adamatch_mu_weak_lab']= mu_pred_weak_aug_lab_before_nms.item()

            results['adamatch_ema_mu_weak_unlab']= self.ema_pred_weak_aug_unlab_before_nms.item()
            results['adamatch_ema_mu_weak_lab']= self.ema_pred_weak_aug_lab_before_nms.item()

            results['adamatch_mu_weak_lab_rt']= self.relative_threshold.item()
            results['adamatch_ema_mu_weak_lab_rt']= self.relative_ema_threshold.item()

            if self.enable_plots:
                HIST_BIN = np.linspace(self.pre_filtering_thresh, 1, 30)
                palettes = {t: c for t, c in zip(['fp', 'tn', 'tp', 'fn'], sns.color_palette("hls", 4))}
                BS = len(self.pred_weak_aug_lab_before_nms[0])
                WS = self.reset_state_interval * BS
                info = f"Iter: {self.iteration_count}    Interval: {self.reset_state_interval}    BS: {BS}    W: {(self.iteration_count - 1) * WS} - {self.iteration_count * WS}"
                
                # plot states
                num_rows = 2
                num_cols = len(self.states_name) // 2
                fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 6), sharex='col', sharey='row', layout="compressed")
                before_nms_states = [state_name for state_name in self.states_name if 'before_nms' in state_name]
                after_nms_states = [state_name for state_name in self.states_name if 'after_nms' in state_name]
                for col, state_name in enumerate(before_nms_states + after_nms_states):
                    row = 0 if col < num_cols else 1
                    col %= num_cols
                    current_metric = accumulated_metrics[state_name].view(-1).cpu().numpy()
                    axs[row, col].hist(current_metric, bins=HIST_BIN, alpha=0.7, label=state_name, edgecolor='black', color=palettes['fp'])
                    axs[row, col].axvline(current_metric.mean().item(), linestyle='--', label='mu', color=palettes['fp'], alpha=0.9)
                    if 'before_nms' in state_name:
                        axs[row, col].axvline(eval(f"self.ema_{state_name}").item(), linestyle='--', label='ema', color=palettes['tp'], alpha=0.9)
                    axs[row, col].legend(loc='upper right', fontsize='x-small')
                    axs[row, col].set_xlabel('score', fontsize='x-small')
                    axs[row, col].set_ylabel('count', fontsize='x-small')
                    axs[row, col].set_ylim(0, 800)
                plt.suptitle(info, fontsize='x-small')
                #fig_title = f'iteration_acc_{self.iteration_count}.png'
                #fig.get_figure().savefig(fig_title)
                results['adamatch_acc_states_plot'] = fig.get_figure()
                plt.close()

                fig, axs = plt.subplots(1, 1, figsize=(12, 6), layout="compressed")
                axs.hist(pred_strong_aug_unlab_before_nms.cpu().numpy(), bins=HIST_BIN, alpha=0.5, label='strong-aug unlab', edgecolor='black', color=palettes['fp'])
                axs.hist(pred_strong_aug_lab_before_nms.cpu().numpy(), bins=HIST_BIN, alpha=0.5, label='strong-aug lab', edgecolor='black', color=palettes['tp'])
                axs.hist(pred_strong_aug_unlab_before_nms_aligned.cpu().numpy(), bins=HIST_BIN, alpha=0.8, label='rectified strong-aug unlab', edgecolor='black', color=palettes['fn'])
                axs.axvline(self.ema_pred_strong_aug_unlab_before_nms.item(), linestyle='--', label='ema unlab', color=palettes['fp'], alpha=0.9)
                axs.axvline(self.ema_pred_strong_aug_lab_before_nms.item(), linestyle='--', label='ema lab (target)', color=palettes['tp'], alpha=0.9)
                axs.axvline(mu_pred_strong_aug_unlab_before_nms.item(), linestyle='--', label='mu unlab', color=palettes['fn'], alpha=0.9)
                axs.axvline(mu_pred_strong_aug_lab_before_nms.item(), linestyle='--', label='mu lab (target)', color=palettes['tn'], alpha=0.9)
                axs.legend(loc='upper right', fontsize='x-small')
                axs.set_xlabel('score', fontsize='x-small')
                axs.set_ylabel('count', fontsize='x-small')
                axs.set_ylim(0, 800)
                plt.suptitle(info, fontsize='x-small')
                #fig_title = f'iteration_acc_{self.iteration_count}.png'
                #fig.get_figure().savefig(fig_title)
                results['adamatch_strong_align_plot'] = fig.get_figure()
                plt.close()

                fig, axs = plt.subplots(1, 1, figsize=(12, 6), layout="compressed")
                axs.hist(pred_weak_aug_unlab_before_nms.cpu().numpy(), bins=HIST_BIN, alpha=0.5, label='weak-aug unlab', edgecolor='black', color=palettes['fp'])
                axs.hist(pred_weak_aug_lab_before_nms.cpu().numpy(), bins=HIST_BIN, alpha=0.5, label='weak-aug lab', edgecolor='black', color=palettes['tp'])
                axs.hist(pred_weak_aug_unlab_before_nms_aligned.cpu().numpy(), bins=HIST_BIN, alpha=0.8, label='rectified weak-aug unlab', edgecolor='black', color=palettes['fn'])
                axs.axvline(self.ema_pred_weak_aug_unlab_before_nms.item(), linestyle='--', label='ema unlab', color=palettes['fp'], alpha=0.9)
                axs.axvline(self.ema_pred_weak_aug_lab_before_nms.item(), linestyle='--', label='ema lab (target)', color=palettes['tp'], alpha=0.9)
                axs.axvline(mu_pred_weak_aug_unlab_before_nms.item(), linestyle='--', label='mu unlab', color=palettes['fn'], alpha=0.9)
                axs.axvline(mu_pred_weak_aug_lab_before_nms.item(), linestyle='--', label='mu lab (target)', color=palettes['tn'], alpha=0.9)
                axs.legend(loc='upper right', fontsize='x-small')
                axs.set_xlabel('score', fontsize='x-small')
                axs.set_ylabel('count', fontsize='x-small')
                axs.set_ylim(0, 800)
                plt.suptitle(info, fontsize='x-small')
                #fig_title = f'iteration_acc_{self.iteration_count}.png'
                #fig.get_figure().savefig(fig_title)
                results['adamatch_weak_align_plot'] = fig.get_figure()
                plt.close()

            self.reset()

        return results
    

class AdaMatchCWThreshold(Metric):
    full_state_update: bool = False
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.config = kwargs.get('config', None)
        self.reset_state_interval = self.config.ROI_HEAD.ADAPTIVE_THRESH_CONFIG.get('RESET_STATE_INTERVAL', 32)
        self.pre_filtering_thresh=self.config.ROI_HEAD.ADAPTIVE_THRESH_CONFIG.get('PRE_FILTERING_THRESH', 0.1)
        self.enable_plots=self.config.ROI_HEAD.ADAPTIVE_THRESH_CONFIG.get('ENABLE_PLOTS', False)
        self.enable_clipping = self.config.ROI_HEAD.ADAPTIVE_THRESH_CONFIG.get('ENABLE_CLIPPING', False)
        self.relative_val= self.config.ROI_HEAD.ADAPTIVE_THRESH_CONFIG.get('RELATIVE_VAL', 0.95)
        self.momentum= self.config.ROI_HEAD.ADAPTIVE_THRESH_CONFIG.get('MOMENTUM', 0.99)
        self.num_classes = 3
        self.iteration_count=0
        self.states_name = ['pred_weak_aug_unlab_before_nms', 'roi_label_weak_aug_unlab_before_nms',
                            'pred_weak_aug_lab_before_nms', 'roi_label_weak_aug_lab_before_nms']

        for name in self.states_name:
            self.add_state(name, default=[], dist_reduce_fx='cat')

        self.relative_threshold = None
        self.relative_ema_threshold = None

        self.ema_pred_weak_aug_unlab_before_nms = None
        self.ema_pred_weak_aug_lab_before_nms = None
        
        self.ema_pred_strong_aug_unlab_before_nms = None
        self.ema_pred_strong_aug_lab_before_nms = None


    def update(self, **kwargs):
        for state_name in self.states_name:
            value = kwargs.get(state_name)
            if value is not None:
                getattr(self, state_name).append(value)


    def _accumulate_metrics(self, prefilter=False):
        accumulated_metrics = {}
        for mname in self.states_name:
            mstate = getattr(self, mname)
            if isinstance(mstate, torch.Tensor):
                mstate = [mstate]
            if isinstance(mstate[0], list):  # Check if mstate is a list of pairs
                mstate = [torch.cat(pair, dim=0) for pair in mstate]
            mstate = torch.cat(mstate, dim=0)
            if prefilter and 'pred' in mname:
                mstate = mstate[mstate>self.pre_filtering_thresh]
            accumulated_metrics[mname]=mstate
        
        return accumulated_metrics
        
    
    def normalize_(self, data_in, prefilter=False):
        min_val = data_in.min()
        max_val = data_in.max()
        data_out = (data_in - min_val) / (max_val - min_val)
        if prefilter:
            data_out = data_out[data_out>self.pre_filtering_thresh]
        return data_out
    
    
    def compute(self):
        results = {}

        if  len(self.pred_weak_aug_lab_before_nms) >= self.reset_state_interval:
            self.iteration_count+=1
            
            accumulated_metrics = self._accumulate_metrics(prefilter=False)  # shape (N, 1)
            
            pred_weak_aug_unlab_before_nms   = accumulated_metrics['pred_weak_aug_unlab_before_nms'].view(-1)
            pred_weak_aug_lab_before_nms     = accumulated_metrics['pred_weak_aug_lab_before_nms'].view(-1)
            roi_label_weak_aug_unlab_before_nms   = accumulated_metrics['roi_label_weak_aug_unlab_before_nms'].view(-1) - 1
            roi_label_weak_aug_lab_before_nms     = accumulated_metrics['roi_label_weak_aug_lab_before_nms'].view(-1) - 1

            mu_cw_lab = torch.ones((self.num_classes)) / self.num_classes
            mu_cw_unlab = torch.ones((self.num_classes)) / self.num_classes

            for cind in range(self.num_classes):
                cw_pseudo_score_lab = pred_weak_aug_lab_before_nms[roi_label_weak_aug_lab_before_nms == cind]
                cw_pseudo_score_lab = cw_pseudo_score_lab[cw_pseudo_score_lab>self.pre_filtering_thresh]
                if not cw_pseudo_score_lab.shape[0]: continue
                mu_cw_lab[cind] = cw_pseudo_score_lab.mean()

                cw_pseudo_score_unlab = pred_weak_aug_unlab_before_nms[roi_label_weak_aug_unlab_before_nms == cind]
                cw_pseudo_score_unlab = cw_pseudo_score_unlab[cw_pseudo_score_unlab>self.pre_filtering_thresh]
                if not cw_pseudo_score_unlab.shape[0]: continue
                mu_cw_unlab[cind] = cw_pseudo_score_unlab.mean()


            if self.ema_cw_lab is None:
                self.ema_cw_lab = mu_cw_lab
            else:
                self.ema_cw_lab = self.momentum  * self.ema_cw_lab + (1 - self.momentum) * mu_cw_lab

            if self.ema_cw_unlab is None:
                self.ema_cw_unlab = mu_cw_unlab
            else:
                self.ema_cw_lab = self.momentum  * self.ema_cw_lab + (1 - self.momentum) * mu_cw_unlab

            # 1. relative classwise threshold using pred_weak_aug_lab_before_nms
            self.relative_cw_threshold = self.relative_val * mu_cw_lab
            self.relative_cw_ema_threshold = self.relative_val * self.ema_cw_lab

            mu_pred_weak_aug_unlab_before_nms   = pred_weak_aug_unlab_before_nms.mean()
            mu_pred_weak_aug_lab_before_nms     = pred_weak_aug_lab_before_nms.mean()

            if self.ema_pred_weak_aug_unlab_before_nms is None:
                self.ema_pred_weak_aug_unlab_before_nms =  mu_pred_weak_aug_unlab_before_nms
            else:    
                self.ema_pred_weak_aug_unlab_before_nms = self.momentum  * \
                    self.ema_pred_weak_aug_unlab_before_nms + (1 - self.momentum) * mu_pred_weak_aug_unlab_before_nms
            
            if self.ema_pred_weak_aug_lab_before_nms is None:
                self.ema_pred_weak_aug_lab_before_nms = mu_pred_weak_aug_lab_before_nms
            else:
                self.ema_pred_weak_aug_lab_before_nms = self.momentum  * \
                self.ema_pred_weak_aug_lab_before_nms + (1 - self.momentum) * mu_pred_weak_aug_lab_before_nms


            # 2. relative threshold using pred_weak_aug_lab_before_nms
            self.relative_threshold = self.relative_val * mu_pred_weak_aug_lab_before_nms
            self.relative_ema_threshold = self.relative_val * self.ema_pred_weak_aug_lab_before_nms
            
            # 3. DA of weak-augmnted-unlabeled data using target as weak-augmnted-labled (using Teacher predictions)
            pred_weak_aug_unlab_before_nms_aligned = pred_weak_aug_unlab_before_nms * \
                (self.ema_pred_weak_aug_lab_before_nms + 1e-6) / (self.ema_pred_weak_aug_unlab_before_nms + 1e-6)
            pred_weak_aug_unlab_before_nms_aligned = self.normalize_(pred_weak_aug_unlab_before_nms_aligned, prefilter=True)
            
            

            if self.enable_clipping:
                self.relative_threshold = torch.clip(self.relative_threshold, 0.1, 0.9)
                self.relative_ema_threshold = torch.clip(self.relative_ema_threshold, 0.1, 0.9)

            results['adamatch_mu_weak_unlab']= mu_pred_weak_aug_unlab_before_nms.item()
            results['adamatch_mu_weak_lab']= mu_pred_weak_aug_lab_before_nms.item()

            results['adamatch_ema_mu_weak_unlab']= self.ema_pred_weak_aug_unlab_before_nms.item()
            results['adamatch_ema_mu_weak_lab']= self.ema_pred_weak_aug_lab_before_nms.item()

            results['adamatch_mu_weak_lab_rt']= self.relative_threshold.item()
            results['adamatch_ema_mu_weak_lab_rt']= self.relative_ema_threshold.item()

            if self.enable_plots:
                HIST_BIN = np.linspace(self.pre_filtering_thresh, 1, 30)
                palettes = {t: c for t, c in zip(['fp', 'tn', 'tp', 'fn'], sns.color_palette("hls", 4))}
                BS = len(self.pred_weak_aug_lab_before_nms[0])
                WS = self.reset_state_interval * BS
                info = f"Iter: {self.iteration_count}    Interval: {self.reset_state_interval}    BS: {BS}    W: {(self.iteration_count - 1) * WS} - {self.iteration_count * WS}"
                
                # plot states
                num_rows = 2
                num_cols = len(self.states_name) // 2
                fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 6), sharex='col', sharey='row', layout="compressed")
                before_nms_states = [state_name for state_name in self.states_name if 'before_nms' in state_name]
                after_nms_states = [state_name for state_name in self.states_name if 'after_nms' in state_name]
                for col, state_name in enumerate(before_nms_states + after_nms_states):
                    row = 0 if col < num_cols else 1
                    col %= num_cols
                    current_metric = accumulated_metrics[state_name].view(-1).cpu().numpy()
                    axs[row, col].hist(current_metric, bins=HIST_BIN, alpha=0.7, label=state_name, edgecolor='black', color=palettes['fp'])
                    axs[row, col].axvline(current_metric.mean().item(), linestyle='--', label='mu', color=palettes['fp'], alpha=0.9)
                    if 'before_nms' in state_name:
                        axs[row, col].axvline(eval(f"self.ema_{state_name}").item(), linestyle='--', label='ema', color=palettes['tp'], alpha=0.9)
                    axs[row, col].legend(loc='upper right', fontsize='x-small')
                    axs[row, col].set_xlabel('score', fontsize='x-small')
                    axs[row, col].set_ylabel('count', fontsize='x-small')
                    axs[row, col].set_ylim(0, 800)
                plt.suptitle(info, fontsize='x-small')
                #fig_title = f'iteration_acc_{self.iteration_count}.png'
                #fig.get_figure().savefig(fig_title)
                results['adamatch_acc_states_plot'] = fig.get_figure()
                plt.close()


                fig, axs = plt.subplots(1, 1, figsize=(12, 6), layout="compressed")
                axs.hist(pred_weak_aug_unlab_before_nms.cpu().numpy(), bins=HIST_BIN, alpha=0.5, label='weak-aug unlab', edgecolor='black', color=palettes['fp'])
                axs.hist(pred_weak_aug_lab_before_nms.cpu().numpy(), bins=HIST_BIN, alpha=0.5, label='weak-aug lab', edgecolor='black', color=palettes['tp'])
                axs.hist(pred_weak_aug_unlab_before_nms_aligned.cpu().numpy(), bins=HIST_BIN, alpha=0.8, label='rectified weak-aug unlab', edgecolor='black', color=palettes['fn'])
                axs.axvline(self.ema_pred_weak_aug_unlab_before_nms.item(), linestyle='--', label='ema unlab', color=palettes['fp'], alpha=0.9)
                axs.axvline(self.ema_pred_weak_aug_lab_before_nms.item(), linestyle='--', label='ema lab (target)', color=palettes['tp'], alpha=0.9)
                axs.axvline(mu_pred_weak_aug_unlab_before_nms.item(), linestyle='--', label='mu unlab', color=palettes['fn'], alpha=0.9)
                axs.axvline(mu_pred_weak_aug_lab_before_nms.item(), linestyle='--', label='mu lab (target)', color=palettes['tn'], alpha=0.9)
                axs.legend(loc='upper right', fontsize='x-small')
                axs.set_xlabel('score', fontsize='x-small')
                axs.set_ylabel('count', fontsize='x-small')
                axs.set_ylim(0, 800)
                plt.suptitle(info, fontsize='x-small')
                #fig_title = f'iteration_acc_{self.iteration_count}.png'
                #fig.get_figure().savefig(fig_title)
                results['adamatch_weak_align_plot'] = fig.get_figure()
                plt.close()

            self.reset()

        return results