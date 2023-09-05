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
    probs_x_ulb_w = accumulated_metrics['pred_weak_aug_unlab'].view(-1)
    probs_x_lb_s = accumulated_metrics['pred_weak_aug_lab'].view(-1)
    self.p_model = self.momentum  * self.p_model + (1 - self.momentum) * torch.mean(probs_x_ulb_w)
    self.p_target = self.momentum  * self.p_target + (1 - self.momentum) * torch.mean(probs_x_lb_s)
    probs_x_ulb_aligned = probs_x_ulb_w * (self.p_target + 1e-6) / (self.p_model + 1e-6)
"""
class AdaMatchThreshold(Metric):
    full_state_update: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.config = kwargs.get('config', None)
        config_roi_head = self.config.ROI_HEAD.ADAPTIVE_THRESH_CONFIG
        self.reset_state_interval = config_roi_head.get('RESET_STATE_INTERVAL', 32)
        self.pre_filtering_thresh = config_roi_head.get('PRE_FILTERING_THRESH', 0.1)
        self.enable_plots = config_roi_head.get('ENABLE_PLOTS', False)
        self.enable_clipping = config_roi_head.get('ENABLE_CLIPPING', False)
        self.relative_val = config_roi_head.get('RELATIVE_VAL', 0.8)
        self.momentum = config_roi_head.get('MOMENTUM', 0.9)
        self.num_classes = 3
        self.iteration_count = 0
        self.states_name = []
        for batch_type in ['lab', 'unlab']:
            #    self.states_name.append(f'roi_score_weak_aug_{batch_type}')
            #    self.states_name.append(f'roi_labels_weak_aug_{batch_type}')
           self.states_name.append(f'pred_weak_aug_{batch_type}')
           self.states_name.append(f'pred_strong_aug_{batch_type}')
           self.states_name.append(f'rect_pred_weak_aug_{batch_type}')
           self.states_name.append(f'pred_pre_gt_sample_{batch_type}')

        for name in self.states_name:
            self.add_state(name, default=[], dist_reduce_fx='cat')

        self.relative_threshold = None
        self.relative_ema_threshold = None

        self.means = {}
        self.emas = {}

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
            if isinstance(mstate[0], list):
                mstate = [torch.cat(pair, dim=0) for pair in mstate]
            mstate = torch.cat(mstate, dim=0)
            if prefilter:
                mstate = mstate[mstate > self.pre_filtering_thresh]
            accumulated_metrics[mname] = mstate

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

        if len(self.pred_pre_gt_sample_lab) >= self.reset_state_interval:
            self.iteration_count += 1
            
            accumulated_metrics = self._accumulate_metrics(prefilter=True)
            for key, val in accumulated_metrics.items():
                self.means[key] = val.mean()
                self._update_ema(key)

            self._update_relative_thresholds(tag='pred_pre_gt_sample_lab')
            
            results.update(self._get_results_dict())
            if self.enable_plots:
                results['acc_states_plot'] = self._generate_histogram_plot(accumulated_metrics)

            self.reset()

        return results

    def _update_ema(self, tag):
        if self.emas.get(tag) is None:
            self.emas[tag] = self.means[tag]
        else:
            self.emas[tag] = self.momentum * self.emas[tag] + (1 - self.momentum) * self.means[tag]

    def _update_relative_thresholds(self, tag='pred_pre_gt_sample_lab'):
        self.relative_threshold = self.relative_val * self.means[tag]
        self.relative_ema_threshold = self.relative_val * self.emas[tag]

        if self.enable_clipping:
            self.relative_threshold = torch.clip(self.relative_threshold, 0.0, 1.0)
            self.relative_ema_threshold = torch.clip(self.relative_ema_threshold, 0.0, 1.0)

    def _generate_histogram_plot(self, accumulated_metrics):
        HIST_BIN = np.linspace(self.pre_filtering_thresh, 1, 30)
        palettes = {t: c for t, c in zip(['fp', 'tn', 'tp', 'fn'], sns.color_palette("hls", 4))}
        BS = len(self.pred_weak_aug_lab[0])
        WS = self.reset_state_interval * BS
        info = f"Iter: {self.iteration_count}    Interval: {self.reset_state_interval}    BS: {BS}    W: {(self.iteration_count - 1) * WS} - {self.iteration_count * WS}"
        num_rows = 2
        num_cols = len(self.states_name) // 2
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 6), sharex='col', sharey='row', layout="compressed")
        before_nms_states = [state_name for state_name in self.states_name if 'weak' in state_name]
        after_nms_states = [state_name for state_name in self.states_name if 'strong' in state_name or 'pred_pre_gt' in state_name]
        for col, state_name in enumerate(before_nms_states + after_nms_states):
            row = 0 if col < num_cols else 1
            col %= num_cols
            current_metric = accumulated_metrics[state_name].view(-1).cpu().numpy()
            axs[row, col].hist(current_metric, bins=HIST_BIN, alpha=0.7, label=state_name, edgecolor='black', color=palettes['fp'])
            axs[row, col].axvline(self.means[state_name].item(), linestyle='--', label='mu', color=palettes['fn'], alpha=0.9)
            axs[row, col].axvline(self.emas[state_name].item(), linestyle='--', label='ema', color=palettes['tn'], alpha=0.9)
            axs[row, col].legend(loc='upper right', fontsize='x-small')
            axs[row, col].set_xlabel('score', fontsize='x-small')
            axs[row, col].set_ylabel('count', fontsize='x-small')
            axs[row, col].set_ylim(0, 800)
        plt.suptitle(info, fontsize='x-small')
        hh = fig.get_figure()
        # fig_title = f'iteration_acc_{self.iteration_count}.png'
        # hh.savefig(fig_title)
        plt.close()
        return hh

    def _get_results_dict(self):
        results={}
        results['ema']={k: v.item() for k, v in self.emas.items()}
        results['mu']={k: v.item() for k, v in self.means.items()}
        results.update({'rt_mu_pred_pre_gt_sample_lab': self.relative_threshold.item(),
                        'rt_ema_pred_pre_gt_sample_lab': self.relative_ema_threshold.item(),
                        'ratio_mu' : (self.means['pred_pre_gt_sample_lab']/self.means['pred_weak_aug_unlab']).item(),
                        'ratio_ema': (self.emas['pred_pre_gt_sample_lab']/self.emas['pred_weak_aug_unlab']).item()})
        return results
    


'Needs revision'
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
        # TODO Discuss with Farzad, should only keep before nms metrics and ignore strong augmented (student dict)
        self.states_name = ['pred_weak_aug_unlab', 'roi_score_weak_aug_unlab', 'roi_labels_weak_aug_unlab',
                            'pred_weak_aug_lab', 'roi_score_weak_aug_lab', 'roi_labels_weak_aug_lab', 
                            'rectified_pred_weak_aug_unlab', 'roi_score_strong_aug_unlab', 'roi_labels_strong_aug_unlab',
                            'pred_weak_aug_lab', 'roi_score_strong_aug_lab', 'roi_labels_strong_aug_lab']

        for name in self.states_name:
            self.add_state(name, default=[], dist_reduce_fx='cat')

        self.relative_threshold = None
        self.relative_ema_threshold = None

        self.ema_pred_weak_aug_unlab = None
        self.ema_pred_weak_aug_lab = None
        
        self.ema_rectified_pred_weak_aug_unlab = None
        self.ema_pred_weak_aug_lab = None


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

        if  len(self.pred_weak_aug_lab) >= self.reset_state_interval:
            self.iteration_count+=1
            accumulated_metrics = self._accumulate_metrics(prefilter=False)  # shape (N, 1)
            
            pred_weak_aug_unlab = accumulated_metrics['pred_weak_aug_unlab'].view(-1)
            pred_weak_aug_lab  = accumulated_metrics['pred_weak_aug_lab'].view(-1)
            roi_score_weak_aug_unlab = accumulated_metrics['roi_score_weak_aug_unlab'].view(-1)
            roi_score_weak_aug_lab = accumulated_metrics['roi_score_weak_aug_lab'].view(-1)
            roi_labels_weak_aug_unlab = accumulated_metrics['roi_labels_weak_aug_unlab'].view(-1) 
            roi_labels_weak_aug_lab = accumulated_metrics['roi_labels_weak_aug_lab'].view(-1) 

            # Class-wise
            cw_mu_pseudo_score_weak_aug_lab = torch.ones((self.num_classes)) / self.num_classes
            cw_mu_pseudo_score_weak_aug_unlab = torch.ones((self.num_classes)) / self.num_classes
            cw_mu_roi_score_weak_aug_lab = torch.ones((self.num_classes)) / self.num_classes
            cw_mu_roi_score_weak_aug_unlab = torch.ones((self.num_classes)) / self.num_classes

            for cind in range(self.num_classes):
                lab_mask = roi_labels_weak_aug_lab == cind
                
                cw_pseudo_score_lab = pred_weak_aug_lab[lab_mask]
                cw_pseudo_score_lab = cw_pseudo_score_lab[cw_pseudo_score_lab > self.pre_filtering_thresh]
                if cw_pseudo_score_lab.shape[0]: 
                    cw_mu_pseudo_score_weak_aug_lab[cind] = cw_pseudo_score_lab.mean()

                cw_roi_score_lab = roi_score_weak_aug_lab[lab_mask]
                cw_roi_score_lab = cw_roi_score_lab[cw_roi_score_lab > self.pre_filtering_thresh]
                if cw_roi_score_lab.shape[0]:
                    cw_mu_roi_score_weak_aug_lab[cind] = cw_roi_score_lab.mean()

                unlab_mask = roi_labels_weak_aug_unlab == cind
                cw_pseudo_score_unlab = pred_weak_aug_unlab[unlab_mask]
                cw_pseudo_score_unlab = cw_pseudo_score_unlab[cw_pseudo_score_unlab > self.pre_filtering_thresh]
                if cw_pseudo_score_unlab.shape[0]:
                    cw_mu_pseudo_score_weak_aug_unlab[cind] = cw_pseudo_score_unlab.mean()

                cw_roi_score_unlab = roi_score_weak_aug_unlab[unlab_mask]
                cw_roi_score_unlab = cw_roi_score_unlab[cw_roi_score_unlab > self.pre_filtering_thresh]
                if cw_roi_score_unlab.shape[0]:
                    cw_mu_roi_score_weak_aug_unlab[cind] = cw_roi_score_unlab.mean()
            
            # EMA of classwise Mean ==> Class-Expectation
            if self.ema_cw_mu_pseudo_score_weak_aug_lab is None:
                self.ema_cw_mu_pseudo_score_weak_aug_lab = cw_mu_pseudo_score_weak_aug_lab
            else:
                self.ema_cw_mu_pseudo_score_weak_aug_lab = self.momentum * \
                    self.ema_cw_mu_pseudo_score_weak_aug_lab + (1 - self.momentum) * cw_mu_pseudo_score_weak_aug_lab

            if self.ema_cw_mu_pseudo_score_weak_aug_unlab is None:
                self.ema_cw_mu_pseudo_score_weak_aug_unlab = cw_mu_pseudo_score_weak_aug_unlab
            else:
                self.ema_cw_mu_pseudo_score_weak_aug_unlab = self.momentum * \
                    self.ema_cw_mu_pseudo_score_weak_aug_unlab + (1 - self.momentum) * cw_mu_pseudo_score_weak_aug_unlab

            if self.ema_cw_mu_roi_score_weak_aug_lab is None:
                self.ema_cw_mu_roi_score_weak_aug_lab = cw_mu_roi_score_weak_aug_lab
            else:
                self.ema_cw_mu_roi_score_weak_aug_lab = self.momentum * \
                    self.ema_cw_mu_roi_score_weak_aug_lab + (1 - self.momentum) * cw_mu_roi_score_weak_aug_lab

            if self.ema_cw_mu_roi_score_weak_aug_unlab is None:
                self.ema_cw_mu_roi_score_weak_aug_unlab = cw_mu_roi_score_weak_aug_unlab
            else:
                self.ema_cw_mu_roi_score_weak_aug_unlab = self.momentum * \
                    self.ema_cw_mu_roi_score_weak_aug_unlab + (1 - self.momentum) * cw_mu_roi_score_weak_aug_unlab

            # (Noisy) Relative classwise threshold using labeled pseudo_score and roi_score
            self.relative_threshold_cw_mu_pseudo_score_weak_aug_lab = self.relative_val * cw_mu_pseudo_score_weak_aug_lab
            self.relative_threshold_cw_mu_roi_score_weak_aug_lab = self.relative_val * cw_mu_roi_score_weak_aug_lab

            # (EMA) Relative classwise threshold using labeled pseudo_score and roi_score
            self.relative_threshold_ema_cw_mu_pseudo_score_weak_aug_lab = self.relative_val * self.ema_cw_mu_pseudo_score_weak_aug_lab
            self.relative_threshold_ema_cw_mu_roi_score_weak_aug_lab = self.relative_val * self.ema_cw_mu_roi_score_weak_aug_lab




            ## Class-agnostic
            mu_pseudo_score_weak_aug_unlab   = pred_weak_aug_unlab.mean()
            mu_pseudo_score_weak_aug_lab     = pred_weak_aug_lab.mean()
            mu_roi_score_weak_aug_unlab   = roi_score_weak_aug_unlab.mean()
            mu_roi_score_weak_aug_lab     = roi_score_weak_aug_lab.mean()

            if self.ema_mu_pseudo_score_weak_aug_unlab is None:
                self.ema_mu_pseudo_score_weak_aug_unlab=  mu_pseudo_score_weak_aug_unlab
            else:    
                self.ema_mu_pseudo_score_weak_aug_unlab = self.momentum  * \
                    self.ema_mu_pseudo_score_weak_aug_unlab + (1 - self.momentum) * mu_pseudo_score_weak_aug_unlab
            
            if self.ema_mu_pseudo_score_weak_aug_lab is None:
                self.ema_mu_pseudo_score_weak_aug_lab=  mu_pseudo_score_weak_aug_lab
            else:    
                self.ema_mu_pseudo_score_weak_aug_lab = self.momentum  * \
                    self.ema_mu_pseudo_score_weak_aug_lab + (1 - self.momentum) * mu_pseudo_score_weak_aug_lab

            if self.ema_mu_roi_score_weak_aug_unlab is None:
                self.ema_mu_roi_score_weak_aug_unlab=  mu_roi_score_weak_aug_unlab
            else:    
                self.ema_mu_roi_score_weak_aug_unlab = self.momentum  * \
                    self.ema_mu_roi_score_weak_aug_unlab + (1 - self.momentum) * mu_roi_score_weak_aug_unlab
            
            if self.ema_mu_roi_score_weak_aug_lab is None:
                self.ema_mu_roi_score_weak_aug_lab=  mu_roi_score_weak_aug_lab
            else:    
                self.ema_mu_roi_score_weak_aug_lab = self.momentum  * \
                    self.ema_mu_roi_score_weak_aug_lab + (1 - self.momentum) * mu_roi_score_weak_aug_lab
                

            # (Noisy) Relative classagnostic threshold using labeled pseudo_score and roi_score
            self.relative_threshold_mu_pseudo_score_weak_aug_lab = self.relative_val * mu_pseudo_score_weak_aug_lab
            self.relative_threshold_mu_roi_score_weak_aug_lab = self.relative_val * mu_roi_score_weak_aug_lab

            # (EMA) Relative classwise threshold using labeled pseudo_score and roi_score
            self.relative_threshold_ema_mu_pseudo_score_weak_aug_lab = self.relative_val * self.ema_mu_pseudo_score_weak_aug_lab
            self.relative_threshold_ema_mu_roi_score_weak_aug_lab = self.relative_val * self.ema_mu_roi_score_weak_aug_lab
            
            # 3. DA of weak-augmnted-unlabeled data using target as weak-augmnted-labled (using Teacher predictions)
            pred_weak_aug_unlab_aligned = pred_weak_aug_unlab * \
                (self.ema_mu_pseudo_score_weak_aug_lab + 1e-6) / (self.ema_mu_pseudo_score_weak_aug_unlab + 1e-6)
            pred_weak_aug_unlab_aligned = self.normalize_(pred_weak_aug_unlab_aligned, prefilter=True)

            roi_score_weak_aug_unlab_aligned = roi_score_weak_aug_unlab * \
                (self.ema_mu_roi_score_weak_aug_lab + 1e-6) / (self.ema_mu_roi_score_weak_aug_unlab + 1e-6)
            roi_score_weak_aug_unlab_aligned = self.normalize_(roi_score_weak_aug_unlab_aligned, prefilter=True)            
            
            mu_aligned_pseudo_score_weak_aug_unlab   = pred_weak_aug_unlab_aligned.mean()
            mu_aligned_roi_score_weak_aug_unlab   = roi_score_weak_aug_unlab_aligned.mean()

            if self.enable_clipping:
                self.relative_threshold_cw_mu_pseudo_score_weak_aug_lab  = torch.clip(self.relative_threshold_cw_mu_pseudo_score_weak_aug_lab , 0.1, 0.9)
                self.relative_threshold_cw_mu_roi_score_weak_aug_lab  = torch.clip(self.relative_threshold_cw_mu_roi_score_weak_aug_lab , 0.1, 0.9)

                self.relative_threshold_ema_cw_mu_pseudo_score_weak_aug_lab  = torch.clip(self.relative_threshold_ema_cw_mu_pseudo_score_weak_aug_lab , 0.1, 0.9)
                self.relative_threshold_ema_cw_mu_roi_score_weak_aug_lab  = torch.clip(self.relative_threshold_ema_cw_mu_roi_score_weak_aug_lab , 0.1, 0.9)

                self.relative_mu_pseudo_score_weak_aug_lab = torch.clip(self.relative_mu_pseudo_score_weak_aug_lab, 0.1, 0.9)
                self.relative_mu_roi_score_weak_aug_lab = torch.clip(self.relative_mu_roi_score_weak_aug_lab, 0.1, 0.9)

                self.relative_ema_mu_pseudo_score_weak_aug_lab = torch.clip(self.relative_ema_mu_pseudo_score_weak_aug_lab, 0.1, 0.9)
                self.relative_ema_mu_roi_score_weak_aug_lab = torch.clip(self.relative_ema_mu_roi_score_weak_aug_lab, 0.1, 0.9)
              
                
               
            

            class_agnostic = {
            'mu_pseudo_score_weak_aug_unlab': mu_pseudo_score_weak_aug_unlab.item(),
            'mu_pseudo_score_weak_aug_lab': mu_pseudo_score_weak_aug_lab.item(),
            'mu_roi_score_weak_aug_unlab' : mu_roi_score_weak_aug_unlab.item(),
            'mu_roi_score_weak_aug_lab': mu_roi_score_weak_aug_lab.item(),
            'mu_aligned_pseudo_score_weak_aug_unlab': mu_aligned_pseudo_score_weak_aug_unlab.item(),
            'mu_aligned_roi_score_weak_aug_unlab': mu_aligned_roi_score_weak_aug_unlab.item(),
            'relative_mu_pseudo_score_weak_aug_lab': self.relative_mu_pseudo_score_weak_aug_lab.item(),
            'relative_ema_mu_pseudo_score_weak_aug_lab': self.relative_ema_mu_pseudo_score_weak_aug_lab.item(),
            }
            results.update(
                **{
                    'cw_mu_pseudo_score_weak_aug_unlab': {cls_name: cw_mu_pseudo_score_weak_aug_unlab[i].item() for i, cls_name in self.class_names.items()},
                    'cw_mu_pseudo_score_weak_aug_lab': {cls_name: mu_pseudo_score_weak_aug_lab[i].item() for i, cls_name in self.class_names.items()},
                    'cw_mu_roi_score_weak_aug_unlab': {cls_name: mu_roi_score_weak_aug_unlab[i].item() for i, cls_name in self.class_names.items()},
                    'cw_mu_roi_score_weak_aug_lab': {cls_name:mu_roi_score_weak_aug_lab[i].item() for i, cls_name in self.class_names.items()},
                    'relative_mu_pseudo_score_weak_aug_lab': {cls_name: self.relative_mu_pseudo_score_weak_aug_lab[i].item() for i, cls_name in self.class_names.items()},
                    'relative_ema_mu_pseudo_score_weak_aug_lab': {cls_name: self.relative_ema_mu_pseudo_score_weak_aug_lab[i].item() for i, cls_name in self.class_names.items()},
            })


            results['adamatch_mu_weak_unlab']= mu_pred_weak_aug_unlab.item()
            results['adamatch_mu_weak_lab']= mu_pred_weak_aug_labitem()

            results['adamatch_ema_mu_weak_unlab']= self.ema_pred_weak_aug_unlab.item()
            results['adamatch_ema_mu_weak_lab']= self.ema_pred_weak_aug_lab.item()

            results['adamatch_mu_weak_lab_rt']= self.relative_threshold.item()
            results['adamatch_ema_mu_weak_lab_rt']= self.relative_ema_threshold.item()

            

            self.reset()

        return results