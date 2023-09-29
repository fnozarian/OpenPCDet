import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.mixture import GaussianMixture
from torchmetrics import Metric
"""
Consistent-Teacher Adaptive Local Thresholding using GMM
"""


class AdaptiveThresholdGMM(Metric):
    full_state_update: bool = False
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.config = kwargs.get('config', None)
        self.reset_state_interval = self.config.ROI_HEAD.ADAPTIVE_THRESH_CONFIG.get('RESET_STATE_INTERVAL', 32)
        self.pre_filtering_thresh=self.config.ROI_HEAD.ADAPTIVE_THRESH_CONFIG.get('PRE_FILTERING_THRESH', 0.1)
        self.enable_plots=self.config.ROI_HEAD.ADAPTIVE_THRESH_CONFIG.get('ENABLE_PLOTS', False)
        self.enable_clipping = self.config.ROI_HEAD.ADAPTIVE_THRESH_CONFIG.get('ENABLE_CLIPPING', False)
        self.momentum= self.config.ROI_HEAD.ADAPTIVE_THRESH_CONFIG.get('MOMENTUM', 0.99)
        self.gmm_policy=self.config.ROI_HEAD.ADAPTIVE_THRESH_CONFIG.get('GMM_POLICY','high')
        self.mu1=self.config.ROI_HEAD.ADAPTIVE_THRESH_CONFIG.get('MU1',0.1)
        self.mu2=self.config.ROI_HEAD.ADAPTIVE_THRESH_CONFIG.get('MU2',0.9)

        self.class_names = {0: 'Car', 1: 'Pedestrian', 2: 'Cyclist'}
        self.num_classes = 3
        self.states_name = ["pred_labels", "pseudo_score", "pseudo_sem_score"]
        for name in self.states_name:
            self.add_state(name, default=[], dist_reduce_fx='cat')
        
        self.pseudo_score_cw_thresholds = torch.ones((self.num_classes)) / self.num_classes
        self.pseudo_sem_score_cw_thresholds = torch.ones((self.num_classes)) / self.num_classes
        self.iteration_count = 0
        self.gmm = GaussianMixture(
            n_components=2,
            weights_init=[0.5, 0.5],
            means_init=[[self.mu1], [self.mu2]],
            precisions_init=[[[1.0]], [[1.0]]],
            init_params='k-means++',
            tol=1e-9,
            max_iter=1000
        )
        
    def _accumulate_metrics(self):
        accumulated_metrics = {}
        for mname in self.states_name:
            mstate = getattr(self, mname)
            if isinstance(mstate, torch.Tensor):
                mstate = [mstate]
            if isinstance(mstate[0], list):  # Check if mstate is a list of pairs
                mstate = [torch.cat(pair, dim=0) for pair in mstate]
            accumulated_metrics[mname] = torch.cat(mstate, dim=0)
        return accumulated_metrics

    def update(self, pred_labels: torch.Tensor,  
               pseudo_score: torch.Tensor , pseudo_sem_score: torch.Tensor) -> None:

        self.pred_labels.append(pred_labels)
        self.pseudo_score.append(pseudo_score)
        self.pseudo_sem_score.append(pseudo_sem_score)

    def compute(self):
        results = {}

        if  len(self.pred_labels) >= self.reset_state_interval:
            self.iteration_count+=1
            accumulated_metrics = self._accumulate_metrics()  # shape (N, 1)
            pred_labels = accumulated_metrics["pred_labels"].view(-1)
            pseudo_score = accumulated_metrics["pseudo_score"].view(-1)
            pseudo_sem_score = accumulated_metrics["pseudo_sem_score"].view(-1)
            pred_labels -= 1
            
            # default initialaisation
            pseudo_score_loc_thr = np.ones((self.num_classes), dtype=np.float32) / self.num_classes  
            pseudo_sem_score_loc_thr = np.ones((self.num_classes), dtype=np.float32) / self.num_classes  

            pseudo_score_plot_handler = {cind:None for cind in range(self.num_classes)}
            pseudo_sem_score_plot_handler = {cind:None for cind in range(self.num_classes)}
            
            for cind in range(self.num_classes):
                info_=f"{self.class_names[cind]} Iter {(self.iteration_count-1)*self.reset_state_interval*len(self.pred_labels[0])} : {self.iteration_count*self.reset_state_interval*len(self.pred_labels[0])}\n"
                
                # pseudo_score
                cw_pseudo_score = pseudo_score[pred_labels == cind]
                cw_pseudo_score =cw_pseudo_score[cw_pseudo_score>self.pre_filtering_thresh].cpu().numpy()
                
                if cw_pseudo_score.shape[0]<4: continue # we dont have much samples to proceed
                
                cw_pseudo_score = cw_pseudo_score.reshape(-1, 1)
                self.gmm.fit(cw_pseudo_score)  
                gmm_assignment = self.gmm.predict(cw_pseudo_score)  
                gmm_scores = self.gmm.score_samples(cw_pseudo_score) 
                adaptive_thr = apply_policy_gmm(cw_pseudo_score, gmm_assignment, gmm_scores, gmm_policy=self.gmm_policy)
                if self.enable_plots:
                    pseudo_score_plot_handler[cind]=plot_gmm_modes(
                        cw_pseudo_score, self.gmm, gmm_assignment, 
                        default_thr=pseudo_score_loc_thr[cind], 
                        adaptive_thr=adaptive_thr, 
                        ema_thr=self.pseudo_score_cw_thresholds[cind], 
                        info=info_ + f" Pseudo Score GMM Modeling\nConverged: {self.gmm.converged_}\nLogLikelihood: {self.gmm.lower_bound_:.2f}\nGMM-Niter: {self.gmm.n_iter_}"
                    )
                if adaptive_thr is not None: pseudo_score_loc_thr[cind] = adaptive_thr

                # pseudo_sem_score
                cw_pseudo_sem_score = pseudo_sem_score[pred_labels == cind]
                cw_pseudo_sem_score =cw_pseudo_sem_score[cw_pseudo_sem_score>self.pre_filtering_thresh].cpu().numpy()
                
                if cw_pseudo_sem_score.shape[0]<4: continue # we dont have much samples to proceed
                
                cw_pseudo_sem_score = cw_pseudo_sem_score.reshape(-1, 1)
                self.gmm.fit(cw_pseudo_sem_score)  
                gmm_assignment = self.gmm.predict(cw_pseudo_sem_score)  
                gmm_scores = self.gmm.score_samples(cw_pseudo_sem_score) 
                adaptive_thr = apply_policy_gmm(cw_pseudo_sem_score, gmm_assignment, gmm_scores, gmm_policy=self.gmm_policy)
                if self.enable_plots:
                    pseudo_sem_score_plot_handler[cind]=plot_gmm_modes(
                        cw_pseudo_sem_score, self.gmm, gmm_assignment, 
                        default_thr=pseudo_sem_score_loc_thr[cind], 
                        adaptive_thr=adaptive_thr, 
                        ema_thr=self.pseudo_sem_score_cw_thresholds[cind], 
                        info=info_ + f" Pseudo Sem Score GMM Modeling\nConverged: {self.gmm.converged_}\nLogLikelihood: {self.gmm.lower_bound_:.2f}\nGMM-Niter: {self.gmm.n_iter_}"
                    )
                if adaptive_thr is not None:
                    pseudo_sem_score_loc_thr[cind] = adaptive_thr


            
            self.pseudo_score_cw_thresholds = self.momentum  * self.pseudo_score_cw_thresholds + (1 - self.momentum) * pseudo_score_loc_thr
            self.pseudo_sem_score_cw_thresholds = self.momentum  * self.pseudo_sem_score_cw_thresholds + (1 - self.momentum) * pseudo_sem_score_loc_thr
            if self.enable_clipping:
                self.pseudo_score_cw_thresholds = torch.clip(self.pseudo_score_cw_thresholds, 0.1, 0.9)
                self.pseudo_sem_score_cw_thresholds = torch.clip(self.pseudo_sem_score_cw_thresholds, 0.1, 0.9)
            
            results.update(**{'cons_teacher_pseudo_score_local_thr': {cls_name: self.pseudo_score_cw_thresholds[i].item() for i, cls_name in self.class_names.items()},
                       'cons_teacher_pseudo_sem_score_local_thr': {cls_name: self.pseudo_sem_score_cw_thresholds[i].item() for i, cls_name in self.class_names.items()},
                       })
            if self.enable_plots:
                results.update(**{
                'pseudo_score_gmm_modes': {cls_name: pseudo_score_plot_handler[i] for i, cls_name in self.class_names.items() if pseudo_score_plot_handler[i] is not None},
                'pseudo_sem_score_gmm_modes': {cls_name: pseudo_sem_score_plot_handler[i] for i, cls_name in self.class_names.items() if pseudo_sem_score_plot_handler[i] is not None}
                })

            self.reset()

        return results          


def apply_policy_gmm(scores, gmm_assignment, gmm_scores, gmm_policy='high'):
    adaptive_thr = None
    if np.any(gmm_assignment == 1):  
        if gmm_policy == 'high':
            gmm_scores[gmm_assignment == 0] = -np.inf  
            index = np.argmax(gmm_scores, axis=0) 
            pos_indx = ((gmm_assignment == 1) & (scores >= scores[index]).squeeze())  
            if np.sum(pos_indx):  adaptive_thr = np.min(scores[pos_indx])
        elif gmm_policy == 'middle': adaptive_thr = np.min(scores[gmm_assignment == 1])
        elif gmm_policy == 'percentile75': adaptive_thr = np.percentile(scores[gmm_assignment == 1], 75)
        elif gmm_policy == 'percentile25': adaptive_thr = np.percentile(scores[gmm_assignment == 1], 25)
        else:
            raise ValueError("Invalid policy. Policy can be 'high', 'middle', 'percentile75', or 'percentile25'.")
    return adaptive_thr

def plot_gmm_modes(scores, gmm, gmm_assignment, default_thr=None, adaptive_thr=None, ema_thr=None, info=None):    
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    _, bins = np.histogram(scores, bins=50)

    # Plot GMM components on the histogram
    gmm_scores = gmm.score_samples(bins.reshape(-1, 1))
    component_0_scores = scores[gmm_assignment == 0]
    component_1_scores = scores[gmm_assignment == 1]
    axs[0].hist(component_0_scores, bins=bins, color='b', alpha=0.5, label='Component 0')
    axs[0].hist(component_1_scores, bins=bins, color='r', alpha=0.5, label='Component 1')
    axs[0].set_xlabel('Scores',fontsize='x-small')
    axs[0].set_ylabel('Count',fontsize='x-small')
    axs[0].set_title('Histogram of GMM Components',fontsize='x-small')
    axs[0].legend(fontsize='x-small')
    axs[0].grid(True) 

    # Plot GMM PDF and mean values with decision boundaries
    gmm_x = np.linspace(np.min(scores), np.max(scores), 1000).reshape(-1, 1)
    gmm_scores = gmm.score_samples(gmm_x)
    axs[1].plot(gmm_x, np.exp(gmm_scores), color='k', label='GMM PDF')
    axs[1].axvline(x=gmm.means_[0], color='b', linestyle='--', label='mu 0')
    axs[1].axvline(x=gmm.means_[1], color='r', linestyle='--', label='mu 1')
    if adaptive_thr is not None:
        axs[1].axvline(x=adaptive_thr, color='m', linestyle='--', label='Ada')
    if default_thr is not None:
        axs[1].axvline(x=default_thr, color='c', linestyle='--', label='1/C')
    if ema_thr is not None:
        axs[1].axvline(x=ema_thr, color='y', linestyle='--', label='EMA')
    axs[1].set_xlabel('Scores',fontsize='x-small')
    axs[1].set_ylabel('Density',fontsize='x-small')
    axs[1].set_title('GMM PDF',fontsize='x-small')
    axs[1].legend(fontsize='x-small')
    axs[1].grid(True) 
    if info is not None:
        fig.suptitle(info, fontsize='x-small')
    fig.tight_layout()
    hh_ = fig.get_figure()
    plt.close()
    return hh_