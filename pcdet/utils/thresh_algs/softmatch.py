import matplotlib.pyplot as plt
import numpy as np
import torch
from torchmetrics import Metric
"""
SoftMatch based Thresholding (Local and Global) Not finalised
"""
class SoftMatchThreshold(Metric):
    full_state_update: bool = False
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reset_state_interval = kwargs.get('RESET_STATE_INTERVAL', 64)
        self.interval_size = kwargs.get('INTERVAL_SIZE', 16)
        self.PRE_FILTERING_THRESH = kwargs.get('PRE_FILTERING_THRESH', 0.1)
        self.enable_plots = kwargs.get('ENABLE_PLOTS', True)
        self.enable_clipping = kwargs.get('enable_clipping', True)
        self.momentum = kwargs.get('momentum', 0.99)
        self.quantile = kwargs.get('quantile', False)
        self.lambdax = kwargs.get('lambdax', 0.9)
        self.n_sigma = kwargs.get('n_sigma', 2)
        self.dist_align = kwargs.get('dist_align', False)

        self.class_names = {0: 'Car', 1: 'Pedestrian', 2: 'Cyclist'}
        self.num_classes = 3
        self.add_state("lab_iou_score", default=[], dist_reduce_fx='cat')
        self.add_state("lab_iou_label", default=[], dist_reduce_fx='cat')
        self.add_state("unlab_iou_score", default=[], dist_reduce_fx='cat')
        self.add_state("unlab_iou_label", default=[], dist_reduce_fx='cat')

        self.weights = None
        self.local_thresholds = None
        self.global_thresholds = None
        self.iteration_count = 0
        self.prob_max_mu_t = torch.tensor(1.0 / self.num_classes)
        self.prob_max_var_t = torch.tensor(1.0)
        self.lab_prob_t = torch.ones((self.num_classes)) / self.num_classes
        self.unlab_prob_t = torch.ones((self.num_classes)) / self.num_classes



    def update(self, lab_iou_label: torch.Tensor, lab_iou_score: torch.Tensor, unlab_iou_label: torch.Tensor,
               unlab_iou_score: torch.Tensor) -> None:
        # Unsqueeze for DDP
        if lab_iou_label.ndim == 1:
            lab_iou_label = lab_iou_label.unsqueeze(dim=0)
        if lab_iou_score.ndim == 1:
            lab_iou_score = lab_iou_score.unsqueeze(dim=0)
        if unlab_iou_label.ndim == 1:
            unlab_iou_label = unlab_iou_label.unsqueeze(dim=0)
        if unlab_iou_score.ndim == 1:
            unlab_iou_score = unlab_iou_score.unsqueeze(dim=0)
        self.lab_iou_score.append(lab_iou_score)
        self.lab_iou_label.append(lab_iou_label)
        self.unlab_iou_score.append(unlab_iou_score)
        self.unlab_iou_label.append(unlab_iou_label)


    def compute(self):
        results = {}
        # sliding window with a step size of interval_size//2
        step_size = self.interval_size // 2
        start_index = self.iteration_count
        end_index = self.iteration_count + self.interval_size

        sample_length = len(self.lab_iou_score)
        if sample_length >= end_index:
            
            self.iteration_count += step_size  # update for next iteration
            softmatch_dict={c:{} for _, c in self.class_names.items()} 

            unlab_iou_score = [i.detach().cpu() for i in self.unlab_iou_score[start_index:end_index]]
            unlab_iou_label = [i.detach().cpu() for i in self.unlab_iou_label[start_index:end_index]]
            lab_iou_score = [i.detach().cpu() for i in self.lab_iou_score[start_index:end_index]]
            lab_iou_label = [i.detach().cpu() for i in self.lab_iou_label[start_index:end_index]]

            unlab_iou_score = torch.cat(unlab_iou_score, dim=0)
            unlab_iou_label = torch.cat(unlab_iou_label, dim=0)
            lab_iou_score = torch.cat(lab_iou_score, dim=0)
            lab_iou_label = torch.cat(lab_iou_label, dim=0)

            self.weights = torch.ones_like(unlab_iou_score) 
            if self.lambdax is not None:
                self.weights *=self.lambdax
            valid_mask = lab_iou_score > self.PRE_FILTERING_THRESH
            lab_iou_score = lab_iou_score[valid_mask]
            lab_iou_label = lab_iou_label[valid_mask]

            valid_mask = unlab_iou_score > self.PRE_FILTERING_THRESH
            unlab_iou_score = unlab_iou_score[valid_mask]
            unlab_iou_label = unlab_iou_label[valid_mask]

            
            
            for class_index, class_name in self.class_names.items():
                
                lab_mask = lab_iou_label == (class_index + 1)
                lab_class_score = lab_iou_score[lab_mask]
                self.lab_prob_t[class_index] = self.momentum * self.lab_prob_t[class_index] + (1 - self.momentum) * (
                    torch.quantile(lab_class_score, 0.8) if self.quantile else torch.mean(lab_class_score))

                unlab_mask = unlab_iou_label == (class_index + 1)
                unlab_class_score = unlab_iou_score[unlab_mask]
                self.unlab_prob_t[class_index] = self.momentum * self.unlab_prob_t[class_index] + (1 - self.momentum) * (
                    torch.quantile(unlab_class_score, 0.8) if self.quantile else torch.mean(unlab_class_score))

                unlab_classwise_max =  torch.max(unlab_class_score)# classwise max score
                # domain alignment: adjust scores based on dist
                if self.dist_align:
                    unlab_class_score = unlab_class_score * self.lab_prob_t[class_index] / self.unlab_prob_t[class_index]
                    #unlab_class_score = unlab_class_score / unlab_class_score.sum()# makes it 0-1 which might be not required for iou-scores
                    softmatch_dict[class_name]['unlab_class_score_DA'] = unlab_class_score
                    
                softmatch_dict[class_name]['lab_mask'] = lab_mask
                softmatch_dict[class_name]['lab_class_score'] = lab_class_score
                softmatch_dict[class_name]['unlab_mask'] = unlab_mask
                softmatch_dict[class_name]['unlab_class_score'] = unlab_iou_score[unlab_mask].clone()
                # scalars
                softmatch_dict[class_name]['unlab_classwise_max'] = unlab_classwise_max
                softmatch_dict[class_name]['lab_unlab_ratio'] = self.lab_prob_t[class_index] / self.unlab_prob_t[class_index]

            # Update EMA
            max_un_score =torch.stack([softmatch_dict[class_name]['unlab_classwise_max'] for _, class_name in self.class_names.items()])
            self.prob_max_mu_t = self.momentum * self.prob_max_mu_t + (1 - self.momentum) * torch.mean(max_un_score)
            self.prob_max_var_t = self.momentum * self.prob_max_var_t + (1 - self.momentum) * torch.var(max_un_score, unbiased=True)

            # Calculate mask
            for class_index, class_name in self.class_names.items():
                diff = torch.square(softmatch_dict[class_name]['unlab_class_score'] - self.prob_max_mu_t)
                scaled_var_t = self.prob_max_var_t / (self.n_sigma ** 2)
                self.weights[valid_mask][softmatch_dict[class_name]['unlab_mask']] *= torch.exp(-diff / (2 * scaled_var_t))


            
            results.update(**{
                'lab_prob_t': {cls_name: self.lab_prob_t[i] for i, cls_name in self.class_names.items()},
                'unlab_prob_t': {cls_name: self.unlab_prob_t[i] for i, cls_name in self.class_names.items()},
                'prob_max_mu_t': self.prob_max_mu_t,
                'prob_max_var_t': self.prob_max_var_t
            })
            if self.enable_plots:
                # Create figures and axes
                fig, axs = plt.subplots(1, 3, figsize=(20, 8), sharey=True)
                info = f"Iter {start_index} : {end_index}"

                for class_index, class_name in self.class_names.items():
                    # Plot histograms and prob_max_mu_t
                    bins = 20
                    alpha = 0.6  # Adjust transparency for the unlabeled scores histogram
                    
                    axs[class_index].hist(softmatch_dict[class_name]['lab_class_score'], bins=bins, edgecolor='black',
                                        label='Labeled Scores', alpha=alpha)
                    axs[class_index].hist(softmatch_dict[class_name]['unlab_class_score'], bins=bins, edgecolor='black',
                                        label='Unlabeled Scores', alpha=alpha)
                    if self.dist_align:
                        axs[class_index].hist(softmatch_dict[class_name]['unlab_class_score_DA'], bins=bins, edgecolor='black',
                                            label='Unlabeled Scores with DA', alpha=alpha)
                    axs[class_index].axvline(self.prob_max_mu_t, color='red', linestyle='--', label='EMA (max-mu)')
                    axs[class_index].axvline(torch.tensor(1.0 / self.num_classes), color='c', linestyle='--', label='1/C')
                    axs[class_index].set_xlabel('RoI IoU wrt PL', fontsize=12)
                    axs[class_index].set_ylabel('Count', fontsize=12)
                    axs[class_index].set_title(f'{class_name} Scores', fontsize=14)
                    axs[class_index].grid(True, alpha=0.2)  # Add grid lines
                    
                    # Adjust legend position to avoid overlapping with bars
                    axs[class_index].legend(loc='upper right', bbox_to_anchor=(1, 1))

                fig.suptitle(info, fontsize='small')
                plt.tight_layout()
                results['iou_cls_wrt_pl_softmatch'] = fig.get_figure()


        
            if  sample_length >= self.reset_state_interval:
                # reset state/records and iteration_count
                self.reset()
                self.iteration_count=0    
        
        return results   

    def get_value(self):
        self.compute()
        return self.global_thresholds, self.local_thresholds, self.weights
    


