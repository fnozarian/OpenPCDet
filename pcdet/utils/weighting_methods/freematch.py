import torch
from torchmetrics import Metric

"""
FreeMatch based Thresholding (Local and Global)
"""
class FreeMatchThreshold(Metric):
    full_state_update: bool = False
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.config = kwargs.get('config', None)
        self.reset_state_interval = self.config.ROI_HEAD.ADAPTIVE_THRESH_CONFIG.get('RESET_STATE_INTERVAL', 32)
        self.pre_filtering_thresh=self.config.ROI_HEAD.ADAPTIVE_THRESH_CONFIG.get('PRE_FILTERING_THRESH', 0.1)
        self.enable_clipping = self.config.ROI_HEAD.ADAPTIVE_THRESH_CONFIG.get('ENABLE_CLIPPING', False)
        self.momentum= self.config.ROI_HEAD.ADAPTIVE_THRESH_CONFIG.get('MOMENTUM', 0.99)

        self.class_names = {0: 'Car', 1: 'Pedestrian', 2: 'Cyclist'}
        self.num_classes = 3
       
        self.add_state("unlab_roi_labels", default=[], dist_reduce_fx='cat')
        self.add_state("unlab_roi_scores", default=[], dist_reduce_fx='cat')
        self.add_state("unlab_gt_iou_of_rois", default=[], dist_reduce_fx='cat') #iou_wrt_pl

        self.cls_local_thresholds = torch.ones((self.num_classes)) / self.num_classes
        self.iou_local_thresholds = torch.ones((self.num_classes)) / self.num_classes
        self.cls_global_thresholds = torch.tensor(1.0 / self.num_classes)
        self.iou_global_thresholds = torch.tensor(1.0 / self.num_classes)
        self.cls_expected_score = torch.ones((self.num_classes)) / self.num_classes
        self.iou_expected_score = torch.ones((self.num_classes)) / self.num_classes



    def update(self, unlab_roi_labels: torch.Tensor, 
               unlab_roi_scores: torch.Tensor, 
               unlab_gt_iou_of_rois: torch.Tensor ) -> None:

        self.unlab_roi_labels.append(unlab_roi_labels)
        self.unlab_roi_scores.append(unlab_roi_scores)
        self.unlab_gt_iou_of_rois.append(unlab_gt_iou_of_rois)
        


    def compute(self):
        results = {}

        if  len(self.unlab_gt_iou_of_rois) >= self.reset_state_interval:

            unlab_roi_labels = [i.clone().detach() for i in self.unlab_roi_labels]
            unlab_roi_scores = [i.clone().detach() for i in self.unlab_roi_scores]    
            unlab_gt_iou_of_rois = [i.clone().detach() for i in self.unlab_gt_iou_of_rois]
            unlab_roi_labels = torch.cat(unlab_roi_labels, dim=0)
            unlab_roi_scores = torch.cat(unlab_roi_scores, dim=0)
            unlab_gt_iou_of_rois = torch.cat(unlab_gt_iou_of_rois, dim=0)
            unlab_roi_labels -= 1

            self.iou_global_thresholds = self.momentum  * self.iou_global_thresholds + (
                                1 - self.momentum) * unlab_gt_iou_of_rois[unlab_gt_iou_of_rois>self.pre_filtering_thresh].mean()
            self.cls_global_thresholds = self.momentum  * self.cls_global_thresholds + (
                                1 - self.momentum) * unlab_roi_scores[unlab_roi_scores>self.pre_filtering_thresh].mean()

            for cind in range(self.num_classes):
                cls_mask = unlab_roi_labels == cind
                cls_score = unlab_roi_scores[cls_mask]
                cls_score = cls_score[cls_score>self.pre_filtering_thresh]
                iou_score = unlab_gt_iou_of_rois[cls_mask]
                iou_score = iou_score[iou_score>self.pre_filtering_thresh]

                self.cls_expected_score = self.momentum  * self.cls_expected_score + (1 - self.momentum) * cls_score.mean()
                self.iou_expected_score = self.momentum  * self.iou_expected_score + (1 - self.momentum) * iou_score.mean()

            self.cls_local_thresholds = (self.cls_expected_score / self.cls_expected_score.max()) * self.cls_global_thresholds
            self.iou_local_thresholds = (self.iou_expected_score / self.iou_expected_score.max()) * self.iou_global_thresholds

            if self.enable_clipping:
                self.cls_local_thresholds = torch.clip(self.cls_local_thresholds, 0.1, 0.9)
                self.iou_local_thresholds = torch.clip(self.iou_local_thresholds, 0.1, 0.9)

            results.update(**{'freematch_cls_local_thr': {cls_name: self.cls_local_thresholds[i].item() for i, cls_name in self.class_names.items()},
                       'freematch_iou_local_thr': {cls_name: self.iou_local_thresholds[i].item() for i, cls_name in self.class_names.items()},
                       })
            
                
            self.reset()

        return results          