import torch
from torchmetrics import Metric

"""
Adamatch based relative Thresholding
"""
class AdaMatchThreshold(Metric):
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
        
        self.add_state("lab_roi_labels", default=[], dist_reduce_fx='cat')
        self.add_state("lab_roi_scores", default=[], dist_reduce_fx='cat')
        self.add_state("lab_gt_iou_of_rois", default=[], dist_reduce_fx='cat')

        self.cls_local_thresholds = torch.ones((self.num_classes)) / self.num_classes
        self.iou_local_thresholds = torch.ones((self.num_classes)) / self.num_classes

    def update(self, lab_roi_labels: torch.Tensor, 
               lab_roi_scores: torch.Tensor,
              lab_gt_iou_of_rois: torch.Tensor, ) -> None:

        self.lab_roi_labels.append(lab_roi_labels)
        self.lab_roi_scores.append(lab_roi_scores)
        self.lab_gt_iou_of_rois.append(lab_gt_iou_of_rois)
        


    def compute(self):
        results = {}

        if  len(self.unlab_gt_iou_of_rois) >= self.reset_state_interval:

            lab_roi_labels = [i.clone().detach() for i in self.lab_roi_labels]
            lab_roi_scores = [i.clone().detach() for i in self.lab_roi_scores]    
            lab_gt_iou_of_rois = [i.clone().detach() for i in self.lab_gt_iou_of_rois]
            lab_roi_labels = torch.cat(lab_roi_labels, dim=0)
            lab_roi_scores = torch.cat(lab_roi_scores, dim=0)
            lab_gt_iou_of_rois = torch.cat(lab_gt_iou_of_rois, dim=0)
            lab_roi_labels -= 1
            

            # estimation using labeled data
            for cind in range(self.num_classes):
                cls_mask = lab_roi_labels == cind
                cls_score = lab_roi_scores[cls_mask]
                cls_score = cls_score[cls_score>self.pre_filtering_thresh]
                iou_score = lab_gt_iou_of_rois[cls_mask]
                iou_score = iou_score[iou_score>self.pre_filtering_thresh]

                self.cls_local_thresholds[cind] = 0.9 * cls_score.mean()
                self.iou_local_thresholds[cind] = 0.9 * iou_score.mean()


            if self.enable_clipping:
                self.cls_local_thresholds = torch.clip(self.cls_local_thresholds, 0.1, 0.9)
                self.iou_local_thresholds = torch.clip(self.iou_local_thresholds, 0.1, 0.9)

            results.update(**{'adamatch_cls_local_thr': {cls_name: self.cls_local_thresholds[i].item() for i, cls_name in self.class_names.items()},
                       'adamatch_iou_local_thr': {cls_name: self.iou_local_thresholds[i].item() for i, cls_name in self.class_names.items()},
                       })
            
                
            self.reset()

        return results