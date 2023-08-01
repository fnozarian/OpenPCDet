from collections import defaultdict
from torchmetrics import Metric
import torch
import numpy as np
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from sklearn.metrics import average_precision_score

__all__ = ["metrics_registry"]

def _detach(tensor: [torch.Tensor] = None) -> [torch.Tensor]:
    return [t.clone().detach() for t in tensor]

def _assert_inputs_are_valid(rois: [torch.Tensor], roi_scores: [torch.Tensor], ground_truths: [torch.Tensor]) -> None:
    assert [len(sample_rois) != 0 for sample_rois in rois], "rois should not be empty"
    assert isinstance(rois, list) and isinstance(ground_truths, list) and isinstance(roi_scores, list)
    assert (
            all(roi.dim() == 2 for roi in rois)
            and all(roi.dim() == 2 for roi in ground_truths)
            and all(roi.dim() == 1 for roi in roi_scores)
    )
    assert all(roi.shape[-1] == 8 for roi in rois) and all(
        gt.shape[-1] == 8 for gt in ground_truths
    )
    assert torch.all([torch.logical_not(torch.all(sample_rois == 0, dim=-1)) for sample_rois in rois]), \
        "rois should not contains all zero boxes"

def _average_precision_score(y_true, y_scores, sample_weight=None):
    y_true = y_true.int().cpu().numpy()
    y_scores = y_scores.float().cpu().numpy()
    return average_precision_score(y_true, y_scores, sample_weight=sample_weight)

class PredQualityMetrics(Metric):
    full_state_update: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reset_state_interval = kwargs.get('reset_state_interval', 64)
        self.tag = kwargs.get('tag', None)
        self.dataset = kwargs.get('dataset', None)
        self.config = kwargs.get('config', None)
        # TODO(farzad): add "roi_matched_pl_score" to states_name
        self.states_name = ["roi_scores", "roi_iou_wrt_gt", "roi_iou_wrt_pl", "roi_pred_labels", "roi_true_labels",
                            "roi_weights", "roi_target_scores", "roi_sim_scores", "roi_sim_labels"]
        self.fg_threshs = kwargs.get('fg_threshs', None) or self.config.ROI_HEAD.TARGET_CONFIG.UNLABELED_CLS_FG_THRESH
        self.bg_thresh = self.config.ROI_HEAD.TARGET_CONFIG.UNLABELED_CLS_BG_THRESH
        self.min_overlaps = np.array([0.7, 0.5, 0.5])

        self.add_state('num_samples', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('num_gts', default=torch.zeros((3,)), dist_reduce_fx='sum')
        for name in self.states_name:
            self.add_state(name, default=[], dist_reduce_fx='cat')

    def update(self, rois: [torch.Tensor], roi_scores: [torch.Tensor], roi_weights: [torch.Tensor],
               roi_iou_wrt_pl: [torch.Tensor], roi_target_scores: [torch.Tensor], ground_truths: [torch.Tensor],
               roi_sim_scores: [torch.Tensor], roi_sim_labels: [torch.Tensor]) -> None:

        _assert_inputs_are_valid(rois, roi_scores, ground_truths)

        num_gts = torch.zeros((3,), dtype=torch.int8, device=rois[0].device)
        for sample_rois, i in enumerate(rois):
            sample_roi_pred_labels = sample_rois[:, -1] - 1

            valid_gts_mask = torch.logical_not(torch.all(ground_truths[i] == 0, dim=-1))
            sample_gts = ground_truths[i][valid_gts_mask]

            if len(sample_gts) > 0:
                overlap = iou3d_nms_utils.boxes_iou3d_gpu(sample_rois[:, 0:7], sample_gts[:, 0:7])
                sample_roi_iou_wrt_gt, matched_gt_inds = overlap.max(dim=1)
                sample_roi_true_labels = sample_gts[matched_gt_inds, -1] - 1
                num_gts += torch.bincount(sample_gts[:, -1] - 1, minlength=3)
            else:
                sample_roi_iou_wrt_gt = torch.zeros_like(sample_rois[:, 0])
                sample_roi_true_labels = -1 * torch.ones_like(sample_rois[:, 0])

            self.roi_scores.append(roi_scores[i])
            self.roi_sim_scores.append(roi_sim_scores[i])
            self.roi_sim_labels.append(roi_sim_labels[i])
            self.roi_iou_wrt_gt.append(sample_roi_iou_wrt_gt)
            self.roi_iou_wrt_pl.append(roi_iou_wrt_pl[i])
            self.roi_pred_labels.append(sample_roi_pred_labels)
            self.roi_true_labels.append(sample_roi_true_labels)
            self.roi_weights.append(roi_weights[i])
            self.roi_target_scores.append(roi_target_scores[i])

        self.num_samples += len(rois)
        self.num_gts += num_gts

    def _accumulate_metrics(self):
        accumulated_metrics = {}
        for mname in self.states_name:
            mstate = getattr(self, mname)
            if isinstance(mstate, torch.Tensor):
                mstate = [mstate]
            accumulated_metrics[mname] = torch.cat(mstate, dim=0)
        return accumulated_metrics

    @property
    def compute(self):
        if self.num_samples < self.reset_state_interval:
            return None

        classwise_metrics = defaultdict(list)
        accumulated_metrics = self._accumulate_metrics()  # shape (N, 1)

        scores = accumulated_metrics["roi_scores"].view(-1)
        iou_wrt_gt = accumulated_metrics["roi_iou_wrt_gt"].view(-1)
        iou_wrt_pl = accumulated_metrics["roi_iou_wrt_pl"].view(-1)
        pred_labels = accumulated_metrics["roi_pred_labels"].view(-1)
        true_labels = accumulated_metrics["roi_true_labels"].view(-1)
        weights = accumulated_metrics["roi_weights"].view(-1)
        target_scores = accumulated_metrics["roi_target_scores"].view(-1)
        sim_scores = accumulated_metrics["roi_sim_scores"].view(-1)
        sim_labels = accumulated_metrics["roi_sim_labels"].view(-1)

        for cind in range(len(self.dataset.class_names)):
            cls_pred_mask = pred_labels == cind
            cls_sim_mask = sim_labels == cind
            cls_true_mask = true_labels == cind

            # By using cls_true_mask we assume that the performance of RPN classification is perfect.
            cls_roi_scores = scores[cls_true_mask]
            cls_roi_sim_scores = sim_scores[cls_sim_mask]
            cls_roi_iou_wrt_gt = iou_wrt_gt[cls_true_mask]
            cls_roi_iou_wrt_pl = iou_wrt_pl[cls_true_mask]
            cls_roi_weights = weights[cls_true_mask]
            cls_roi_target_scores = target_scores[cls_true_mask]

            # Multiclass classification average precision score based on different scores.
            y_labels = cls_true_mask.int().cpu().numpy()
            y_scores = (cls_pred_mask.float() * scores).cpu().numpy()
            y_sim_scores = (cls_sim_mask.float() * sim_scores).cpu().numpy()
            cls_avg_precision_score = average_precision_score(y_labels, y_scores)
            cls_avg_precision_sim_score = average_precision_score(y_labels, y_sim_scores)
            tag = 'multiclass_clf_avg_precision_score_using_sem_score'
            classwise_metrics[tag][cind] = cls_avg_precision_score
            tag = 'multiclass_clf_avg_precision_score_using_sim_score'
            classwise_metrics[tag][cind] = cls_avg_precision_sim_score
            sem_clf_pr_curve_sem_score_data = {'labels': y_labels, 'predictions': y_scores}
            sem_clf_pr_curve_sim_score_data = {'labels': y_labels, 'predictions': y_sim_scores}
            classwise_metrics['sem_clf_pr_curve_sem_score'][cind] = sem_clf_pr_curve_sem_score_data
            classwise_metrics['sem_clf_pr_curve_sim_score'][cind] = sem_clf_pr_curve_sim_score_data

            # Using kitti test class-wise fg thresholds.
            fg_thresh = self.min_overlaps[cind]
            cls_fg_mask = cls_roi_iou_wrt_gt >= fg_thresh
            cls_bg_mask = cls_roi_iou_wrt_gt <= self.bg_thresh
            cls_uc_mask = ~(cls_bg_mask | cls_fg_mask)
            def add_avg_metric(key, metric):
                classwise_metrics[f'fg_{key}'][cind].append((metric * cls_fg_mask.float()).sum() / cls_fg_mask.sum())
                classwise_metrics[f'uc_{key}'][cind].append((metric * cls_uc_mask.float()).sum() / cls_uc_mask.sum())
                classwise_metrics[f'bg_{key}'][cind].append((metric * cls_bg_mask.float()).sum() / cls_bg_mask.sum())

            classwise_metrics['avg_num_true_rois_per_sample'][cind].append(cls_true_mask.sum() / self.num_samples)
            classwise_metrics['avg_num_pred_rois_using_sem_score_per_sample'][cind].append(cls_pred_mask.sum() / self.num_samples)
            classwise_metrics['avg_num_pred_rois_using_sim_score_per_sample'][cind].append(cls_sim_mask.sum() / self.num_samples)
            classwise_metrics['avg_num_gts_per_sample'][cind].append()

            classwise_metrics['rois_fg_ratio'][cind].append(cls_fg_mask.sum() / cls_true_mask.sum())
            classwise_metrics['rois_uc_ratio'][cind].append(cls_uc_mask.sum() / cls_true_mask.sum())
            classwise_metrics['rois_bg_ratio'][cind].append(cls_bg_mask.sum() / cls_true_mask.sum())

            add_avg_metric('rois_avg_score', cls_roi_scores)
            add_avg_metric('rois_avg_sim_score', cls_roi_sim_scores)
            add_avg_metric('rois_avg_iou_wrt_gt', cls_roi_iou_wrt_gt)
            add_avg_metric('rois_avg_iou_wrt_pl', cls_roi_iou_wrt_pl)
            add_avg_metric('rois_avg_weight', cls_roi_weights)
            add_avg_metric('rois_avg_target_score', cls_roi_target_scores)

            tag = 'bin_clf_avg_precision_score_using_target_score'
            classwise_metrics[tag][cind].append(_average_precision_score(cls_fg_mask, cls_roi_target_scores))
            tag = 'bin_clf_avg_precision_score_using_target_score_weighted'
            classwise_metrics[tag][cind].append(_average_precision_score(cls_fg_mask, cls_roi_target_scores, cls_roi_weights))
            tag = 'bin_clf_avg_precision_score_using_roi_iou_wrt_pl'
            classwise_metrics[tag][cind].append(_average_precision_score(cls_fg_mask, cls_roi_iou_wrt_pl))
            tag = 'bin_clf_avg_precision_score_using_roi_iou_wrt_pl_weighted'
            classwise_metrics[tag][cind].append(_average_precision_score(cls_fg_mask, cls_roi_iou_wrt_pl, cls_roi_weights))

        self.reset()

        return classwise_metrics


class MetricRegistry(object):
    def __init__(self, **kwargs):
        self._tag_metrics = {}

    def get(self, tag=None, **kwargs):
        if tag is None:
            tag = 'default'
        if tag in self._tag_metrics.keys():
            metric = self._tag_metrics[tag]
        else:
            metric = PredQualityMetrics(**kwargs)
            self._tag_metrics[tag] = metric
        return metric

    def tags(self):
        return self._tag_metrics.keys()

metrics_registry = MetricRegistry()
