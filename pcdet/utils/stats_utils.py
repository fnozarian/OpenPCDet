from collections import defaultdict
from torchmetrics import Metric
import torch
from torch.distributions import Categorical
import numpy as np
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from pcdet.utils import box_utils
from sklearn.metrics import average_precision_score, confusion_matrix, precision_score
from matplotlib import pyplot as plt
import itertools
# from tools.visual_utils import open3d_vis_utils as V
# from pcdet.models.roi_heads.target_assigner.proposal_target_layer import ProposalTargetLayer

__all__ = ["metrics_registry"]


def _detach(tensor: [torch.Tensor] = None) -> [torch.Tensor]:
    return [t.clone().detach() for t in tensor]


def _assert_inputs_are_valid(rois: [torch.Tensor], roi_scores: [torch.Tensor], ground_truths: [torch.Tensor]) -> None:
    assert [len(sample_rois) != 0 for sample_rois in rois], "rois should not be empty"
    assert isinstance(rois, list) and isinstance(ground_truths, list) and isinstance(roi_scores, list)
    assert (
            all(roi.dim() == 2 for roi in rois)
            and all(roi.dim() == 2 for roi in ground_truths)
            and all(roi.dim() == 2 for roi in roi_scores)
    )
    assert all(roi.shape[-1] == 8 for roi in rois) and all(
        gt.shape[-1] == 8 for gt in ground_truths
    )
    num_gts = torch.stack([torch.logical_not(torch.all(sample_gts == 0, dim=-1)).sum() for sample_gts in ground_truths])
    if num_gts.eq(0).any():
        print(f"\nWARNING! Unlabeled sample has no ground truths!")

def get_max_iou_with_same_class(rois, roi_labels, gt_boxes, gt_labels):
    """
    Args:
        rois: (N, 7)
        roi_labels: (N)
        gt_boxes: (N, )
        gt_labels:

    Returns:

    """
    """
    :param rois: (N, 7)
    :param roi_labels: (N)
    :param gt_boxes: (N, 8)
    :return:
    """
    max_overlaps = rois.new_zeros(rois.shape[0])
    gt_assignment = roi_labels.new_zeros(roi_labels.shape[0])

    for k in range(gt_labels.min().item(), gt_labels.max().item() + 1):
        roi_mask = (roi_labels == k)
        gt_mask = (gt_labels == k)
        if roi_mask.sum() > 0 and gt_mask.sum() > 0:
            cur_roi = rois[roi_mask]
            cur_gt = gt_boxes[gt_mask]
            original_gt_assignment = gt_mask.nonzero().view(-1)

            iou3d = iou3d_nms_utils.boxes_iou3d_gpu(cur_roi, cur_gt)  # (M, N)
            cur_max_overlaps, cur_gt_assignment = torch.max(iou3d, dim=1)
            max_overlaps[roi_mask] = cur_max_overlaps
            gt_assignment[roi_mask] = original_gt_assignment[cur_gt_assignment]

    return max_overlaps, gt_assignment

def get_max_iou(anchors, gt_boxes, gt_classes, matched_threshold=0.6):
    num_anchors = anchors.shape[0]
    num_gts = gt_boxes.shape[0]

    ious = torch.zeros((num_anchors,), dtype=torch.float, device=anchors.device)
    labels = torch.ones((num_anchors,), dtype=torch.int64, device=anchors.device) * -1
    gt_to_anchor_max = torch.zeros((num_gts,), dtype=torch.float, device=anchors.device)

    if len(gt_boxes) > 0 and anchors.shape[0] > 0:
        anchor_by_gt_overlap = iou3d_nms_utils.boxes_iou3d_gpu(anchors[:, 0:7], gt_boxes[:, 0:7])
        gt_to_anchor_max = anchor_by_gt_overlap.max(dim=0)[0]
        anchor_to_gt_argmax = anchor_by_gt_overlap.argmax(dim=1)
        anchor_to_gt_max = anchor_by_gt_overlap[torch.arange(num_anchors, device=anchors.device), anchor_to_gt_argmax]

        pos_inds = anchor_to_gt_max >= matched_threshold
        gt_inds_over_thresh = anchor_to_gt_argmax[pos_inds]
        labels[pos_inds] = gt_classes[gt_inds_over_thresh]
        ious[:] = anchor_to_gt_max

    return ious, labels, gt_to_anchor_max

def plot_confusion_matrix(cm, class_names):
  """
  Returns a matplotlib figure containing the plotted confusion matrix.

  Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
  """
  fig, ax = plt.subplots(figsize=(4, 4))
  ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
  ax.set_title("Confusion matrix")
  fig.colorbar(ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues))
  tick_marks = np.arange(len(class_names))
  ax.set_xticks(tick_marks)
  ax.set_xticklabels(class_names, rotation=45)
  ax.set_yticks(tick_marks)
  ax.set_yticklabels(class_names)

  # Compute the labels from the normalized confusion matrix.
  labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

  # Use white text if squares are dark; otherwise black.
  threshold = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      color = "white" if cm[i, j] > threshold else "black"
      ax.text(j, i, labels[i, j], horizontalalignment="center", color=color)

  ax.set_ylabel('True label')
  ax.set_xlabel('Predicted label')
  fig.tight_layout()

  return fig

def _arr2dict(array, ignore_zeros=False, ignore_nan=False):
    def should_include(value):
        return not ((ignore_zeros and value == 0) or (ignore_nan and np.isnan(value)))

    classes = ['Bg', 'Fg'] if array.shape[-1] == 2 else ['Car', 'Pedestrian', 'Cyclist']
    return {cls: array[cind] for cind, cls in enumerate(classes) if should_include(array[cind])}
class PredQualityMetrics(Metric):
    full_state_update: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reset_state_interval = kwargs.get('RESET_STATE_INTERVAL', 64)
        self.tag = kwargs.get('tag', None)
        self.dataset = kwargs.get('dataset', None)

        self.states_name = ["roi_scores", "roi_weights", "roi_labels", "roi_iou_wrt_gt", "roi_assigned_labels"]
        self.fg_threshs = kwargs.get('fg_threshs', None)
        self.bg_thresh = kwargs.get('BG_THRESH', 0.25)
        self.min_overlaps = np.array([0.7, 0.5, 0.5])

        self.add_state('num_samples', default=torch.tensor(0).cuda(), dist_reduce_fx='sum')
        self.add_state('num_gts', default=torch.zeros((3,)).cuda(), dist_reduce_fx='sum')
        self.add_state('num_gts_matched', default=torch.zeros((3,3)).cuda(), dist_reduce_fx='sum')  # 3 recall thresholds, 3 classes
        for name in self.states_name:
            self.add_state(name, default=[], dist_reduce_fx='cat')

    def update(self, rois: [torch.Tensor], roi_scores: [torch.Tensor], roi_weights: [torch.Tensor], ground_truths: [torch.Tensor]) -> None:

        _assert_inputs_are_valid(rois, roi_scores, ground_truths)

        for i, sample_rois in enumerate(rois):

            valid_roi_mask = torch.logical_not(torch.all(sample_rois == 0, dim=-1))
            sample_rois = sample_rois[valid_roi_mask]
            roi_scores[i] = roi_scores[i][valid_roi_mask]
            roi_weights[i] = roi_weights[i][valid_roi_mask]

            valid_gts_mask = torch.logical_not(torch.all(ground_truths[i] == 0, dim=-1))
            sample_gts = ground_truths[i][valid_gts_mask]

            if len(sample_rois) == 0 and len(sample_gts) > 0:  # Skip empty samples
                sample_gts_labels = sample_gts[:, -1].long() - 1
                self.num_gts += torch.bincount(sample_gts_labels, minlength=3)
                continue
            elif len(sample_rois) > 0 and len(sample_gts) > 0:
                sample_gts_labels = sample_gts[:, -1].long() - 1
                sample_roi_labels = sample_rois[:, -1].long() - 1
                matched_threshold = torch.tensor(self.min_overlaps, dtype=torch.float, device=sample_roi_labels.device)[sample_roi_labels]
                sample_roi_iou_wrt_gt, assigned_label, gt_to_roi_max_iou = get_max_iou(sample_rois[:, 0:7], sample_gts[:, 0:7],
                                                                                       sample_gts_labels, matched_threshold=matched_threshold)
                self.num_gts += torch.bincount(sample_gts_labels, minlength=3)

                for t, thresh in enumerate([0.3, 0.5, 0.7]):
                    for c in range(3):
                        self.num_gts_matched[t, c] += gt_to_roi_max_iou[sample_gts_labels == c].ge(thresh).sum()
            elif len(sample_rois) > 0 and len(sample_gts) == 0:
                sample_roi_labels = sample_rois[:, -1].long() - 1
                sample_roi_iou_wrt_gt = torch.zeros_like(sample_rois[:, 0])
                assigned_label = torch.ones_like(sample_roi_iou_wrt_gt, dtype=torch.int64) * -1
            else:
                print("WARNING! Both rois and gts are empty!")
                continue

            self.roi_scores.append(roi_scores[i])
            # self.roi_sim_scores.append(roi_sim_scores[i])
            self.roi_iou_wrt_gt.append(sample_roi_iou_wrt_gt.view(-1, 1))
            self.roi_assigned_labels.append(assigned_label.view(-1, 1))
            self.roi_labels.append(sample_roi_labels.view(-1, 1))
            self.roi_weights.append(roi_weights[i])

        # Draw the last sample in batch
        # pred_boxes = sample_rois[:, :-1].clone().cpu().numpy()
        # pred_labels = sample_roi_labels.clone().int().cpu().numpy()
        # pred_scores = sample_roi_iou_wrt_gt.clone().cpu().numpy()
        # gts = sample_gts[:, :-1].clone().cpu().numpy()
        # gt_labels = sample_gts[:, -1].clone().int().cpu().numpy() - 1
        # pts = points[i].clone().cpu().numpy()
        # V.draw_scenes(points=pts, gt_boxes=gts, gt_labels=gt_labels,
        #               ref_boxes=pred_boxes, ref_scores=pred_scores, ref_labels=pred_labels)

        self.num_samples += len(rois)

    def _accumulate_metrics(self):
        accumulated_metrics = {}
        for mname in self.states_name:
            mstate = getattr(self, mname)
            accumulated_metrics[mname] = torch.cat(mstate, dim=0) if len(mstate) > 0 else []
        return accumulated_metrics

    # @staticmethod
    # def draw_sim_matrix_figure(sim_matrix, lbls):
        # fig, ax = plt.subplots(figsize=(20, 20))
        # cax = ax.matshow(sim_matrix, interpolation='nearest')
        # ax.grid(True)
        # plt.xticks(range(len(lbls)), lbls)
        # plt.yticks(range(len(lbls)), lbls)
        # plt.show()

    def compute(self):
        if self.num_samples < self.reset_state_interval:
            return None

        classwise_metrics = defaultdict(dict)

        accumulated_metrics = self._accumulate_metrics()  # shape (N, 1)
        if len(accumulated_metrics["roi_scores"]) == 0:  # No valid samples
            self.reset()
            return None
        scores = accumulated_metrics["roi_scores"]
        iou_wrt_gt = accumulated_metrics["roi_iou_wrt_gt"].view(-1)
        pred_labels = accumulated_metrics["roi_labels"].view(-1)
        assigned_labels = accumulated_metrics["roi_assigned_labels"].view(-1)
        max_scores, argmax_scores = scores.max(dim=-1)
        weights = accumulated_metrics["roi_weights"].view(-1)

        # Multiclass classification average precision score based on different scores.
        y_labels = torch.where(assigned_labels == -1, 3, assigned_labels)
        # one_hot_labels = argmax_scores.new_zeros(len(y_labels), 4, dtype=torch.long, device=scores.device)
        # one_hot_labels.scatter_(-1, y_labels.unsqueeze(dim=-1).long(), 1.0).cpu().numpy()

        y_labels = y_labels.cpu().numpy()
        pred_labels = pred_labels.cpu().numpy()
        classwise_metrics['mean_p_max_model'] = (max_scores * weights).mean().item()
        mean_p_max_model_classwise = scores.new_zeros((3,)).scatter_add_(0, argmax_scores, max_scores * weights)
        mean_p_max_model_classwise /= torch.bincount(argmax_scores, weights=weights, minlength=3)
        classwise_metrics['mean_p_max_model_classwise'] = _arr2dict(mean_p_max_model_classwise.cpu().numpy(), ignore_nan=True)
        mean_p_model = (scores * weights.unsqueeze(-1)).sum(dim=0) / weights.sum()
        classwise_metrics['mean_p_model'] = _arr2dict(mean_p_model.cpu().numpy())
        label_hist = torch.bincount(argmax_scores, minlength=3)
        classwise_metrics['label_hist'] = _arr2dict(label_hist.cpu().numpy(), ignore_zeros=True)
        precision = precision_score(y_labels, pred_labels, sample_weight=weights.cpu().numpy(), average=None, labels=range(3), zero_division=np.nan)
        classwise_metrics['avg_precision_sem_score'] = _arr2dict(precision[:3], ignore_nan=True)

        # sim_scores = accumulated_metrics["roi_sim_scores"]
        # sim_labels = torch.argmax(sim_scores, dim=-1)
        # y_sim_scores = np.zeros((len(y_labels), 4))
        # y_sim_scores[:, :3] = sim_scores.cpu().numpy()

        # cm = confusion_matrix(y_labels, pred_labels)
        # print("\n Confusion Matrix: \n", cm)
        # figure = plot_confusion_matrix(cm, class_names=['Car', 'Pedestrian', 'Cyclist', 'BG'])
        # classwise_metrics['matrix'] = figure

        for cind, cls in enumerate(self.dataset.class_names):
            cls_label_mask = y_labels == cind
            cls_pred_mask = pred_labels == cind
            # cls_pred_mask = argmax_scores == cind
            # tp_mask = (cls_pred_mask & cls_label_mask)

            sem_clf_pr_curve_data = {'labels': cls_label_mask, 'predictions': scores[:, cind].cpu().numpy()}
            classwise_metrics['sem_clf_pr_curve'][cls] = sem_clf_pr_curve_data

            cls_roi_scores = max_scores[cls_pred_mask]
            cls_roi_iou_wrt_gt = iou_wrt_gt[cls_pred_mask]

            # Using kitti test class-wise fg thresholds.
            fg_thresh = self.min_overlaps[cind]
            cls_fg_mask = cls_roi_iou_wrt_gt >= fg_thresh
            cls_bg_mask = cls_roi_iou_wrt_gt <= self.bg_thresh
            cls_uc_mask = ~(cls_bg_mask | cls_fg_mask)

            def add_avg_metric(key, metric):
                if cls_fg_mask.sum() > 0:
                    classwise_metrics[f'fg_{key}'][cls] = (metric * cls_fg_mask.float()).sum() / torch.clip(cls_fg_mask.sum(), min=1)
                # if cls_fg_mask.sum() > 0:
                #     classwise_metrics[f'uc_{key}'][cls] = (metric * cls_uc_mask.float()).sum() / cls_uc_mask.sum()
                # classwise_metrics[f'bg_{key}'][cls] = (metric * cls_bg_mask.float()).sum() / cls_bg_mask.sum()

            classwise_metrics['avg_num_rois_per_sample'][cls] = cls_pred_mask.sum() / self.num_samples
            classwise_metrics['avg_num_gts_per_sample'][cls] = self.num_gts[cind] / self.num_samples

            # classwise_metrics['rois_fg_ratio'][cls] = cls_fg_mask.sum() / max(cls_pred_mask.sum(), 1)
            # classwise_metrics['rois_uc_ratio'][cls] = cls_uc_mask.sum() / max(cls_pred_mask.sum(), 1)
            # classwise_metrics['rois_bg_ratio'][cls] = cls_bg_mask.sum() / max(cls_pred_mask.sum(), 1)

            add_avg_metric('rois_avg_score', cls_roi_scores)
            add_avg_metric('rois_avg_iou_wrt_gt', cls_roi_iou_wrt_gt)
            # add_avg_metric('rois_avg_weight', cls_roi_weights)

            # recall
            for t, thresh in enumerate([0.3, 0.5, 0.7]):
                classwise_metrics[f'recall_{thresh}'][cls] = self.num_gts_matched[t, cind] / torch.clip(self.num_gts[cind], min=1)

            # y_sim_scores = sim_scores[:, cind].cpu().numpy()
            # sem_clf_pr_curve_sim_score_data = {'labels': cls_label_mask, 'predictions': y_sim_scores}
            # classwise_metrics['sem_clf_pr_curve_sim_score'][cls] = sem_clf_pr_curve_sim_score_data
            # classwise_metrics['avg_precision_sim_score'][cls] = average_precision_score(cls_label_mask, y_sim_scores)
            # cls_sim_mask = sim_labels == cind
            # cls_roi_sim_scores = sim_scores[cls_pred_mask, cind]
            # cls_roi_sim_scores_entropy = Categorical(sim_scores[cls_pred_mask] + torch.finfo(torch.float32).eps).entropy()
            # classwise_metrics['avg_num_pred_rois_using_sim_score_per_sample'][cls] = cls_sim_mask.sum() / self.num_samples
            # add_avg_metric('rois_avg_sim_score', cls_roi_sim_scores)
            # add_avg_metric('rois_avg_sim_score_entropy', cls_roi_sim_scores_entropy)


        self.reset()
        # Torchmetrics has an issue with defaultdicts. I have to pass the keys and values separately.
        return classwise_metrics.keys(), classwise_metrics.values()


class MetricRegistry(object):
    def __init__(self, **kwargs):
        self._metrics_bank = {}

    def register(self, tag=None, **metrics_configs):
        if tag is None:
            tag = 'default'
        if tag in self.tags():
            raise ValueError(f'Metrics with tag {tag} already exists')
        metrics = PredQualityMetrics(**metrics_configs)
        self._metrics_bank[tag] = metrics
        return self._metrics_bank[tag]

    def get(self, tag=None):
        if tag is None:
            tag = 'default'
        if tag not in self.tags():
            raise ValueError(f'Metrics with tag {tag} does not exist')
        return self._metrics_bank[tag]

    def tags(self):
        return self._metrics_bank.keys()


metrics_registry = MetricRegistry()
