from collections import defaultdict
from torchmetrics import Metric
import torch
import numpy as np
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from sklearn.metrics import average_precision_score
from matplotlib import pyplot as plt
# from tools.visual_utils import open3d_vis_utils as V
# from pcdet.models.roi_heads.target_assigner.proposal_target_layer import ProposalTargetLayer
# get_max_iou_with_same_class = ProposalTargetLayer.get_max_iou_with_same_class
from torch.distributions import Categorical
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
    assert all(
        torch.logical_not(torch.all(sample_rois == 0, dim=-1).any())
        for sample_rois in rois
    ), "rois should not contains all zero boxes"


def _average_precision_score(y_true, y_scores, sample_weight=None):
    y_true = y_true.int().cpu().numpy()
    y_scores = y_scores.float().cpu().numpy()
    if sample_weight is not None:
        sample_weight = sample_weight.cpu().numpy()
    return average_precision_score(y_true, y_scores, sample_weight=sample_weight)


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


class PredQualityMetrics(Metric):
    full_state_update: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reset_state_interval = kwargs.get('RESET_STATE_INTERVAL', 64)
        self.tag = kwargs.get('tag', None)
        self.dataset = kwargs.get('dataset', None)
        # TODO(farzad): add "roi_matched_pl_score" to states_name
        self.states_name = ["roi_scores", "roi_labels", "roi_iou_wrt_gt", "roi_iou_wrt_pl", "roi_weights",
                            "roi_target_scores", "roi_sim_scores"]
        self.fg_threshs = kwargs.get('fg_threshs', None)
        self.bg_thresh = kwargs.get('BG_THRESH', 0.25)
        self.min_overlaps = np.array([0.7, 0.5, 0.5])

        self.add_state('num_samples', default=torch.tensor(0).cuda(), dist_reduce_fx='sum')
        self.add_state('num_gts', default=torch.zeros((3,)).cuda(), dist_reduce_fx='sum')
        for name in self.states_name:
            self.add_state(name, default=[], dist_reduce_fx='cat')

    def update(self, rois: [torch.Tensor], roi_scores: [torch.Tensor], roi_weights: [torch.Tensor],
               roi_iou_wrt_pl: [torch.Tensor], roi_target_scores: [torch.Tensor], ground_truths: [torch.Tensor],
               roi_sim_scores: [torch.Tensor], points: [torch.Tensor]) -> None:

        _assert_inputs_are_valid(rois, roi_scores, ground_truths)

        num_gts = torch.zeros((3,), dtype=torch.int8, device=rois[0].device)
        for i, sample_rois in enumerate(rois):
            sample_roi_labels = sample_rois[:, -1].long() - 1

            valid_gts_mask = torch.logical_not(torch.all(ground_truths[i] == 0, dim=-1))
            sample_gts = ground_truths[i][valid_gts_mask]
            sample_gts_labels = sample_gts[:, -1].long() - 1
            if len(sample_gts) > 0:
                sample_roi_iou_wrt_gt, _ = get_max_iou_with_same_class(sample_rois[:, 0:7], sample_roi_labels,
                                                                       sample_gts[:, 0:7], sample_gts_labels)
                num_gts += torch.bincount(sample_gts[:, -1].int() - 1, minlength=3)
            else:
                sample_roi_iou_wrt_gt = torch.zeros_like(sample_rois[:, 0])

            self.roi_scores.append(roi_scores[i])
            self.roi_sim_scores.append(roi_sim_scores[i])
            self.roi_iou_wrt_gt.append(sample_roi_iou_wrt_gt)
            self.roi_iou_wrt_pl.append(roi_iou_wrt_pl[i])
            self.roi_labels.append(sample_roi_labels)
            self.roi_weights.append(roi_weights[i])
            self.roi_target_scores.append(roi_target_scores[i])

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
        self.num_gts += num_gts

    def _accumulate_metrics(self):
        accumulated_metrics = {}
        for mname in self.states_name:
            mstate = getattr(self, mname)
            if isinstance(mstate, torch.Tensor):
                mstate = [mstate]
            accumulated_metrics[mname] = torch.cat(mstate, dim=0)
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

        scores = accumulated_metrics["roi_scores"]
        sim_scores = accumulated_metrics["roi_sim_scores"]
        sim_labels = torch.argmax(sim_scores, dim=-1)
        iou_wrt_gt = accumulated_metrics["roi_iou_wrt_gt"].view(-1)
        iou_wrt_pl = accumulated_metrics["roi_iou_wrt_pl"].view(-1)
        pred_labels = accumulated_metrics["roi_labels"].view(-1)
        weights = accumulated_metrics["roi_weights"].view(-1)
        target_scores = accumulated_metrics["roi_target_scores"].view(-1)

        true_mask = (iou_wrt_gt >= iou_wrt_gt.new_tensor(self.min_overlaps)[pred_labels]).long()

        # Multiclass classification average precision score based on different scores.
        one_hot_labels = pred_labels.new_zeros(len(pred_labels), 3, dtype=torch.long, device=scores.device)
        one_hot_labels.scatter_(-1, pred_labels.unsqueeze(dim=-1).long(), 1.0)
        one_hot_labels = one_hot_labels * true_mask.unsqueeze(dim=-1)

        y_labels = one_hot_labels.int().cpu().numpy()
        y_scores = scores.cpu().numpy()
        y_sim_scores = sim_scores.cpu().numpy()
        classwise_metrics['multiclass_avg_precision_sem_score_micro'] = average_precision_score(y_labels, y_scores, average='micro')
        classwise_metrics['multiclass_avg_precision_sem_score_weighted'] = average_precision_score(y_labels, y_scores, average='weighted')
        classwise_metrics['multiclass_avg_precision_sem_score_macro'] = average_precision_score(y_labels, y_scores, average='macro')
        classwise_metrics['multiclass_avg_precision_sim_score_micro'] = average_precision_score(y_labels, y_sim_scores, average='micro')
        classwise_metrics['multiclass_avg_precision_sim_score_weighted'] = average_precision_score(y_labels, y_sim_scores, average='weighted')
        classwise_metrics['multiclass_avg_precision_sim_score_macro'] = average_precision_score(y_labels, y_sim_scores, average='macro')

        for cind, cls in enumerate(self.dataset.class_names):
            cls_pred_mask = pred_labels == cind
            # cls_sim_mask = sim_labels == cind

            # By using cls_true_mask we assume that the performance of RPN classification is perfect.
            cls_roi_scores = scores[cls_pred_mask, cind]
            cls_roi_sim_scores = sim_scores[cls_pred_mask, cind]
            cls_roi_sim_scores_entropy = Categorical(sim_scores[cls_pred_mask] + torch.finfo(torch.float32).eps).entropy()
            cls_roi_iou_wrt_gt = iou_wrt_gt[cls_pred_mask]
            cls_roi_iou_wrt_pl = iou_wrt_pl[cls_pred_mask]
            cls_roi_weights = weights[cls_pred_mask]
            cls_roi_target_scores = target_scores[cls_pred_mask]

            sem_clf_pr_curve_sem_score_data = {'labels': y_labels, 'predictions': y_scores}
            sem_clf_pr_curve_sim_score_data = {'labels': y_labels, 'predictions': y_sim_scores}
            classwise_metrics['sem_clf_pr_curve_sem_score'][cls] = sem_clf_pr_curve_sem_score_data
            classwise_metrics['sem_clf_pr_curve_sim_score'][cls] = sem_clf_pr_curve_sim_score_data

            # Using kitti test class-wise fg thresholds.
            fg_thresh = self.min_overlaps[cind]
            cls_fg_mask = cls_roi_iou_wrt_gt >= fg_thresh
            cls_bg_mask = cls_roi_iou_wrt_gt <= self.bg_thresh
            cls_uc_mask = ~(cls_bg_mask | cls_fg_mask)

            def add_avg_metric(key, metric):
                classwise_metrics[f'fg_{key}'][cls] = (metric * cls_fg_mask.float()).sum() / cls_fg_mask.sum()
                classwise_metrics[f'uc_{key}'][cls] = (metric * cls_uc_mask.float()).sum() / cls_uc_mask.sum()
                # classwise_metrics[f'bg_{key}'][cls] = (metric * cls_bg_mask.float()).sum() / cls_bg_mask.sum()

            classwise_metrics['avg_num_true_rois_per_sample'][cls] = cls_pred_mask.sum() / self.num_samples
            classwise_metrics['avg_num_pred_rois_using_sem_score_per_sample'][cls] = cls_pred_mask.sum() / self.num_samples
            classwise_metrics['avg_num_pred_rois_using_sim_score_per_sample'][cls] = cls_pred_mask.sum() / self.num_samples
            # classwise_metrics['avg_num_gts_per_sample'].append()

            classwise_metrics['rois_fg_ratio'][cls] = cls_fg_mask.sum() / cls_pred_mask.sum()
            classwise_metrics['rois_uc_ratio'][cls] = cls_uc_mask.sum() / cls_pred_mask.sum()
            classwise_metrics['rois_bg_ratio'][cls] = cls_bg_mask.sum() / cls_pred_mask.sum()

            add_avg_metric('rois_avg_score', cls_roi_scores)
            add_avg_metric('rois_avg_sim_score', cls_roi_sim_scores)
            add_avg_metric('rois_avg_iou_wrt_gt', cls_roi_iou_wrt_gt)
            add_avg_metric('rois_avg_iou_wrt_pl', cls_roi_iou_wrt_pl)
            add_avg_metric('rois_avg_weight', cls_roi_weights)
            add_avg_metric('rois_avg_target_score', cls_roi_target_scores)
            add_avg_metric('rois_avg_sim_score_entropy', cls_roi_sim_scores_entropy)
            # tag = 'bin_clf_avg_precision_score_using_target_score'
            # classwise_metrics[tag][cls] = _average_precision_score(cls_fg_mask, cls_roi_target_scores)
            # tag = 'bin_clf_avg_precision_score_using_target_score_weighted'
            # classwise_metrics[tag][cls] = _average_precision_score(cls_fg_mask, cls_roi_target_scores, cls_roi_weights)
            # tag = 'bin_clf_avg_precision_score_using_roi_iou_wrt_pl'
            # classwise_metrics[tag][cls] = _average_precision_score(cls_fg_mask, cls_roi_iou_wrt_pl)
            # tag = 'bin_clf_avg_precision_score_using_roi_iou_wrt_pl_weighted'
            # classwise_metrics[tag][cls] = _average_precision_score(cls_fg_mask, cls_roi_iou_wrt_pl, cls_roi_weights)

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
