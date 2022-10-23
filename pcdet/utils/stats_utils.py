# arguments example:
# --pred_infos
# <OpenPCDet_HOME>/output/cfgs/kitti_models/pv_rcnn_ssl/enabled_st_all_bs8_dist4_split_1_2_trial3_169035d/eval/eval_with_train/epoch_60/val/result.pkl
# --gt_infos
# <OpenPCDet_HOME>/data/kitti/kitti_infos_val.pkl

import argparse
import pickle

from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval
from torchmetrics import Metric
import torch
import numpy as np
from pcdet.ops.iou3d_nms import iou3d_nms_utils
import math
from pcdet.config import cfg
from matplotlib import pyplot as plt


# TODO(farzad): Pass only scores and labels?
#               Calculate overlap inside update or compute?
#               Change the states to TP, FP, FN, etc?
#               Calculate incrementally based on summarized value?
class KittiEvalMetric:
    def __init__(self):
        # TODO(farzad): Move some of these to init args
        current_classes = ['Car', 'Pedestrian', 'Cyclist']
        self.metric = 2  # evaluation only for 3D metric (2)
        overlap_0_7 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.7],
                                [0.7, 0.5, 0.5, 0.7, 0.5, 0.7],
                                [0.7, 0.5, 0.5, 0.7, 0.5, 0.7]])
        self.min_overlaps = np.expand_dims(overlap_0_7, axis=0)  # [1, num_metrics, num_cls][1, 3, 6]
        class_to_name = {0: 'Car', 1: 'Pedestrian', 2: 'Cyclist', 3: 'Van', 4: 'Person_sitting', 5: 'Truck'}
        name_to_class = {v: n for n, v in class_to_name.items()}
        if not isinstance(current_classes, (list, tuple)):
            current_classes = [current_classes]
        current_classes_int = []
        for curcls in current_classes:
            if isinstance(curcls, str):
                current_classes_int.append(name_to_class[curcls])
            else:
                current_classes_int.append(curcls)
        self.current_classes = current_classes_int
        self.min_overlaps = self.min_overlaps[:, :, self.current_classes]

        self.detections= []
        self.groundtruths= []
        self.overlaps= []

    def update(self, valid_pred_boxes,valid_gt_boxes, overlap):
        # The following states are accumulated over updates
        self.detections.append(valid_pred_boxes)
        self.groundtruths.append(valid_gt_boxes)
        self.overlaps.append(overlap)
    def reset(self):
        self.detections= []
        self.groundtruths= []
        self.overlaps= []
    def compute(self):
        results = eval_class(self.groundtruths, self.detections, self.current_classes,
                                self.metric, self.min_overlaps, self.overlaps)
        mAP_3d = get_mAP(results["precision"])
        mAP_3d_R40 = get_mAP_R40(results["precision"])
        results.update({'mAP_3d': mAP_3d, 'mAP_3d_R40': mAP_3d_R40, 'total_num_samples': len(self.detections)})
        return results
class MeanMetric(Metric):
    full_state_update: bool = False
    def __init__(self):
        super().__init__()
        self.add_state("pred_ious", default=torch.tensor((0,)), dist_reduce_fx='cat')
        self.add_state("pred_accs", default=torch.tensor((0,)), dist_reduce_fx='cat')
        self.add_state("pred_fgs", default=torch.tensor((0,)), dist_reduce_fx='cat')
        self.add_state("sem_score_fgs", default=torch.tensor((0,)), dist_reduce_fx='cat')
        self.add_state("sem_score_bgs", default=torch.tensor((0,)), dist_reduce_fx='cat')
        self.add_state("num_pred_boxes", default=torch.tensor((0,)), dist_reduce_fx='cat')
        self.add_state("num_gt_boxes", default=torch.tensor((0,)), dist_reduce_fx='cat')

    def update(self, pred_ious, pred_accs,pred_fgs,sem_score_fgs,sem_score_bgs,num_pred_boxes,num_gt_boxes):
        # The following states are reset on every update (per-batch states)
        self.pred_ious = torch.tensor(pred_ious).cuda().mean()
        self.pred_accs = torch.tensor(pred_accs).cuda().mean()
        self.pred_fgs = torch.tensor(pred_fgs).cuda().mean()
        self.sem_score_fgs = torch.tensor(sem_score_fgs).cuda().mean()
        self.sem_score_bgs = torch.tensor(sem_score_bgs).cuda().mean()
        self.num_pred_boxes = torch.tensor(num_pred_boxes).cuda().float().mean()
        self.num_gt_boxes = torch.tensor(num_gt_boxes).cuda().float().mean()



        
    def compute(self):
        results = {'pred_ious': self.pred_ious.mean(), 'pred_accs': self.pred_accs.mean(),
                   'pred_fgs': self.pred_fgs.mean(), 'sem_score_fgs': self.sem_score_fgs.mean(),
                   'sem_score_bgs': self.sem_score_bgs.mean(), 'num_pred_boxes': self.num_pred_boxes.mean(),
                   'num_gt_boxes': self.num_gt_boxes.mean()}
        return results
class CombinedMetric:
    def __init__(self):
        self.kitti_eval_metric = KittiEvalMetric()
        self.mean_metric = MeanMetric()
        self.update_count=0
        self.kitti_eval_repeat_counter=cfg.MODEL.POST_PROCESSING.METRIC_CONFIG.KITTI_EVAL_UPDATE_COUNT

    def update(self, preds: [torch.Tensor], targets: [torch.Tensor], pred_scores: [torch.Tensor], pred_sem_scores: [torch.Tensor]) -> None:
        assert all([pred.shape[-1] == 8 for pred in preds]) and all([tar.shape[-1] == 8 for tar in targets])
        assert len(pred_scores) == len(pred_sem_scores)
        # TODO(farzad) should we clone and detach? if so, then profile memory consumption.
        preds = [pred_box.clone().detach() for pred_box in preds]
        targets = [gt_box.clone().detach() for gt_box in targets]
        pred_scores = [ps_score.clone().detach() for ps_score in pred_scores]
        pred_sem_scores = [score.clone().detach() for score in pred_sem_scores]
        pred_ious = []
        pred_accs = []
        pred_fgs = []
        sem_score_fgs = []
        sem_score_bgs = []
        num_pred_boxes = []
        num_gt_boxes = []
        for i in range(len(preds)):
            valid_preds_mask = torch.logical_not(torch.all(preds[i] == 0, dim=-1))
            valid_gts_mask = torch.logical_not(torch.all(targets[i] == 0, dim=-1))
            if pred_scores[i].ndim == 1:
                pred_scores[i] = pred_scores[i].unsqueeze(dim=-1)
            if pred_sem_scores[i].ndim == 1:
                pred_sem_scores[i] = pred_sem_scores[i].unsqueeze(dim=-1)

            valid_pred_boxes = preds[i][valid_preds_mask]
            valid_gt_boxes = targets[i][valid_gts_mask]
            valid_pred_scores = pred_scores[i][valid_preds_mask.nonzero().view(-1)]
            valid_sem_scores = pred_sem_scores[i][valid_preds_mask.nonzero().view(-1)]

            # Starting class indices from zero
            valid_pred_boxes[:, -1] -= 1
            valid_gt_boxes[:, -1] -= 1

            # Adding predicted scores as the last column
            valid_pred_boxes = torch.cat([valid_pred_boxes, valid_pred_scores], dim=-1)

            num_gts = valid_gts_mask.sum()
            num_preds = valid_preds_mask.sum()
            num_pred_boxes.append(num_preds)
            num_gt_boxes.append(num_gts)
            overlap = valid_gts_mask.new_zeros((num_preds, num_gts))
            if num_gts > 0 and num_preds > 0:
                # TODO(farzad) IoU computation in CPU might be faster due to smaller preds_x_gts matrix
                overlap = iou3d_nms_utils.boxes_iou3d_gpu(valid_pred_boxes[:, 0:7], valid_gt_boxes[:, 0:7])
                preds_iou_max, assigned_gt_inds = overlap.max(dim=1)
                pred_ious.append(preds_iou_max.mean())
                pred_labels = valid_pred_boxes[:, -2]
                acc = (pred_labels == valid_gt_boxes[assigned_gt_inds, -1]).float().mean()
                pred_accs.append(acc)

                # Using kitti test class-wise fg threshold instead of thresholds used during train.
                classwise_fg_thresh = self.kitti_eval_metric.min_overlaps[0, self.kitti_eval_metric.metric]
                classwise_fg_thresh = pred_labels.new_tensor(classwise_fg_thresh).unsqueeze(dim=0).repeat(pred_labels.shape[0], 1)
                fg_thresh = classwise_fg_thresh.gather(dim=-1, index=pred_labels.unsqueeze(dim=-1).long()).squeeze()

                bg_thresh = cfg['MODEL']['ROI_HEAD']['TARGET_CONFIG']['CLS_BG_THRESH']

                fg = (preds_iou_max > fg_thresh).float().mean().item()
                pred_fgs.append(fg)

                # Using clamp with min=1 in the denominator makes the final results zero when there's no FG,
                # while without clamp it is N/A, which makes more sense.
                sem_score_fg = (valid_sem_scores.squeeze() * (preds_iou_max > fg_thresh).float()).sum() / (preds_iou_max > fg_thresh).sum()
                sem_score_bg = (valid_sem_scores.squeeze() * (preds_iou_max < bg_thresh).float()).sum() \
                               / torch.clamp((preds_iou_max < bg_thresh).float().sum(), min=1.0)

                sem_score_fgs.append(sem_score_fg)
                sem_score_bgs.append(sem_score_bg)

            #The following states are accumulated over updates
            self.kitti_eval_metric.update(valid_pred_boxes,valid_gt_boxes, overlap)
        # The following states are reset on every update (per-batch states)
        self.mean_metric.update(pred_ious, pred_accs,pred_fgs,sem_score_fgs,sem_score_bgs,num_pred_boxes,num_gt_boxes)
        self.update_count+=1
    
    def compute(self):
        statistics={}
        mean_metrics = self.mean_metric.compute()
        other_stats = ['pred_ious', 'pred_accs', 'pred_fgs', 'sem_score_fgs',
                       'sem_score_bgs', 'num_pred_boxes', 'num_gt_boxes']
        for m, metric_name in enumerate(other_stats):
            if metric_name in mean_metrics.keys():
                statistics[metric_name] = mean_metrics[metric_name]


        if self.update_count%self.kitti_eval_repeat_counter==0:  # TODO(farzad) epoch length is hardcoded.
            kitti_eval_metrics = self.kitti_eval_metric.compute()
            # Get calculated TPs, FPs, FNs
            # Early results might not be correct as the 41 values are initialized with zero
            # and only a few predictions are available and thus a few thresholds are non-zero.
            # Therefore, mean over several zero values results in low final value.
            # detailed_stats shape (3, 1, 41, 5) where last dim is
            # {0: 'tp', 1: 'fp', 2: 'fn', 3: 'similarity', 4: 'precision thresholds'}
            total_num_samples = kitti_eval_metrics['total_num_samples']#max(self.metrics[tag].kitti_eval_metric._update_count, 1)
            detailed_stats = kitti_eval_metrics['detailed_stats']
            for m, metric_name in enumerate(['tps', 'fps', 'fns', 'sim', 'thresh', 'trans_err', 'orient_err', 'scale_err']):
                if metric_name == 'sim' or metric_name == 'thresh':
                    continue
                class_metrics_all = {}
                class_metrics_batch = {}
                for c, cls_name in enumerate(['Car', 'Pedestrian', 'Cyclist']):
                    metric_value = np.nanmax(detailed_stats[c, 0, :, m])
                    if not np.isnan(metric_value):
                        # class_metrics_all[cls_name] = metric_value  # commented to reduce complexity.
                        class_metrics_batch[cls_name] = metric_value / total_num_samples
                # statistics['all_' + metric_name] = class_metrics_all
                statistics[metric_name + '_per_sample'] = class_metrics_batch

            # Get calculated Precision
            for m, metric_name in enumerate(['mAP_3d', 'mAP_3d_R40']):
                class_metrics_all = {}
                for c, cls_name in enumerate(['Car', 'Pedestrian', 'Cyclist']):
                    metric_value = kitti_eval_metrics[metric_name][c].item()
                    if not np.isnan(metric_value):
                        class_metrics_all[cls_name] = metric_value
                statistics[metric_name] = class_metrics_all

            # Get calculated recall
            class_metrics_all = {}
            for c, cls_name in enumerate(['Car', 'Pedestrian', 'Cyclist']):
                metric_value = np.nanmax(kitti_eval_metrics['raw_recall'][c])
                if not np.isnan(metric_value):
                    class_metrics_all[cls_name] = metric_value
            statistics['max_recall'] = class_metrics_all

            # Draw Precision-Recall curves
            fig, axs = plt.subplots(1, 3, figsize=(12, 4), gridspec_kw={'wspace': 0.5})
            # plt.tight_layout()
            for c, cls_name in enumerate(['Car', 'Pedestrian', 'Cyclist']):
                thresholds = kitti_eval_metrics['detailed_stats'][c, 0, ::-1, 4]
                prec = kitti_eval_metrics['raw_precision'][c, 0, ::-1]
                rec = kitti_eval_metrics['raw_recall'][c, 0, ::-1]
                valid_mask = ~((rec == 0) | (prec == 0))

                ax_c = axs[c]
                ax_c_twin = ax_c.twinx()
                ax_c.plot(thresholds[valid_mask], prec[valid_mask], 'b-')
                ax_c_twin.plot(thresholds[valid_mask], rec[valid_mask], 'r-')
                ax_c.set_title(cls_name)
                ax_c.set_xlabel('Foreground score')
                ax_c.set_ylabel('Precision', color='b')
                ax_c_twin.set_ylabel('Recall', color='r')

            prec_rec_fig = fig.get_figure()
            statistics['prec_rec_fig'] = prec_rec_fig

            # reset kitti eval stats
            self.kitti_eval_metric.reset()
            

        # reset other stats
        self.mean_metric.reset()

        return statistics


def eval_class(gt_annos,
               dt_annos,
               current_classes,
               metric,
               min_overlaps,
               overlaps):
    """Kitti eval. support 2d/bev/3d/aos eval. support 0.5:0.05:0.95 coco AP.
    Args:
        gt_annos: dict, must from get_label_annos() in kitti_common.py
        dt_annos: dict, must from get_label_annos() in kitti_common.py
        current_classes: list of int, 0: car, 1: pedestrian, 2: cyclist
        difficultys: list of int. eval difficulty, 0: easy, 1: normal, 2: hard
        metric: eval type. 0: bbox, 1: bev, 2: 3d
        min_overlaps: float, min overlap. format: [num_overlap, metric, class].
        num_parts: int. a parameter for fast calculate algorithm

    Returns:
        dict of recall, precision and aos
    """
    # TODO(farzad) Assuming class labels in gt and dt start from 0

    N_SAMPLE_PTS = 41
    num_minoverlap = len(min_overlaps)
    num_class = len(current_classes)
    precision = np.nan * np.zeros([num_class, num_minoverlap, N_SAMPLE_PTS])
    recall = np.nan * np.zeros([num_class, num_minoverlap, N_SAMPLE_PTS])
    detailed_stats = np.nan * np.zeros([num_class, num_minoverlap, N_SAMPLE_PTS, 8])  # TP, FP, FN, Similarity, thresholds
    raw_precision = np.nan * np.zeros_like(precision)
    raw_recall = np.nan * np.zeros_like(recall)

    for m, current_class in enumerate(current_classes):
        rets = _prepare_data(gt_annos, dt_annos, current_class)
        (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets,
         dontcares, total_dc_num, total_num_valid_gt) = rets
        for k, min_overlap in enumerate(min_overlaps[:, metric, m]):
            thresholdss = []
            for i in range(len(gt_annos)):
                rets = compute_statistics_jit(overlaps[i], gt_datas_list[i], dt_datas_list[i], ignored_gts[i],
                                              ignored_dets[i], dontcares[i], metric, min_overlap=min_overlap,
                                              thresh=0.0, compute_fp=False)
                tp, fp, fn, similarity, thresholds, *_ = rets
                thresholdss += thresholds.tolist()
            thresholdss = np.array(thresholdss)
            thresholds = get_thresholds(thresholdss, total_num_valid_gt)
            thresholds = np.array(thresholds)
            pr = np.zeros([len(thresholds), 7])
            for i in range(len(gt_annos)):
                for t, thresh in enumerate(thresholds):
                    tp, fp, fn, similarity, _, tp_indices, gt_indices = compute_statistics_jit(overlaps[i], gt_datas_list[i],
                                                                       dt_datas_list[i], ignored_gts[i],
                                                                       ignored_dets[i], dontcares[i],
                                                                       metric, min_overlap=min_overlap,
                                                                       thresh=thresh, compute_fp=True)
                    if 0 < tp:
                        assignment_err = cal_tp_metric(dt_datas_list[i][tp_indices], gt_datas_list[i][gt_indices])
                        pr[t, 4] += assignment_err[0]
                        pr[t, 5] += assignment_err[1]
                        pr[t, 6] += assignment_err[2]

                    pr[t, 0] += tp
                    pr[t, 1] += fp
                    pr[t, 2] += fn
                    if similarity != -1:
                        pr[t, 3] += similarity

            for i in range(len(thresholds)):
                detailed_stats[m, k, i, 0] = pr[i, 0]
                detailed_stats[m, k, i, 1] = pr[i, 1]
                detailed_stats[m, k, i, 2] = pr[i, 2]
                detailed_stats[m, k, i, 3] = pr[i, 3]
                detailed_stats[m, k, i, 4] = thresholds[i]
                detailed_stats[m, k, i, 5] = pr[i, 4] / pr[i, 0]
                detailed_stats[m, k, i, 6] = pr[i, 5] / pr[i, 0]
                detailed_stats[m, k, i, 7] = pr[i, 6] / pr[i, 0]

            for i in range(len(thresholds)):
                recall[m, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 2])
                precision[m, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 1])

            raw_recall[m, k] = recall[m, k]
            raw_precision[m, k] = precision[m, k]
            for i in range(len(thresholds)):
                precision[m, k, i] = np.nanmax(
                    precision[m, k, i:], axis=-1)
                recall[m, k, i] = np.nanmax(recall[m, k, i:], axis=-1)

    ret_dict = {
        "recall": recall,
        "precision": precision,
        "detailed_stats": detailed_stats,
        "raw_recall": raw_recall,
        "raw_precision": raw_precision
    }
    return ret_dict


def _prepare_data(gt_annos, dt_annos, current_class):
    gt_datas_list = []
    dt_datas_list = []
    total_dc_num = []
    ignored_gts, ignored_dets, dontcares = [], [], []
    total_num_valid_gt = 0
    for i in range(len(gt_annos)):
        rets = clean_data(gt_annos[i], dt_annos[i], current_class)
        num_valid_gt, ignored_gt, ignored_det, dc_bboxes = rets
        ignored_gts.append(np.array(ignored_gt, dtype=np.int64))
        ignored_dets.append(np.array(ignored_det, dtype=np.int64))
        if len(dc_bboxes) == 0:
            dc_bboxes = np.zeros((0, 4)).astype(np.float64)
        else:
            dc_bboxes = np.stack(dc_bboxes, 0).astype(np.float64)
        total_dc_num.append(dc_bboxes.shape[0])
        dontcares.append(dc_bboxes)
        total_num_valid_gt += num_valid_gt
        gt_datas_list.append(gt_annos[i])
        dt_datas_list.append(dt_annos[i])
    total_dc_num = np.stack(total_dc_num, axis=0)
    return (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets, dontcares,
            total_dc_num, total_num_valid_gt)


def clean_data(gt_anno, dt_anno, current_class):
    dc_bboxes, ignored_gt, ignored_dt = [], [], []

    num_gt = len(gt_anno)  # len(gt_anno["name"])
    num_dt = len(dt_anno)  # len(dt_anno["name"])
    num_valid_gt = 0
    # TODO(farzad) cleanup and parallelize
    for i in range(num_gt):
        gt_cls_ind = gt_anno[i][-1]
        if (gt_cls_ind == current_class):
            ignored_gt.append(0)
            num_valid_gt += 1
        else:
            ignored_gt.append(-1)

    for i in range(num_dt):
        dt_cls_ind = dt_anno[i][-2]
        if (dt_cls_ind == current_class):
            ignored_dt.append(0)
        else:
            ignored_dt.append(-1)

    return num_valid_gt, ignored_gt, ignored_dt, dc_bboxes


def compute_statistics_jit(overlaps,
                           gt_datas,
                           dt_datas,
                           ignored_gt,
                           ignored_det,
                           dc_bboxes,
                           metric,
                           min_overlap,
                           thresh=0,
                           compute_fp=False):
    det_size = dt_datas.shape[0]
    gt_size = gt_datas.shape[0]
    dt_scores = dt_datas[:, -1]

    assigned_detection = [False] * det_size
    ignored_threshold = [False] * det_size
    if compute_fp:
        for i in range(det_size):
            if (dt_scores[i] < thresh):
                ignored_threshold[i] = True
    NO_DETECTION = -10000000
    tp, fp, fn, similarity = 0, 0, 0, 0
    tp_indices = []
    gt_indices = []
    thresholds = np.zeros((gt_size,))
    thresh_idx = 0

    for i in range(gt_size):
        if ignored_gt[i] == -1:
            continue
        det_idx = -1
        valid_detection = NO_DETECTION
        max_overlap = 0
        assigned_ignored_det = False

        for j in range(det_size):
            if (ignored_det[j] == -1):
                continue
            if (assigned_detection[j]):
                continue
            if (ignored_threshold[j]):
                continue
            overlap = overlaps[j, i]
            dt_score = dt_scores[j]
            if (not compute_fp and (overlap > min_overlap)
                    and dt_score > valid_detection):
                det_idx = j
                valid_detection = dt_score
            elif (compute_fp and (overlap > min_overlap)
                  and (overlap > max_overlap or assigned_ignored_det)
                  and ignored_det[j] == 0):
                max_overlap = overlap
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = False
            elif (compute_fp and (overlap > min_overlap)
                  and (valid_detection == NO_DETECTION)
                  and ignored_det[j] == 1):
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = True

        if (valid_detection == NO_DETECTION) and ignored_gt[i] == 0:
            fn += 1
        elif ((valid_detection != NO_DETECTION)
              and (ignored_gt[i] == 1 or ignored_det[det_idx] == 1)):
            assigned_detection[det_idx] = True
        elif valid_detection != NO_DETECTION:
            tp += 1
            tp_indices.append(det_idx)
            gt_indices.append(i)
            # thresholds.append(dt_scores[det_idx])
            thresholds[thresh_idx] = dt_scores[det_idx]
            thresh_idx += 1
            assigned_detection[det_idx] = True
    if compute_fp:
        for i in range(det_size):
            if (not (assigned_detection[i] or ignored_det[i] == -1
                     or ignored_det[i] == 1 or ignored_threshold[i])):
                fp += 1

    return tp, fp, fn, similarity, thresholds[:thresh_idx], np.array(tp_indices), np.array(gt_indices)


def cor_angle_range(angle):
    """ correct angle range to [-pi, pi]
    Args:
        angle:
    Returns:
    """
    gt_pi_mask = angle > np.pi
    lt_minus_pi_mask = angle < - np.pi
    angle[gt_pi_mask] = angle[gt_pi_mask] - 2 * np.pi
    angle[lt_minus_pi_mask] = angle[lt_minus_pi_mask] + 2 * np.pi

    return angle


def cal_angle_diff(angle1, angle2):
    # angle is from x to y, anti-clockwise
    angle1 = cor_angle_range(angle1)
    angle2 = cor_angle_range(angle2)

    diff = torch.abs(angle1 - angle2)
    gt_pi_mask = diff > math.pi
    diff[gt_pi_mask] = 2 * math.pi - diff[gt_pi_mask]

    return diff


def cal_tp_metric(tp_boxes, gt_boxes):
    assert tp_boxes.shape[0] == gt_boxes.shape[0]
    # L2 distance xy only
    center_distance = torch.norm(tp_boxes[:, :2] - gt_boxes[:, :2], dim=1)
    trans_err = center_distance.sum().item()

    # Angle difference
    angle_diff = cal_angle_diff(tp_boxes[:, 6], gt_boxes[:, 6])
    assert angle_diff.sum() >= 0
    orient_err = angle_diff.sum().item()

    # Scale difference
    aligned_tp_boxes = tp_boxes.detach().clone()
    # shift their center together
    aligned_tp_boxes[:, 0:3] = gt_boxes[:, 0:3]
    # align their angle
    aligned_tp_boxes[:, 6] = gt_boxes[:, 6]
    iou_matrix = iou3d_nms_utils.boxes_iou3d_gpu(aligned_tp_boxes[:, 0:7], gt_boxes[:, 0:7])
    max_ious, _ = torch.max(iou_matrix, dim=1)
    scale_err = (1 - max_ious).sum().item()

    return trans_err, orient_err, scale_err


def get_thresholds(scores: np.ndarray, num_gt, num_sample_pts=41):
    scores.sort()
    scores = scores[::-1]
    current_recall = 0
    thresholds = []
    for i, score in enumerate(scores):
        l_recall = (i + 1) / num_gt
        if i < (len(scores) - 1):
            r_recall = (i + 2) / num_gt
        else:
            r_recall = l_recall
        if (((r_recall - current_recall) < (current_recall - l_recall))
                and (i < (len(scores) - 1))):
            continue
        # recall = l_recall
        thresholds.append(score)
        current_recall += 1 / (num_sample_pts - 1.0)
    return thresholds


def get_mAP(prec):
    # sums = 0
    # for i in range(0, prec.shape[-1], 4):
    #     sums = sums + prec[..., i]
    # return sums / 11 * 100
    return np.nanmean(prec[..., ::4], axis=-1) * 100

def get_mAP_R40(prec):
    # sums = 0
    # for i in range(1, prec.shape[-1]):
    #     sums = sums + prec[..., i]
    # return sums / 40 * 100
    return np.nanmean(prec[..., 1:], axis=-1) * 100



def _stats(pred_infos, gt_infos):
    pred_infos = pickle.load(open(pred_infos, 'rb'))
    gt_infos = pickle.load(open(gt_infos, 'rb'))
    gt_annos = [info['annos'] for info in gt_infos]
    PR_detail_dict = {}
    ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
        gt_annos, pred_infos, current_classes=['Car', 'Pedestrian', 'Cyclist'], PR_detail_dict=PR_detail_dict
    )

    detailed_stats_3d = PR_detail_dict['3d']['detailed_stats']
    # detailed_stats_3d is a tensor of size [num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS, NUM_STATS] where
    # num_class in [0..6], and {0: 'Car', 1: 'Pedestrian', 2: 'Cyclist', 3: 'Van', 4: 'Person_sitting', 5: 'Truck'},
    # num_difficulty is 3, and {0: 'easy', 1: 'normal', 2: 'hard'},
    # num_minoverlap is 2, and {0: 'overlap_0_7', 1: 'overlap_0_5'} and overlap_0_7 (for 3D metric),
    # is [0.7, 0.5, 0.5, 0.7, 0.5, 0.7] for 'Car', 'Pedestrian', etc. correspondingly,
    # N_SAMPLE_PTS is 41,
    # NUM_STATS is 5, and {0:'tp', 1:'fp', 2:'fn', 3:'similarity', 4:'precision thresholds'}
    # for example [0, 1, 0, :, 0] means number of TPs of Car class with normal difficulty and overlap@0.7 for all 41 sample points

    # Example of extracting overlap between gts and dets of an example based on specific class and difficulty combination
    example_idx = 1  # second example in our dataset
    class_idx = 0  # class Car
    difficulty_idx = 1  # medium difficulty
    import numpy as np
    overlaps = PR_detail_dict['3d']['overlaps']
    class_difficulty_ignored_gts_mask = PR_detail_dict['3d']['class_difficulty_ignored_gts_mask']
    class_difficulty_ignored_dets_mask = PR_detail_dict['3d']['class_difficulty_ignored_dets_mask']
    valid_gts_inds = np.where(class_difficulty_ignored_gts_mask[class_idx, difficulty_idx, example_idx] == 0)[0]
    valid_dets_inds = np.where(class_difficulty_ignored_dets_mask[class_idx, difficulty_idx, example_idx] == 0)[0]
    valid_inds = np.ix_(valid_dets_inds, valid_gts_inds)
    cls_diff_overlaps = overlaps[example_idx][valid_inds]
    print("cls_diff_overlaps: ", cls_diff_overlaps)
    print("cls_diff_overlaps.shape: ", cls_diff_overlaps.shape)

    # Reproducing fig. 3 of soft-teacher as an example
    from matplotlib import pyplot as plt
    fig, ax1 = plt.subplots()
    precision = PR_detail_dict['3d']['precision']
    recall = PR_detail_dict['3d']['recall']
    thresholds = detailed_stats_3d[0, 1, 0, ::-1, -1]
    prec = precision[0, 1, 0, ::-1]
    rec = recall[0, 1, 0, ::-1]
    ax2 = ax1.twinx()
    valid_mask = ~((rec == 0) | (prec == 0))
    ax1.plot(thresholds[valid_mask], prec[valid_mask], 'b-')
    ax2.plot(thresholds[valid_mask], rec[valid_mask], 'r-')
    ax1.set_xlabel('Foreground score')
    ax1.set_ylabel('Precision', color='b')
    ax2.set_ylabel('Recall', color='r')

    plt.show()


def main():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--pred_infos', type=str, default=None, help='pickle file')
    parser.add_argument('--gt_infos', type=str, default=None, help='pickle file')
    args = parser.parse_args()

    _stats(args.pred_infos, args.gt_infos)


if __name__ == '__main__':
    main()
