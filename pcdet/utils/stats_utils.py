import argparse
import pickle

import torch
import numpy as np
from pcdet.ops.iou3d_nms import iou3d_nms_utils
import math
from pcdet.config import cfg

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
    from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval

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
