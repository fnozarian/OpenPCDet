import copy
import os
import pickle
import numpy as np
import torch
from pcdet.datasets.augmentor import augmentor_utils
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from .detector3d_template import Detector3DTemplate
from .pv_rcnn import PVRCNN

import common_utils
from stats_utils import metrics_registry
from prototype_utils import feature_bank_registry
from collections import defaultdict
from visual_utils import open3d_vis_utils as V

class PVRCNN_SSL(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        # something changes so need deep copy
        model_cfg_copy = copy.deepcopy(model_cfg)
        dataset_copy = copy.deepcopy(dataset)
        self.pv_rcnn = PVRCNN(model_cfg=model_cfg, num_class=num_class, dataset=dataset)

        self.pv_rcnn_ema = PVRCNN(model_cfg=model_cfg_copy, num_class=num_class, dataset=dataset_copy)
        for param in self.pv_rcnn_ema.parameters():
            param.detach_()
        self.add_module('pv_rcnn', self.pv_rcnn)
        self.add_module('pv_rcnn_ema', self.pv_rcnn_ema)
        self.accumulated_itr = 0

        self.thresh = model_cfg.THRESH
        self.sem_thresh = model_cfg.SEM_THRESH
        self.unlabeled_supervise_cls = model_cfg.UNLABELED_SUPERVISE_CLS
        self.unlabeled_supervise_refine = model_cfg.UNLABELED_SUPERVISE_REFINE
        self.unlabeled_weight = model_cfg.UNLABELED_WEIGHT
        self.no_nms = model_cfg.NO_NMS
        self.supervise_mode = model_cfg.SUPERVISE_MODE

        for bank_configs in model_cfg.get("FEATURE_BANK_LIST", []):
            feature_bank_registry.register(tag=bank_configs["NAME"], **bank_configs)

        for metrics_configs in model_cfg.get("METRICS_BANK_LIST", []):
            metrics_registry.register(tag=metrics_configs["NAME"], dataset=self.dataset, **metrics_configs)

        vals_to_store = ['iou_roi_pl', 'iou_roi_gt', 'pred_scores', 'teacher_pred_scores', 
                        'weights', 'roi_scores', 'pcv_scores', 'num_points_in_roi', 'class_labels',
                        'iteration']
        self.val_dict = {val: [] for val in vals_to_store}

    def _update_feature_bank(self, batch_dict, labeled_inds):
        # Update the bank with student's features from augmented labeled data
        bank = feature_bank_registry.get('gt_aug_lbl_prototypes')
        bs = batch_dict['batch_size']
        num_gts = batch_dict['gt_boxes'].shape[1]

        batch_dict_temp = {
            "batch_size": batch_dict['batch_size'],
            "gt_boxes": batch_dict['gt_boxes'].clone().detach(),
            "point_coords": batch_dict['point_coords'].clone().detach(),
            "point_features": batch_dict['point_features'].clone().detach(),
            "point_cls_scores": batch_dict['point_cls_scores'].clone().detach()
        }
        with torch.no_grad():
            batch_gt_feats = self.pv_rcnn.roi_head.get_pooled_features(batch_dict_temp, pool_gtboxes=True).view(bs, num_gts, -1)

        bank_input = defaultdict(list)
        for i in labeled_inds:
            gt_boxes = batch_dict_temp['gt_boxes'][i]
            nonzero_mask = torch.logical_not(torch.eq(gt_boxes, 0).all(dim=-1))
            if nonzero_mask.sum() == 0:
                print(f"no gt instance in frame {batch_dict['frame_id'][i]}")
                continue
            gt_boxes = gt_boxes[nonzero_mask].clone().detach()
            bs_mask = batch_dict['points'][:, 0].int() == i
            points = batch_dict['points'][bs_mask, 1:4]
            gt_feat = batch_gt_feats[i][nonzero_mask]
            gt_labels = gt_boxes[:, -1].int() - 1
            gt_boxes = gt_boxes[:, :7]
            ins_idxs = batch_dict['instance_idx'][i][nonzero_mask].int()
            smpl_id = torch.from_numpy(batch_dict['frame_id'].astype(np.int32))[i].int()

            # filter out gt instances with too few points when updating the bank
            num_points_in_gt = roiaware_pool3d_utils.points_in_boxes_cpu(points.cpu(), gt_boxes.cpu()).sum(dim=-1)
            valid_gts_mask = (num_points_in_gt >= bank.num_points_thresh)
            print(f"{(~valid_gts_mask).sum()} gt instance(s) with id(s) {ins_idxs[~valid_gts_mask].tolist()}"
                  f" and num points {num_points_in_gt[~valid_gts_mask].tolist()} are filtered")
            if valid_gts_mask.sum() == 0:
                print(f"no valid gt instances with enough points in frame {batch_dict['frame_id'][i]}")
                continue
            bank_input['feats'].append(gt_feat[valid_gts_mask])
            bank_input['labels'].append(gt_labels[valid_gts_mask])
            bank_input['ins_ids'].append(ins_idxs[valid_gts_mask])
            bank_input['smpl_ids'].append(smpl_id)

            # valid_boxes = gt_boxes[valid_gts_mask]
            # valid_box_labels = gt_labels[valid_gts_mask]
            # self.vis(valid_boxes, valid_box_labels, points)

        bank.update(**bank_input, iteration=batch_dict['cur_iteration'])

    def forward(self, batch_dict):
        if self.training:
            labeled_mask = batch_dict['labeled_mask'].view(-1)
            labeled_inds = torch.nonzero(labeled_mask).squeeze(1).long()
            unlabeled_inds = torch.nonzero(1-labeled_mask).squeeze(1).long()
            batch_dict['unlabeled_inds'] = unlabeled_inds
            batch_dict['labeled_inds'] = labeled_inds
            batch_dict_ema = {}
            keys = list(batch_dict.keys())
            for k in keys:
                if k + '_ema' in keys:
                    continue
                if k.endswith('_ema'):
                    batch_dict_ema[k[:-4]] = batch_dict[k]
                else:
                    batch_dict_ema[k] = batch_dict[k]

            with torch.no_grad():
                # self.pv_rcnn_ema.eval()  # TODO(farzad) Why this should be commented out?
                for cur_module in self.pv_rcnn_ema.module_list:
                    try:
                        batch_dict_ema = cur_module(batch_dict_ema, test_only=True)
                    except TypeError as e:
                        batch_dict_ema = cur_module(batch_dict_ema)

            pred_dicts_ens, recall_dicts_ema = self.pv_rcnn_ema.post_processing(batch_dict_ema, no_recall_dict=True)

            # Used for calc stats before and after filtering
            ori_unlabeled_boxes = batch_dict['gt_boxes'][unlabeled_inds, ...]
            # if self.model_cfg.ROI_HEAD.get("ENABLE_EVAL", False):
            #     # PL metrics before filtering
            #     self.update_metrics(batch_dict, pred_dicts_ens, unlabeled_inds, labeled_inds)

            pseudo_boxes, pseudo_scores, pseudo_sem_scores, pseudo_boxes_var, pseudo_scores_var = \
                self._filter_pseudo_labels(pred_dicts_ens, unlabeled_inds)

            self._fill_with_pseudo_labels(batch_dict, pseudo_boxes, unlabeled_inds, labeled_inds)

            # apply student's augs on teacher's pseudo-labels (filtered) only (not points)
            batch_dict = self.apply_augmentation(batch_dict, batch_dict, unlabeled_inds, key='gt_boxes')

            batch_dict['ori_unlabeled_boxes'] = ori_unlabeled_boxes

            for cur_module in self.pv_rcnn.module_list:
                batch_dict = cur_module(batch_dict)

            self._update_feature_bank(batch_dict, labeled_inds)

            # For metrics calculation
            self.pv_rcnn.roi_head.forward_ret_dict['unlabeled_inds'] = unlabeled_inds
            # self.pv_rcnn.roi_head.forward_ret_dict['pl_boxes'] = batch_dict['gt_boxes']
            # self.pv_rcnn.roi_head.forward_ret_dict['pl_scores'] = pseudo_scores

            if self.model_cfg['ROI_HEAD'].get('ENABLE_SOFT_TEACHER', False):
                # using teacher to evaluate student's bg/fg proposals through its rcnn head
                with torch.no_grad():
                    batch_dict_std = {}
                    batch_dict_std['unlabeled_inds'] = batch_dict['unlabeled_inds']
                    batch_dict_std['labeled_inds'] = batch_dict['labeled_inds']
                    batch_dict_std['rois'] = batch_dict['rois'].data.clone()
                    batch_dict_std['roi_scores'] = batch_dict['roi_scores'].data.clone()
                    batch_dict_std['roi_labels'] = batch_dict['roi_labels'].data.clone()
                    batch_dict_std['has_class_labels'] = batch_dict['has_class_labels']
                    batch_dict_std['batch_size'] = batch_dict['batch_size']
                    batch_dict_std['point_features'] = batch_dict_ema['point_features'].data.clone()
                    batch_dict_std['point_coords'] = batch_dict_ema['point_coords'].data.clone()
                    batch_dict_std['point_cls_scores'] = batch_dict_ema['point_cls_scores'].data.clone()

                    batch_dict_std = self.reverse_augmentation(batch_dict_std, batch_dict, unlabeled_inds)

                    # Perturb Student's ROIs before using them for Teacher's ROI head
                    if self.model_cfg.ROI_HEAD.ROI_AUG.get('ENABLE', False):
                        augment_rois = getattr(augmentor_utils, self.model_cfg.ROI_HEAD.ROI_AUG.AUG_TYPE, augmentor_utils.roi_aug_ros)
                        # rois_before_aug is used only for debugging, can be removed later
                        batch_dict_std['rois_before_aug'] = batch_dict_std['rois'].clone().detach()
                        batch_dict_std['rois'][unlabeled_inds] = \
                            augment_rois(batch_dict_std['rois'][unlabeled_inds], self.model_cfg.ROI_HEAD)
                    self.pv_rcnn_ema.roi_head.forward(batch_dict_std, test_only=True)
                    batch_dict_std = self.apply_augmentation(batch_dict_std, batch_dict, unlabeled_inds, key='batch_box_preds')

                    pred_dicts_std, recall_dicts_std = self.pv_rcnn_ema.post_processing(batch_dict_std,
                                                                                        no_recall_dict=True,
                                                                                        no_nms_for_unlabeled=True)
                    rcnn_cls_score_teacher = -torch.ones_like(self.pv_rcnn.roi_head.forward_ret_dict['rcnn_cls_labels'])
                    batch_box_preds_teacher = torch.zeros_like(self.pv_rcnn.roi_head.forward_ret_dict['batch_box_preds'])
                    for uind in unlabeled_inds:
                        rcnn_cls_score_teacher[uind] = pred_dicts_std[uind]['pred_scores']
                        batch_box_preds_teacher[uind] = pred_dicts_std[uind]['pred_boxes']
                    self.pv_rcnn.roi_head.forward_ret_dict['rcnn_cls_score_teacher'] = rcnn_cls_score_teacher
                    # For metrics
                    self.pv_rcnn.roi_head.forward_ret_dict['batch_box_preds_teacher'] = batch_box_preds_teacher

            disp_dict = {}
            loss_rpn_cls, loss_rpn_box, tb_dict = self.pv_rcnn.dense_head.get_loss(scalar=False)
            loss_point, tb_dict = self.pv_rcnn.point_head.get_loss(tb_dict, scalar=False)
            loss_rcnn_cls, loss_rcnn_box, ulb_loss_cls_dist, tb_dict = self.pv_rcnn.roi_head.get_loss(tb_dict, scalar=False)

            # Use the same reduction method as the baseline model (3diou) by the default
            reduce_loss = getattr(torch, self.model_cfg.REDUCE_LOSS, 'sum')
            if not self.unlabeled_supervise_cls:
                loss_rpn_cls = reduce_loss(loss_rpn_cls[labeled_inds, ...])
            else:
                loss_rpn_cls = reduce_loss(loss_rpn_cls[labeled_inds, ...]) + reduce_loss(loss_rpn_cls[unlabeled_inds, ...]) * self.unlabeled_weight

            loss_rpn_box = reduce_loss(loss_rpn_box[labeled_inds, ...]) + reduce_loss(loss_rpn_box[unlabeled_inds, ...]) * self.unlabeled_weight
            loss_point = reduce_loss(loss_point[labeled_inds, ...])
            if self.model_cfg['ROI_HEAD'].get('ENABLE_SOFT_TEACHER', False) or self.model_cfg.get('UNLABELED_SUPERVISE_OBJ', False):
                loss_rcnn_cls = reduce_loss(loss_rcnn_cls[labeled_inds, ...]) + reduce_loss(loss_rcnn_cls[unlabeled_inds, ...]) * self.unlabeled_weight
            else:
                loss_rcnn_cls = reduce_loss(loss_rcnn_cls[labeled_inds, ...])
            if not self.unlabeled_supervise_refine:
                loss_rcnn_box = reduce_loss(loss_rcnn_box[labeled_inds, ...])
            else:
                loss_rcnn_box = reduce_loss(loss_rcnn_box[labeled_inds, ...]) + reduce_loss(loss_rcnn_box[unlabeled_inds, ...]) * self.unlabeled_weight
            if self.model_cfg['ROI_HEAD'].get('ENABLE_ULB_CLS_DIST_LOSS', False):
                loss = loss_rpn_cls + loss_rpn_box + loss_point + loss_rcnn_cls + loss_rcnn_box + ulb_loss_cls_dist
            else:
                loss = loss_rpn_cls + loss_rpn_box + loss_point + loss_rcnn_cls + loss_rcnn_box
            tb_dict_ = {}
            for key in tb_dict.keys():
                if 'loss' in key:
                    tb_dict_[key+"_labeled"] = reduce_loss(tb_dict[key][labeled_inds, ...])
                    tb_dict_[key + "_unlabeled"] = reduce_loss(tb_dict[key][unlabeled_inds, ...])
                elif 'acc' in key:
                    tb_dict_[key+"_labeled"] = reduce_loss(tb_dict[key][labeled_inds, ...])
                    tb_dict_[key + "_unlabeled"] = reduce_loss(tb_dict[key][unlabeled_inds, ...])
                elif 'point_pos_num' in key:
                    tb_dict_[key + "_labeled"] = reduce_loss(tb_dict[key][labeled_inds, ...])
                    tb_dict_[key + "_unlabeled"] = reduce_loss(tb_dict[key][unlabeled_inds, ...])
                else:
                    tb_dict_[key] = tb_dict[key]

            if self.model_cfg.get('STORE_SCORES_IN_PKL', False) :
                self.dump_statistics(batch_dict, unlabeled_inds)

            for tag in feature_bank_registry.tags():
                feature_bank_registry.get(tag).compute()

            for tag in metrics_registry.tags():
                results = metrics_registry.get(tag).compute()
                if results is not None:
                    tb_dict_.update({tag + "/" + k: v for k, v in zip(*results)})

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict_, disp_dict

        else:
            for cur_module in self.pv_rcnn.module_list:
                batch_dict = cur_module(batch_dict)

            pred_dicts, recall_dicts = self.pv_rcnn.post_processing(batch_dict)

            return pred_dicts, recall_dicts, {}

    def vis(self, boxes, box_labels, points):
        boxes = boxes.cpu().numpy()
        points = points.cpu().numpy()
        box_labels = box_labels.cpu().numpy()
        V.draw_scenes(points=points, gt_boxes=boxes, gt_labels=box_labels)

    def dump_statistics(self, batch_dict, unlabeled_inds):
        # Store different types of scores over all itrs and epochs and dump them in a pickle for offline modeling
        # TODO (shashank) : Can be optimized later to save computational time, currently takes about 0.002sec
        batch_roi_labels = self.pv_rcnn.roi_head.forward_ret_dict['roi_labels'][unlabeled_inds]
        batch_roi_labels = [roi_labels.clone().detach() for roi_labels in batch_roi_labels]

        batch_rois = self.pv_rcnn.roi_head.forward_ret_dict['rois'][unlabeled_inds]
        batch_rois = [rois.clone().detach() for rois in batch_rois]

        batch_ori_gt_boxes = self.pv_rcnn.roi_head.forward_ret_dict['ori_unlabeled_boxes']
        batch_ori_gt_boxes = [ori_gt_boxes.clone().detach() for ori_gt_boxes in batch_ori_gt_boxes]

        for i in range(len(batch_rois)):
            valid_rois_mask = torch.logical_not(torch.all(batch_rois[i] == 0, dim=-1))
            valid_rois = batch_rois[i][valid_rois_mask]
            valid_roi_labels = batch_roi_labels[i][valid_rois_mask]
            valid_roi_labels -= 1  # Starting class indices from zero

            valid_gt_boxes_mask = torch.logical_not(torch.all(batch_ori_gt_boxes[i] == 0, dim=-1))
            valid_gt_boxes = batch_ori_gt_boxes[i][valid_gt_boxes_mask]
            valid_gt_boxes[:, -1] -= 1  # Starting class indices from zero

            num_gts = valid_gt_boxes_mask.sum()
            num_preds = valid_rois_mask.sum()

            cur_unlabeled_ind = unlabeled_inds[i]
            if num_gts > 0 and num_preds > 0:
                # Find IoU between Student's ROI v/s Original GTs
                overlap = iou3d_nms_utils.boxes_iou3d_gpu(valid_rois[:, 0:7], valid_gt_boxes[:, 0:7])
                preds_iou_max, assigned_gt_inds = overlap.max(dim=1)
                self.val_dict['iou_roi_gt'].extend(preds_iou_max.tolist())

                cur_iou_roi_pl = self.pv_rcnn.roi_head.forward_ret_dict['gt_iou_of_rois'][cur_unlabeled_ind]
                self.val_dict['iou_roi_pl'].extend(cur_iou_roi_pl.tolist())

                cur_pred_score = torch.sigmoid(batch_dict['batch_cls_preds'][cur_unlabeled_ind]).squeeze()
                self.val_dict['pred_scores'].extend(cur_pred_score.tolist())

                if 'rcnn_cls_score_teacher' in self.pv_rcnn.roi_head.forward_ret_dict:
                    cur_teacher_pred_score = self.pv_rcnn.roi_head.forward_ret_dict['rcnn_cls_score_teacher'][
                        cur_unlabeled_ind]
                    self.val_dict['teacher_pred_scores'].extend(cur_teacher_pred_score.tolist())

                    cur_weight = self.pv_rcnn.roi_head.forward_ret_dict['rcnn_cls_weights'][cur_unlabeled_ind]
                    self.val_dict['weights'].extend(cur_weight.tolist())

                cur_roi_score = torch.sigmoid(self.pv_rcnn.roi_head.forward_ret_dict['roi_scores'][cur_unlabeled_ind])
                self.val_dict['roi_scores'].extend(cur_roi_score.tolist())

                cur_pcv_score = self.pv_rcnn.roi_head.forward_ret_dict['pcv_scores'][cur_unlabeled_ind]
                self.val_dict['pcv_scores'].extend(cur_pcv_score.tolist())

                cur_num_points_roi = self.pv_rcnn.roi_head.forward_ret_dict['num_points_in_roi'][cur_unlabeled_ind]
                self.val_dict['num_points_in_roi'].extend(cur_num_points_roi.tolist())

                cur_roi_label = self.pv_rcnn.roi_head.forward_ret_dict['roi_labels'][cur_unlabeled_ind].squeeze()
                self.val_dict['class_labels'].extend(cur_roi_label.tolist())

                cur_iteration = torch.ones_like(preds_iou_max) * (batch_dict['cur_iteration'])
                self.val_dict['iteration'].extend(cur_iteration.tolist())

        # replace old pickle data (if exists) with updated one
        output_dir = os.path.split(os.path.abspath(batch_dict['ckpt_save_dir']))[0]
        file_path = os.path.join(output_dir, 'scores.pkl')
        pickle.dump(self.val_dict, open(file_path, 'wb'))

    # def update_metrics(self, input_dict, pred_dict, unlabeled_inds, labeled_inds):
    #     """
    #     Recording PL vs GT statistics BEFORE filtering
    #     """
    #     if 'pl_gt_metrics_before_filtering' in self.model_cfg.ROI_HEAD.METRICS_PRED_TYPES:
    #         pseudo_boxes, pseudo_labels, pseudo_scores, pseudo_sem_scores, _, _ = self._unpack_predictions(
    #             pred_dict, unlabeled_inds)
    #         pseudo_boxes = [torch.cat([pseudo_box, pseudo_label.view(-1, 1).float()], dim=1) \
    #                         for (pseudo_box, pseudo_label) in zip(pseudo_boxes, pseudo_labels)]
    #
    #         # Making consistent # of pseudo boxes in each batch
    #         # NOTE: Need to store them in batch_dict in a new key, which can be removed later
    #         input_dict['pseudo_boxes_prefilter'] = torch.zeros_like(input_dict['gt_boxes'])
    #         self._fill_with_pseudo_labels(input_dict, pseudo_boxes, unlabeled_inds, labeled_inds,
    #                                       key='pseudo_boxes_prefilter')
    #
    #         # apply student's augs on teacher's pseudo-boxes (w/o filtered)
    #         batch_dict = self.apply_augmentation(input_dict, input_dict, unlabeled_inds, key='pseudo_boxes_prefilter')
    #
    #         tag = f'pl_gt_metrics_before_filtering'
    #         metrics = metrics_registry.get(tag)
    #
    #         preds_prefilter = [batch_dict['pseudo_boxes_prefilter'][uind] for uind in unlabeled_inds]
    #         gts_prefilter = [batch_dict['gt_boxes'][uind] for uind in unlabeled_inds]
    #         metric_inputs = {'preds': preds_prefilter, 'pred_scores': pseudo_scores, 'roi_scores': pseudo_sem_scores,
    #                          'ground_truths': gts_prefilter}
    #         metrics.update(**metric_inputs)
    #         batch_dict.pop('pseudo_boxes_prefilter')

    # TODO(farzad) refactor and remove this!
    def _unpack_predictions(self, pred_dicts, unlabeled_inds):
        pseudo_boxes = []
        pseudo_scores = []
        pseudo_sem_scores = []
        pseudo_labels = []
        pseudo_boxes_var = []
        pseudo_scores_var = []
        for ind in unlabeled_inds:
            pseudo_score = pred_dicts[ind]['pred_scores']
            pseudo_box = pred_dicts[ind]['pred_boxes']
            pseudo_label = pred_dicts[ind]['pred_labels']
            pseudo_sem_score = pred_dicts[ind]['pred_sem_scores']
            # TODO(farzad) REFACTOR LATER!
            pseudo_box_var = -1 * torch.ones_like(pseudo_box)
            if "pred_boxes_var" in pred_dicts[ind].keys():
                pseudo_box_var = pred_dicts[ind]['pred_boxes_var']
            pseudo_score_var = -1 * torch.ones_like(pseudo_score)
            if "pred_scores_var" in pred_dicts[ind].keys():
                pseudo_score_var = pred_dicts[ind]['pred_scores_var']
            if len(pseudo_label) == 0:
                pseudo_boxes.append(pseudo_label.new_zeros((1, 7)).float())
                pseudo_boxes_var.append(pseudo_label.new_zeros((1, 7)).float())
                pseudo_sem_scores.append(pseudo_label.new_zeros((1,)).float())
                pseudo_scores.append(pseudo_label.new_zeros((1,)).float())
                pseudo_scores_var.append(pseudo_label.new_zeros((1,)).float())
                pseudo_labels.append(pseudo_label.new_zeros((1,)).float())
                continue

            pseudo_boxes.append(pseudo_box)
            pseudo_boxes_var.append(pseudo_box_var)
            pseudo_sem_scores.append(pseudo_sem_score)
            pseudo_scores.append(pseudo_score)
            pseudo_scores_var.append(pseudo_score_var)
            pseudo_labels.append(pseudo_label)

        return pseudo_boxes, pseudo_labels, pseudo_scores, pseudo_sem_scores, pseudo_boxes_var, pseudo_scores_var

    # TODO(farzad) refactor and remove this!
    def _filter_pseudo_labels(self, pred_dicts, unlabeled_inds):
        pseudo_boxes = []
        pseudo_scores = []
        pseudo_sem_scores = []
        pseudo_scores_var = []
        pseudo_boxes_var = []
        for pseudo_box, pseudo_label, pseudo_score, pseudo_sem_score, pseudo_box_var, pseudo_score_var in zip(
                *self._unpack_predictions(pred_dicts, unlabeled_inds)):

            if pseudo_label[0] == 0:
                pseudo_boxes.append(torch.cat([pseudo_box, pseudo_label.view(-1, 1).float()], dim=1))
                pseudo_sem_scores.append(pseudo_sem_score)
                pseudo_scores.append(pseudo_score)
                pseudo_scores_var.append(pseudo_score_var)
                pseudo_boxes_var.append(pseudo_box_var)
                continue

            conf_thresh = torch.tensor(self.thresh, device=pseudo_label.device).unsqueeze(
                0).repeat(len(pseudo_label), 1).gather(dim=1, index=(pseudo_label - 1).unsqueeze(-1))

            sem_conf_thresh = torch.tensor(self.sem_thresh, device=pseudo_label.device).unsqueeze(
                0).repeat(len(pseudo_label), 1).gather(dim=1, index=(pseudo_label - 1).unsqueeze(-1))

            valid_inds = pseudo_score > conf_thresh.squeeze()

            valid_inds = valid_inds & (pseudo_sem_score > sem_conf_thresh.squeeze())

            # TODO(farzad) can this be similarly determined by tag-based stats before and after filtering?
            # rej_labels = pseudo_label[~valid_inds]
            # rej_labels_per_class = torch.bincount(rej_labels, minlength=len(self.thresh) + 1)
            # for class_ind, class_key in enumerate(self.metric_table.metric_record):
            #     if class_key == 'class_agnostic':
            #         self.metric_table.metric_record[class_key].metrics['rej_pseudo_lab'].update(
            #             rej_labels_per_class[1:].sum().item())
            #     else:
            #         self.metric_table.metric_record[class_key].metrics['rej_pseudo_lab'].update(
            #             rej_labels_per_class[class_ind].item())

            pseudo_sem_score = pseudo_sem_score[valid_inds]
            pseudo_box = pseudo_box[valid_inds]
            pseudo_label = pseudo_label[valid_inds]
            pseudo_score = pseudo_score[valid_inds]
            pseudo_box_var = pseudo_box_var[valid_inds]
            pseudo_score_var = pseudo_score_var[valid_inds]
            # TODO : Two stage filtering instead of applying NMS
            # Stage1 based on size of bbox, Stage2 is objectness thresholding
            # Note : Two stages happen sequentially, and not independently.
            # vol_boxes = ((pseudo_box[:, 3] * pseudo_box[:, 4] * pseudo_box[:, 5])/torch.abs(pseudo_box[:,2][0])).view(-1)
            # vol_boxes, _ = torch.sort(vol_boxes, descending=True)
            # # Set volume threshold to 10% of the maximum volume of the boxes
            # keep_ind = int(self.model_cfg.PSEUDO_TWO_STAGE_FILTER.MAX_VOL_PROP * len(vol_boxes))
            # keep_vol = vol_boxes[keep_ind]
            # valid_inds = vol_boxes > keep_vol # Stage 1
            # pseudo_sem_score = pseudo_sem_score[valid_inds]
            # pseudo_box = pseudo_box[valid_inds]
            # pseudo_label = pseudo_label[valid_inds]
            # pseudo_score = pseudo_score[valid_inds]

            # valid_inds = pseudo_score > self.model_cfg.PSEUDO_TWO_STAGE_FILTER.THRESH # Stage 2
            # pseudo_sem_score = pseudo_sem_score[valid_inds]
            # pseudo_box = pseudo_box[valid_inds]
            # pseudo_label = pseudo_label[valid_inds]
            # pseudo_score = pseudo_score[valid_inds]

            pseudo_boxes.append(torch.cat([pseudo_box, pseudo_label.view(-1, 1).float()], dim=1))
            pseudo_sem_scores.append(pseudo_sem_score)
            pseudo_scores.append(pseudo_score)
            pseudo_scores_var.append(pseudo_score_var)
            pseudo_boxes_var.append(pseudo_box_var)

        return pseudo_boxes, pseudo_scores, pseudo_sem_scores, pseudo_boxes_var, pseudo_scores_var

    def _fill_with_pseudo_labels(self, batch_dict, pseudo_boxes, unlabeled_inds, labeled_inds, key=None):
        key = 'gt_boxes' if key is None else key
        max_box_num = batch_dict['gt_boxes'].shape[1]

        # Ignore the count of pseudo boxes if filled with default values(zeros) when no preds are made
        max_pseudo_box_num = max(
            [torch.logical_not(torch.all(ps_box == 0, dim=-1)).sum().item() for ps_box in pseudo_boxes])

        if max_box_num >= max_pseudo_box_num:
            for i, pseudo_box in enumerate(pseudo_boxes):
                diff = max_box_num - pseudo_box.shape[0]
                if diff > 0:
                    pseudo_box = torch.cat([pseudo_box, torch.zeros((diff, 8), device=pseudo_box.device)], dim=0)
                batch_dict[key][unlabeled_inds[i]] = pseudo_box
        else:
            ori_boxes = batch_dict['gt_boxes']
            new_boxes = torch.zeros((ori_boxes.shape[0], max_pseudo_box_num, ori_boxes.shape[2]),
                                    device=ori_boxes.device)
            for i, inds in enumerate(labeled_inds):
                diff = max_pseudo_box_num - ori_boxes[inds].shape[0]
                new_box = torch.cat([ori_boxes[inds], torch.zeros((diff, 8), device=ori_boxes[inds].device)], dim=0)
                new_boxes[inds] = new_box
            for i, pseudo_box in enumerate(pseudo_boxes):

                diff = max_pseudo_box_num - pseudo_box.shape[0]
                if diff > 0:
                    pseudo_box = torch.cat([pseudo_box, torch.zeros((diff, 8), device=pseudo_box.device)], dim=0)
                new_boxes[unlabeled_inds[i]] = pseudo_box
            batch_dict[key] = new_boxes

    def apply_augmentation(self, batch_dict, batch_dict_org, unlabeled_inds, key='rois'):
        batch_dict[key][unlabeled_inds] = augmentor_utils.random_flip_along_x_bbox(
            batch_dict[key][unlabeled_inds], batch_dict_org['flip_x'][unlabeled_inds])
        batch_dict[key][unlabeled_inds] = augmentor_utils.random_flip_along_y_bbox(
            batch_dict[key][unlabeled_inds], batch_dict_org['flip_y'][unlabeled_inds])
        batch_dict[key][unlabeled_inds] = augmentor_utils.global_rotation_bbox(
            batch_dict[key][unlabeled_inds], batch_dict_org['rot_angle'][unlabeled_inds])
        batch_dict[key][unlabeled_inds] = augmentor_utils.global_scaling_bbox(
            batch_dict[key][unlabeled_inds], batch_dict_org['scale'][unlabeled_inds])

        batch_dict[key][unlabeled_inds, :, 6] = common_utils.limit_period(
            batch_dict[key][unlabeled_inds, :, 6], offset=0.5, period=2 * np.pi
        )

        return batch_dict

    def reverse_augmentation(self, batch_dict, batch_dict_org, unlabeled_inds, key='rois'):
        batch_dict[key][unlabeled_inds] = augmentor_utils.global_scaling_bbox(
            batch_dict[key][unlabeled_inds], 1.0 / batch_dict_org['scale'][unlabeled_inds])
        batch_dict[key][unlabeled_inds] = augmentor_utils.global_rotation_bbox(
            batch_dict[key][unlabeled_inds], - batch_dict_org['rot_angle'][unlabeled_inds])
        batch_dict[key][unlabeled_inds] = augmentor_utils.random_flip_along_y_bbox(
            batch_dict[key][unlabeled_inds], batch_dict_org['flip_y'][unlabeled_inds])
        batch_dict[key][unlabeled_inds] = augmentor_utils.random_flip_along_x_bbox(
            batch_dict[key][unlabeled_inds], batch_dict_org['flip_x'][unlabeled_inds])

        batch_dict[key][unlabeled_inds, :, 6] = common_utils.limit_period(
            batch_dict[key][unlabeled_inds, :, 6], offset=0.5, period=2 * np.pi
        )

        return batch_dict

    def get_supervised_training_loss(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_point, tb_dict = self.point_head.get_loss(tb_dict)
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss_rpn + loss_point + loss_rcnn
        return loss, tb_dict, disp_dict

    def update_global_step(self):
        self.global_step += 1
        self.accumulated_itr += 1
        if self.accumulated_itr % self.model_cfg.EMA_UPDATE_INTERVAL != 0:
            return
        alpha = self.model_cfg.EMA_ALPHA
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (self.global_step + 1), alpha)
        for ema_param, param in zip(self.pv_rcnn_ema.parameters(), self.pv_rcnn.parameters()):
            # TODO(farzad) check this
            ema_param.data.mul_(alpha).add_((1 - alpha) * param.data)
        self.accumulated_itr = 0

    def load_params_from_file(self, filename, logger, to_cpu=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']

        if 'version' in checkpoint:
            logger.info('==> Checkpoint trained from version: %s' % checkpoint['version'])

        update_model_state = {}
        for key, val in model_state_disk.items():
            new_key = 'pv_rcnn.' + key
            if new_key in self.state_dict() and self.state_dict()[new_key].shape == model_state_disk[key].shape:
                update_model_state[new_key] = val
                # logger.info('Update weight %s: %s' % (key, str(val.shape)))
            new_key = 'pv_rcnn_ema.' + key
            if new_key in self.state_dict() and self.state_dict()[new_key].shape == model_state_disk[key].shape:
                update_model_state[new_key] = val
            new_key = key
            if new_key in self.state_dict() and self.state_dict()[new_key].shape == model_state_disk[key].shape:
                update_model_state[new_key] = val

        state_dict = self.state_dict()
        state_dict.update(update_model_state)
        self.load_state_dict(state_dict)

        for key in state_dict:
            if key not in update_model_state:
                logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

        logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(self.state_dict())))