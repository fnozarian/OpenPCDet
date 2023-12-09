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

from pcdet.utils import common_utils
from pcdet.utils.stats_utils import metrics_registry
from pcdet.utils.prototype_utils import feature_bank_registry
from collections import defaultdict
from pcdet.utils.thresh_algs import thresh_registry
#from visual_utils import open3d_vis_utils as V


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
        self.thresh_config = self.model_cfg.ADAPTIVE_THRESH_CONFIG

        self.thresh_alg = None
        self.adapt_thresholding = False
        for key, confs in self.thresh_config.items():
            thresh_registry.register(key, **confs)
            if confs['ENABLE']:
                self.thresh_alg = thresh_registry.get(key)
                self.adapt_thresholding = True

        for bank_configs in model_cfg.get("FEATURE_BANK_LIST", []):
            feature_bank_registry.register(tag=bank_configs["NAME"], **bank_configs)

        for metrics_configs in model_cfg.get("METRICS_BANK_LIST", []):
            metrics_registry.register(tag=metrics_configs["NAME"], dataset=self.dataset, **metrics_configs)

        vals_to_store = ['iou_roi_pl', 'iou_roi_gt', 'pred_scores', 'teacher_pred_scores',
                         'weights', 'roi_scores', 'pcv_scores', 'num_points_in_roi', 'class_labels', 'iteration']
        self.val_dict = {val: [] for val in vals_to_store}

    @staticmethod
    def _clone_gt_boxes_and_feats(batch_dict):
        return {
            "batch_size": batch_dict['batch_size'],
            "gt_boxes": batch_dict['gt_boxes'].clone().detach(),
            "point_coords": batch_dict['point_coords'].clone().detach(),
            "point_features": batch_dict['point_features'].clone().detach(),
            "point_cls_scores": batch_dict['point_cls_scores'].clone().detach()
        }

    def _prep_bank_inputs(self, batch_dict, inds, num_points_threshold=20):
        selected_batch_dict = self._clone_gt_boxes_and_feats(batch_dict)
        with torch.no_grad():
            batch_gt_feats = self.pv_rcnn.roi_head.pool_features(selected_batch_dict, use_gtboxes=True)

        batch_gt_feats = batch_gt_feats.view(*batch_dict['gt_boxes'].shape[:2], -1)
        bank_inputs = defaultdict(list)
        for ix in inds:
            gt_boxes = selected_batch_dict['gt_boxes'][ix]
            nonzero_mask = torch.logical_not(torch.eq(gt_boxes, 0).all(dim=-1))
            if nonzero_mask.sum() == 0:
                print(f"no gt instance in frame {batch_dict['frame_id'][ix]}")
                continue
            gt_boxes = gt_boxes[nonzero_mask]
            sample_mask = batch_dict['points'][:, 0].int() == ix
            points = batch_dict['points'][sample_mask, 1:4]
            gt_feat = batch_gt_feats[ix][nonzero_mask]
            gt_labels = gt_boxes[:, -1].int() - 1
            gt_boxes = gt_boxes[:, :7]
            ins_idxs = batch_dict['instance_idx'][ix][nonzero_mask].int()
            smpl_id = torch.from_numpy(batch_dict['frame_id'].astype(np.int32))[ix].to(gt_boxes.device)

            # filter out gt instances with too few points when updating the bank
            num_points_in_gt = roiaware_pool3d_utils.points_in_boxes_cpu(points.cpu(), gt_boxes.cpu()).sum(dim=-1)
            valid_gts_mask = (num_points_in_gt >= num_points_threshold)
            # print(f"{(~valid_gts_mask).sum()} gt instance(s) with id(s) {ins_idxs[~valid_gts_mask].tolist()}"
            #       f" and num points {num_points_in_gt[~valid_gts_mask].tolist()} are filtered")
            if valid_gts_mask.sum() == 0:
                print(f"no valid gt instances with enough points in frame {batch_dict['frame_id'][ix]}")
                continue
            bank_inputs['feats'].append(gt_feat[valid_gts_mask])
            bank_inputs['labels'].append(gt_labels[valid_gts_mask])
            bank_inputs['ins_ids'].append(ins_idxs[valid_gts_mask])
            bank_inputs['smpl_ids'].append(smpl_id)

            # valid_boxes = gt_boxes[valid_gts_mask]
            # valid_box_labels = gt_labels[valid_gts_mask]
            # self.vis(valid_boxes, valid_box_labels, points)

        return bank_inputs

    def forward(self, batch_dict):
        if self.training:
            return self._forward_training(batch_dict)

        for cur_module in self.pv_rcnn.module_list:
            batch_dict = cur_module(batch_dict)
        pred_dicts, recall_dicts = self.pv_rcnn.post_processing(batch_dict)

        return pred_dicts, recall_dicts, {}
    @torch.no_grad()
    def _gen_pseudo_labels(self, batch_dict_ema):
        # self.pv_rcnn_ema.eval()  # https://github.com/yezhen17/3DIoUMatch-PVRCNN/issues/6
        for cur_module in self.pv_rcnn_ema.module_list:
            try:
                batch_dict_ema = cur_module(batch_dict_ema, test_only=True)
            except TypeError as e:
                batch_dict_ema = cur_module(batch_dict_ema)


    @staticmethod
    def _split_batch(batch_dict, tag='ema'):
        assert tag in ['ema', 'pre_gt_sample'], f'{tag} not in list [ema, pre_gt_sample]'
        batch_dict_out = {}
        keys = list(batch_dict.keys())
        for k in keys:
            if f'{k}_{tag}' in keys:
                continue
            if k.endswith(f'_{tag}'):
                batch_dict_out[k[:-(len(tag)+1)]] = batch_dict[k]
                batch_dict.pop(k)
            if k in ['batch_size']:
                batch_dict_out[k] = batch_dict[k]
        return batch_dict_out

    @staticmethod
    def _prep_batch_dict(batch_dict):
        labeled_mask = batch_dict['labeled_mask'].view(-1)
        labeled_inds = torch.nonzero(labeled_mask).squeeze(1).long()
        unlabeled_inds = torch.nonzero(1 - labeled_mask).squeeze(1).long()
        batch_dict['unlabeled_inds'] = unlabeled_inds
        batch_dict['labeled_inds'] = labeled_inds
        batch_dict['ori_unlabeled_boxes'] = batch_dict['gt_boxes'][unlabeled_inds, ...].clone().detach()
        return labeled_inds, unlabeled_inds

    @staticmethod
    def pad_tensor(tensor_in, max_len=50):
        diff_ = max_len - tensor_in.shape[1]
        if diff_>0:
            tensor_in = torch.cat([tensor_in, torch.zeros((tensor_in.shape[0], diff_, tensor_in.shape[-1]), device=tensor_in.device)], dim=1)
        return tensor_in

    def _forward_training(self, batch_dict):
        lbl_inds, ulb_inds = self._prep_batch_dict(batch_dict)
        batch_dict_ema = self._split_batch(batch_dict, tag='ema')

        self._gen_pseudo_labels(batch_dict_ema)

        pseudo_labels_teacher_wa, _ = self.pv_rcnn_ema.post_processing(batch_dict_ema, no_recall_dict=True)

        ulb_pred_labels = torch.cat([pseudo_labels_teacher_wa[ind]['pred_labels'] for ind in ulb_inds]).int().detach()
        pl_cls_count_pre_filter = torch.bincount(ulb_pred_labels, minlength=4)[1:]

        pseudo_boxes, pseudo_scores, pseudo_sem_scores, pseudo_sem_scores_multi, sem_scores_multi_rect = self._filter_pls(pseudo_labels_teacher_wa, ulb_inds)
        self._fill_with_pseudo_labels(batch_dict, pseudo_boxes, ulb_inds, lbl_inds)

        pl_cls_count_post_filter = torch.bincount(batch_dict['gt_boxes'][ulb_inds][...,-1].view(-1).int().detach(), minlength=4)[1:]
        gt_cls_count = torch.bincount(batch_dict['ori_unlabeled_boxes'][...,-1].view(-1).int().detach(), minlength=4)[1:]

        pl_count_dict = {'avg_num_gts_per_sample': self._arr2dict(gt_cls_count / len(ulb_inds)),
                         'avg_num_pls_pre_filter_per_sample': self._arr2dict(pl_cls_count_pre_filter / len(ulb_inds)),
                         'avg_num_pls_post_filter_per_sample': self._arr2dict(pl_cls_count_post_filter / len(ulb_inds))}

        if sem_scores_multi_rect:
            pl_count_dict['sem_scores_multi_org'] = self._arr2dict(torch.softmax(pseudo_sem_scores_multi / self.thresh_alg.temperature, dim=-1).mean(dim=0).tolist())
            pl_count_dict['sem_scores_multi_rect'] = self._arr2dict(torch.cat(sem_scores_multi_rect, dim=0).mean(dim=0).tolist())

        # apply student's augs on teacher's pseudo-labels (filtered) only (not points)
        batch_dict = self.apply_augmentation(batch_dict, batch_dict, ulb_inds, key='gt_boxes')

        for cur_module in self.pv_rcnn.module_list:
            batch_dict = cur_module(batch_dict)

        if self.adapt_thresholding:
            batch_dict_pre_gt_sample = self._split_batch(batch_dict, tag='pre_gt_sample')
            self._gen_pseudo_labels(batch_dict_pre_gt_sample)

            pseudo_labels_student, _ = self.pv_rcnn_ema.post_processing(batch_dict, no_recall_dict=True)
            pseudo_labels_teacher_pre_gt_sample, _ = self.pv_rcnn_ema.post_processing(batch_dict_pre_gt_sample, no_recall_dict=True)
            metrics_input_ = defaultdict(list)
            for ind in range(len(pseudo_labels_teacher_wa)):
                metrics_input_['conf_scores_wa'].append(self.pad_tensor(pseudo_labels_teacher_wa[ind]['pred_scores'].unsqueeze(0).unsqueeze(2), max_len=100))
                metrics_input_['sem_scores_wa'].append(self.pad_tensor(pseudo_labels_teacher_wa[ind]['pred_sem_scores_multiclass'].unsqueeze(0), max_len=100))
                metrics_input_['conf_scores_pre_gt_wa'].append(self.pad_tensor(pseudo_labels_teacher_pre_gt_sample[ind]['pred_scores'].unsqueeze(0).unsqueeze(2), max_len=100))
                metrics_input_['sem_scores_pre_gt_wa'].append(self.pad_tensor(pseudo_labels_teacher_pre_gt_sample[ind]['pred_sem_scores_multiclass'].unsqueeze(0), max_len=100))
                metrics_input_['conf_scores_sa'].append(self.pad_tensor(pseudo_labels_student[ind]['pred_scores'].unsqueeze(0).unsqueeze(2), max_len=128))
                metrics_input_['sem_scores_sa'].append(self.pad_tensor(pseudo_labels_student[ind]['pred_sem_scores_multiclass'].unsqueeze(0), max_len=128))

            metrics_input = {'gt_labels_wa': self.pad_tensor(batch_dict_ema['gt_boxes'][..., 7:8].detach().clone(), max_len=100),  # (B, 100, 1)
                             'sem_scores_wa': torch.cat(metrics_input_['sem_scores_wa']).detach().clone(),  # (B, 100, 3)
                             'conf_scores_wa': torch.cat(metrics_input_['conf_scores_wa']).detach().clone(), # (B, 100, 1)

                             'gt_labels_pre_gt_wa': self.pad_tensor(batch_dict_pre_gt_sample['gt_boxes'][..., 7:8].detach().clone(), max_len=100),  # (B, 100, 1)
                             'sem_scores_pre_gt_wa': torch.cat(metrics_input_['sem_scores_pre_gt_wa']).detach().clone(),  # (B, 100, 3)
                             'conf_scores_pre_gt_wa': torch.cat(metrics_input_['conf_scores_pre_gt_wa']).detach().clone(), # (B, 100, 1)

                             'gt_labels_sa': self.pad_tensor(batch_dict['gt_boxes'][..., 7:8].detach().clone(), max_len=128),  # (B, 128, 1)
                             'sem_scores_sa': torch.cat(metrics_input_['sem_scores_sa']).detach().clone(),  # (B, 128, 3)
                             'conf_scores_sa': torch.cat(metrics_input_['conf_scores_sa']).detach().clone(), # (B, 128, 1)
                             }
            self.thresh_alg.update(**metrics_input)

        if self.model_cfg['ROI_HEAD'].get('ENABLE_PROTOTYPING', False):
            # Update the bank with student's features from augmented labeled data
            bank = feature_bank_registry.get('gt_aug_lbl_prototypes')
            sa_gt_lbl_inputs = self._prep_bank_inputs(batch_dict, lbl_inds, bank.num_points_thresh)
            bank.update(**sa_gt_lbl_inputs, iteration=batch_dict['cur_iteration'])

        # For metrics calculation
        self.pv_rcnn.roi_head.forward_ret_dict['unlabeled_inds'] = ulb_inds

        if self.model_cfg['ROI_HEAD'].get('ENABLE_SOFT_TEACHER', False):
            # using teacher to evaluate student's bg/fg proposals through its rcnn head
            with torch.no_grad():
                self._add_teacher_scores(batch_dict, batch_dict_ema, ulb_inds)

        disp_dict = {}
        loss_rpn_cls, loss_rpn_box, tb_dict = self.pv_rcnn.dense_head.get_loss(scalar=False)
        loss_point, tb_dict = self.pv_rcnn.point_head.get_loss(tb_dict, scalar=False)
        loss_rcnn_cls, loss_rcnn_box, ulb_loss_cls_dist, tb_dict = self.pv_rcnn.roi_head.get_loss(tb_dict, scalar=False)

        loss = 0
        # Use the same reduction method as the baseline model (3diou) by the default
        reduce_loss_fn = getattr(torch, self.model_cfg.REDUCE_LOSS, 'sum')
        loss += reduce_loss_fn(loss_rpn_cls[lbl_inds, ...])
        loss += reduce_loss_fn(loss_rpn_box[lbl_inds, ...]) + reduce_loss_fn(loss_rpn_box[ulb_inds, ...]) * self.unlabeled_weight
        loss += reduce_loss_fn(loss_point[lbl_inds, ...])
        loss += reduce_loss_fn(loss_rcnn_cls[lbl_inds, ...])
        loss += reduce_loss_fn(loss_rcnn_box[lbl_inds, ...])

        if self.unlabeled_supervise_cls:
            loss += reduce_loss_fn(loss_rpn_cls[ulb_inds, ...]) * self.unlabeled_weight
        if self.model_cfg['ROI_HEAD'].get('ENABLE_SOFT_TEACHER', False) or self.model_cfg.get('UNLABELED_SUPERVISE_OBJ', False):
            loss += reduce_loss_fn(loss_rcnn_cls[ulb_inds, ...]) * self.unlabeled_weight
        if self.unlabeled_supervise_refine:
            loss += reduce_loss_fn(loss_rcnn_box[ulb_inds, ...]) * self.unlabeled_weight
        if self.model_cfg['ROI_HEAD'].get('ENABLE_ULB_CLS_DIST_LOSS', False):
            loss += ulb_loss_cls_dist
        if self.model_cfg['ROI_HEAD'].get('ENABLE_PROTO_CONTRASTIVE_LOSS', False):
            proto_cont_loss = self._get_proto_contrastive_loss(batch_dict, bank, ulb_inds)
            if proto_cont_loss is not None:
                loss += proto_cont_loss * self.model_cfg['ROI_HEAD']['PROTO_CONTRASTIVE_LOSS_WEIGHT']
                tb_dict['proto_cont_loss'] = proto_cont_loss.item()

        tb_dict_ = self._prep_tb_dict(tb_dict, lbl_inds, ulb_inds, reduce_loss_fn)
        tb_dict_.update(**pl_count_dict)
        if self.model_cfg.get('STORE_SCORES_IN_PKL', False):
            self.dump_statistics(batch_dict, ulb_inds)

        if self.model_cfg['ROI_HEAD'].get('ENABLE_PROTOTYPING', False):
            for tag in feature_bank_registry.tags():
                feature_bank_registry.get(tag).compute()

        # update dynamic thresh alg
        if self.adapt_thresholding and (results := self.thresh_alg.compute()):
            tb_dict_.update(results)

        for tag in metrics_registry.tags():
            results = metrics_registry.get(tag).compute()
            if results is not None:
                tb_dict_.update({f"{tag}/{k}": v for k, v in zip(*results)})

        ret_dict = {
            'loss': loss
        }
        return ret_dict, tb_dict_, disp_dict

    def _arr2dict(self, array):
        return {cls: array[cind] for cind, cls in enumerate(self.class_names)}

    def _get_proto_contrastive_loss(self, batch_dict, bank, ulb_inds):
        gt_boxes = batch_dict['gt_boxes']
        B, N = gt_boxes.shape[:2]
        sa_pl_feats = self.pv_rcnn.roi_head.pool_features(batch_dict, use_gtboxes=True).view(B * N, -1)
        pl_labels = batch_dict['gt_boxes'][..., -1].view(-1).long() - 1
        proto_cont_loss = bank.get_proto_contrastive_loss(sa_pl_feats, pl_labels)
        if proto_cont_loss is None:
            return
        nonzero_mask = torch.logical_not(torch.eq(gt_boxes, 0).all(dim=-1))
        ulb_nonzero_mask = nonzero_mask[ulb_inds]
        if ulb_nonzero_mask.sum() == 0:
            print(f"No pl instances predicted for strongly augmented frame(s) {batch_dict['frame_id'][ulb_inds]}")
            return
        return proto_cont_loss.view(B, N)[ulb_inds][ulb_nonzero_mask].mean()

    @staticmethod
    def _prep_tb_dict(tb_dict, lbl_inds, ulb_inds, reduce_loss_fn):
        tb_dict_ = {}
        for key in tb_dict.keys():
            if key == 'proto_cont_loss':
                tb_dict_[key] = tb_dict[key]
            elif 'loss' in key or 'acc' in key or 'point_pos_num' in key:
                tb_dict_[f"{key}_labeled"] = reduce_loss_fn(tb_dict[key][lbl_inds, ...])
                tb_dict_[f"{key}_unlabeled"] = reduce_loss_fn(tb_dict[key][ulb_inds, ...])
            else:
                tb_dict_[key] = tb_dict[key]

        return tb_dict_

    def _add_teacher_scores(self, batch_dict, batch_dict_ema, ulb_inds):
        batch_dict_std = {'unlabeled_inds': batch_dict['unlabeled_inds'],
                          'labeled_inds': batch_dict['labeled_inds'],
                          'rois': batch_dict['rois'].data.clone(),
                          'roi_scores': batch_dict['roi_scores'].data.clone(),
                          'roi_labels': batch_dict['roi_labels'].data.clone(),
                          'has_class_labels': batch_dict['has_class_labels'],
                          'batch_size': batch_dict['batch_size'],
                          # using teacher features
                          'point_features': batch_dict_ema['point_features'].data.clone(),
                          'point_coords': batch_dict_ema['point_coords'].data.clone(),
                          'point_cls_scores': batch_dict_ema['point_cls_scores'].data.clone()
        }

        batch_dict_std = self.reverse_augmentation(batch_dict_std, batch_dict, ulb_inds)

        # Perturb Student's ROIs before using them for Teacher's ROI head
        if self.model_cfg.ROI_HEAD.ROI_AUG.get('ENABLE', False):
            augment_rois = getattr(augmentor_utils, self.model_cfg.ROI_HEAD.ROI_AUG.AUG_TYPE, augmentor_utils.roi_aug_ros)
            # rois_before_aug is used only for debugging, can be removed later
            batch_dict_std['rois_before_aug'] = batch_dict_std['rois'].clone().detach()
            batch_dict_std['rois'][ulb_inds] = augment_rois(batch_dict_std['rois'][ulb_inds], self.model_cfg.ROI_HEAD)

        self.pv_rcnn_ema.roi_head.forward(batch_dict_std, test_only=True)
        batch_dict_std = self.apply_augmentation(batch_dict_std, batch_dict, ulb_inds, key='batch_box_preds')
        pred_dicts_std, recall_dicts_std = self.pv_rcnn_ema.post_processing(batch_dict_std,
                                                                            no_recall_dict=True,
                                                                            no_nms_for_unlabeled=True)
        rcnn_cls_score_teacher = -torch.ones_like(self.pv_rcnn.roi_head.forward_ret_dict['rcnn_cls_labels'])
        batch_box_preds_teacher = torch.zeros_like(self.pv_rcnn.roi_head.forward_ret_dict['batch_box_preds'])
        for uind in ulb_inds:
            rcnn_cls_score_teacher[uind] = pred_dicts_std[uind]['pred_scores']
            batch_box_preds_teacher[uind] = pred_dicts_std[uind]['pred_boxes']

        self.pv_rcnn.roi_head.forward_ret_dict['rcnn_cls_score_teacher'] = rcnn_cls_score_teacher
        self.pv_rcnn.roi_head.forward_ret_dict['batch_box_preds_teacher'] = batch_box_preds_teacher # for metrics

    @staticmethod
    def vis(boxes, box_labels, points):
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

    @staticmethod
    def _unpack_preds(pred_dicts, ulb_inds):
        pseudo_boxes = []
        pseudo_scores = []
        pseudo_sem_scores = []
        pseudo_sem_scores_multi = []
        pseudo_labels = []
        pseudo_pre_nms_thresh_masks = []
        for ind in ulb_inds:
            pseudo_score = pred_dicts[ind]['pred_scores']
            pseudo_box = pred_dicts[ind]['pred_boxes']
            pseudo_label = pred_dicts[ind]['pred_labels']
            pseudo_sem_score = pred_dicts[ind]['pred_sem_scores']
            pseudo_sem_score_multi = pred_dicts[ind]['pred_sem_scores_multiclass']
            pseudo_masks = pred_dicts[ind]['pred_thresh_masks'] if 'pred_thresh_masks' in pred_dicts[ind] else None
            if len(pseudo_label) == 0:
                pseudo_boxes.append(pseudo_label.new_zeros((1, 7)).float())
                pseudo_sem_scores.append(pseudo_label.new_zeros((1,)).float())
                pseudo_sem_scores_multi.append(pseudo_label.new_zeros((1, 3)).float())
                pseudo_pre_nms_thresh_masks.append(pseudo_label.new_zeros((1,)).float())
                pseudo_scores.append(pseudo_label.new_zeros((1,)).float())
                pseudo_labels.append(pseudo_label.new_zeros((1,)).float())
                continue

            pseudo_boxes.append(pseudo_box)
            pseudo_sem_scores.append(pseudo_sem_score)
            pseudo_scores.append(pseudo_score)
            pseudo_labels.append(pseudo_label)
            pseudo_sem_scores_multi.append(pseudo_sem_score_multi)
            pseudo_pre_nms_thresh_masks.append(pseudo_masks)

        return pseudo_boxes, pseudo_labels, pseudo_scores, pseudo_sem_scores, pseudo_sem_scores_multi, pseudo_pre_nms_thresh_masks

    def _filter_pls(self, pls_dict, ulb_inds):
        pseudo_boxes = []
        pseudo_scores = []
        pseudo_sem_scores = []
        sem_scores_multi = []
        sem_scores_multi_rect = []
        for boxs, labels, scores, sem_scores, sem_scores_multi, pseudo_pre_nms_thresh_masks in zip(*self._unpack_preds(pls_dict, ulb_inds)):

            if labels[0] == 0:
                pseudo_boxes.append(torch.cat([boxs, labels.view(-1, 1).float()], dim=1))
                pseudo_sem_scores.append(sem_scores)
                pseudo_scores.append(scores)
                continue

            conf_thresh = torch.tensor(self.thresh, device=labels.device).unsqueeze(
                0).repeat(len(labels), 1).gather(dim=1, index=(labels - 1).unsqueeze(-1))

            reliable_mask = scores > conf_thresh.squeeze()

            if self.adapt_thresholding and self.thresh_alg.iteration_count > 0:
                # apply dynamic thresholding
                sem_score_mask, rect_scores = self.thresh_alg.get_mask(sem_scores_multi)
                if self.thresh_alg.thresh_method == 'AdaMatch':
                    sem_scores_multi_rect.append(rect_scores)
            else:
                sem_conf_thresh = torch.tensor(self.sem_thresh, device=labels.device).unsqueeze(
                    0).repeat(len(labels), 1).gather(dim=1, index=(labels - 1).unsqueeze(-1))
                sem_score_mask = sem_scores > sem_conf_thresh.squeeze()

            reliable_mask = torch.logical_and(reliable_mask, sem_score_mask)

            boxs = boxs[reliable_mask]
            labels = labels[reliable_mask]
            scores = scores[reliable_mask]
            sem_scores = sem_scores[reliable_mask]

            pseudo_boxes.append(torch.cat([boxs, labels.view(-1, 1).float()], dim=1))
            pseudo_sem_scores.append(sem_scores)
            pseudo_scores.append(scores)

        return pseudo_boxes, pseudo_scores, pseudo_sem_scores, sem_scores_multi, sem_scores_multi_rect

    @staticmethod
    def _fill_with_pseudo_labels(batch_dict, pseudo_boxes, unlabeled_inds, labeled_inds, key=None):
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
            ori_ins_ids = batch_dict['instance_idx']
            new_boxes = torch.zeros((ori_boxes.shape[0], max_pseudo_box_num, ori_boxes.shape[2]),
                                    device=ori_boxes.device)
            new_ins_idx = torch.full((ori_boxes.shape[0], max_pseudo_box_num), fill_value=-1, device=ori_boxes.device)
            for idx in labeled_inds:
                diff = max_pseudo_box_num - ori_boxes[idx].shape[0]
                new_box = torch.cat([ori_boxes[idx], torch.zeros((diff, 8), device=ori_boxes[idx].device)], dim=0)
                new_boxes[idx] = new_box
                new_ins_idx[idx] = torch.cat([ori_ins_ids[idx], -torch.ones((diff,), device=ori_boxes[idx].device)], dim=0)
            for i, pseudo_box in enumerate(pseudo_boxes):

                diff = max_pseudo_box_num - pseudo_box.shape[0]
                if diff > 0:
                    pseudo_box = torch.cat([pseudo_box, torch.zeros((diff, 8), device=pseudo_box.device)], dim=0)
                new_boxes[unlabeled_inds[i]] = pseudo_box
            batch_dict[key] = new_boxes
            batch_dict['instance_idx'] = new_ins_idx

    @staticmethod
    def apply_augmentation(batch_dict, batch_dict_org, unlabeled_inds, key='rois'):
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

    @staticmethod
    def reverse_augmentation(batch_dict, batch_dict_org, unlabeled_inds, key='rois'):
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