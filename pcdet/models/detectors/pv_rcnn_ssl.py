import copy
import os
import numpy as np
import torch
import torch.nn.functional as F
from pcdet.datasets.augmentor import augmentor_utils
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from .detector3d_template import Detector3DTemplate
from .pv_rcnn import PVRCNN
from matplotlib import pyplot as plt
from pcdet.utils import common_utils
from pcdet.utils.stats_utils import metrics_registry
from pcdet.utils.prototype_utils import FeatureBankV2
from pcdet.ssod import AdaptiveThresholding
from tools.visual_utils import open3d_vis_utils as V
from pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu


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

        # Overwrite the threshold values from the config/args.
        self.model_cfg.ADAPTIVE_THRESHOLDING['CONF_THRESH'] = self.thresh
        self.model_cfg.ADAPTIVE_THRESHOLDING['SEM_THRESH'] = self.sem_thresh
        self.thresh_alg = AdaptiveThresholding(**self.model_cfg.ADAPTIVE_THRESHOLDING)

        data_sampler = dataset.data_augmentor.data_augmentor_queue[0]
        instance_ids = data_sampler.instance_ids
        self.bank = FeatureBankV2(instance_ids, **model_cfg.FEATURE_BANK)

        self.temperature = self.model_cfg['ROI_HEAD']['INST_CONT_LOSS']['TEMPERATURE']
        self.iou_pos_thresh = self.model_cfg['ROI_HEAD']['INST_CONT_LOSS']['IOU_POS_THRESH']

        for metrics_configs in model_cfg.get("METRICS_BANK_LIST", []):
            if metrics_configs.ENABLE:
                metrics_registry.register(tag=metrics_configs["NAME"], dataset=self.dataset, **metrics_configs)

    @staticmethod
    def _clone_gt_boxes_and_feats(batch_dict):
        return {
            "batch_size": batch_dict['batch_size'],
            "instance_idx": batch_dict['instance_idx'],
            "gt_boxes": batch_dict['gt_boxes'].clone().detach(),
            "point_coords": batch_dict['point_coords'].clone().detach(),
            "point_features": batch_dict['point_features'].clone().detach(),
            "point_cls_scores": batch_dict['point_cls_scores'].clone().detach()
        }

    @torch.no_grad()
    def get_roi_feats_wa(self, batch_dict, chunk=False):
        batch_dict = self._clone_gt_boxes_and_feats(batch_dict)  # TODO(farzad): is cloning (all) required?
        batch_feats = self.pv_rcnn_ema.roi_head.pool_features(batch_dict, use_gtboxes=True, use_projector=True)
        batch_feats = batch_feats.view(*batch_dict['gt_boxes'].shape[:2], -1)
        pad_inds = torch.where(torch.eq(batch_dict['gt_boxes'], 0).all(dim=-1))
        batch_feats[pad_inds] = 0
        batch_labels = batch_dict['gt_boxes'][..., 7].long() - 1
        batch_ins_ids = batch_dict['instance_idx'].long()
        if chunk:
            batch_feats = batch_feats.chunk(2, dim=0)
            batch_labels = batch_labels.chunk(2, dim=0)
            batch_ins_ids = batch_ins_ids.chunk(2, dim=0)
            lbl_chunk = {'feats': batch_feats[0], 'labels': batch_labels[0], 'ins_ids': batch_ins_ids[0]}
            ulb_chunk = {'feats': batch_feats[1], 'labels': batch_labels[1], 'ins_ids': batch_ins_ids[1]}
            return lbl_chunk, ulb_chunk
        out_dict = {'feats': batch_feats, 'labels': batch_labels, 'ins_ids': batch_ins_ids}

        return out_dict

    def get_roi_feats_sa(self, batch_dict, bbox_name='gt_boxes'):
        # TODO(farzad): check if the feats of the padded rois are set to zero.
        if bbox_name == 'gt_boxes':
            roi_feats_sa = self.pv_rcnn.roi_head.pool_features(batch_dict, use_gtboxes=True, use_projector=True)
        else:
            roi_feats_sa = self.pv_rcnn.roi_head.pool_features(batch_dict, use_ori_gtboxes=True, use_projector=True)
        roi_boxes_sa = batch_dict[bbox_name]
        roi_feats_sa = roi_feats_sa.view(*roi_boxes_sa.shape[:2], -1)
        roi_feats_sa_ulb = roi_feats_sa.chunk(2, dim=0)[1]  # get unlabeled feats
        roi_boxes_ulb = roi_boxes_sa.chunk(2, dim=0)[1]
        nonzero_mask = torch.logical_not(torch.eq(roi_boxes_ulb, 0).all(dim=-1))
        roi_feats_sa_ulb = roi_feats_sa_ulb[nonzero_mask]
        labels_ulb = roi_boxes_ulb[..., 7].long() - 1
        labels_ulb = labels_ulb[nonzero_mask]
        return roi_feats_sa_ulb, labels_ulb

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
        batch_dict['ori_gt_boxes'] = batch_dict['gt_boxes'].clone().detach()
        batch_dict['ori_gt_boxes_ema'] = batch_dict['gt_boxes_ema'].clone().detach()
        return labeled_inds, unlabeled_inds

    @staticmethod
    def pad_tensor_dim2(tensor_in, max_len=100):
        if tensor_in.dim() == 1:
            tensor_in = tensor_in.unsqueeze(0).unsqueeze(2)
        if tensor_in.dim() == 2:
            tensor_in = tensor_in.unsqueeze(0)
        if tensor_in.dim() == 3:
            diff = max_len - tensor_in.shape[1]
            if diff > 0:
                zero_pad = torch.zeros((tensor_in.shape[0], diff, tensor_in.shape[-1]), device=tensor_in.device)
                tensor_out = torch.cat([tensor_in, zero_pad], dim=1)
                return tensor_out
        else:
            raise ValueError(f"tensor_in has wrong shape {tensor_in.shape}")

    # This is being used for debugging the loss functions, specially the new ones,
    # to see if they can be minimized to zero or converged to their lowest expected value or not.
    def _get_fixed_batch_dict(self):
        batch_dict_out = {}
        if self.fixed_batch_dict is None:
            return
        for k, v in self.fixed_batch_dict.items():
            if isinstance(v, torch.Tensor):
                batch_dict_out[k] = v.clone().detach()
            else:
                batch_dict_out[k] = copy.deepcopy(v)
        return batch_dict_out

    def _forward_training(self, batch_dict):
        lbl_inds, ulb_inds = self._prep_batch_dict(batch_dict)
        batch_dict_ema = self._split_batch(batch_dict, tag='ema')

        self._gen_pseudo_labels(batch_dict_ema)
        preds_ema, _ = self.pv_rcnn_ema.post_processing(batch_dict_ema, no_recall_dict=True)
        pls = self._filter_pls(preds_ema, ulb_inds)
        self._fill_with_pls(batch_dict, pls['boxes'], pls['masks'], ulb_inds, lbl_inds)

        # Note! Form now on, the `gt_boxes` of `batch_dict_ema` for unlabeled samples are replaced with filtered pls.
        # TODO: find usages of `gt_boxes` in `batch_dict_ema` for unlabeled samples and adapt them with this new change.
        self._fill_with_pls(batch_dict_ema, pls['boxes'], pls['masks'], ulb_inds, lbl_inds)

        # apply student's augs on teacher's pseudo-labels (filtered) only (not points)
        batch_dict = self.apply_augmentation(batch_dict, batch_dict, ulb_inds, key='gt_boxes')

        for cur_module in self.pv_rcnn.module_list:
            batch_dict = cur_module(batch_dict)

        if self.model_cfg['ROI_HEAD'].get('ENABLE_PROTOTYPING', False):
            lbl_roi_feats_wa, _ = self.get_roi_feats_wa(batch_dict_ema, chunk=True)
            self.bank.update(**lbl_roi_feats_wa)

        if self.model_cfg['ROI_HEAD'].get('ENABLE_SOFT_TEACHER', False):
            # using teacher to evaluate student's bg/fg proposals through its rcnn head
            with torch.no_grad():
                self._add_teacher_scores(batch_dict, batch_dict_ema, ulb_inds)

        disp_dict = {}
        loss_rpn_cls, loss_rpn_box, tb_dict = self.pv_rcnn.dense_head.get_loss()
        loss_point, tb_dict = self.pv_rcnn.point_head.get_loss(tb_dict)
        loss_rcnn_cls, loss_rcnn_box, tb_dict = self.pv_rcnn.roi_head.get_loss(tb_dict)

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
            roi_head_forward_dict = self.pv_rcnn.roi_head.forward_ret_dict
            ulb_loss_cls_dist, cls_dist_dict = self.pv_rcnn.roi_head.get_ulb_cls_dist_loss(roi_head_forward_dict)
            loss += ulb_loss_cls_dist
            tb_dict.update(cls_dist_dict)
        if self.model_cfg['ROI_HEAD'].get('ENABLE_PROTO_CONTRASTIVE_LOSS', False):
            roi_feats_sa, roi_labels = self.get_roi_feats_sa(batch_dict)
            proto_cont_loss, pl_sim_logits, bank_labels = self.bank.get_proto_contrastive_loss(roi_feats_sa, roi_labels)
            loss += proto_cont_loss * self.model_cfg['ROI_HEAD']['PROTO_CONTRASTIVE_LOSS_WEIGHT']
            tb_dict['proto_cont_loss'] = proto_cont_loss.item()
            pls['sim_logits'] = pl_sim_logits
        if self.model_cfg['ROI_HEAD'].get('ENABLE_LPCONT_LOSS', False):
            roi_feats_sa, roi_labels = self.get_roi_feats_sa(batch_dict)
            # NOTE: we temporarily use true roi ious between PLs and GTs as weights
            # to find the upper bound performance of the loss and to make it independent of the FG/BG scores
            ulb_gt_boxes = batch_dict['gt_boxes'].chunk(2)[1]
            ulb_ori_gt_boxes = batch_dict['ori_gt_boxes'].chunk(2)[1]
            _, true_ious = self._calc_roi_ious(ulb_gt_boxes, ulb_ori_gt_boxes)
            lpcont_loss, sim_matrix, proto_labels = self.bank.get_lpcont_loss(roi_feats_sa, roi_labels, weights=true_ious)
            if lpcont_loss is not None:
                loss += lpcont_loss * self.model_cfg['ROI_HEAD']['LPCONT_LOSS_WEIGHT']
                tb_dict['lpcont_loss'] = lpcont_loss.item()
                tb_dict['sim_matrix_info'] = self._get_sim_matrix_fig(sim_matrix, roi_labels, proto_labels)
        if self.model_cfg['ROI_HEAD']['INST_CONT_LOSS'].get('ENABLE', False):
            with torch.no_grad():
                ulb_pl_dict_wa = self.get_roi_feats_wa(batch_dict_ema, chunk=True)[1]
                ulb_pl_feats_wa = F.normalize(ulb_pl_dict_wa['feats'], dim=-1)
                keep_pl_idns = torch.eq(ulb_pl_feats_wa, 0).all(dim=-1).logical_not().nonzero(as_tuple=True)
                ulb_pl_feats_wa = ulb_pl_feats_wa[keep_pl_idns]

                ulb_pls_sa = batch_dict['gt_boxes'][..., :7].chunk(2)[1]
                ulb_pls_sa = ulb_pls_sa[keep_pl_idns]
                ulb_rois_sa = self.pv_rcnn.roi_head.forward_ret_dict['rois'].chunk(2)[1]

                # filter out gt instances with too few points when updating the bank
                num_points_thresh = self.model_cfg['ROI_HEAD']['INST_CONT_LOSS'].get('NUM_POINTS_THRESHOLD', 10)
                ulb_valid_rois_mask = []
                for i, ui in enumerate(ulb_inds):
                    mask = batch_dict['points'][:, 0] == ui
                    points = batch_dict['points'][mask, 1:4]
                    box_idxs = points_in_boxes_gpu(points.unsqueeze(0), ulb_rois_sa[i].unsqueeze(0))  # (num_points,)
                    box_idxs = box_idxs[box_idxs >= 0]  # remove points that are not in any box
                    # Count the number of points in each box
                    box_point_counts = torch.bincount(box_idxs, minlength=ulb_rois_sa[i].shape[0])
                    valid_roi_mask = box_point_counts >= num_points_thresh
                    ulb_valid_rois_mask.append(valid_roi_mask)
                ulb_valid_rois_mask = torch.vstack(ulb_valid_rois_mask)

                ulb_rois_sa = ulb_rois_sa[ulb_valid_rois_mask]
                # TODO: add wa feats of ulb_rois_sa with enough points to the bank!
                keep_roi_inds = ulb_valid_rois_mask.nonzero(as_tuple=True)
                ious = iou3d_nms_utils.boxes_iou3d_gpu(ulb_rois_sa, ulb_pls_sa)
                # `mask` sets the iou between rois of different samples to zero.
                smpl_pl_ids = keep_pl_idns[0]
                smpl_roi_ids = keep_roi_inds[0]
                mask = smpl_roi_ids.unsqueeze(1) == smpl_pl_ids.unsqueeze(0)
                ious = ious * mask.float()

            ulb_roi_feats_sa = self.pv_rcnn.roi_head.forward_ret_dict['proj_feats'].chunk(2)[1]
            ulb_roi_feats_sa = ulb_roi_feats_sa[ulb_valid_rois_mask.view(-1)]
            ulb_roi_feats_sa = F.normalize(ulb_roi_feats_sa, dim=-1)
            sim_matrix = ulb_roi_feats_sa @ ulb_pl_feats_wa.T
            logits = sim_matrix / self.temperature
            positive_mask = (ious > self.iou_pos_thresh).float()
            log_prob = F.log_softmax(logits, dim=1)
            log_prob = log_prob * positive_mask
            num_pos = positive_mask.sum(dim=1)
            inst_cont_loss = -log_prob.sum(dim=1) / (num_pos + 1e-8)
            inst_cont_loss = inst_cont_loss.mean() * self.model_cfg['ROI_HEAD']['INST_CONT_LOSS'].get('WEIGHT', 1.0)
            loss += inst_cont_loss
            plt.clf()
            plt.imshow(sim_matrix.detach().cpu().numpy(), cmap='plasma', vmin=0, vmax=1, aspect='auto')
            plt.colorbar()

            if self.model_cfg['ROI_HEAD']['INST_CONT_LOSS'].get('VISUALIZE', False):
                points_sa = batch_dict['points']
                ulb_points_mask = points_sa[:, 0] == ulb_inds[0]  # only visualize the first ulb sample
                ulb_points_sa = points_sa[ulb_points_mask, 1:4]
                ulb_pls_labels = batch_dict['gt_boxes'][..., -1].chunk(2)[1] - 1
                ulb_pls_labels = ulb_pls_labels[keep_pl_idns]
                ulb_roi_labels_sa = self.pv_rcnn.roi_head.forward_ret_dict['roi_labels'].chunk(2)[1] - 1
                ulb_roi_labels_sa = ulb_roi_labels_sa[ulb_valid_rois_mask]
                ulb_roi_scores_sa = torch.zeros(ulb_rois_sa.shape[0])
                attributes = {"id": np.arange(ulb_rois_sa.shape[0]),
                              'positive': (positive_mask.sum(dim=-1) > 0).cpu().numpy()}
                self.vis(ulb_points_sa, ulb_pls_sa.squeeze(), ulb_pls_labels.squeeze(),
                         ulb_rois_sa.squeeze(), ulb_roi_labels_sa.squeeze(), ulb_roi_scores_sa, attributes=attributes)

            tb_dict['inst_cont_loss_unlabeled'] = inst_cont_loss.item()
            tb_dict['sim_matrix_info'] = plt.gcf()

        # update dynamic thresh alg
        if self.model_cfg.ADAPTIVE_THRESHOLDING.ENABLE:
            # Note that the thresholding algorithms use pl information *before* filtering.
            gt_labels = batch_dict['ori_gt_boxes'][..., -1].long() - 1
            self._update_thresh_alg(pls['boxes'], pls['scores'], pls['sem_logits'],  gt_labels, preds_ema, lbl_inds)
            if results := self.thresh_alg.compute():
                tb_dict.update(results)

        # update metrics
        for tag in metrics_registry.tags():
            if tag == 'pl_metrics':
                pl_sim_logits = pls['sim_logits'] if 'sim_logits' in pls.keys() else None
                # Note that the metrics use pl information *after* filtering.
                self._update_pl_metrics(pls['boxes'], pls['rect_scores'], pls['weights'], pls['masks'],
                                        batch_dict_ema['ori_gt_boxes'][ulb_inds], pl_sim_logits=pl_sim_logits)
            if results := metrics_registry.get(tag).compute():
                tb_dict.update({f"{tag}/{k}": v for k, v in zip(*results)})

        if self.model_cfg.get('STORE_SCORES_IN_PKL', False):
            self.dump_statistics(batch_dict, ulb_inds)

        tb_dict = self._split_lbl_ulb_logs(tb_dict, lbl_inds, ulb_inds, reduce_loss_fn)
        ret_dict = {
            'loss': loss
        }

        return ret_dict, tb_dict, disp_dict

    def get_max_iou(self, anchors, gt_boxes, gt_classes, matched_threshold=0.6):
        num_anchors = anchors.shape[0]
        num_gts = gt_boxes.shape[0]

        ious = torch.zeros((num_anchors,), dtype=torch.float, device=anchors.device)
        labels = torch.ones((num_anchors,), dtype=torch.int64, device=anchors.device) * -1
        gt_to_anchor_max = torch.zeros((num_gts,), dtype=torch.float, device=anchors.device)

        if len(gt_boxes) > 0 and anchors.shape[0] > 0:
            anchor_by_gt_overlap = iou3d_nms_utils.boxes_iou3d_gpu(anchors[:, 0:7], gt_boxes[:, 0:7])
            gt_to_anchor_max = anchor_by_gt_overlap.max(dim=0)[0]
            anchor_to_gt_argmax = anchor_by_gt_overlap.argmax(dim=1)
            anchor_to_gt_max = anchor_by_gt_overlap[
                torch.arange(num_anchors, device=anchors.device), anchor_to_gt_argmax]

            pos_inds = anchor_to_gt_max >= matched_threshold
            gt_inds_over_thresh = anchor_to_gt_argmax[pos_inds]
            labels[pos_inds] = gt_classes[gt_inds_over_thresh]
            ious[:len(anchor_to_gt_max)] = anchor_to_gt_max

        return ious, labels, gt_to_anchor_max

    def _get_gt_pls(self, batch_dict, ulb_inds):
        pl_boxes = []
        pl_rect_scores = []
        pl_weights = []
        masks = []
        pl_sem_logits = []
        pl_conf_scores = []
        pl_sem_scores = []
        for i in ulb_inds:
            pboxes = batch_dict['gt_boxes'][i]
            pboxes = pboxes[pboxes[:, -1] != 0]
            pl_boxes.append(pboxes)
            sem_scores = torch.zeros((pboxes.shape[0], 3), device=pboxes.device).scatter_(1, pboxes[:, -1].long().view(-1, 1) - 1, 1)
            sem_logits = torch.zeros((pboxes.shape[0], 3), device=pboxes.device).scatter_(1, pboxes[:, -1].long().view(-1, 1) - 1, 1)
            pl_sem_logits.append(sem_logits)
            pl_rect_scores.append(sem_scores)
            pl_weights.append(torch.ones((pboxes.shape[0], 1), device=pboxes.device))
            pl_conf_scores.append(torch.zeros((pboxes.shape[0], 1), device=pboxes.device))
            pl_sem_scores.append(torch.zeros((pboxes.shape[0], 1), device=pboxes.device))
            masks.append(torch.ones((pboxes.shape[0],), dtype=torch.bool, device=pboxes.device))
        return pl_boxes, pl_conf_scores, pl_sem_scores, pl_sem_logits, pl_rect_scores, masks

    def _calc_roi_ious(self, batch_rois1: torch.Tensor, batch_rois2: torch.Tensor):
        assert batch_rois1.shape[0] == batch_rois2.shape[0]
        ious = torch.zeros(batch_rois1.shape[:2], dtype=torch.float, device=batch_rois1.device)
        for i, (rois1, rois2) in enumerate(zip(batch_rois1, batch_rois2)):
            mask_rois1 = torch.logical_not(torch.all(rois1 == 0, dim=-1))
            mask_rois2 = torch.logical_not(torch.all(rois2 == 0, dim=-1))
            rois1 = rois1[mask_rois1]
            rois2 = rois2[mask_rois2]

            if len(rois2) > 0 and len(rois1) > 0:
                roi1_labels = rois1[:, -1].long() - 1
                roi2_labels = rois2[:, -1].long() - 1
                thresh = torch.tensor([0.7, 0.5, 0.5], device=roi1_labels.device)[roi1_labels]
                roi_ious, _, _ = self.get_max_iou(rois1[:, 0:7], rois2[:, 0:7], roi2_labels, matched_threshold=thresh)
                inds = mask_rois1.nonzero().squeeze()
                ious[i, inds] = roi_ious
        inds = torch.logical_not(torch.all(batch_rois1 == 0, dim=-1)).nonzero(as_tuple=True)
        return ious, ious[inds]

    def _arr2dict(self, array):
        return {cls: array[cind] for cind, cls in enumerate(self.class_names)}

    @staticmethod
    def _get_sim_matrix_fig(sim_matrix, roi_labels, proto_labels):
        sim_matrix = sim_matrix.detach().cpu().numpy()
        roi_labels = roi_labels.cpu().numpy()
        proto_labels = proto_labels.cpu().numpy()
        fig, ax = plt.subplots(figsize=(max(4, len(proto_labels) * 0.2), max(4, len(roi_labels) * 0.2)))
        img = ax.imshow(sim_matrix, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
        x_tick_marks = np.arange(len(proto_labels))
        ax.set_xticks(x_tick_marks)
        ax.set_xticklabels(proto_labels, rotation=45)
        y_tick_marks = np.arange(len(roi_labels))
        ax.set_yticks(y_tick_marks)
        ax.set_yticklabels(roi_labels)
        ax.set_title("Similarity matrix")
        ax.set_ylabel('Unlabeled RoI features')
        ax.set_xlabel('Labeled Bank features')
        plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.1)
        fig.colorbar(img)
        # fig.tight_layout()
        return fig

    @staticmethod
    def _split_lbl_ulb_logs(tb_dict, lbl_inds, ulb_inds, reduce_loss_fn):
        tb_dict_ = {}
        for key in tb_dict.keys():
            if isinstance(tb_dict[key], torch.Tensor) and ('loss' in key or 'acc' in key or 'point_pos_num' in key):
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
    def vis(points, gt_boxes, gt_labels, ref_boxes=None, ref_labels=None, ref_scores=None, attributes=None):
        gt_boxes = gt_boxes.cpu().numpy()
        points = points.cpu().numpy()
        gt_labels = gt_labels.cpu().numpy()
        ref_boxes = ref_boxes.cpu().numpy() if ref_boxes is not None else None
        ref_labels = ref_labels.cpu().numpy() if ref_labels is not None else None
        ref_scores = ref_scores.cpu().numpy() if ref_scores is not None else None
        V.draw_scenes(points=points, gt_boxes=gt_boxes, gt_labels=gt_labels, ref_boxes=ref_boxes,
                      ref_labels=ref_labels, ref_scores=ref_scores, attributes=attributes)

    def _update_thresh_alg(self, pl_boxes, pl_conf_scores, pl_sem_logits, gt_labels, preds_ema, lbl_inds):
        thresh_inputs = dict()

        conf_scores_wa_lbl = [self.pad_tensor_dim2(preds_ema[i]['pred_scores']) for i in lbl_inds]
        sem_logits_wa_lbl = [self.pad_tensor_dim2(preds_ema[i]['pred_sem_logits']) for i in lbl_inds]
        conf_scores_wa_lbl = torch.cat(conf_scores_wa_lbl).detach().clone()
        sem_logits_wa_lbl = torch.cat(sem_logits_wa_lbl).detach().clone()

        conf_scores_wa_ulb = torch.cat([self.pad_tensor_dim2(score) for score in pl_conf_scores]).detach().clone()
        sem_logits_wa_ulb = torch.cat([self.pad_tensor_dim2(score) for score in pl_sem_logits]).detach().clone()

        thresh_inputs['conf_scores_wa'] = torch.cat([conf_scores_wa_lbl, conf_scores_wa_ulb])
        # Note: sem_scores_wa is actually sem_logits_wa
        thresh_inputs['sem_scores_wa'] = torch.cat([sem_logits_wa_lbl, sem_logits_wa_ulb])

        thresh_inputs['gt_labels'] = gt_labels
        thresh_inputs['pls_wa'] = torch.cat([self.pad_tensor_dim2(pl) for pl in pl_boxes]).detach().clone()

        self.thresh_alg.update(**thresh_inputs)

    @staticmethod
    def _update_pl_metrics(pl_boxes, pl_scores, pl_weights, masks, gts, pl_sim_logits=None):
        metrics_input = dict()
        metrics_input['rois'] = [pbox[mask] for pbox, mask in zip(pl_boxes, masks)]
        metrics_input['roi_scores'] = [score[mask] for score, mask in zip(pl_scores, masks)]
        metrics_input['roi_sim_logits'] = pl_sim_logits
        metrics_input['roi_weights'] = [weight[mask] for weight, mask in zip(pl_weights, masks)]
        metrics_input['ground_truths'] = [gtb for gtb in gts]
        metrics_registry.get('pl_metrics').update(**metrics_input)

    def _filter_pls(self, pls_dict, ulb_inds):
        pl_boxes = []
        pl_scores = []
        pl_sem_scores = []
        pl_sem_logits = []
        pl_rect_scores = []
        pl_weights = []
        masks = []

        def _fill_with_zeros():
            pl_boxes.append(labels.new_zeros((1, 8)).float())
            pl_scores.append(labels.new_zeros((1,)).float())
            pl_sem_scores.append(labels.new_zeros((1,)).float())
            pl_sem_logits.append(labels.new_zeros((1, 3)).float())
            pl_rect_scores.append(labels.new_zeros((1, 3)).float())
            pl_weights.append(labels.new_ones((1,)))
            masks.append(labels.new_ones((1,), dtype=torch.bool))

        for ind in ulb_inds:
            scores = pls_dict[ind]['pred_scores']  # Using gt scores for now
            boxs = pls_dict[ind]['pred_boxes']
            labels = pls_dict[ind]['pred_labels']
            sem_scores = pls_dict[ind]['pred_sem_scores']
            sem_logits = pls_dict[ind]['pred_sem_logits']
            if len(labels) == 0:
                _fill_with_zeros()
                continue

            # Uncomment the following two lines to use the true ious as conf scores for the adaptive thresholding
            # pl_bboxes = torch.cat([boxs, labels.view(-1, 1).float()], dim=1)
            # scores = self._calc_true_ious([pl_bboxes], [batch_dict_ema['gt_boxes'][ind]])[0]
            assert torch.all(labels == torch.argmax(sem_logits, dim=1) + 1), f"labels: {labels}, sem_scores: {sem_scores}"  # sanity check
            mask, rect_scores, weights = self.thresh_alg.get_mask(scores, sem_logits)

            if mask.sum() == 0:
                _fill_with_zeros()
                continue

            pl_boxes.append(torch.cat([boxs, labels.view(-1, 1).float()], dim=1))
            pl_scores.append(scores)
            pl_sem_scores.append(sem_scores)
            pl_sem_logits.append(sem_logits)
            pl_rect_scores.append(rect_scores)
            pl_weights.append(weights)
            masks.append(mask)

        pls = {'boxes': pl_boxes, 'scores': pl_scores, 'sem_scores': pl_sem_scores,
               'sem_logits': pl_sem_logits, 'rect_scores': pl_rect_scores, 'masks': masks, 'weights': pl_weights}
        return pls

    @staticmethod
    def _fill_with_pls(batch_dict, pseudo_boxes, masks, ulb_inds, lb_inds, key=None):
        key = 'gt_boxes' if key is None else key
        max_box_num = batch_dict[key].shape[1]
        pseudo_boxes = [pboxes[mask] for pboxes, mask in zip(pseudo_boxes, masks)]

        # Ignore the count of pseudo boxes if filled with default values(zeros) when no preds are made
        max_pseudo_box_num = max(
            [torch.logical_not(torch.all(ps_box == 0, dim=-1)).sum().item() for ps_box in pseudo_boxes])

        if max_box_num >= max_pseudo_box_num:
            for i, pseudo_box in enumerate(pseudo_boxes):
                diff = max_box_num - pseudo_box.shape[0]
                if diff > 0:
                    pseudo_box = torch.cat([pseudo_box, torch.zeros((diff, pseudo_box.shape[-1]), device=pseudo_box.device)], dim=0)
                batch_dict[key][ulb_inds[i]] = pseudo_box
        else:
            ori_boxes = batch_dict['gt_boxes']
            ori_ins_ids = batch_dict['instance_idx']
            new_boxes = torch.zeros((ori_boxes.shape[0], max_pseudo_box_num, ori_boxes.shape[-1]), device=ori_boxes.device)
            new_ins_idx = torch.full((ori_boxes.shape[0], max_pseudo_box_num), fill_value=-1, device=ori_boxes.device)
            for idx in lb_inds:
                diff = max_pseudo_box_num - ori_boxes[idx].shape[0]
                new_box = torch.cat([ori_boxes[idx], torch.zeros((diff, ori_boxes.shape[-1]), device=ori_boxes[idx].device)], dim=0)
                new_boxes[idx] = new_box
                new_ins_idx[idx] = torch.cat([ori_ins_ids[idx], -torch.ones((diff,), device=ori_boxes[idx].device)], dim=0)
            for i, pseudo_box in enumerate(pseudo_boxes):

                diff = max_pseudo_box_num - pseudo_box.shape[0]
                if diff > 0:
                    pseudo_box = torch.cat([pseudo_box, torch.zeros((diff, pseudo_box.shape[-1]), device=pseudo_box.device)], dim=0)
                new_boxes[ulb_inds[i]] = pseudo_box
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