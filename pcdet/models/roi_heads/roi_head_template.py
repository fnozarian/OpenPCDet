import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils import box_coder_utils, common_utils, loss_utils
from ..model_utils.model_nms_utils import class_agnostic_nms
from .target_assigner.proposal_target_layer import ProposalTargetLayer
from pcdet.utils.stats_utils import metrics_registry
from ...ops.iou3d_nms import iou3d_nms_utils


from collections import defaultdict
from pcdet.config import cfg # temporarily adding cfg for temprature scaling

def get_roi_metrics_input(targets_dict, mask_type='cls'):
    metrics_input = defaultdict(list)

    for i, uind in enumerate(targets_dict['unlabeled_inds']):
        mask = (targets_dict['reg_valid_mask'][uind] > 0) if mask_type == 'reg' else (
                    targets_dict['rcnn_cls_labels'][uind] >= 0)
        if mask.sum() == 0:
            # print(f'Warning: No {mask_type} rois for unlabeled index {uind}')
            continue

        # (Proposals) ROI info
        rois = targets_dict['rois'][uind][mask].detach().clone()
        roi_labels = targets_dict['roi_labels'][uind][mask].unsqueeze(-1).detach().clone()
        temperature = cfg.MODEL.ADAPTIVE_THRESHOLDING.TEMPERATURE_SA
        roi_scores_multiclass = torch.softmax(targets_dict['roi_scores_logits'][uind][mask].detach().clone() / temperature, dim=-1)
        # roi_sim_scores_multiclass = targets_dict['roi_sim_scores'][uind][mask].detach().clone()
        roi_labeled_boxes = torch.cat([rois, roi_labels], dim=-1)
        metrics_input['rois'].append(roi_labeled_boxes)
        metrics_input['roi_scores'].append(roi_scores_multiclass)
        metrics_input['roi_weights'].append(torch.ones_like(roi_labels))
        # metrics_input['roi_sim_scores'].append(roi_sim_scores_multiclass)
        # gt_iou_of_rois = targets_dict['gt_iou_of_rois'][uind][mask].unsqueeze(-1).detach().clone()
        # metrics_input['roi_iou_wrt_pl'].append(gt_iou_of_rois)

        # (Real labels) GT info
        gt_labeled_boxes = targets_dict['ori_unlabeled_boxes'][i]
        metrics_input['ground_truths'].append(gt_labeled_boxes)

        # RoI weights
        # target_weights = targets_dict['rcnn_cls_weights'][uind][mask].detach().clone()
        # metrics_input['roi_weights'].append(target_weights)

        # bs_id = targets_dict['points'][:, 0] == uind
        # points = targets_dict['points'][bs_id, 1:].detach().clone()
        # metrics_input['points'].append(points)
    if len(metrics_input['rois']) == 0:
        # print(f'Warning: No {mask_type} rois for any unlabeled index')
        return
    return metrics_input


class RoIHeadTemplate(nn.Module):
    def __init__(self, num_class, model_cfg, predict_boxes_when_training=True):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.box_coder = getattr(box_coder_utils, self.model_cfg.TARGET_CONFIG.BOX_CODER)(
            **self.model_cfg.TARGET_CONFIG.get('BOX_CODER_CONFIG', {})
        )
        self.proposal_target_layer = ProposalTargetLayer(roi_sampler_cfg=self.model_cfg.TARGET_CONFIG)
        self.build_losses(self.model_cfg.LOSS_CONFIG)
        self.forward_ret_dict = None
        self.mean_p_model_sa = None
        self.mean_p_cls = None
        self.ulb_cls_dist = None
        self.mean_cls_labels = None
        self.mean_p_cls_shadow = None
        self.ulb_cls_dist_shadow = None
        self.mean_cls_labels_shadow = None
        self.avg_num_fgs_per_sample = 4.5
        self.iteration_count = 0
        self.target_dist = torch.tensor([0.82, 0.13, 0.05])
        self.momentum = 0.99
        self.predict_boxes_when_training = predict_boxes_when_training

    def build_losses(self, losses_cfg):
        self.add_module(
            'reg_loss_func',
            loss_utils.WeightedSmoothL1Loss(code_weights=losses_cfg.LOSS_WEIGHTS['code_weights'])
        )

    def make_fc_layers(self, input_channels, output_channels, fc_list):
        fc_layers = []
        pre_channel = input_channels
        for k in range(0, fc_list.__len__()):
            fc_layers.extend([
                nn.Conv1d(pre_channel, fc_list[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(fc_list[k]),
                nn.ReLU()
            ])
            pre_channel = fc_list[k]
            if self.model_cfg.DP_RATIO >= 0 and k == 0:
                fc_layers.append(nn.Dropout(self.model_cfg.DP_RATIO))
        fc_layers.append(nn.Conv1d(pre_channel, output_channels, kernel_size=1, bias=True))
        fc_layers = nn.Sequential(*fc_layers)
        return fc_layers

    @torch.no_grad()
    def proposal_layer(self, batch_dict,  nms_config):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
            nms_config:

        Returns:
            batch_dict:
                rois: (B, num_rois, 7+C)
                roi_scores: (B, num_rois)
                roi_labels: (B, num_rois)

        """
        if batch_dict.get('rois', None) is not None:
            return batch_dict
        batch_size = batch_dict['batch_size']
        batch_box_preds = batch_dict['batch_box_preds']
        batch_cls_preds = batch_dict['batch_cls_preds']
        rois = batch_box_preds.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE, batch_box_preds.shape[-1]))
        roi_scores = batch_box_preds.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE))
        roi_labels = batch_box_preds.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE), dtype=torch.long)
        roi_scores_logits = batch_box_preds.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE, batch_cls_preds.shape[-1]))
        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_cls_preds.shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_cls_preds'].shape.__len__() == 3
                batch_mask = index
            box_preds = batch_box_preds[batch_mask]
            cls_preds = batch_cls_preds[batch_mask]

            cur_roi_scores, cur_roi_labels = torch.max(cls_preds, dim=1)
            if nms_config.MULTI_CLASSES_NMS:
                raise NotImplementedError
            else:
                selected, selected_scores = class_agnostic_nms(
                    box_scores=cur_roi_scores, box_preds=box_preds, nms_config=nms_config
                )

            rois[index, :len(selected), :] = box_preds[selected]
            roi_scores[index, :len(selected)] = cur_roi_scores[selected]
            roi_labels[index, :len(selected)] = cur_roi_labels[selected]
            roi_scores_logits[index, :len(selected), :] = cls_preds[selected]
        batch_dict['rois'] = rois
        batch_dict['roi_scores'] = roi_scores
        batch_dict['roi_scores_logits'] = roi_scores_logits
        batch_dict['roi_labels'] = roi_labels + 1
        batch_dict['has_class_labels'] = batch_cls_preds.shape[-1] > 1
        batch_dict.pop('batch_index', None)
        return batch_dict
    def assign_targets(self, batch_dict):

        with torch.no_grad():
            targets_dict = self.proposal_target_layer.forward(batch_dict)

        batch_size = batch_dict['batch_size']

        # Adding points temporarily to the targets_dict for visualization inside update_metrics
        targets_dict['points'] = batch_dict['points']
        rois = targets_dict['rois']  # (B, N, 7 + C)
        gt_of_rois = targets_dict['gt_of_rois']  # (B, N, 7 + C + 1)
        targets_dict['gt_of_rois_src'] = gt_of_rois.clone().detach()

        # canonical transformation
        roi_center = rois[:, :, 0:3]
        roi_ry = rois[:, :, 6] % (2 * np.pi)
        gt_of_rois[:, :, 0:3] = gt_of_rois[:, :, 0:3] - roi_center
        gt_of_rois[:, :, 6] = gt_of_rois[:, :, 6] - roi_ry

        # transfer LiDAR coords to local coords
        gt_of_rois = common_utils.rotate_points_along_z(
            points=gt_of_rois.view(-1, 1, gt_of_rois.shape[-1]), angle=-roi_ry.view(-1)
        ).view(batch_size, -1, gt_of_rois.shape[-1])

        # flip orientation if rois have opposite orientation
        heading_label = gt_of_rois[:, :, 6] % (2 * np.pi)  # 0 ~ 2pi
        opposite_flag = (heading_label > np.pi * 0.5) & (heading_label < np.pi * 1.5)
        heading_label[opposite_flag] = (heading_label[opposite_flag] + np.pi) % (2 * np.pi)  # (0 ~ pi/2, 3pi/2 ~ 2pi)
        flag = heading_label > np.pi
        heading_label[flag] = heading_label[flag] - np.pi * 2  # (-pi/2, pi/2)
        heading_label = torch.clamp(heading_label, min=-np.pi / 2, max=np.pi / 2)

        gt_of_rois[:, :, 6] = heading_label
        targets_dict['gt_of_rois'] = gt_of_rois
        return targets_dict

    def get_ulb_cls_dist_loss(self, forward_ret_dict):
        # ulb_num_cls = torch.bincount(ulb_sem_preds.view(-1), minlength=4)[1:].float()
        # ulb_cls_dist = ulb_num_cls / ulb_num_cls.sum()
        ulb_cls_dist_weight = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS.get('ulb_cls_dist_weight', 0.2)
        mse_loss_weight = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS.get('ulb_fg_mse_loss_weight', 0.2)
        batch_size = forward_ret_dict['rcnn_cls_labels'].shape[0]
        ulb_inds = forward_ret_dict['unlabeled_inds']
        ulb_roi_labels = forward_ret_dict['roi_labels'][ulb_inds].detach() - 1
        ulb_sem_logits = forward_ret_dict['roi_scores_logits'][ulb_inds]
        # ulb_sem_scores = torch.softmax(ulb_sem_logits / 4, dim=-1)

        rcnn_cls_labels = forward_ret_dict['rcnn_cls_labels']
        ulb_conf_logits = forward_ret_dict['rcnn_cls'].reshape(batch_size, -1)[ulb_inds]
        ulb_conf_preds_scores = torch.sigmoid(ulb_conf_logits)

        # ulb_sem_conf_scores = ulb_sem_scores * ulb_conf_preds_scores.unsqueeze(-1).expand_as(ulb_sem_scores)

        ulb_conf_dist = torch.zeros((3,), device=ulb_sem_logits.device).scatter_add_(0, ulb_roi_labels.view(-1), ulb_conf_preds_scores.view(-1))
        ulb_conf_dist = ulb_conf_dist / ulb_conf_dist.sum()

        num_ulb_samples = batch_size // 2
        avg_num_ulb_fgs_pred = ulb_conf_preds_scores.sum() / num_ulb_samples
        avg_num_ulb_fgs_true = torch.tensor(num_ulb_samples * self.avg_num_fgs_per_sample, dtype=torch.float32, device=avg_num_ulb_fgs_pred.device)

        target_dist = self.target_dist.to(ulb_conf_dist.device)
        # The following loss equals to F.kl_div(torch.log(ulb_conf_dist), target_dist)
        ulb_cls_dist_loss = torch.mean(target_dist * torch.log((target_dist + 1e-6) / (ulb_conf_dist + 1e-6)))
        ulb_cls_dist_loss = ulb_cls_dist_loss * ulb_cls_dist_weight

        ulb_fg_per_sample_loss = nn.functional.mse_loss(avg_num_ulb_fgs_pred, avg_num_ulb_fgs_true)
        ulb_fg_per_sample_loss = ulb_fg_per_sample_loss * mse_loss_weight

        loss = ulb_cls_dist_loss + ulb_fg_per_sample_loss
        # ulb_cls_dist_loss = torch.clamp(ulb_cls_dist_loss, min=0.0, max=2.0)
        # ulb_cls_dist_loss = torch.tensor(0.0).to(ulb_sem_logits.device)
        tb_dict = {
            'cls_dist_loss_unlabeled': ulb_cls_dist_loss.item(),
            'ulb_fg_per_sample_loss': ulb_fg_per_sample_loss.item(),
            'ulb_conf_dist': self._arr2dict(ulb_conf_dist.detach().cpu().numpy()),
            'avg_num_ulb_fgs_pred': avg_num_ulb_fgs_pred.item(),
        }
        return loss, tb_dict

    def _arr2dict(self, array):
        if array.shape[-1] == 2:
            return {cls: array[cind] for cind, cls in enumerate(['Bg', 'Fg'])}
        elif array.shape[-1] == 3:
            return {cls: array[cind] for cind, cls in enumerate(['Car', 'Ped', 'Cyc'])}
        else:
            raise ValueError(f"Invalid array shape: {array.shape}")

    def get_box_reg_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        code_size = self.box_coder.code_size

        reg_valid_mask = forward_ret_dict['reg_valid_mask'].view(-1)
        gt_boxes3d_ct = forward_ret_dict['gt_of_rois'][..., 0:code_size]
        gt_of_rois_src = forward_ret_dict['gt_of_rois_src'][..., 0:code_size].view(-1, code_size)
        # gt_of_rois_loss_weight = forward_ret_dict['gt_of_rois'][..., -1].view(-1)
        rcnn_reg = forward_ret_dict['rcnn_reg']  # (rcnn_batch_size, C)
        roi_boxes3d = forward_ret_dict['rois']
        rcnn_batch_size = gt_boxes3d_ct.view(-1, code_size).shape[0]
        batch_size = forward_ret_dict['reg_valid_mask'].shape[0]

        fg_mask = (reg_valid_mask > 0)
        fg_sum = fg_mask.long().sum().item()

        tb_dict = {}

        if loss_cfgs.REG_LOSS == 'smooth-l1':
            rois_anchor = roi_boxes3d.clone().detach().view(-1, code_size)
            rois_anchor[:, 0:3] = 0
            rois_anchor[:, 6] = 0
            reg_targets = self.box_coder.encode_torch(
                gt_boxes3d_ct.view(rcnn_batch_size, code_size), rois_anchor
            )

            rcnn_loss_reg = self.reg_loss_func(
                rcnn_reg.view(rcnn_batch_size, -1).unsqueeze(dim=0),
                reg_targets.unsqueeze(dim=0),
            )  # [B, M, 7]

            fg_sum_ = fg_mask.reshape(batch_size, -1).long().sum(-1)
            # assert gt_of_rois_loss_weight[:rcnn_batch_size // 2].sum() == rcnn_batch_size // 2
            rcnn_loss_reg = (rcnn_loss_reg.view(rcnn_batch_size, -1) *
                             # gt_of_rois_loss_weight.view(rcnn_batch_size, -1) *
                             fg_mask.unsqueeze(dim=-1).float()
                             ).reshape(batch_size, -1).sum(-1) / torch.clamp(fg_sum_.float(), min=1.0)
            rcnn_loss_reg = rcnn_loss_reg * loss_cfgs.LOSS_WEIGHTS['rcnn_reg_weight']
            tb_dict['rcnn_loss_reg'] = rcnn_loss_reg

            if loss_cfgs.CORNER_LOSS_REGULARIZATION and fg_sum > 0:
                split_size = []
                fg_mask_batch = fg_mask.reshape(batch_size, -1)
                for i in range(batch_size):
                    split_size.append(len(torch.nonzero(fg_mask_batch[i])))
                # TODO: need further check
                fg_rcnn_reg = rcnn_reg.view(rcnn_batch_size, -1)[fg_mask]
                fg_roi_boxes3d = roi_boxes3d.view(-1, code_size)[fg_mask]

                fg_roi_boxes3d = fg_roi_boxes3d.view(1, -1, code_size)
                batch_anchors = fg_roi_boxes3d.clone().detach()
                roi_ry = fg_roi_boxes3d[:, :, 6].view(-1)
                roi_xyz = fg_roi_boxes3d[:, :, 0:3].view(-1, 3)
                batch_anchors[:, :, 0:3] = 0
                rcnn_boxes3d = self.box_coder.decode_torch(
                    fg_rcnn_reg.view(batch_anchors.shape[0], -1, code_size), batch_anchors
                ).view(-1, code_size)

                rcnn_boxes3d = common_utils.rotate_points_along_z(
                    rcnn_boxes3d.unsqueeze(dim=1), roi_ry
                ).squeeze(dim=1)
                rcnn_boxes3d[:, 0:3] += roi_xyz

                loss_corner = loss_utils.get_corner_loss_lidar(
                    rcnn_boxes3d[:, 0:7],
                    gt_of_rois_src[fg_mask][:, 0:7]
                )

                loss_corner = torch.split(loss_corner, split_size, dim=0)
                zero = torch.zeros([1], device=fg_mask.device)
                loss_corner = [x.mean(dim=0, keepdim=True) if len(x) > 0 else zero for x in loss_corner]
                loss_corner = torch.cat(loss_corner, dim=0)
                loss_corner = loss_corner * loss_cfgs.LOSS_WEIGHTS['rcnn_corner_weight']

                rcnn_loss_reg += loss_corner
                tb_dict['rcnn_loss_corner'] = loss_corner
        else:
            raise NotImplementedError

        return rcnn_loss_reg, tb_dict

    def get_box_cls_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        rcnn_cls = forward_ret_dict['rcnn_cls']
        rcnn_cls_labels = forward_ret_dict['rcnn_cls_labels'].view(-1)
        if loss_cfgs.CLS_LOSS == 'BinaryCrossEntropy':
            rcnn_cls_flat = rcnn_cls.view(-1)
            batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat), rcnn_cls_labels.float(), reduction='none')
            cls_valid_mask = (rcnn_cls_labels >= 0).float()
            batch_size = forward_ret_dict['rcnn_cls_labels'].shape[0]
            batch_loss_cls = batch_loss_cls.reshape(batch_size, -1)
            cls_valid_mask = cls_valid_mask.reshape(batch_size, -1)
            if 'rcnn_cls_weights' in forward_ret_dict:
                rcnn_cls_weights = forward_ret_dict['rcnn_cls_weights']
                rcnn_loss_cls_norm = (cls_valid_mask * rcnn_cls_weights).sum(-1)
                rcnn_loss_cls = (batch_loss_cls * cls_valid_mask * rcnn_cls_weights).sum(-1) / torch.clamp(rcnn_loss_cls_norm, min=1.0)
            else:
                rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum(-1) / torch.clamp(cls_valid_mask.sum(-1), min=1.0)
        elif loss_cfgs.CLS_LOSS == 'CrossEntropy':
            batch_loss_cls = F.cross_entropy(rcnn_cls, rcnn_cls_labels, reduction='none', ignore_index=-1)
            cls_valid_mask = (rcnn_cls_labels >= 0).float()
            batch_size = forward_ret_dict['rcnn_cls_labels'].shape[0]
            batch_loss_cls = batch_loss_cls.reshape(batch_size, -1)
            cls_valid_mask = cls_valid_mask.reshape(batch_size, -1)
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum(-1) / torch.clamp(cls_valid_mask.sum(-1), min=1.0)
        elif loss_cfgs.CLS_LOSS == 'UnbiasedCrossEntropy':
            self.iteration_count += 1
            tau = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS.get('unbiased_ce_tau', 1.0)
            batch_size = forward_ret_dict['rcnn_cls_labels'].shape[0]
            roi_labels = forward_ret_dict['roi_labels']

            rcnn_cls_labels = forward_ret_dict['rcnn_cls_labels']
            rcnn_cls_logits = forward_ret_dict['rcnn_cls'].view(batch_size, -1)
            cls_valid_mask = (rcnn_cls_labels >= 0).float()
            assert cls_valid_mask.sum() == cls_valid_mask.numel(), "All the labels should be valid if valid mask is not used"

            roi_labels = forward_ret_dict['roi_labels']
            ulb_roi_labels = roi_labels.chunk(2)[1].long() - 1
            ulb_label_hist = torch.bincount(ulb_roi_labels.view(-1), minlength=3)
            ulb_rcnn_cls_labels = forward_ret_dict['rcnn_cls_labels'].chunk(2)[1]
            ulb_sum_cls_labels = roi_labels.new_zeros((3,), dtype=torch.float).scatter_add_(0, ulb_roi_labels.view(-1), ulb_rcnn_cls_labels.view(-1))
            ulb_rcnn_cls_dist = ulb_sum_cls_labels / torch.clamp(ulb_label_hist, min=1.0)

            self._ema_update_p('mean_p_cls', ulb_rcnn_cls_dist)

            # ulb_cls_dist = ulb_sum_cls_labels / ulb_sum_cls_labels.sum()
            # self._ema_update_p('ulb_cls_dist', ulb_cls_dist)
            # self.target_dist = self.target_dist.to(ulb_cls_dist.device)
            # divergence_offsets = self.target_dist * torch.log(ulb_cls_dist / self.target_dist)
            # ulb_logit_offsets = divergence_offsets.unsqueeze(0).repeat(ulb_roi_labels.shape[0], 1).gather(1, ulb_roi_labels)
            # ulb_logit_offsets = ulb_logit_offsets * (1 - ulb_rcnn_cls_labels)

            # average values when using fully supervised setting: torch.log(torch.tensor([0.2, 0.1, 0.1]))
            # tensor([-1.6094, -2.3026, -2.3026])
            # torch.log(torch.tensor([0.2, 0.1, 1e-6])) with epsilon 1e-6 if the value is 0
            # tensor([ -1.6094,  -2.3026, -13.8155])
            log_target = torch.log(torch.clamp(self.mean_p_cls, min=1e-6)).to(ulb_roi_labels.device)
            ulb_logit_offsets = log_target.unsqueeze(0).repeat(ulb_roi_labels.shape[0], 1).gather(1, ulb_roi_labels)
            ulb_logit_offsets = ulb_logit_offsets * ulb_rcnn_cls_labels
            logit_offsets = torch.cat([torch.zeros_like(ulb_logit_offsets), ulb_logit_offsets])
            unbiased_logits = rcnn_cls_logits + tau * logit_offsets
            batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(unbiased_logits).view(-1), rcnn_cls_labels.view(-1), reduction='none')
            rcnn_loss_cls = (batch_loss_cls.unsqueeze(-1).view_as(cls_valid_mask) * cls_valid_mask).sum(-1) / torch.clamp(cls_valid_mask.sum(-1), min=1.0)
        else:
            raise NotImplementedError

        rcnn_loss_cls = rcnn_loss_cls * loss_cfgs.LOSS_WEIGHTS['rcnn_cls_weight']
        tb_dict = {
            'rcnn_loss_cls': rcnn_loss_cls
        }

        if loss_cfgs.CLS_LOSS == 'UnbiasedCrossEntropy':
          tb_dict.update({'rcnn_fg_dist_unlabeled': self._arr2dict(self.mean_p_cls.detach().cpu().numpy())})
          tb_dict.update({'rcnn_cls_dist_unlabeled': self._arr2dict(self.ulb_cls_dist.detach().cpu().numpy())})

        return rcnn_loss_cls, tb_dict

    def _ema_update_p(self, prob_name, probs):
        probs_shadow = getattr(self, prob_name + '_shadow')
        if probs_shadow is None:
            probs_shadow = torch.zeros_like(probs)
        probs_shadow = self.momentum * probs_shadow + (1 - self.momentum) * probs
        new_probs = probs_shadow / (1 - self.momentum ** self.iteration_count)
        setattr(self, prob_name + '_shadow', probs_shadow)
        setattr(self, prob_name, new_probs)

    def _get_reliability_weight(self, unlabeled_inds):
        # Initialize reliability weights with 1s
        self.forward_ret_dict['rcnn_cls_weights'] = torch.ones_like(self.forward_ret_dict['rcnn_cls_labels'])

        # Compute the background scores from the teacher based on student proposals
        rcnn_bg_score_teacher = 1 - self.forward_ret_dict['rcnn_cls_score_teacher']
        unlabeled_rcnn_cls_weights = self.forward_ret_dict['rcnn_cls_weights'][unlabeled_inds]
        ul_interval_mask = self.forward_ret_dict['interval_mask'][unlabeled_inds]
        ulb_fg_mask = self.forward_ret_dict['rcnn_cls_labels'][unlabeled_inds] == 1
        ulb_bg_mask = self.forward_ret_dict['rcnn_cls_labels'][unlabeled_inds] == 0

        if self.model_cfg['LOSS_CONFIG']['UL_RCNN_CLS_WEIGHT_TYPE'] == 'all':
            unlabeled_rcnn_cls_weights[ul_interval_mask] = rcnn_bg_score_teacher[unlabeled_inds][ul_interval_mask]
        # Use Teacher's FG scores instead of BG scores for UCs (its the reverse of "all")
        # (assuming, we have more FPs > FNs in UC)
        elif self.model_cfg['LOSS_CONFIG']['UL_RCNN_CLS_WEIGHT_TYPE'] == 'rev_uc':
            unlabeled_rcnn_cls_weights[ul_interval_mask] = self.forward_ret_dict['rcnn_cls_score_teacher'][unlabeled_inds][ul_interval_mask]

        elif self.model_cfg['LOSS_CONFIG']['UL_RCNN_CLS_WEIGHT_TYPE'] == 'interval-only':
            unlabeled_rcnn_cls_weights = torch.zeros_like(self.forward_ret_dict['rcnn_cls_labels'][unlabeled_inds])
            unlabeled_rcnn_cls_weights[ul_interval_mask] = rcnn_bg_score_teacher[unlabeled_inds][ul_interval_mask]
        elif self.model_cfg['LOSS_CONFIG']['UL_RCNN_CLS_WEIGHT_TYPE'] == 'bg':
            ulb_bg_mask = self.forward_ret_dict['rcnn_cls_labels'][unlabeled_inds] == 0
            unlabeled_rcnn_cls_weights[ulb_bg_mask] = rcnn_bg_score_teacher[unlabeled_inds][ulb_bg_mask]
        elif self.model_cfg['LOSS_CONFIG']['UL_RCNN_CLS_WEIGHT_TYPE'] == 'fg':
            unlabeled_rcnn_cls_weights[ulb_fg_mask] = self.forward_ret_dict['rcnn_cls_score_teacher'][unlabeled_inds][ulb_fg_mask]

        elif self.model_cfg['LOSS_CONFIG']['UL_RCNN_CLS_WEIGHT_TYPE'] == 'ignore_interval':  # Naive baseline
            unlabeled_rcnn_cls_weights[ul_interval_mask] = 0
        elif self.model_cfg['LOSS_CONFIG']['UL_RCNN_CLS_WEIGHT_TYPE'] == 'ignore-bg':  # Naive baseline
            unlabeled_rcnn_cls_weights[ulb_bg_mask] = 0
        elif self.model_cfg['LOSS_CONFIG']['UL_RCNN_CLS_WEIGHT_TYPE'] == 'full-ema':
            unlabeled_rcnn_cls_weights = rcnn_bg_score_teacher[unlabeled_inds]

        # Use 1s for FG mask, teacher's BG scores for UC+BG mask
        elif self.model_cfg['LOSS_CONFIG']['UL_RCNN_CLS_WEIGHT_TYPE'] == 'uc-bg':
            ulb_fg_mask = self.forward_ret_dict['rcnn_cls_labels'][unlabeled_inds] == 1
            unlabeled_rcnn_cls_weights[~ulb_fg_mask] = rcnn_bg_score_teacher[unlabeled_inds][~ulb_fg_mask]
        # Use 1s for FG mask, suppress false positives from UC, false negatives from BG regions
        elif self.model_cfg['LOSS_CONFIG']['UL_RCNN_CLS_WEIGHT_TYPE'] == 'rev_uc-bg':
            unlabeled_rcnn_cls_weights[ul_interval_mask] = self.forward_ret_dict['rcnn_cls_score_teacher'][unlabeled_inds][ul_interval_mask]
            ulb_bg_mask = self.forward_ret_dict['rcnn_cls_labels'][unlabeled_inds] == 0
            unlabeled_rcnn_cls_weights[ulb_bg_mask] = rcnn_bg_score_teacher[unlabeled_inds][ulb_bg_mask]

        # Use teacher's FG scores for FG mask, teacher's BG scores for UC+BG mask
        elif self.model_cfg['LOSS_CONFIG']['UL_RCNN_CLS_WEIGHT_TYPE'] == 'fg-uc-bg':
            ulb_fg_mask = self.forward_ret_dict['rcnn_cls_labels'][unlabeled_inds] == 1
            unlabeled_rcnn_cls_weights[ulb_fg_mask] = self.forward_ret_dict['rcnn_cls_score_teacher'][unlabeled_inds][ulb_fg_mask]
            unlabeled_rcnn_cls_weights[~ulb_fg_mask] = rcnn_bg_score_teacher[unlabeled_inds][~ulb_fg_mask]
        # Use teacher's FG scores for FG+UC mask, teacher's BG scores for BG mask
        elif self.model_cfg['LOSS_CONFIG']['UL_RCNN_CLS_WEIGHT_TYPE'] == 'fg-rev_uc-bg':
            unlabeled_rcnn_cls_weights[~ulb_bg_mask] = self.forward_ret_dict['rcnn_cls_score_teacher'][unlabeled_inds][~ulb_bg_mask]
            unlabeled_rcnn_cls_weights[ulb_bg_mask] = rcnn_bg_score_teacher[unlabeled_inds][ulb_bg_mask]

        # Use 1s for UC, FG scores for FG, BG scores for BG
        elif self.model_cfg['LOSS_CONFIG']['UL_RCNN_CLS_WEIGHT_TYPE'] == 'fg-bg':
            unlabeled_rcnn_cls_weights[ulb_fg_mask] = self.forward_ret_dict['rcnn_cls_score_teacher'][unlabeled_inds][ulb_fg_mask]
            unlabeled_rcnn_cls_weights[ulb_bg_mask] = rcnn_bg_score_teacher[unlabeled_inds][ulb_bg_mask]

        # Use cos scores for UC, 1s for FG and BG
        elif self.model_cfg['LOSS_CONFIG']['UL_RCNN_CLS_WEIGHT_TYPE'] == 'cos-uc':
            unlabeled_rcnn_cls_weights[ul_interval_mask] = self.forward_ret_dict['cos_scores'][unlabeled_inds][ul_interval_mask]

        # Use cos scores for FG,UC, 1 - cos_scores for BG
        elif self.model_cfg['LOSS_CONFIG']['UL_RCNN_CLS_WEIGHT_TYPE'] == 'cos-uc-bg':
            unlabeled_rcnn_cls_weights[ul_interval_mask] = self.forward_ret_dict['cos_scores'][unlabeled_inds][ul_interval_mask]
            unlabeled_rcnn_cls_weights[ulb_bg_mask] = 1 - self.forward_ret_dict['cos_scores'][unlabeled_inds][ulb_bg_mask]
            unlabeled_rcnn_cls_weights[ulb_fg_mask] = torch.clamp(self.forward_ret_dict['cos_scores'][unlabeled_inds][ulb_fg_mask]+0.22,max=1.0)

        # Use cos scores for FG, 1-cos scores for UC and BG
        elif self.model_cfg['LOSS_CONFIG']['UL_RCNN_CLS_WEIGHT_TYPE'] == 'cos-rev-uc-bg':
            unlabeled_rcnn_cls_weights[ul_interval_mask] = 1 - self.forward_ret_dict['cos_scores'][unlabeled_inds][ul_interval_mask]
            unlabeled_rcnn_cls_weights[ulb_bg_mask] = 1 - self.forward_ret_dict['cos_scores'][unlabeled_inds][ulb_bg_mask]
            unlabeled_rcnn_cls_weights[ulb_fg_mask] = torch.clamp((self.forward_ret_dict['cos_scores'][unlabeled_inds][ulb_fg_mask]+0.22),max=1.0)

        else:
            raise ValueError

        return unlabeled_rcnn_cls_weights

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict

        if self.model_cfg.ENABLE_SOFT_TEACHER and not self.model_cfg.TARGET_CONFIG.DISABLE_ST_WEIGHTS:
            # Get reliability weights for unlabeled samples
            unlabeled_inds = self.forward_ret_dict['unlabeled_inds']
            self.forward_ret_dict['rcnn_cls_weights'][unlabeled_inds] = self._get_reliability_weight(unlabeled_inds)

        if 'rcnn_roi_metrics' in metrics_registry.tags():
            # update_metrics(self.forward_ret_dict, mask_type='reg')
            metrics_input = get_roi_metrics_input(self.forward_ret_dict, mask_type='cls')
            metrics_registry.get('rcnn_roi_metrics').update(**metrics_input)

        rcnn_loss_cls, cls_tb_dict = self.get_box_cls_layer_loss(self.forward_ret_dict)
        rcnn_loss_reg, reg_tb_dict = self.get_box_reg_layer_loss(self.forward_ret_dict)
        rcnn_loss = rcnn_loss_cls + rcnn_loss_reg
        tb_dict.update(cls_tb_dict)
        tb_dict.update(reg_tb_dict)
        tb_dict['rcnn_loss'] = rcnn_loss

        return rcnn_loss_cls, rcnn_loss_reg, tb_dict

    def generate_predicted_boxes(self, batch_size, rois, cls_preds, box_preds):
        """
        Args:
            batch_size:
            rois: (B, N, 7)
            cls_preds: (BN, num_class)
            box_preds: (BN, code_size)

        Returns:

        """
        code_size = self.box_coder.code_size
        # batch_cls_preds: (B, N, num_class or 1)
        batch_cls_preds = cls_preds.view(batch_size, -1, cls_preds.shape[-1])
        batch_box_preds = box_preds.view(batch_size, -1, code_size)

        roi_ry = rois[:, :, 6].view(-1)
        roi_xyz = rois[:, :, 0:3].view(-1, 3)
        local_rois = rois.clone().detach()
        local_rois[:, :, 0:3] = 0

        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, local_rois).view(-1, code_size)

        batch_box_preds = common_utils.rotate_points_along_z(
            batch_box_preds.unsqueeze(dim=1), roi_ry
        ).squeeze(dim=1)
        batch_box_preds[:, 0:3] += roi_xyz
        batch_box_preds = batch_box_preds.view(batch_size, -1, code_size)
        return batch_cls_preds, batch_box_preds
