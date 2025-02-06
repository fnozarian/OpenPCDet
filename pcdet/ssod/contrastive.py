import torch
from pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu
from pcdet.utils.loss_utils import DINOLoss
from train_utils.semi_utils import transform_aug, load_data_to_gpu
from tools.visual_utils import open3d_vis_utils as V
from pcdet.models import build_network
import copy
from torch import nn
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from torch.nn import functional as F


class Contrastive(nn.Module):
    def __init__(self, cfgs, datasets, *args, **kwargs):
        super().__init__(*args, **kwargs)

        model_cfgs = copy.deepcopy(cfgs.MODEL)
        self.teacher = build_network(model_cfg=model_cfgs, num_class=len(cfgs.CLASS_NAMES), dataset=datasets['labeled'])
        model_cfgs = copy.deepcopy(cfgs.MODEL)
        self.student = build_network(model_cfg=model_cfgs, num_class=len(cfgs.CLASS_NAMES), dataset=datasets['labeled'])

        if cfgs.OPTIMIZATION.SEMI_SUP_LEARNING.TEACHER.NUM_ITERS_PER_UPDATE == -1:  # for pseudo label
            self.teacher.eval()  # Set to eval mode to avoid BN update and dropout
        else:  # for EMA teacher with consistency
            self.teacher.train()  # Set to train mode
        for t_param in self.teacher.parameters():
            t_param.requires_grad = False

        self.add_module('teacher', self.teacher)
        self.add_module('student', self.student)

        self.conf_thresh = torch.tensor(cfgs.OPTIMIZATION.SEMI_SUP_LEARNING.CONF_THRESH, dtype=torch.float32)
        self.sem_thresh = torch.tensor(cfgs.OPTIMIZATION.SEMI_SUP_LEARNING.SEM_THRESH, dtype=torch.float32)
        self.cat_lbl_ulb = cfgs.OPTIMIZATION.SEMI_SUP_LEARNING.CAT_LBL_ULB
        self.rpn_cls_ulb = cfgs.MODEL.RPN_CLS_LOSS_ULB
        self.rpn_reg_ulb = cfgs.MODEL.RPN_REG_LOSS_ULB
        self.rcnn_cls_ulb = cfgs.MODEL.RCNN_CLS_LOSS_ULB
        self.rcnn_reg_ulb = cfgs.MODEL.RCNN_REG_LOSS_ULB
        self.unlabeled_weight = cfgs.MODEL.UNLABELED_WEIGHT
        self.cfgs = cfgs

        self.cls_weights = torch.tensor(model_cfgs['ROI_HEAD']['CLS_WEIGHT'])

        if cfgs.MODEL.DINO_HEAD.get('ENABLE', False):
            self.dino_loss = DINOLoss(cfgs.MODEL.DINO_HEAD)

        self.mask_gpoint = self.cfgs.MODEL.DINO_HEAD.get('MASK_GPOINTS', False)

    @torch.no_grad()
    def _forward_test_teacher(self, batch_dict):
        # self.pv_rcnn_ema.eval()  # https://github.com/yezhen17/3DIoUMatch-PVRCNN/issues/6
        for cur_module in self.teacher.module_list:
            try:
                batch_dict = cur_module(batch_dict, test_only=True)
            except TypeError as e:
                batch_dict = cur_module(batch_dict)

    def _forward_student(self, batch_dict):
        for cur_module in self.student.module_list:
            batch_dict = cur_module(batch_dict)

    @staticmethod
    def pack_boxes(pseudo_boxes):
        # Ignore the count of pseudo boxes if filled with default values(zeros) when no preds are made
        max_pseudo_box_num = max([torch.logical_not(torch.all(ps_box == 0, dim=-1)).sum().item() for ps_box in pseudo_boxes])
        bs = len(pseudo_boxes)
        dim = pseudo_boxes[0].shape[-1]
        new_boxes = torch.zeros((bs, max_pseudo_box_num, dim), device=pseudo_boxes[0].device)
        for i, pseudo_box in enumerate(pseudo_boxes):
            new_boxes[i, :pseudo_box.shape[0]] = pseudo_box
        return new_boxes

    def mask_dense_rois(self, batch_points, rois, inds=None, num_points=5):
        if inds is None:
            inds = torch.arange(rois.shape[0], device=rois.device)
        ulb_valid_rois_mask = []
        for i, ui in enumerate(inds):
            mask = batch_points[:, 0] == ui
            points = batch_points[mask, 1:4]
            box_idxs = points_in_boxes_gpu(points.unsqueeze(0), rois[i].unsqueeze(0))  # (num_points,)
            box_idxs = box_idxs[box_idxs >= 0]  # remove points that are not in any box
            # Count the number of points in each box
            box_point_counts = torch.bincount(box_idxs, minlength=rois[i].shape[0])
            valid_roi_mask = box_point_counts >= num_points
            ulb_valid_rois_mask.append(valid_roi_mask)
        ulb_valid_rois_mask = torch.vstack(ulb_valid_rois_mask)
        return ulb_valid_rois_mask

    @staticmethod
    def clone_dict(batch_dict, keys, by_ref=False):
        new_dict = {}
        for k in keys:
            if by_ref or not isinstance(batch_dict[k], torch.Tensor):
                new_dict[k] = batch_dict[k]
            else:
                new_dict[k] = batch_dict[k].clone()
        return new_dict

    def get_rois_cls_token(self, batch_dict, rois, model='teacher', full_forward_pass=False, apply_mask=False):
        rois = rois.clone().detach()[..., :7]  # remove the 8th column (cls) if exists
        if full_forward_pass:
            input_keys = ['points', 'voxels', 'voxel_coords', 'voxel_num_points', 'batch_size']
        else:
            input_keys = ["point_features", "point_coords", "batch_size"]
        batch_dict_tmp = self.clone_dict(batch_dict, input_keys, by_ref=True)
        if model == 'teacher':
            if full_forward_pass:
                self._forward_test_teacher(batch_dict_tmp)
            batch_dict_tmp['rois'] = rois  # replace with the transformed rois
            with torch.no_grad():
                gpoint_feats = self.teacher.roi_head.roi_grid_pool(batch_dict_tmp, use_point_cls_score=False)  # (BxN, 6x6x6, C)
                B_N = gpoint_feats.shape[0]
                gpoint_feats = gpoint_feats.permute(0, 2, 1).contiguous().view(B_N, -1, 6, 6, 6)  # (BxN, C, 6, 6, 6)
                shared_features = self.teacher.roi_head.shared_fc_layer(gpoint_feats.view(B_N, -1, 1))
                batch_feats = self.teacher.dino_head.get_cls_token(shared_features)
        else:
            if full_forward_pass:
                batch_dict_tmp['gt_boxes'] = torch.zeros((rois.shape[0], 1, 8), device=rois.device)  # dummy
                self._forward_student(batch_dict_tmp)
            batch_dict_tmp['rois'] = rois  # replace with the transformed rois
            gpoint_feats = self.student.roi_head.roi_grid_pool(batch_dict_tmp, use_point_cls_score=False)  # (BxN, 6x6x6, C)
            B_N = gpoint_feats.shape[0]
            gpoint_feats = gpoint_feats.permute(0, 2, 1).contiguous().view(B_N, -1, 6, 6, 6)  # (BxN, C, 6, 6, 6)

            if apply_mask:
                gpoint_feats = self.student.dino_head.get_masked_feats(gpoint_feats)

            shared_features = self.student.roi_head.shared_fc_layer(gpoint_feats.view(B_N, -1, 1))
            batch_feats = self.student.dino_head.get_cls_token(shared_features)

        return batch_feats  # (BxN, out_dim)

    def forward(self, batch_dict_wa_lbl, batch_dict_wa_ulb, batch_dict_sa_lbl, batch_dict_sa_ulb, epoch_id):
        load_data_to_gpu(batch_dict_wa_ulb)
        load_data_to_gpu(batch_dict_sa_lbl)
        load_data_to_gpu(batch_dict_sa_ulb)

        self._forward_test_teacher(batch_dict_wa_ulb)
        preds_ema, _ = self.teacher.post_processing(batch_dict_wa_ulb, no_recall_dict=True)
        pls = self.filter_pls(preds_ema)
        pls = self.pack_boxes(pls['boxes'])
        # TODO(farzad): transform_aug changes the rotation (7th column) padding rows, making the row non-zero.
        pl_boxes_sa = transform_aug(pls, batch_dict_wa_ulb, batch_dict_sa_ulb)
        batch_dict_sa_ulb['gt_boxes'] = pl_boxes_sa

        self._forward_student(batch_dict_sa_lbl)

        loss = 0
        tb_dict, disp_dict = {}, {}
        loss_lbl, tb_dict_lbl, disp_dict_lbl = self.student.get_training_loss()
        loss += loss_lbl * self.cfgs.MODEL.LABELED_WEIGHT

        for cur_module in self.student.module_list:
            batch_dict_sa_ulb = cur_module(batch_dict_sa_ulb)

        loss_rpn_cls, loss_rpn_reg, tb_dict_rpn_ulb = self.student.dense_head.get_loss(separate_losses=True)
        loss_rcnn_cls, loss_rcnn_reg, tb_dict_rcnn_ulb = self.student.roi_head.get_loss(separate_losses=True)

        if self.rpn_cls_ulb:
            loss += loss_rpn_cls * self.unlabeled_weight
        if self.rpn_reg_ulb:
            loss += loss_rpn_reg * self.unlabeled_weight
        if self.rcnn_cls_ulb:
            loss += loss_rcnn_cls * self.unlabeled_weight
        if self.rcnn_reg_ulb:
            loss += loss_rcnn_reg * self.unlabeled_weight

        # TODO(farzad): Think about the point loss: Note that the point fg/bg predictions
        #  are directly used in roi_grid_pool and thus affect the roi proj features
        # loss_point, tb_dict = self.point_head.get_loss(tb_dict)
        # loss += loss_point

        # TODO: Design decision: use only teacher's pls for both teacher and student? Check the quality of RoIs.
        # TODO: Pool from BEV or point-only or any simpler backbone feature
        if self.cfgs.MODEL.DINO_HEAD.get('ENABLE', False):
            with torch.no_grad():
                if self.cfgs.MODEL.DINO_HEAD.INPUT == 'ROI':
                    sa_rois = batch_dict_sa_ulb['rois']
                    sa_roi_labels = batch_dict_sa_ulb['roi_labels'].long().view(-1) - 1
                    keep_mask = self.mask_dense_rois(batch_dict_sa_ulb['points'], sa_rois).view(-1).float()
                    wa_rois = transform_aug(sa_rois, batch_dict_sa_ulb, batch_dict_wa_ulb)
                elif self.cfgs.MODEL.DINO_HEAD.INPUT == 'PL':
                    sa_rois = batch_dict_sa_ulb['gt_boxes'][..., :7]
                    sa_roi_labels = pls[..., -1].long().view(-1) - 1  # 0-based
                    wa_rois = pls[..., :7]
                    keep_mask = torch.logical_not(torch.eq(wa_rois, 0).all(dim=-1)).view(-1).float()

                cls_weights = self.cls_weights.to(sa_roi_labels.device)[sa_roi_labels].view(-1)
                avg_keep_rois = keep_mask.sum() / sa_rois.shape[0]
                tb_dict.update({'avg_keep_rois': avg_keep_rois.item()})

                t2 = self.get_rois_cls_token(batch_dict_sa_ulb, sa_rois, full_forward_pass=True)
                t1 = self.get_rois_cls_token(batch_dict_wa_ulb, wa_rois)
            s2 = self.get_rois_cls_token(batch_dict_sa_ulb, sa_rois, model='student', apply_mask=self.mask_gpoint)
            s1 = self.get_rois_cls_token(batch_dict_wa_ulb, wa_rois, model='student', full_forward_pass=True, apply_mask=self.mask_gpoint)
            teacher_output = torch.cat([t1, t2], dim=0).view(-1, t1.shape[-1])  # (BxN, C) N=128
            t1_centered, t2_centered = self.dino_loss.softmax_center_teacher(teacher_output).chunk(2)
            self.dino_loss.update_center(teacher_output, keep_mask.repeat(2))
            dino_loss = self.dino_loss.forward(s1, s2, t1_centered, t2_centered, cls_weights, keep_mask)
            tb_dict.update({'dino_loss_unlabeled': dino_loss.item()})
            loss += dino_loss * self.cfgs.MODEL.DINO_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.get('dino_loss_weight', 1.0)

            kl_div = F.kl_div(F.log_softmax(s2 / self.dino_loss.student_temp, dim=-1), t1_centered, reduction='batchmean')
            tb_dict.update({'kl_div_t1_s2': kl_div.mean().item()})
            kl_div2 = F.kl_div(F.log_softmax(s1 / self.dino_loss.student_temp, dim=-1), t2_centered, reduction='batchmean')
            tb_dict.update({'kl_div_t1_s1': kl_div2.mean().item()})
            entropy = -torch.sum(t1_centered * torch.log(t1_centered + 1e-9), dim=-1)
            tb_dict.update({'entropy_t1': entropy.mean().item()})
            entropy2 = -torch.sum(t2_centered * torch.log(t2_centered + 1e-9), dim=-1)
            tb_dict.update({'entropy_t2': entropy2.mean().item()})

            # t1_dist = torch.sum(t1_centered * keep_mask.unsqueeze(-1), dim=0) / keep_mask.sum()
            # s2_dist = torch.sum(torch.softmax(s2 / self.dino_loss.student_temp, dim=-1) * keep_mask.unsqueeze(-1), dim=0) / keep_mask.sum()
            # t1_dist = self.plot_soft_class_distribution(t1_dist.cpu().numpy())
            # s2_dist = self.plot_soft_class_distribution(s2_dist.detach().cpu().numpy())
            # tb_dict.update({'t1_cls_dist_fig': t1_dist})
            # tb_dict.update({'s2_cls_dist_fig': s2_dist})

            # t2_dist = torch.sum(t2_centered * keep_mask.unsqueeze(-1), dim=0) / keep_mask.sum()
            # s1_dist = torch.sum(s1 * keep_mask.unsqueeze(-1), dim=0) / keep_mask.sum()
            # t2_dist = self.plot_soft_class_distribution(t2_dist.cpu().numpy())
            # s1_dist = self.plot_soft_class_distribution(s1_dist.detach().cpu().numpy())
            # tb_dict.update({'t2_cls_dist_fig': t2_dist})
            # tb_dict.update({'s1_cls_dist_fig': s1_dist})

        self.merge_tb_dicts(tb_dict_lbl, tb_dict, 'labeled')
        self.merge_tb_dicts(tb_dict_rpn_ulb, tb_dict, 'unlabeled')
        self.merge_tb_dicts(tb_dict_rcnn_ulb, tb_dict, 'unlabeled')

        return loss, tb_dict, disp_dict

    def plot_soft_class_distribution(self, distribution):
        num_classes = len(distribution)
        indices = np.arange(num_classes)

        # Use Figure directly
        fig = Figure(figsize=(12, 6))
        ax = fig.subplots()
        ax.bar(indices, distribution, color='skyblue', edgecolor='black', linewidth=0.5)

        ax.set_xlabel("Class Index")
        ax.set_ylabel("Probability")
        ax.set_title("Soft Class Distribution")
        ax.set_xticks(indices[::4])  # Show every 4th index
        ax.set_xticklabels(indices[::4], rotation=45)
        ax.set_ylim(0, 1)
        fig.tight_layout()

        return fig

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

    def filter_pls(self, pls_dict):
        pl_boxes, pl_scores, pl_sem_scores, pl_sem_logits = [], [], [], []

        def _fill_with_zeros():
            pl_boxes.append(labels.new_zeros((1, 8)).float())
            pl_scores.append(labels.new_zeros((1,)).float())
            pl_sem_scores.append(labels.new_zeros((1,)).float())
            pl_sem_logits.append(labels.new_zeros((1, 3)).float())

        for pl_dict in pls_dict:
            scores, boxs, labels = pl_dict['pred_scores'], pl_dict['pred_boxes'], pl_dict['pred_labels']
            sem_scores, sem_logits = pl_dict['pred_sem_scores'], pl_dict['pred_sem_logits']
            if len(labels) == 0:
                _fill_with_zeros()
                continue

            assert torch.all(labels == torch.argmax(sem_logits, dim=1) + 1), \
                f"labels: {labels}, sem_scores: {sem_scores}"  # sanity check
            mask = self.iou_match_3d_filtering(scores, sem_logits)

            if mask.sum() == 0:
                _fill_with_zeros()
                continue

            pl_boxes.append(torch.cat([boxs, labels.view(-1, 1).float()], dim=1)[mask])
            pl_scores.append(scores[mask])
            pl_sem_scores.append(sem_scores[mask])
            pl_sem_logits.append(sem_logits[mask])

        return {'boxes': pl_boxes, 'scores': pl_scores, 'sem_scores': pl_sem_scores, 'sem_logits': pl_sem_logits}

    def iou_match_3d_filtering(self, conf_scores, sem_logits):
        labels = torch.argmax(sem_logits, dim=1)
        sem_scores = sem_logits.sigmoid().max(dim=1)[0]
        conf_thresh = self.conf_thresh.to(conf_scores.device).expand_as(sem_logits).gather(dim=1, index=labels.unsqueeze(-1)).squeeze()
        sem_thresh = self.sem_thresh.to(sem_logits.device).expand_as(sem_logits).gather(dim=1, index=labels.unsqueeze(-1)).squeeze()
        return (conf_scores > conf_thresh) & (sem_scores > sem_thresh)
    
    @staticmethod
    def merge_tb_dicts(source_tb_dict, target_tb_dict, postfix=None):
        for key, val in source_tb_dict.items():
            target_tb_dict[f"{key}_{postfix}"] = val
