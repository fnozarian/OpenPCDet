import copy
import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from pcdet.datasets.augmentor import augmentor_utils
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from .detector3d_template import Detector3DTemplate
from .pv_rcnn import PVRCNN
import matplotlib.pyplot as plt
from pcdet.utils import common_utils
from pcdet.utils.stats_utils import metrics_registry
from pcdet.utils.prototype_utils import feature_bank_registry
# from tools.visual_utils import open3d_vis_utils as V
from collections import defaultdict
from pcdet.utils.weighting_methods import build_thresholding_method

class DynamicThreshRegistry(object):
    def __init__(self, **kwargs):
        self._tag_metrics = {}
        self.dataset = kwargs.get('dataset', None)
        self.model_cfg = kwargs.get('model_cfg', None)

    def get(self, tag=None):
        if tag is None:
            tag = 'default'
        if tag in self._tag_metrics.keys():
            metric = self._tag_metrics[tag]
        else:
            metric = build_thresholding_method(tag=tag, dataset=self.dataset, config=self.model_cfg)
            self._tag_metrics[tag] = metric
        return metric

    def tags(self):
        return self._tag_metrics.keys()


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
        self.thresh_registry = DynamicThreshRegistry(dataset=self.dataset, model_cfg=model_cfg)
        for bank_configs in model_cfg.get("FEATURE_BANK_LIST", []):
            feature_bank_registry.register(tag=bank_configs["NAME"], **bank_configs)

        for metrics_configs in model_cfg.get("METRICS_BANK_LIST", []):
            metrics_registry.register(tag=metrics_configs["NAME"], dataset=self.dataset, **metrics_configs)

        vals_to_store = ['iou_roi_pl', 'iou_roi_gt', 'pred_scores', 'teacher_pred_scores',
                         'weights', 'roi_scores', 'num_points_in_roi', 'class_labels', 'iteration','lbl_inst_freq', 'positive_pairs_duped', 'negative_pairs_duped',
                         'unscaled_instloss_car','unscaled_instloss_ped','unscaled_instloss_cyc','instloss_car','instloss_ped','instloss_cyc']

        self.val_dict = {val: [] for val in vals_to_store}
        self.val_dict['lbl_inst_freq'] = [0,0,0]
        self.val_dict['positive_pairs_duped'] = [0,0,0]
        self.val_dict['negative_pairs_duped'] = [1,1,1]


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

    def _rectify_pl_scores(self, batch_dict_ema, unlabeled_inds):
        thresh_reg = self.thresh_registry.get(tag='pl_adaptive_thresh')
        pred_wa = torch.sigmoid(batch_dict_ema['batch_cls_preds']).detach().clone()
        # to be used later for updating the EMA (p_model/p_target)
        pred_weak_aug_before_nms_org = pred_wa.clone()
        if thresh_reg.iteration_count > 0:
            pred_wa_ulb = pred_wa[unlabeled_inds, ...]
            pred_wa_ulb_aligned = pred_wa_ulb * thresh_reg.ema_pred_wa_lab / (thresh_reg.ema_pred_wa_ulb + 1e-6)
            pred_wa_ulb_aligned = thresh_reg.normalize_(pred_wa_ulb_aligned)
            pred_wa[unlabeled_inds, ...] = pred_wa_ulb_aligned

        batch_dict_ema['batch_cls_preds_org'] = pred_weak_aug_before_nms_org
        batch_dict_ema['batch_cls_preds'] = pred_wa
        batch_dict_ema['cls_preds_normalized'] = True

    def _gen_pseudo_labels(self, batch_dict_ema, ulb_inds,lbl_inds):
        with torch.no_grad():
            # self.pv_rcnn_ema.eval()  # https://github.com/yezhen17/3DIoUMatch-PVRCNN/issues/6
            for cur_module in self.pv_rcnn_ema.module_list:
                try:
                    batch_dict_ema = cur_module(batch_dict_ema, test_only=True)
                except TypeError as e:
                    batch_dict_ema = cur_module(batch_dict_ema)

        if self.model_cfg.ROI_HEAD.ADAPTIVE_THRESH_CONFIG.get('ENABLE', False):
            self._rectify_pl_scores(batch_dict_ema, ulb_inds)

        pseudo_labels, _ = self.pv_rcnn_ema.post_processing(batch_dict_ema, no_recall_dict=True)

        return batch_dict_ema,pseudo_labels 

    @staticmethod
    def _split_ema_batch(batch_dict):
        batch_dict_ema = {}
        keys = list(batch_dict.keys())
        for k in keys:
            if f'{k}_ema' in keys:
                continue
            if k.endswith('_ema'):
                batch_dict_ema[k[:-4]] = batch_dict[k]
            else:
                batch_dict_ema[k] = batch_dict[k]
        return batch_dict_ema

    @staticmethod
    def _prep_batch_dict(batch_dict):
        labeled_mask = batch_dict['labeled_mask'].view(-1)
        labeled_inds = torch.nonzero(labeled_mask).squeeze(1).long()
        unlabeled_inds = torch.nonzero(1 - labeled_mask).squeeze(1).long()
        batch_dict['unlabeled_inds'] = unlabeled_inds
        batch_dict['labeled_inds'] = labeled_inds
        batch_dict['ori_unlabeled_boxes'] = batch_dict['gt_boxes'][unlabeled_inds, ...].clone().detach()
        return labeled_inds, unlabeled_inds

    def _forward_training(self, batch_dict):
        lbl_inds, ulb_inds = self._prep_batch_dict(batch_dict)
        batch_dict_ema = self._split_ema_batch(batch_dict)

        batch_dict_ema, pseudo_labels = self._gen_pseudo_labels(batch_dict_ema, ulb_inds,lbl_inds)
        pseudo_boxes, pseudo_scores, pseudo_sem_scores = self._filter_pseudo_labels(pseudo_labels, ulb_inds)

        self._fill_with_pseudo_labels(batch_dict, pseudo_boxes, ulb_inds, lbl_inds)

        # apply student's augs on teacher's pseudo-labels (filtered) only (not points)
        batch_dict = self.apply_augmentation(batch_dict, batch_dict, ulb_inds, key='gt_boxes')

        for cur_module in self.pv_rcnn.module_list:
            batch_dict = cur_module(batch_dict)


        if self.model_cfg.ROI_HEAD.ADAPTIVE_THRESH_CONFIG.get('ENABLE', False):
            pred_strong_aug_before_nms_org = torch.sigmoid(batch_dict['batch_cls_preds']).detach().clone()
            pred_dicts_std, recall_dicts_std = self.pv_rcnn_ema.post_processing(batch_dict, no_recall_dict=True)

            metrics_input = defaultdict(list)
            for ind in range(len(pred_dicts_std)):
                batch_type = 'unlab' if ind in ulb_inds else 'lab'
                metrics_input[f'pred_weak_aug_{batch_type}_before_nms'].append(batch_dict_ema['batch_cls_preds_org'][ind])
                metrics_input[f'pred_weak_aug_{batch_type}_after_nms'].append(pseudo_labels[ind]['pred_scores'].clone().detach())
                metrics_input[f'pred_strong_aug_{batch_type}_before_nms'].append(pred_strong_aug_before_nms_org[ind])
                metrics_input[f'pred_strong_aug_{batch_type}_after_nms'].append(pred_dicts_std[ind]['pred_scores'].clone().detach())
            self.thresh_registry.get(tag='pl_adaptive_thresh').update(**metrics_input)

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
        if self.model_cfg['ROI_HEAD'].get('ENABLE_INSTANCE_SUP_LOSS', False):
            assert 'projected_features_gt' in batch_dict.keys()
            assert 'projected_features_gt' in batch_dict_ema.keys()
            lbl_inst_cont_loss, tb_dict = self._get_instance_contrastive_loss(tb_dict,batch_dict,batch_dict_ema,lbl_inds,ulb_inds)
            if lbl_inst_cont_loss is not None:
                loss +=  lbl_inst_cont_loss * self.model_cfg['ROI_HEAD']['INSTANCE_CONTRASTIVE_LOSS_WEIGHT']
        tb_dict_ = self._prep_tb_dict(tb_dict, lbl_inds, lbl_inds, reduce_loss_fn)

        if self.model_cfg.get('STORE_SCORES_IN_PKL', False):
            self.dump_statistics(batch_dict, ulb_inds,lbl_inds)

        for tag in feature_bank_registry.tags():
            feature_bank_registry.get(tag).compute()

        # # update dynamic thresh results
        # for tag in self.thresh_registry.tags():
        #     if results := self.thresh_registry.get(tag).compute():
        #         tag = f"{tag}/" if tag else ''
        #         tb_dict_.update({tag + key: val for key, val in results.items()})

        for tag in metrics_registry.tags():
            results = metrics_registry.get(tag).compute()
            if results is not None:
                tb_dict_.update({f"{tag}/{k}": v for k, v in zip(*results)})

        ret_dict = {
            'loss': loss
        }
        return ret_dict, tb_dict_, disp_dict


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


    def _sort_instance_pairs(self, batch_dict, batch_dict_wa, indices):
        
        embed_size = batch_dict['shared_features_gt'].squeeze().shape[-1]
        shared_ft_sa = batch_dict['shared_features_gt'].view(batch_dict['batch_size'],-1,embed_size)         # shared_ft_sa = batch_dict['projected_features_gt'].view(batch_dict['batch_size'],-1,embed_size)[indices]

        shared_ft_sa = shared_ft_sa[indices].view(-1,256)
        shared_ft_wa = batch_dict_wa['shared_features_gt'].view(batch_dict['batch_size'],-1,embed_size)         # shared_ft_wa = batch_dict_wa['projected_features_gt'].view(batch_dict['batch_size'],-1,embed_size)[indices]
        shared_ft_wa = shared_ft_wa[indices].view(-1,256)
        labels_sa = batch_dict['gt_boxes'][indices][:,:,-1].view(-1)
        labels_wa = batch_dict_wa['gt_boxes'][indices][:,:,-1].view(-1)
        instance_idx_sa = batch_dict['instance_idx'][indices].view(-1)
        instance_idx_wa = batch_dict_wa['instance_idx'][indices].view(-1)
        nonzero_mask_sa = torch.logical_not(torch.eq(instance_idx_sa, 0))
        nonzero_mask_wa = torch.logical_not(torch.eq(instance_idx_wa, 0))

        # Strip off zero masking
        instance_idx_sa = instance_idx_sa.masked_select(nonzero_mask_sa)
        labels_sa = labels_sa.masked_select(nonzero_mask_sa)
        assert instance_idx_sa.size(0)==labels_sa.size(0)
        instance_idx_wa = instance_idx_wa.masked_select(nonzero_mask_wa)
        labels_wa = labels_wa.masked_select(nonzero_mask_wa)
        assert instance_idx_wa.size(0)==labels_wa.size(0)
        shared_ft_sa = shared_ft_sa.masked_select(nonzero_mask_sa.unsqueeze(-1).expand(-1,embed_size))
        shared_ft_sa = shared_ft_sa.view(-1,embed_size)
        shared_ft_wa = shared_ft_wa.masked_select(nonzero_mask_wa.unsqueeze(-1).expand(-1,embed_size))
        shared_ft_wa = shared_ft_wa.view(-1,embed_size)

        # Finds correspondences of common instances between SA and WA, return corresponding labels and features

        common_instnaces, ids_sa_common, ids_wa_common = np.intersect1d(instance_idx_sa.cpu().numpy(),instance_idx_wa.cpu().numpy(), return_indices = True) 
        valid_instances = torch.from_numpy(common_instnaces).to(labels_sa.device)
        ids_sa_common = torch.from_numpy(ids_sa_common).to(labels_sa.device)
        ids_wa_common = torch.from_numpy(ids_wa_common).to(labels_sa.device)

        instance_idx_sa_common = valid_instances.clone()
        instance_idx_wa_common = valid_instances.clone()

        labels_sa_common = torch.index_select(labels_sa, 0,ids_sa_common)
        labels_wa_common = torch.index_select(labels_wa, 0, ids_wa_common)
        assert torch.equal(labels_sa_common, labels_wa_common)

        shared_ft_sa_common = shared_ft_sa[ids_sa_common]
        shared_ft_wa_common = shared_ft_wa[ids_wa_common]
        assert shared_ft_sa_common.size(0) == labels_wa_common.size(0)

        return  (labels_sa_common, labels_wa_common, shared_ft_sa_common, shared_ft_wa_common)


    def _get_instance_contrastive_loss(self, tb_dict, batch_dict, batch_dict_wa, lbl_inds, temperature=1.0, base_temperature=1.0):
        '''
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: roi_labels[B,N].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        '''
        start_epoch = self.model_cfg['ROI_HEAD'].get('INSTANCE_CONTRASTIVE_LOSS_START_EPOCH', 0)
        stop_epoch = self.model_cfg['ROI_HEAD'].get('INSTANCE_CONTRASTIVE_LOSS_STOP_EPOCH', 60)
        tb_dict = {} if tb_dict is None else tb_dict
        # To examine effects of stopping supervised contrastive loss
        if not start_epoch<=batch_dict['cur_epoch']<stop_epoch:
            return None, None
        temperature = self.model_cfg['ROI_HEAD'].get('TEMPERATURE', 1.0)
        instance_idx_sa = batch_dict['instance_idx'].view(-1)
        instance_idx_wa = batch_dict_wa['instance_idx'].view(-1)
        
        sorted_pairs = self._sort_instance_pairs(batch_dict, batch_dict_wa, lbl_inds)
        
        if sorted_pairs is None:
            return None, tb_dict

        labels_sa, labels_wa, embed_ft_sa, embed_ft_wa = sorted_pairs
        batch_size_labeled = labels_sa.shape[0]
        device = embed_ft_sa.device
        labels = torch.cat((labels_sa,labels_wa), dim=0)

        '''# Record stats of num_pairs per batch'''
        # batch_dict['lbl_inst_freq'] =  torch.bincount(labels_sa.view(-1).int().detach(),minlength = 4).tolist()[1:]       #Record freq of each class in batch
        # batch_dict['positive_pairs_duped'] = [(2*n-1) * 2*n for n in batch_dict['lbl_inst_freq']]
        # batch_dict['negative_pairs_duped'] = [sum(batch_dict['lbl_inst_freq']) - k  for k in batch_dict['lbl_inst_freq']]
        # batch_dict['negative_pairs_duped'] = [4*k*i for k,i in zip(batch_dict['negative_pairs_duped'],batch_dict['lbl_inst_freq'])]


        combined_embed_features = torch.cat([embed_ft_sa.unsqueeze(1), embed_ft_wa.unsqueeze(1)], dim=1) # B*N,num_pairs,channel_dim
        num_pairs = combined_embed_features.shape[1]
        assert num_pairs == 2  # contrast_count = 2

        '''Create Contrastive Mask'''
        labels_sa = labels_sa.contiguous().view(-1, 1)
        mask = torch.eq(labels_sa, labels_sa.T).float().to(device) # (B*N, B*N)
        mask = mask.repeat(num_pairs, num_pairs)        # Tiling mask from N,N -> 2N, 2N)
        logits_mask = torch.scatter(torch.ones_like(mask),1,torch.arange(batch_size_labeled * num_pairs).view(-1, 1).to(device),0)    # mask-out self-contrast cases
        mask = mask * logits_mask

        ''' Visualize mask'''
        # plt.imshow(mask.cpu().numpy(), cmap='viridis')
        # plt.colorbar()
        # plt.savefig("mask.png")
        # plt.clf()
        
        contrast_feature = torch.cat(torch.unbind(combined_embed_features, dim=1), dim=0)   # B,2,256 -> Tuple(B,256) with len(2) -> (2*B,256)
        contrast_feature = F.normalize(contrast_feature.view(-1,combined_embed_features.shape[-1]),dim=-1) # normalized features for cosine similarity
        sim_supcon_matrix = torch.div(torch.matmul(contrast_feature, contrast_feature.T),temperature)  # compute similarity matrix

        # for numerical stability
        logits_max, _ = torch.max(sim_supcon_matrix, dim=1, keepdim=True)
        logits = sim_supcon_matrix - logits_max.detach()

        exp_logits = torch.exp(logits) * logits_mask # (NOTE:exponent over mostly negative values)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)         # compute mean of log-likelihood over positive        
        
        instance_loss = - ( temperature / base_temperature) * mean_log_prob_pos 
        
        if instance_loss is None:
            return

        instloss_car = instance_loss[labels==1].mean()
        instloss_ped = instance_loss[labels==2].mean()
        instloss_cyc = instance_loss[labels==3].mean()
        instloss_all = instance_loss.mean()


        inst_tb_dict = {
            'instloss_car': instloss_car.unsqueeze(-1),
            'instloss_cyc': instloss_cyc.unsqueeze(-1),
            'instloss_ped': instloss_ped.unsqueeze(-1),
            'instloss_all' : instloss_all.unsqueeze(-1),
        }
        tb_dict.update(inst_tb_dict)
        return instloss_all, tb_dict


    @staticmethod
    def _prep_tb_dict(tb_dict, lbl_inds, ulb_inds, reduce_loss_fn):
        tb_dict_ = {}
        for key in tb_dict.keys():
            if key == 'proto_cont_loss':
                tb_dict_[key] = tb_dict[key]
            elif "instloss" in key:
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

    def dump_statistics(self, batch_dict, unlabeled_inds,labeled_inds):
        # Store different types of scores over all itrs and epochs and dump them in a pickle for offline modeling
        # TODO (shashank) : Can be optimized later to save computational time, currently takes about 0.002sec
        batch_roi_labels = self.pv_rcnn.roi_head.forward_ret_dict['roi_labels'][unlabeled_inds]
        batch_roi_labels = [roi_labels.clone().detach() for roi_labels in batch_roi_labels]

        batch_rois = self.pv_rcnn.roi_head.forward_ret_dict['rois'][unlabeled_inds]
        batch_rois = [rois.clone().detach() for rois in batch_rois]

        batch_ori_gt_boxes = self.pv_rcnn.roi_head.forward_ret_dict['ori_unlabeled_boxes']
        batch_ori_gt_boxes = [ori_gt_boxes.clone().detach() for ori_gt_boxes in batch_ori_gt_boxes]

        gt_labels = batch_dict['gt_boxes'][:,:,7][labeled_inds].view(-1)
        

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

                # cur_pcv_score = self.pv_rcnn.roi_head.forward_ret_dict['pcv_scores'][cur_unlabeled_ind]
                # self.val_dict['pcv_scores'].extend(cur_pcv_score.tolist())

                # cur_num_points_roi = self.pv_rcnn.roi_head.forward_ret_dict['num_points_in_roi'][cur_unlabeled_ind]
                # self.val_dict['num_points_in_roi'].extend(cur_num_points_roi.tolist())

                cur_roi_label = self.pv_rcnn.roi_head.forward_ret_dict['roi_labels'][cur_unlabeled_ind].squeeze()
                self.val_dict['class_labels'].extend(cur_roi_label.tolist())

                cur_iteration = torch.ones_like(preds_iou_max) * (batch_dict['cur_iteration'])
                self.val_dict['iteration'].extend(cur_iteration.tolist())

                '''Stats for instance_contrastive_loss'''
                # start_epoch = self.model_cfg['ROI_HEAD'].get('INSTANCE_CONTRASTIVE_LOSS_START_EPOCH', 0)
                # stop_epoch = self.model_cfg['ROI_HEAD'].get('INSTANCE_CONTRASTIVE_LOSS_STOP_EPOCH', 60)
                # if self.model_cfg['ROI_HEAD']['ENABLE_INSTANCE_SUP_LOSS'] and start_epoch<=batch_dict['cur_epoch']<stop_epoch:
                #     bincount_values = batch_dict['lbl_inst_freq']
                #     # cumu_values = [a + b for a, b in zip(self.val_dict['lbl_inst_freq'][-3:], bincount_values)]
                #     self.val_dict['lbl_inst_freq'].extend(bincount_values)
                #     self.val_dict['positive_pairs_duped'].extend(batch_dict['positive_pairs_duped'])
                #     self.val_dict['negative_pairs_duped'].extend(batch_dict['negative_pairs_duped'])

                #     # self.val_dict['unscaled_instloss_car'].extend(batch_dict['unscaled_instloss_car'])
                #     # self.val_dict['unscaled_instloss_ped'].extend(batch_dict['unscaled_instloss_ped'])
                #     # self.val_dict['unscaled_instloss_cyc'].extend(batch_dict['unscaled_instloss_cyc'])

                #     # self.val_dict['instloss_car'].extend(batch_dict['instloss_car'])
                #     # self.val_dict['instloss_ped'].extend(batch_dict['instloss_ped'])
                #     # self.val_dict['instloss_cyc'].extend(batch_dict['instloss_cyc'])

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
        for pseudo_box, pseudo_label, pseudo_score, pseudo_sem_score, pseudo_box_var, pseudo_score_var in zip(
                *self._unpack_predictions(pred_dicts, unlabeled_inds)):

            if pseudo_label[0] == 0:
                pseudo_boxes.append(torch.cat([pseudo_box, pseudo_label.view(-1, 1).float()], dim=1))
                pseudo_sem_scores.append(pseudo_sem_score)
                pseudo_scores.append(pseudo_score)
                continue

            pl_thresh = self.thresh
            if self.model_cfg.ROI_HEAD.ADAPTIVE_THRESH_CONFIG.get('ENABLE', False):
                thresh_reg = self.thresh_registry.get(tag='pl_adaptive_thresh')
                if thresh_reg.relative_ema_threshold is not None:
                   pl_thresh = [thresh_reg.relative_ema_threshold.item()] * len(self.thresh)

            conf_thresh = torch.tensor(pl_thresh, device=pseudo_label.device).unsqueeze(
                0).repeat(len(pseudo_label), 1).gather(dim=1, index=(pseudo_label - 1).unsqueeze(-1))

            sem_conf_thresh = torch.tensor(self.sem_thresh, device=pseudo_label.device).unsqueeze(
                0).repeat(len(pseudo_label), 1).gather(dim=1, index=(pseudo_label - 1).unsqueeze(-1))

            valid_inds = pseudo_score > conf_thresh.squeeze()

            valid_inds = valid_inds & (pseudo_sem_score > sem_conf_thresh.squeeze())

            pseudo_sem_score = pseudo_sem_score[valid_inds]
            pseudo_box = pseudo_box[valid_inds]
            pseudo_label = pseudo_label[valid_inds]
            pseudo_score = pseudo_score[valid_inds]

            pseudo_boxes.append(torch.cat([pseudo_box, pseudo_label.view(-1, 1).float()], dim=1))
            pseudo_sem_scores.append(pseudo_sem_score)
            pseudo_scores.append(pseudo_score)

        return pseudo_boxes, pseudo_scores, pseudo_sem_scores

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

    def update_adaptive_thresholding_metrics(self, pred_dicts, unlabeled_inds, tag = 'pl_adaptive_thresh'):
        metrics_input = defaultdict(list)
        for ind in unlabeled_inds:
            pseudo_score = pred_dicts[ind]['pred_scores']
            pseudo_label = pred_dicts[ind]['pred_labels']
            pseudo_sem_score = pred_dicts[ind]['pred_sem_scores']
            if len(pseudo_label):
                metrics_input['pred_labels'].append(pseudo_label)
                metrics_input['pseudo_score'].append(pseudo_score)
                metrics_input['pseudo_sem_score'].append(pseudo_sem_score)
        self.thresh_registry.get(tag).update(**metrics_input)

