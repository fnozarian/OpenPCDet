import torch
import torch.nn as nn
from ...ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from ...utils import common_utils
from .roi_head_template import RoIHeadTemplate
from pcdet.utils.prototype_utils import feature_bank_registry
import torch.nn.functional as F     # TODO - refactor imports


class PVRCNNHead(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, num_class=1,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg,
                         predict_boxes_when_training=predict_boxes_when_training)
        self.model_cfg = model_cfg

        self.roi_grid_pool_layer, num_c_out = pointnet2_stack_modules.build_local_aggregation_module(
            input_channels=input_channels, config=self.model_cfg.ROI_GRID_POOL
        )

        GRID_SIZE = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * num_c_out
        pre_channel2 = GRID_SIZE * GRID_SIZE * GRID_SIZE * num_c_out

        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([
                nn.Conv1d(pre_channel, self.model_cfg.SHARED_FC[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

        self.shared_fc_layer = nn.Sequential(*shared_fc_list) 

        projected_fc_list = []
        for k in range(0, self.model_cfg.PROJECTED_FC.__len__()):
            projected_fc_list.extend([
                nn.Conv1d(pre_channel2, self.model_cfg.PROJECTED_FC[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(self.model_cfg.PROJECTED_FC[k]),
                nn.ReLU()
            ])
            pre_channel2 = self.model_cfg.PROJECTED_FC[k]

            if k != self.model_cfg.PROJECTED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                projected_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

        self.projector_fc_layer = nn.Sequential(*projected_fc_list) # Using this layer's projections to calculate instance wise contrastive loss on

        self.cls_layers = self.make_fc_layers(
            input_channels=pre_channel, output_channels=self.num_class, fc_list=self.model_cfg.CLS_FC
        )
        self.reg_layers = self.make_fc_layers(
            input_channels=pre_channel,
            output_channels=self.box_coder.code_size * self.num_class,
            fc_list=self.model_cfg.REG_FC
        )
        self.init_weights(weight_init='xavier')

        self.print_loss_when_eval = False

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layers[-1].weight, mean=0, std=0.001)

    def roi_grid_pool(self, batch_dict, use_gtboxes=False):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:

        """
        batch_size = batch_dict['batch_size']
        if use_gtboxes:
            rois = batch_dict['gt_boxes'][..., 0:7]
        else:
            rois = batch_dict['rois']
        point_coords = batch_dict["point_coords"]
        point_features = batch_dict["point_features"]
        point_cls_scores = batch_dict["point_cls_scores"]

        point_features = point_features * point_cls_scores.view(-1, 1)

        global_roi_grid_points, local_roi_grid_points = self.get_global_grid_points_of_roi(
            rois, grid_size=self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        )  # (BxN, 6x6x6, 3)
        global_roi_grid_points = global_roi_grid_points.view(batch_size, -1, 3)  # (B, Nx6x6x6, 3)

        xyz = point_coords[:, 1:4]
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        batch_idx = point_coords[:, 0]
        for k in range(batch_size):
            xyz_batch_cnt[k] = (batch_idx == k).sum()

        new_xyz = global_roi_grid_points.view(-1, 3)
        new_xyz_batch_cnt = xyz.new_zeros(batch_size).int().fill_(global_roi_grid_points.shape[1])
        pooled_points, pooled_features = self.roi_grid_pool_layer(
            xyz=xyz.contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz,
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=point_features.contiguous(),
        )  # (M1 + M2 ..., C)

        pooled_features = pooled_features.view(
            -1, self.model_cfg.ROI_GRID_POOL.GRID_SIZE ** 3,
            pooled_features.shape[-1]
        )  # (BxN, 6x6x6, C)
        return pooled_features

    def get_global_grid_points_of_roi(self, rois, grid_size):
        rois = rois.view(-1, rois.shape[-1])
        batch_size_rcnn = rois.shape[0]

        local_roi_grid_points = self.get_dense_grid_points(rois, batch_size_rcnn, grid_size)  # (B, 6x6x6, 3)
        global_roi_grid_points = common_utils.rotate_points_along_z(
            local_roi_grid_points.clone(), rois[:, 6]
        ).squeeze(dim=1)
        global_center = rois[:, 0:3].clone()
        global_roi_grid_points += global_center.unsqueeze(dim=1)
        return global_roi_grid_points, local_roi_grid_points

    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
        faked_features = rois.new_ones((grid_size, grid_size, grid_size))
        dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx]
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (B, 6x6x6, 3)

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 6x6x6, 3)
        return roi_grid_points

    def pool_features(self, batch_dict, use_gtboxes=False):
        pooled_features = self.roi_grid_pool(batch_dict, use_gtboxes=use_gtboxes)  # (BxN, 6x6x6, C)
        grid_size = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        batch_size_rcnn = pooled_features.shape[0]
        pooled_features = pooled_features.permute(0, 2, 1). \
            contiguous().view(batch_size_rcnn, -1, grid_size, grid_size, grid_size)  # (BxN, C, 6, 6, 6)

        return pooled_features

    def forward(self, batch_dict, test_only=False,use_gtboxes=False):
        """
        :param input_data: input dict
        :return:
        """
        nms_config = self.model_cfg.NMS_CONFIG['TRAIN' if self.training and not test_only else 'TEST']
        # proposal_layer doesn't continue if the rois are already in the batch_dict.
        # However, for labeled data proposal layer should continue!
        targets_dict = self.proposal_layer(batch_dict, nms_config=nms_config)
        # should not use gt_roi for pseudo label generation
        if (self.training or self.print_loss_when_eval) and not test_only:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_scores'] = targets_dict['roi_scores']
            batch_dict['roi_labels'] = targets_dict['roi_labels']
            # Temporarily add infos to targets_dict for metrics
            targets_dict['unlabeled_inds'] = batch_dict['unlabeled_inds']
            targets_dict['ori_unlabeled_boxes'] = batch_dict['ori_unlabeled_boxes']
            targets_dict['points'] = batch_dict['points']

        pooled_features = self.pool_features(batch_dict,use_gtboxes=use_gtboxes)
        if use_gtboxes == True:
            # batch_dict['pooled_features_gt'] = pooled_features
            batch_size_rcnn = pooled_features.shape[0]
            start_epoch = self.model_cfg['INSTANCE_CONTRASTIVE_LOSS_START_EPOCH']
            stop_epoch = self.model_cfg['INSTANCE_CONTRASTIVE_LOSS_STOP_EPOCH']
            if self.model_cfg.ENABLE_INSTANCE_SUP_LOSS==True and start_epoch<=batch_dict['cur_epoch']<stop_epoch: # normalize embedding and produce projected representation only when instance_sup_loss true.
                
                if self.model_cfg['NORMALIZATION']:
                    # pooled_features dim : [GT_boxes,27648]
                    pooled_features = F.normalize(pooled_features, dim = -1)
                proj_features = pooled_features.clone().detach()
                projected_features_gt = self.projector_fc_layer(proj_features.view(batch_size_rcnn, -1, 1))
                batch_dict['shared_features_gt'] = projected_features_gt
            return batch_dict
        batch_size_rcnn = pooled_features.shape[0]
        shared_features = self.shared_fc_layer(pooled_features.view(batch_size_rcnn, -1, 1))
        rcnn_cls = self.cls_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
        rcnn_reg = self.reg_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)

        if (self.training or self.print_loss_when_eval) and not test_only:
            # RoI-level similarity.
            # calculate cosine similarity between unlabeled augmented RoI features and labeled augmented prototypes.
            roi_features = pooled_features.clone().detach().view(batch_size_rcnn, -1)
            roi_scores_shape = batch_dict['roi_scores'].shape  # (B, N)
            bank = feature_bank_registry.get('gt_aug_lbl_prototypes')
            sim_scores = bank.get_sim_scores(roi_features)
            targets_dict['roi_sim_scores'] = sim_scores.view(*roi_scores_shape, -1)

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            # note that the rpn batch_cls_preds and batch_box_preds are being overridden here by rcnn preds
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
            # Temporarily add infos to targets_dict for metrics
            targets_dict['batch_box_preds'] = batch_box_preds

        if self.training or self.print_loss_when_eval:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg
            self.forward_ret_dict = targets_dict

        return batch_dict
