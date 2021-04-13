import torch

from .point_head_box import PointHeadBox


class _GradientScalarLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.weight = weight
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return ctx.weight*grad_input, None


gradient_scalar = _GradientScalarLayer.apply


class GradientScalarLayer(torch.nn.Module):
    def __init__(self, weight):
        super(GradientScalarLayer, self).__init__()
        self.weight = weight

    def forward(self, input):
        return gradient_scalar(input, self.weight)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "weight=" + str(self.weight)
        tmpstr += ")"
        return tmpstr

class PointHeadBoxMCD(PointHeadBox):

    def __init__(self, num_class, input_channels, model_cfg, predict_boxes_when_training=False, **kwargs):
        super().__init__(num_class=num_class, input_channels=input_channels, model_cfg=model_cfg,
                         predict_boxes_when_training=predict_boxes_when_training, **kwargs)
        self.predict_boxes_when_training = predict_boxes_when_training
        self.second_cls_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.CLS_FC,
            input_channels=input_channels,
            output_channels=num_class
        )
        self.rev_grad = GradientScalarLayer(-1)

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        point_loss_cls, tb_dict_1 = self.get_cls_layer_loss()
        point_loss_box, tb_dict_2 = self.get_box_layer_loss()

        point_loss = point_loss_cls + point_loss_box
        tb_dict.update(tb_dict_1)
        tb_dict.update(tb_dict_2)
        return point_loss, tb_dict

    def get_discrepancy_loss(self, tb_dict=None):
        point_cls_preds1 = self.forward_ret_dict['point_cls_preds1'].view(-1, self.num_class)
        point_cls_preds2 = self.forward_ret_dict['point_cls_preds2'].view(-1, self.num_class)
        if self.num_class == 1:
            score_func = torch.sigmoid
        else:
            score_func = lambda input: torch.softmax(input, dim=-1)
        disc_loss = torch.mean(torch.abs(score_func(point_cls_preds1) - score_func(point_cls_preds2)))
        disc_loss = -1 * disc_loss

        if 'discrepancy_weight' in self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS.keys():
            disc_weight = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['discrepancy_weight']
        else:
            disc_weight = 1.

        disc_loss = disc_loss * disc_weight

        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({
            'discrepancy_loss': disc_loss.item()
        })
        return disc_loss, tb_dict

    def get_cls_layer_loss(self, tb_dict=None):
        point_cls_labels = self.forward_ret_dict['point_cls_labels'].view(-1)
        point_cls_preds1 = self.forward_ret_dict['point_cls_preds1'].view(-1, self.num_class)
        point_cls_preds2 = self.forward_ret_dict['point_cls_preds2'].view(-1, self.num_class)

        positives = (point_cls_labels > 0)
        negative_cls_weights = (point_cls_labels == 0) * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        pos_normalizer = positives.sum(dim=0).float()
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)

        one_hot_targets = point_cls_preds1.new_zeros(*list(point_cls_labels.shape), self.num_class + 1)
        one_hot_targets.scatter_(-1, (point_cls_labels * (point_cls_labels >= 0).long()).unsqueeze(dim=-1).long(), 1.0)
        one_hot_targets = one_hot_targets[..., 1:]
        cls_loss_src1 = self.cls_loss_func(point_cls_preds1, one_hot_targets, weights=cls_weights).sum()
        cls_loss_src2 = self.cls_loss_func(point_cls_preds2, one_hot_targets, weights=cls_weights).sum()

        point_loss_cls = cls_loss_src1 + cls_loss_src2

        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_cls = point_loss_cls * loss_weights_dict['point_cls_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({
            'cls_loss_src1': cls_loss_src1.item(),
            'cls_loss_src2': cls_loss_src2.item(),
            'point_loss_cls': point_loss_cls.item(),
            'point_pos_num': pos_normalizer.item()
        })
        return point_loss_cls, tb_dict

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                point_features: (N1 + N2 + N3 + ..., C) or (B, N, C)
                point_features_before_fusion: (N1 + N2 + N3 + ..., C)
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                point_labels (optional): (N1 + N2 + N3 + ...)
                gt_boxes (optional): (B, M, 8)
        Returns:
            batch_dict:
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        """
        if self.model_cfg.get('USE_POINT_FEATURES_BEFORE_FUSION', False):
            point_features = batch_dict['point_features_before_fusion']
        else:
            point_features = batch_dict['point_features']

        is_source = torch.all(batch_dict['is_source'])
        if not is_source:
            point_features = self.rev_grad(point_features)

        point_cls_preds1 = self.cls_layers(point_features)  # (total_points, num_class)
        point_cls_preds2 = self.second_cls_layers(point_features)  # (total_points, num_class)

        point_box_preds = self.box_layers(point_features)  # (total_points, box_code_size)

        point_cls_preds_max1, _ = point_cls_preds1.max(dim=-1)
        point_cls_preds_max2, _ = point_cls_preds2.max(dim=-1)
        batch_dict['point_cls_scores'] = torch.sigmoid(point_cls_preds_max1)
        batch_dict['point_cls_scores2'] = torch.sigmoid(point_cls_preds_max2)

        ret_dict = {'point_cls_preds1': point_cls_preds1,
                    'point_cls_preds2': point_cls_preds2,
                    'point_box_preds': point_box_preds}
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            ret_dict['point_cls_labels'] = targets_dict['point_cls_labels']
            ret_dict['point_box_labels'] = targets_dict['point_box_labels']

        if not self.training or self.predict_boxes_when_training:
            point_cls_preds1, point_box_preds = self.generate_predicted_boxes(
                points=batch_dict['point_coords'][:, 1:4],
                point_cls_preds=point_cls_preds1, point_box_preds=point_box_preds
            )
            batch_dict['batch_cls_preds'] = point_cls_preds1
            batch_dict['batch_box_preds'] = point_box_preds
            batch_dict['batch_index'] = batch_dict['point_coords'][:, 0]
            batch_dict['cls_preds_normalized'] = False

        self.forward_ret_dict = ret_dict

        return batch_dict
