# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from __future__ import print_function
import torch
import torch.nn.functional as F
from torch import nn


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


class DomainAdaptationHead(nn.Module):
    """
    Module for Domain Adaptation Component. Takes feature maps from the backbone and instance
    feature vectors, domain labels and proposals. Works for both FPN and non-FPN.
    """

    def __init__(self, num_class, class_names, ins_cls_input_channels, img_cls_input_channels, model_cfg, predict_boxes_when_training=False, **kwargs):
        super(DomainAdaptationHead, self).__init__()
        self.num_class = num_class
        self.class_names = class_names
        self.predict_boxes_when_training = predict_boxes_when_training
        self.model_cfg = model_cfg

        gscaler_cfg = self.model_cfg.GRADIENT_SCALER_CONFIG
        self.grl_img = GradientScalarLayer(-1.0 * gscaler_cfg.DA_IMG_GRL_WEIGHT)
        self.grl_ins = GradientScalarLayer(-1.0 * gscaler_cfg.DA_INS_GRL_WEIGHT)
        self.grl_img_consist = GradientScalarLayer(1.0 * gscaler_cfg.DA_IMG_GRL_WEIGHT)
        self.grl_ins_consist = GradientScalarLayer(1.0 * gscaler_cfg.DA_INS_GRL_WEIGHT)

        self.img_domain_cls_layers = self.make_fc_layers(input_channels=img_cls_input_channels,
                                                         output_channels=self.num_class,
                                                         fc_list=self.model_cfg.IMG_CLS_FC)
        self.ins_domain_cls_layers = self.make_fc_layers(input_channels=ins_cls_input_channels,
                                                         output_channels=self.num_class,
                                                         fc_list=self.model_cfg.INS_CLS_FC)
        self.forward_ret_dict = None
        self.init_weights(weight_init='xavier')

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

    def get_img_loss(self, tb_dict=None):

        img_domain_preds = self.forward_ret_dict['img_domain_preds']
        img_domain_labels = self.forward_ret_dict['img_domain_labels']

        da_img_loss = F.binary_cross_entropy_with_logits(img_domain_preds, img_domain_labels)

        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        da_img_loss = da_img_loss * loss_weights_dict['da_img_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({
            'da_img_loss': da_img_loss.item(),
        })
        return da_img_loss, tb_dict

    def get_ins_loss(self, tb_dict=None):
        ins_domain_preds = self.forward_ret_dict['ins_domain_preds']
        ins_domain_labels = self.forward_ret_dict['ins_domain_labels']

        da_ins_loss = F.binary_cross_entropy_with_logits(ins_domain_preds, ins_domain_labels)

        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        da_ins_loss = da_ins_loss * loss_weights_dict['da_ins_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({
            'da_ins_loss': da_ins_loss.item(),
        })
        return da_ins_loss, tb_dict

    def get_consistency_loss(self, tb_dict=None):
        """
        Consistency regularization as stated in the paper
        `Domain Adaptive Faster R-CNN for Object Detection in the Wild`
        L_cst = \sum_{i,j}||\frac{1}{|I|}\sum_{u,v}p_i^{(u,v)}-p_{i,j}||_2
        """

        ins_domain_preds = self.forward_ret_dict['ins_domain_preds']
        da_ins_labels = self.forward_ret_dict['da_ins_labels']
        img_domain_preds = self.forward_ret_dict['img_domain_preds']

        loss = []
        len_ins = ins_domain_preds.size(0)
        intervals = [torch.nonzero(da_ins_labels).size(0), len_ins - torch.nonzero(ins_labels).size(0)]
        for img_fea_per_level in [img_domain_preds]:
            N, A, H, W = img_fea_per_level.shape
            img_fea_per_level = torch.mean(img_fea_per_level.reshape(N, -1), 1)
            img_feas_per_level = []
            assert N == 2, \
                "only batch size=2 is supported for consistency loss now, received batch size: {}".format(N)
            for i in range(N):
                img_fea_mean = img_fea_per_level[i].view(1, 1).repeat(intervals[i], 1)
                img_feas_per_level.append(img_fea_mean)
            img_feas_per_level = torch.cat(img_feas_per_level, dim=0)
            loss_per_level = torch.abs(img_feas_per_level - ins_fea)
            loss.append(loss_per_level)
        da_cst_loss = torch.cat(loss, dim=1)

        # Assuming size_average is True
        da_cst_loss = da_cst_loss.mean()

        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        da_cst_loss = da_cst_loss * loss_weights_dict['da_cst_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({
            'da_cst_loss': da_cst_loss.item(),
        })
        return da_cst_loss, tb_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        da_img_loss, tb_dict = self.get_img_loss(tb_dict)
        da_ins_loss, tb_dict = self.get_ins_loss(tb_dict)
        # da_cst_loss, tb_dict = self.get_cst_loss(tb_dict)

        da_loss = da_img_loss + da_ins_loss  # + da_cst_loss
        tb_dict['da_loss'] = da_loss.item()

        return da_loss, tb_dict

    def forward(self, batch_dict):
        """
        Arguments:
            batch_dict
                img_features (list[Tensor]): features computed from the images that are
                    used for computing the predictions.
                da_ins_feature (Tensor): instance-level feature vectors
                da_ins_labels (Tensor): domain labels for instance-level feature vectors
                targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        if self.training:
            # TODO(farzad) like DA-Faster-RCNN you can divide the 3D space into regions
            #  and align their corresponding features in each source and target regions
            batch_img_features = batch_dict['img_features']  # torch.Size([2, 1024, 64])
            batch_ins_feature = batch_dict['roi_features'].transpose(1, 2)  # torch.Size([2, 512, 128])
            batch_ins_domain_labels = batch_dict['roi_domain_labels'].float()  # torch.Size([2, 128])
            is_source = batch_dict['is_source']
            bs = batch_dict['batch_size']
            img_domain_labels = batch_img_features.new_zeros(bs, batch_img_features.shape[-1])
            source_mask = torch.any(is_source == 1, axis=1)
            img_domain_labels[source_mask] = 1

            img_grl_fea = self.grl_img(batch_img_features)  # torch.Size([2, 512, 1])
            ins_grl_fea = self.grl_ins(batch_ins_feature)  # torch.Size([2, 512, 128])
            img_grl_consist_fea = self.grl_img_consist(batch_img_features)
            ins_grl_consist_fea = self.grl_ins_consist(batch_ins_feature)

            img_domain_preds = self.img_domain_cls_layers(img_grl_fea).squeeze()
            ins_domain_preds = self.ins_domain_cls_layers(ins_grl_fea).squeeze()
            img_domain_cst_preds = self.img_domain_cls_layers(img_grl_consist_fea)
            ins_domain_cst_preds = self.ins_domain_cls_layers(ins_grl_consist_fea)
            img_domain_cst_preds = img_domain_cst_preds.sigmoid()
            ins_domain_cst_preds = ins_domain_cst_preds.sigmoid()

            ret_dict = {'img_domain_preds': img_domain_preds,
                        'ins_domain_preds': ins_domain_preds,
                        'img_domain_cst_preds': img_domain_cst_preds,
                        'ins_domain_cst_preds': ins_domain_cst_preds,
                        'ins_domain_labels': batch_ins_domain_labels,
                        'img_domain_labels': img_domain_labels}

            self.forward_ret_dict = ret_dict

        return batch_dict
