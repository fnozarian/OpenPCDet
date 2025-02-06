import torch.nn as nn
import torch
from torch.nn import BatchNorm1d
import random
from torch.nn.utils import weight_norm
from torch.nn.init import trunc_normal_
from torch.functional import F


class DINOHead(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.cfgs = model_cfg
        mlp_params = model_cfg.get('MLP', {})
        bottleneck_dim = mlp_params.get('bottleneck_dim', 128)
        in_dim = model_cfg.get('in_dim', 256)
        out_dim = model_cfg.get('out_dim', 64)
        # self.mlp = self._build_mlp(**mlp_params)
        self.mlp = self.make_fc_layers(input_channels=in_dim, output_channels=256, fc_list=[256])
        # self.apply(self._init_weights)
        self.last_layer = weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if model_cfg.get('NORM_LAST_LAYER', False):
            self.last_layer.weight_g.requires_grad = False
        self.mask_token = nn.Parameter(torch.randn(1, 128))  # (1, C)
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

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=0.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)

    def get_masked_feats(self, gpoint_feats):
        # grid_features (BxN, C, 6, 6, 6)
        grid_size = gpoint_feats.size(2)
        mask = torch.rand(grid_size, grid_size, grid_size) > 0.5
        mask = mask.unsqueeze(0).unsqueeze(0).to(gpoint_feats.device)
        embedding_expanded = self.mask_token.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(gpoint_feats)
        masked_features = gpoint_feats * mask + embedding_expanded * (~mask)
        return masked_features

    def get_cls_token(self, grid_feats):
        grid_feats = self.mlp(grid_feats)
        grid_feats = grid_feats.squeeze(-1)
        if self.cfgs['NORM_LAST_LAYER']:
            eps = 1e-6 if grid_feats.dtype == torch.float16 else 1e-12
            grid_feats = F.normalize(grid_feats, p=2, dim=-1, eps=eps)
        grid_feats = self.last_layer(grid_feats)
        return grid_feats

    def forward(self, batch_dict):
        return batch_dict

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
            if k == 0:
                fc_layers.append(nn.Dropout(0.3))
        fc_layers.append(nn.Conv1d(pre_channel, output_channels, kernel_size=1, bias=True))
        fc_layers = nn.Sequential(*fc_layers)
        return fc_layers

    def _build_mlp(self, nlayers, in_dim, bottleneck_dim, hidden_dim=None, use_bn=False, bias=True):
        if nlayers == 1:
            return nn.Linear(in_dim, bottleneck_dim, bias=bias)
        else:
            layers = [nn.Linear(in_dim, hidden_dim, bias=bias)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim, bias=bias))
            return nn.Sequential(*layers)


class DINOHeadWithGridTransformer(DINOHead):
    def __init__(self, input_channels, model_cfg, num_class=1, **kwargs):
        super().__init__(input_channels, model_cfg, num_class=num_class, **kwargs)
        self.grid_points_encoder = GridTransformerEncoder(
            feature_dim=self.model_cfg.GRID_TRANSFORMER_ENCODER.FEATURE_DIM,
            num_heads=self.model_cfg.GRID_TRANSFORMER_ENCODER.NUM_HEADS,
            num_layers=self.model_cfg.GRID_TRANSFORMER_ENCODER.NUM_LAYERS,
            hidden_dim=self.model_cfg.GRID_TRANSFORMER_ENCODER.HIDDEN_DIM
        )
        self.cls_layers = nn.Sequential(nn.Linear(256, 256),
                                        BatchNorm1d(256),
                                        nn.ReLU(),
                                        nn.Linear(256, num_class))
        self.reg_layers = nn.Sequential(nn.Linear(256, 256),
                                        BatchNorm1d(256),
                                        nn.ReLU(),
                                        nn.Linear(256, self.box_coder.code_size * num_class))
        self.print_loss_when_eval = False
        self.init_weights(weight_init='xavier')

    def forward(self, batch_dict, test_only=False):
        nms_config = self.model_cfg.NMS_CONFIG['TRAIN' if self.training and not test_only else 'TEST']
        targets_dict = self.proposal_layer(batch_dict, nms_config=nms_config)
        if (self.training or self.print_loss_when_eval) and not test_only:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']

        pooled_features = self.roi_grid_pool(batch_dict)  # (BxN, 6x6x6, C)
        grid_size = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        batch_size_rcnn = pooled_features.shape[0]
        pooled_features = pooled_features.permute(0, 2, 1). \
            contiguous().view(batch_size_rcnn, -1, grid_size, grid_size, grid_size)  # (BxN, C, 6, 6, 6)

        refined_grid_features = self.grid_transformer_encoder(pooled_features)
        shared_features = self.shared_fc_layer(refined_grid_features.contiguous().view(batch_size_rcnn, -1, 1)).squeeze(-1)
        rcnn_cls = self.cls_layers(shared_features)
        rcnn_reg = self.reg_layers(shared_features)

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            # note that the rpn batch_cls_preds and batch_box_preds are being overridden here by rcnn preds
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False

        if self.training or self.print_loss_when_eval:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg
            self.forward_ret_dict = targets_dict

    def get_grid_tokens(self, batch_dict):
        pooled_features = self.roi_grid_pool(batch_dict)  # (BxN, 6x6x6, C)
        grid_size = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        batch_size_rcnn = pooled_features.shape[0]
        pooled_features = pooled_features.permute(0, 2, 1). \
            contiguous().view(batch_size_rcnn, -1, grid_size, grid_size, grid_size)  # (BxN, C, 6, 6, 6)
        cls_token, grid_patches = self.grid_points_encoder(pooled_features)
        return cls_token, grid_patches


class GridTransformerEncoder(nn.Module):
    def __init__(self, feature_dim, num_heads, num_layers, hidden_dim):
        super().__init__()
        self.flatten_dim = 6 * 6 * 6  # Flattened grid size (6x6x6 = 216)
        self.pos_embed = nn.Parameter(torch.randn(1, self.flatten_dim + 1, feature_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))  # (1, 1, C)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            activation="relu"
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)

    def forward(self, batch_dict):
        pooled_features = self.get_masked_patches(batch_dict)  # (B * N, C, 216)
        B_N, C, _ = pooled_features.shape
        pooled_features = pooled_features.permute(0, 2, 1)  # (B * N, 216, C)
        cls_token = self.cls_token.expand(B_N, -1, -1)  # (B * N, 1, C)
        features_with_cls = torch.cat([cls_token, pooled_features], dim=1)  # (B * N, 217, C)
        features_with_pos = features_with_cls + self.pos_embed  # (B * N, 217, C)
        refined_features = self.encoder(features_with_pos)  # (B * N, 217, D)
        cls_token = refined_features[:, 0, :]  # (B * N, D)
        refined_grid_features = refined_features[:, 1:, :]  # (B * N, 216, D)
        refined_grid_features = refined_grid_features.permute(0, 2, 1)  # (B * N, D, 216)
        refined_grid_features = refined_grid_features.view(B_N, C, -1)  # (B * N, D, 6x6x6)
        return cls_token, refined_grid_features

    # TODO: Improve masking
    def get_masked_patches(self, pooled_features, crop_size=4):
        grid_size = self.cfgs.ROI_GRID_POOL.GRID_SIZE
        batch_size_rcnn = pooled_features.shape[0]
        x = random.randint(0, grid_size - crop_size)
        y = random.randint(0, grid_size - crop_size)
        z = random.randint(0, grid_size - crop_size)
        cropped_features = pooled_features[:, :, x:x + crop_size, y:y + crop_size, z:z + crop_size]
        # TODO(farzad): requires clone()?
        padded_features = self.mask_token.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(pooled_features).clone()
        padded_features[:, :, x:x + crop_size, y:y + crop_size, z:z + crop_size] = cropped_features
        pooled_features_flat = padded_features.view(batch_size_rcnn, -1, self.flatten_dim)  # (BxN, Cx6x6x6)
        return pooled_features_flat