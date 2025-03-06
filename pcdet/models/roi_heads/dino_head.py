import torch.nn as nn
import torch
from torch.nn import BatchNorm1d
import random
from torch.nn.utils import weight_norm
from torch.functional import F
from visual_utils.open3d_vis_utils import Open3DRenderer
import numpy as np


class DINOHead(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.cfgs = model_cfg
        mlp_params = model_cfg.get('MLP', {})
        bottleneck_dim = mlp_params.get('bottleneck_dim', 128)
        in_dim = model_cfg.get('in_dim', 256)
        out_dim = model_cfg.get('out_dim', 64)
        self.mlp = self.make_fc_layers(input_channels=in_dim, output_channels=256, fc_list=[256])
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


class DINOHeadWithGPointsFormer(DINOHead):
    def __init__(self, model_cfg):
        super().__init__(model_cfg)
        self.gpoints_encoder = GridPointsFormer(model_cfg=model_cfg.GPOINTS_FORMER)

    def forward(self, batch_dict):
        return batch_dict

    def get_cls_token(self, gpoint_feats):
        return self.gpoints_encoder(gpoint_feats)


class LearnablePositionalEncoder3D(nn.Module):
    def __init__(self, grid_size=6, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.grid_size = grid_size
        self.embed_dim = embed_dim

        # MLP to map (x, y, z) -> high-dimensional embedding
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, local_indices):
        """
        Args:
            local_indices: Tensor of shape (N, 3), containing (x, y, z) indices of grid points.
        Returns:
            pos_embedding: Tensor of shape (N, embed_dim)
        """
        # Normalize indices to range [-1, 1]
        normalized_indices = (local_indices.float() / (self.grid_size - 1)) * 2 - 1

        # Pass through MLP
        pos_embedding = self.mlp(normalized_indices)
        return pos_embedding


class PositionalEncoding3D(nn.Module):
    def __init__(self, channels, grid_size=6):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding3D, self).__init__()
        self.orig_ch = channels
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None, persistent=False)
        self.channels = channels
        self.grid_size = grid_size

    def forward(self, tensor):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """
        self.cached_penc = None

        pos_x = torch.arange(self.grid_size, device=tensor.device, dtype=self.inv_freq.dtype)
        pos_y = torch.arange(self.grid_size, device=tensor.device, dtype=self.inv_freq.dtype)
        pos_z = torch.arange(self.grid_size, device=tensor.device, dtype=self.inv_freq.dtype)
        sin_inp_x = pos_x.unsqueeze(1) * self.inv_freq.unsqueeze(0)
        sin_inp_y = pos_y.unsqueeze(1) * self.inv_freq.unsqueeze(0)
        sin_inp_z = pos_z.unsqueeze(1) * self.inv_freq.unsqueeze(0)
        emb_x = self.get_emb(sin_inp_x).unsqueeze(1).unsqueeze(1)
        emb_y = self.get_emb(sin_inp_y).unsqueeze(1)
        emb_z = self.get_emb(sin_inp_z)
        emb = torch.zeros((self.grid_size, self.grid_size, self.grid_size, self.channels * 3), device=tensor.device, dtype=tensor.dtype)
        emb[:, :, :, : self.channels] = emb_x
        emb[:, :, :, self.channels: 2 * self.channels] = emb_y
        emb[:, :, :, 2 * self.channels:] = emb_z

        self.cached_penc = emb[:, :, :, :self.orig_ch].unsqueeze(0).view(1, -1, self.orig_ch)
        return self.cached_penc

    @staticmethod
    def get_emb(sin_cos_inp):
        """
        Gets a base embedding for one dimension with sin and cos intertwined
        """
        emb = torch.stack((sin_cos_inp.sin(), sin_cos_inp.cos()), dim=-1)
        return torch.flatten(emb, -2, -1)


class GridPointsFormer(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        num_heads = model_cfg.NUM_HEADS  # 8
        num_layers = model_cfg.NUM_LAYERS  # 2
        ff_dim = model_cfg.FF_DIM  # 1024
        self.feature_dim = model_cfg.FEATURE_DIM  # 128
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.feature_dim, nhead=num_heads, dim_feedforward=ff_dim, batch_first=True
        )
        self.fusion_layer = nn.Linear(2 * self.feature_dim, self.feature_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.feature_dim)).cuda()
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_encoder = PositionalEncoding3D(channels=self.feature_dim)

        # self.renderer = Open3DRenderer()

    def forward(self, gpoint_feats):
        """
        :param gpoint_feats: (B*num_rois, 6x6x6, C)
        :return: (B*num_rois, 128)
        """
        batch_size = gpoint_feats.shape[0]
        if self.pos_encoder.cached_penc is not None:
            pe = self.pos_encoder.cached_penc.repeat(batch_size, 1, 1)
        else:
            pe = self.pos_encoder(gpoint_feats.view(-1, 6, 6, 6, self.feature_dim)).repeat(batch_size, 1, 1)
        gpoint_feats = self.fusion_layer(torch.cat((gpoint_feats, pe), dim=-1))
        cls_tokens = self.cls_token.expand(gpoint_feats.shape[0], -1, -1)  # (batch_size*num_rois, 1, C)
        gpoint_feats = torch.cat((cls_tokens, gpoint_feats), dim=1)  # (batch_size*num_rois, 217, C)
        gpoint_feats = self.encoder(gpoint_feats)  # (batch_size*num_rois, 217, C)
        cls_token = gpoint_feats[:, 0, :]  # (batch_size*num_rois, C)
        return cls_token
