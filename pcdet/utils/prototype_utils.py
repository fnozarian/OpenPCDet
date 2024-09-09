import torch
from torch.functional import F
from torchmetrics import Metric
import numpy as np
import torch.distributed as dist
from torch import nn


class FeatureBankV2:
    def __init__(self, instance_ids, **kwargs):

        super().__init__()
        self.tag = kwargs.get('NAME', None)
        self.distributed = dist.is_initialized()
        self.world_size = kwargs.get('WORLD_SIZE', 1)

        self.num_classes = 3
        self.temperature = kwargs.get('TEMPERATURE')
        self.feature_size = kwargs.get('FEATURE_SIZE')
        self.momentum = kwargs.get('MOMENTUM')
        self.direct_update = kwargs.get('DIRECT_UPDATE')
        self.num_points_thresh = kwargs.get('FILTER_MIN_POINTS_IN_GT', 0)
        self.bank_size = len(instance_ids)
        self.ce_loss = nn.CrossEntropyLoss()

        # Globally synchronized tensors
        self.instance_ids = torch.tensor(instance_ids, dtype=torch.long).cuda()
        features = torch.randn((self.bank_size, self.feature_size), dtype=torch.float32).cuda()
        self.features = F.normalize(features, dim=-1)
        self.classwise_prototypes = torch.zeros((3, self.feature_size), dtype=torch.float32).cuda()
        self.labels = -torch.ones(self.bank_size, dtype=torch.long).cuda()

    @torch.no_grad()
    def concat_all_gather(self, tensor):
        tensors_gather = [torch.ones_like(tensor)
                          for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor)
        output = torch.cat(tensors_gather, dim=0)
        return output

    @torch.no_grad()
    def update(self, feats, labels, ins_ids):
        if self.distributed and self.world_size > 1:
            feats = self.concat_all_gather(feats)
            labels = self.concat_all_gather(labels)
            ins_ids = self.concat_all_gather(ins_ids)

        feats = feats.view(-1, feats.size(-1))
        labels = labels.view(-1)
        ins_ids = ins_ids.view(-1)
        # filter out padded samples
        valid_mask = ins_ids != 0
        ins_ids = ins_ids[valid_mask]
        labels = labels[valid_mask]
        feats = feats[valid_mask]
        feat_indices = torch.tensor([torch.where(self.instance_ids == x)[0].item() for x in ins_ids]).cuda()
        feats = F.normalize(feats, dim=-1)
        if self.direct_update:
            self.features[feat_indices, :] = feats.detach()
        else:
            smoothed_feats = self.momentum * self.features[feat_indices, :] + (1 - self.momentum) * feats.detach()
            self.features[feat_indices, :] = F.normalize(smoothed_feats, dim=-1)
        self.labels[feat_indices] = labels.detach()

    @torch.no_grad()
    def _randomly_sample_protos_by_class(self, num_samples):
        sampled_inds_list = []
        for i in range(self.num_classes):
            inds = torch.where(self.labels == i)[0]
            sampled_inds = torch.randperm(len(inds))[:num_samples]
            sampled_inds_list.append(inds[sampled_inds])
        return torch.cat(sampled_inds_list)

    def get_lpcont_loss(self, roi_feats_sa, roi_labels, nppc=10):
        sampled_inds = self._randomly_sample_protos_by_class(nppc)
        bank_labels = self.labels[sampled_inds]
        bank_feats_wa = self.features[sampled_inds]
        if torch.bincount(bank_labels).min() < nppc or roi_labels.size(0) == 0:
            return None, None, None
        roi_feats_sa = F.normalize(roi_feats_sa, dim=-1)

        sim_matrix = roi_feats_sa @ bank_feats_wa.t()  # (N, C) @ (C, M) -> (N, M)
        num_rois = roi_feats_sa.size(0)
        pos_mask = roi_labels.unsqueeze(1) == bank_labels.unsqueeze(0)
        neg_sim_matrix = torch.masked_select(sim_matrix, ~pos_mask).view(num_rois, -1)
        neg_sim_matrix = torch.repeat_interleave(neg_sim_matrix, nppc, dim=0)
        pos_inds = torch.where(pos_mask)
        positives = sim_matrix[pos_inds].view(-1, 1)
        logits = torch.cat((positives, neg_sim_matrix), dim=1)
        logits = logits / self.temperature
        labels = torch.zeros((logits.size(0),), dtype=torch.long).cuda()
        lpcont_loss = self.ce_loss(logits, labels)

        return lpcont_loss, sim_matrix.detach(), bank_labels


class FeatureBank(Metric):
    full_state_update: bool = False

    def __init__(self, **kwargs):

        super().__init__()
        self.tag = kwargs.get('NAME', None)

        self.temperature = kwargs.get('TEMPERATURE')
        self.num_classes = 3
        self.feat_size = kwargs.get('FEATURE_SIZE')
        self.bank_size = kwargs.get('BANK_SIZE')  # e.g., num. of classes or labeled instances
        self.momentum = kwargs.get('MOMENTUM')
        self.direct_update = kwargs.get('DIRECT_UPDATE')
        self.reset_state_interval = kwargs.get('RESET_STATE_INTERVAL')  # reset the state when N unique samples are seen
        self.num_points_thresh = kwargs.get('FILTER_MIN_POINTS_IN_GT', 0)
        self.ce_loss = nn.CrossEntropyLoss()
        self.initialized = False
        self.insId_protoId_mapping = None  # mapping from instance index to prototype index

        # Globally synchronized prototypes used in each process
        self.prototypes = None
        self.classwise_prototypes = None
        self.proto_labels = None
        self.num_updates = None

        # Local feature/label which are used to update the global ones
        self.add_state('feats', default=[], dist_reduce_fx='cat')
        self.add_state('labels', default=[], dist_reduce_fx='cat')
        self.add_state('ins_ids', default=[], dist_reduce_fx='cat')
        self.add_state('smpl_ids', default=[], dist_reduce_fx='cat')
        self.add_state('iterations', default=[], dist_reduce_fx='cat')

    def _init(self, unique_ins_ids, labels):
        self.bank_size = len(unique_ins_ids)
        print(f"Initializing the feature bank with size {self.bank_size} and feature size {self.feat_size}")
        self.prototypes = torch.zeros((self.bank_size, self.feat_size)).cuda()
        self.classwise_prototypes = torch.zeros((3, self.feat_size)).cuda()
        self.proto_labels = labels
        self.num_updates = torch.zeros(self.bank_size).cuda()
        self.insId_protoId_mapping = {unique_ins_ids[i]: i for i in range(len(unique_ins_ids))}

    def update(self, feats: [torch.Tensor], labels: [torch.Tensor], ins_ids: [torch.Tensor], smpl_ids: torch.Tensor,
               iteration: int) -> None:
        for i in range(len(feats)):
            self.feats.append(feats[i])                 # (N, C)
            self.labels.append(labels[i].view(-1))      # (N,)
            self.ins_ids.append(ins_ids[i].view(-1))    # (N,)
            self.smpl_ids.append(smpl_ids[i].view(-1))  # (1,)
            rois_iter = torch.tensor(iteration, device=feats[0].device).expand_as(ins_ids[i].view(-1))
            self.iterations.append(rois_iter)           # (N,)

    def compute(self):
        unique_smpl_ids = torch.unique(torch.cat(self.smpl_ids))
        if len(unique_smpl_ids) < self.reset_state_interval:
            return None

        features = torch.cat(self.feats)
        labels = torch.cat(self.labels).int()
        ins_ids = torch.cat(self.ins_ids).int().cpu().numpy()
        iterations = torch.cat(self.iterations).int().cpu().numpy()
        assert len(features) == len(labels) == len(ins_ids) == len(iterations), \
            "length of features, labels, ins_ids, and iterations should be the same"
        sorted_ins_ids, arg_sorted_ins_ids = np.sort(ins_ids), np.argsort(ins_ids)
        unique_ins_ids, split_indices = np.unique(sorted_ins_ids, return_index=True)

        if not self.initialized:
            self._init(unique_ins_ids, labels[arg_sorted_ins_ids[split_indices]])

        # Group by ins_ids
        inds_groupby_ins_ids = np.split(arg_sorted_ins_ids, split_indices[1:])
        # For each group sort instance ids by iterations in ascending order and apply reduction operation
        for grouped_inds in inds_groupby_ins_ids:
            grouped_inds = grouped_inds[np.argsort(iterations[grouped_inds])]
            ins_id = ins_ids[grouped_inds[0]]
            proto_id = self.insId_protoId_mapping[ins_id]
            assert torch.allclose(labels[grouped_inds[0]], labels[grouped_inds]), "labels should be the same for the same instance id"

            if not self.initialized or self.direct_update:
                self.num_updates[proto_id] += len(grouped_inds)
                new_prototype = torch.mean(features[grouped_inds], dim=0, keepdim=True)  # TODO: maybe it'd be better to replaced it by the EMA
                self.prototypes[proto_id] = new_prototype
            else:
                for ind in grouped_inds:
                    new_prototype = self.momentum * self.prototypes[proto_id] + (1 - self.momentum) * features[ind]
                    self.prototypes[proto_id] = new_prototype
        self._update_classwise_prototypes()
        self.initialized = True
        self.reset()
        return self.prototypes, self.proto_labels, self.num_updates

    def _update_classwise_prototypes(self):
        classwise_prototypes = torch.zeros((3, self.feat_size)).cuda()
        for i in range(self.num_classes):  # TODO: refactor it
            inds = torch.where(self.proto_labels == i)[0]
            print(f"Update classwise prototypes for class {i} with {len(inds)} instances.")
            classwise_prototypes[i] = torch.mean(self.prototypes[inds], dim=0)
        self.classwise_prototypes = self.momentum * self.classwise_prototypes + (1 - self.momentum) * classwise_prototypes

    @torch.no_grad()
    def get_sim_scores(self, input_features, use_classwise_prototypes=True):
        assert input_features.shape[1] == self.feat_size, "input feature size is not equal to the bank feature size"
        if not self.initialized:
            return input_features.new_zeros(input_features.shape[0], 3)
        if use_classwise_prototypes:
            cos_sim = F.normalize(input_features) @ F.normalize(self.classwise_prototypes).t()
            return F.softmax(cos_sim / self.temperature, dim=-1)
        else:
            self._get_sim_scores_with_instance_prototypes(input_features)

    def _get_sim_scores_with_instance_prototypes(self, input_features):
        cos_sim = F.normalize(input_features) @ F.normalize(self.prototypes).t()
        norm_cos_sim = F.softmax(cos_sim / self.temperature, dim=-1)
        classwise_sim = cos_sim.new_zeros(input_features.shape[0], 3)
        lbs = self.proto_labels.expand_as(cos_sim).long()
        classwise_sim.scatter_add_(1, lbs, norm_cos_sim)
        # classwise_sim.scatter_add_(1, lbs, cos_sim)
        # protos_cls_counts = torch.bincount(self.proto_labels).view(1, -1)
        # classwise_sim /= protos_cls_counts  # Note: not probability
        classwise_sim /= classwise_sim.mean(dim=0)
        return classwise_sim

    def get_pairwise_protos_sim_matrix(self):
        sorted_lbs, arg_sorted_lbs = torch.sort(self.proto_labels)
        protos = self.prototypes[arg_sorted_lbs]
        sim_matrix = F.normalize(protos) @ F.normalize(protos).t()

        return sim_matrix.cpu().numpy(), sorted_lbs.cpu().numpy()

    def get_proto_contrastive_loss(self, feats, labels):
        """
        :param feats: pseudo-box features of the strongly augmented unlabeled samples (N, C)
        :param labels: pseudo-labels of the strongly augmented unlabeled samples (N,)
        :return:
        """
        if not self.initialized:
            return None
        sim_scores = F.normalize(feats) @ F.normalize(self.classwise_prototypes).t()
        log_probs = F.log_softmax(sim_scores / self.temperature, dim=-1)
        return -log_probs[torch.arange(len(labels)), labels]

    def is_initialized(self):
        return self.initialized

    def _randomly_sample_protos_by_class(self, num_samples):
        sampled_protos = []
        sampled_labels = []
        for i in range(self.num_classes):
            inds = torch.where(self.proto_labels == i)[0]
            sampled_inds = torch.randperm(len(inds))[:num_samples]
            sampled_protos.append(self.prototypes[inds[sampled_inds]])
            sampled_labels.append(self.proto_labels[inds[sampled_inds]])
        return torch.cat(sampled_protos), torch.cat(sampled_labels)

    def get_lpcont_loss(self, roi_feats_sa, roi_labels):
        # TODO: do we require to filter out the zeros for 'proto' AND 'roi' labels?
        # non_zero_mask = self.proto_labels.long() != 0
        # proto_feats_wa = self.prototypes[non_zero_mask]
        # proto_labels = self.proto_labels[non_zero_mask]

        nppc = 9  # number of prototypes per class
        proto_feats, proto_labels = self._randomly_sample_protos_by_class(nppc)
        proto_feats = F.normalize(proto_feats, dim=-1)
        roi_feats_sa = F.normalize(roi_feats_sa, dim=-1)
        num_rois = roi_feats_sa.size(0)

        sim_matrix = roi_feats_sa @ proto_feats.t()  # (N, C) @ (C, M) -> (N, M)
        pos_mask = roi_labels.unsqueeze(1) == proto_labels.unsqueeze(0)
        neg_sim_matrix = torch.masked_select(sim_matrix, ~pos_mask).view(num_rois, -1)
        neg_sim_matrix = torch.repeat_interleave(neg_sim_matrix, nppc, dim=0)
        pos_inds = torch.where(pos_mask)
        positives = sim_matrix[pos_inds].view(-1, 1)
        logits = torch.cat((positives, neg_sim_matrix), dim=1)
        logits = logits / self.temperature
        labels = torch.zeros((logits.size(0),), dtype=torch.long).cuda()
        lpcont_loss = self.ce_loss(logits, labels)

        return lpcont_loss, sim_matrix.detach(), proto_labels


class FeatureBankRegistry(object):
    def __init__(self, **kwargs):
        self._banks = {}

    def register(self, tag=None, **bank_configs):
        if tag is None:
            tag = 'default'
        if tag in self.tags():
            raise ValueError(f'Feature bank with tag {tag} already exists')
        bank = FeatureBank(**bank_configs)
        self._banks[tag] = bank
        return self._banks[tag]

    def get(self, tag=None):
        if tag is None:
            tag = 'default'
        if tag not in self.tags():
            raise ValueError(f'Feature bank with tag {tag} does not exist')
        return self._banks[tag]

    def tags(self):
        return self._banks.keys()


feature_bank_registry = FeatureBankRegistry()
