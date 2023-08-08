import torch
from torch.functional import F
from torchmetrics import Metric
import numpy as np


class FeatureBank(Metric):
    full_state_update: bool = False

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.tag = kwargs.get('NAME', None)

        self.temperature = kwargs.get('TEMPERATURE')
        self.feat_size = kwargs.get('FEATURE_SIZE')
        self.bank_size = kwargs.get('BANK_SIZE')  # e.g., num. of classes or labeled instances
        self.momentum = kwargs.get('MOMENTUM')
        self.direct_update = kwargs.get('DIRECT_UPDATE')
        self.reset_state_interval = kwargs.get('RESET_STATE_INTERVAL')  # reset the state when N unique samples are seen

        self.initialized = False
        self.insId_protoId_mapping = None  # mapping from instance index to prototype index

        # Globally synchronized prototypes used in each process
        self.prototypes = None
        self.proto_labels = None
        self.num_updates = None

        # Local feature/label which are used to update the global ones
        self.add_state('feats', default=[], dist_reduce_fx='cat')
        self.add_state('labels', default=[], dist_reduce_fx='cat')
        self.add_state('ins_ids', default=[], dist_reduce_fx='cat')
        self.add_state('smpl_ids', default=[], dist_reduce_fx='cat')
        self.add_state('iterations', default=[], dist_reduce_fx='cat')

    def _init(self, unique_ins_ids):
        self.bank_size = len(unique_ins_ids)
        print(f"Initializing the feature bank with size {self.bank_size} and feature size {self.feat_size}")
        self.prototypes = torch.zeros((self.bank_size, self.feat_size)).cuda()
        self.proto_labels = -torch.ones(self.bank_size).cuda()
        self.num_updates = torch.zeros(self.bank_size).cuda()
        self.insId_protoId_mapping = {unique_ins_ids[i]: i for i in range(len(unique_ins_ids))}
    def update(self, feats: torch.Tensor, labels: torch.Tensor, ins_ids: torch.Tensor, smpl_ids: torch.Tensor,
               iteration: int) -> None:
        self.feats.append(feats.view(-1, self.feat_size))  # (BxN, C)
        self.labels.append(labels.view(-1))      # (BxN,)
        self.ins_ids.append(ins_ids.view(-1))    # (BxN,)
        self.smpl_ids.append(smpl_ids.view(-1))  # (B,)
        rois_iter = torch.tensor(iteration, device=feats[0].device).expand_as(ins_ids.view(-1))
        self.iterations.append(rois_iter)        # (BxN,)

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
        sorted_ins_ids, inds = np.sort(ins_ids), np.argsort(ins_ids)
        unique_ins_ids, split_indices = np.unique(sorted_ins_ids, return_index=True)

        if not self.initialized:
            self._init(unique_ins_ids)

        # Groupby ins_ids
        inds_groupby_ins_ids = np.split(inds, split_indices[1:])
        # For each group sort instance ids by iterations in ascending order and apply reduction operation
        for grouped_inds in inds_groupby_ins_ids:
            inds = grouped_inds[np.argsort(iterations[grouped_inds])]
            ins_id = ins_ids[inds[0]]
            proto_id = self.insId_protoId_mapping[ins_id]
            assert torch.allclose(labels[inds[0]], labels[inds]), "labels should be the same for the same instance id"

            if not self.initialized or self.direct_update:
                self.proto_labels[proto_id] = labels[inds[0]] - 1
                self.num_updates[proto_id] += len(grouped_inds)
                new_prototype = torch.mean(features[inds], dim=0, keepdim=True)  # TODO: maybe it'd be better to replaced it by the EMA
                self.prototypes[proto_id] = new_prototype
            else:
                for ind in inds:
                    new_prototype = self.momentum * self.prototypes[proto_id] + (1 - self.momentum) * features[ind]
                    self.prototypes[proto_id] = new_prototype

        self.initialized = True
        self.reset()
        return self.prototypes, self.proto_labels, self.num_updates
    def get_sim_scores(self, input_features):
        assert input_features.shape[1] == self.feat_size, "input feature size is not equal to the bank feature size"
        if not self.initialized:
            return input_features.new_zeros(input_features.shape[0], 3)
        cos_sim = F.normalize(input_features) @ F.normalize(self.prototypes).t()
        norm_cos_sim = F.softmax(cos_sim / self.temperature, dim=-1)
        classwise_sim = cos_sim.new_zeros(input_features.shape[0], 3)
        lbls = self.proto_labels.expand_as(cos_sim).long()
        classwise_sim.scatter_add_(1, lbls, norm_cos_sim)
        return F.softmax(classwise_sim / self.temperature, dim=-1)

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