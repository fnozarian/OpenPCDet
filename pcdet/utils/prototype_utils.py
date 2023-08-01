import torch
import torch.distributed as dist
import pickle
from torch.functional import F
from torchmetrics import Metric
import numpy as np

def _detach(tensor: [torch.Tensor] = None) -> [torch.Tensor]:
    return [t.clone().detach() for t in tensor]

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
        self.update_interval = kwargs.get('UPDATE_INTERVAL')

        self.initialized = False
        self.insId_protoId_mapping = None  # mapping from instance index to prototype index

        # Globally synchronized prototypes used in each process
        self.prototypes = torch.randn((self.bank_size, self.feat_size)).cuda()
        self.proto_labels = -torch.ones(self.bank_size).cuda()
        self.num_updates = torch.zeros(self.bank_size).cuda()

        # Local feature/label banks which are used to update the global banks
        self.add_state('feats', default=[], dist_reduce_fx='cat')
        self.add_state('labels', default=[], dist_reduce_fx='cat')
        self.add_state('ins_ids', default=[], dist_reduce_fx='cat')
        self.add_state('iterations', default=[], dist_reduce_fx='cat')
        self.add_state('num_samples', default=torch.tensor(0), dist_reduce_fx='sum')
    def update(self, feats: [torch.Tensor], labels: [torch.Tensor], ins_ids: [torch.Tensor], iteration: int) -> None:
        feats = _detach(feats)
        labels = _detach(labels)
        ins_ids = _detach(ins_ids)
        for i in range(len(feats)):
            self.feats.append(feats[i])
            self.labels.append(labels[i])
            self.ins_ids.append(ins_ids[i])
            self.iterations.append(torch.tensor(iteration, device=feats[0].device).expand_as(ins_ids[i]))
        self.num_samples += len(feats)

    def compute(self):
        if self.num_samples < self.reset_state_interval:
            return None
        features = torch.cat(self.feats)
        features = F.normalize(features, dim=2).view(-1, self.feat_size)  # normalize along the feature dim
        labels = torch.cat(self.label).view(-1)
        ins_ids = torch.cat(self.instance_ids).view(-1)
        iterations = torch.cat(self.iterations).view(-1)

        sorted_ins_ids, inds = torch.sort(ins_ids)
        unique_ins_ids, split_indices = np.unique(sorted_ins_ids, return_index=True)

        if not self.initialized:
            assert len(unique_ins_ids) == self.bank_size, "Assumes the number of unique indices is equal to the bank size"
            self.insId_protoId_mapping = {unique_ins_ids[i]: i for i in range(len(unique_ins_ids))}

        # Groupby ins_ids
        inds_groupby_ins_ids = np.split(inds, split_indices[1:])

        # For each group soft instance ids by iterations in ascending order and apply reduction operation
        for grouped_inds in inds_groupby_ins_ids:
            inds = grouped_inds[np.argsort(iterations[grouped_inds])]
            ins_id = ins_ids[inds[0]]
            proto_id = self.insId_protoId_mapping[ins_id]
            self.proto_labels[proto_id] = labels[inds[0]]
            self.num_updates[proto_id] += len(grouped_inds)

            if not self.initialized or self.direct_update:
                new_prototype = torch.mean(features[inds], dim=-1)  # TODO: maybe it'd be better to replaced it by the EMA
                new_prototype = F.normalize(new_prototype, dim=-1)
                self.prototypes[proto_id] = new_prototype
                self.initialized = True
            else:
                for ind in inds:
                    new_prototype = self.momentum * self.prototypes[proto_id] + (1 - self.momentum) * features[ind]
                    self.prototypes[proto_id] = F.normalize(new_prototype, dim=-1)

        return self.prototypes, self.proto_labels, self.num_updates

    def get_sim_scores(self, input_features):
        assert input_features.shape[1] == self.feat_size, "input feature size is not equal to the bank feature size"
        assert self.initialized, "prototypes are not initialized"
        cos_sim = F.normalize(input_features) @ F.normalize(self.prototypes).t()
        return F.softmax(cos_sim / self.temperature, dim=-1)


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
    def update(self, global_iteration, **kwargs):
        for tag, bank in self._banks.items():
            if global_iteration % bank.bank_config['update_interval'] == 0:
                pass

    def get(self, tag=None):
        if tag is None:
            tag = 'default'
        if tag not in self.tags():
            raise ValueError(f'Feature bank with tag {tag} does not exist')
        return self._banks[tag]

    def tags(self):
        return self._banks.keys()

feature_bank_registry = FeatureBankRegistry()

class Prototype(object):
    def __init__(self,**kwargs):
        self.file = kwargs.get('file',None)
        self.tag = kwargs.get('tag', None)
        self.classes  = ['Car','Ped','Cyc']
        self.momentum = 0.9
        assert self.file is not None, "File path to the prototype is not provided"
        with open(self.file,'rb') as f:
            self.rcnn_features = pickle.loads(f.read())
        rcnn_sh_mean = []
        avg = "mean"
        param = "sh"
        for cls in self.classes:
            rcnn_sh_mean.append(self.rcnn_features[cls][avg][param].unsqueeze(dim=0).detach().cpu())
        self.rcnn_sh_mean_ = torch.stack(rcnn_sh_mean).cuda()  #TODO: check if this device assignment is correct
        self.rcnn_sh_mean = self.rcnn_sh_mean_.detach().clone()
        self.features = []
        self.labels = []
        self.reset_state_interval = 20 #TODO:Deepika make this configurable
    
    def update(self,features,labels,iter):
        self.features.extend(features) #NOTE: if features is [], extend will not affect the self.features
        self.labels.extend(labels)
        # Compute EMA
        if ((iter+1) % self.reset_state_interval) == 0: 
            print("20th iter")
            if len(self.features)!= 0:
                print("Computing EMA")
                # Gather the tensors (shares tensor among all GPUs)
                features_to_gather = torch.cat(self.features, dim=0).detach().clone() # convert to tensor before gather
                labels_to_gather = torch.cat(self.labels, dim=0).detach().clone()
                print(f"gathering features {features_to_gather.shape} in {self.tag}")
                gathered_features = self.gather_tensors(features_to_gather) # Gather tensors from all GPUs
                gathered_labels = self.gather_tensors(labels_to_gather,labels=True) 
                print(f'gathered_features {gathered_features.shape}')
                # Do EMA update of prototype
                for cls in range(0,3):
                    cls_mask = gathered_labels == (cls+1)
                    if torch.any(cls_mask): 
                        cls_features_mean = (gathered_features[cls_mask]).mean(dim=0)                    
                        self.rcnn_sh_mean[cls] = (self.momentum*self.rcnn_sh_mean[cls]) + ((1-self.momentum)*cls_features_mean)

                # Reset the lists         
                self.features = []
                self.labels = []

        return self.rcnn_sh_mean


    
    def gather_tensors(self,tensor,labels=False):
        """
        Returns the gathered tensor to all GPUs in DDP else returns the tensor as such
        dist.gather_all needs the gathered tensors to be of same size.
        We get the sizes of the tensors first, zero pad them to match the size
        Then gather and filter the padding

        Args:
            tensor: tensor to be gathered
            labels: bool True if the tensor represents label information TODO:Deepika Remove this arg and make function tensor agnostic 
        """
        if labels:
            assert tensor.ndim == 1,"labels should be of shape 1"
        else:
            assert tensor.ndim == 3,"features should be of shape N,1,256"

        if dist.is_initialized(): # check if dist mode is initialized
            # Determine sizes first
            local_size = torch.tensor(tensor.size(), device=tensor.device)
            WORLD_SIZE = dist.get_world_size()
            all_sizes = [torch.zeros_like(local_size) for _ in range(WORLD_SIZE)]
            dist.barrier() 
            dist.all_gather(all_sizes,local_size)
            dist.barrier()
            
            # make zero-padded version https://stackoverflow.com/questions/71433507/pytorch-python-distributed-multiprocessing-gather-concatenate-tensor-arrays-of
            max_length = max([size[0] for size in all_sizes])
            if max_length != local_size[0].item():
                diff = max_length - local_size[0].item()
                pad_size =[diff.item()] #pad with zeros 
                if local_size.ndim >= 1:
                    pad_size.extend(dimension.item() for dimension in local_size[1:])
                padding = torch.zeros(pad_size, device=tensor.device, dtype=tensor.dtype)
                tensor = torch.cat((tensor,padding))
            
            all_tensors_padded = [torch.zeros_like(tensor) for _ in range(WORLD_SIZE)]
            dist.barrier()
            dist.all_gather(all_tensors_padded,tensor)
            dist.barrier()
            gathered_tensor = torch.cat(all_tensors_padded)
            if gathered_tensor.ndim == 1: # diff filtering mechanism for labels TODO:Deepika make this tensor agnostic
                assert gathered_tensor.ndim == 1, "Label dimension should be N"
                non_zero_mask = gathered_tensor > 0
            else:
                non_zero_mask = torch.any(gathered_tensor!=0,dim=-1).squeeze()
            gathered_tensor = gathered_tensor[non_zero_mask]
            return gathered_tensor
        else:
            return tensor
        
