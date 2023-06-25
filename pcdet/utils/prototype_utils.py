import torch
import torch.distributed as dist
import pickle

class Prototype(object):
    def __init__(self,**kwargs):
        self.file = kwargs.get('file',None)
        self.classes  = ['Car','Ped','Cyc']
        self.momentum = 0.9
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
    
    def update(self,features,labels,iter):
        self.features.extend(features)
        self.labels.extend(labels)
        # Computer EMA
        if ((iter+1)%20) == 0:
            print("20th iter")
            if len(self.features)!= 0:
                print("Computing EMA")
                # Gather the tensors (shares tensor among all GPUs)
                features_to_gather = torch.cat(self.features, dim=0).detach().clone() # convert to tensor before gather
                labels_to_gather = torch.cat(self.labels, dim=0).detach().clone()
                print(f"gathering features {features_to_gather.shape}")
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
        
