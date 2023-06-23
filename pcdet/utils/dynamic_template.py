import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.stats import norm
import pickle
from torchmetrics import Metric
from pcdet.config import cfg

class Prototype(Metric):
    full_state_update: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tag = kwargs.get('tag', None)
        self.dataset = kwargs.get('dataset', None)
        self.quantile = kwargs.get('quantile', False)
        self.momentum = kwargs.get('momentum', 0.90)
        # self.template_momentum = kwargs.get('template_momentum',0.8)
        self.enable_clipping = kwargs.get('enable_clipping', True)
        self.metrics_name = ['batchwise_mean','batchwise_variance','ema_mean','ema_variance']
        self.reset_state_interval = kwargs.get('reset_state_interval', 20)
        # INITIALIZATION SIMILAR TO PREDQUALITYMETRIC
        if self.dataset is not None:
            self.class_names  = self.dataset.class_names
            if isinstance(self.class_names, list):
                self.class_names = {i: self.dataset.class_names[i] for i in range(len(self.dataset.class_names))}
            self.num_classes = len(self.dataset.class_names)
        else:
            self.class_names = {0: 'Car', 1: 'Pedestrian', 2: 'Cyclist'}
            self.num_classes = 3
        self.state_list = ['car_template','ped_template','cyc_template','templates','labels']    
        for cls in self.state_list:
            self.add_state(cls, default=[],dist_reduce_fx='cat')
        # Load the Base Prototype
        self.classes = ['Car','Ped','Cyc']
        with open('ema_cls_sh.pkl','rb') as f:
            self.rcnn_features = pickle.loads(f.read())
        rcnn_sh_mean = []
        avg = "mean"
        param = "sh"
        for cls in self.classes:
            rcnn_sh_mean.append(self.rcnn_features[cls][avg][param].unsqueeze(dim=0).detach().cpu())
        self.rcnn_sh_mean_ = torch.stack(rcnn_sh_mean).cuda()  #TODO: check if this device assignment is correct
        self.rcnn_sh_mean = self.rcnn_sh_mean_.detach().clone()

    def update(self,templates=None,labels=None):
        if templates is not None:
            if templates.ndim == 1: # unsqueeze for DDP
                templates = templates.unsqueeze(dim=0)
            self.templates.append(templates)
        if labels is not None:
            if labels.ndim == 1: # unsqueeze for DDP
                labels = labels.unsqueeze(dim=0)
            self.labels.append(labels)
                 
    def compute(self):
        batch_template = {cls:None for cls in self.classes} 
        if len(self.templates) == 0:
            return self.rcnn_sh_mean
        assert self.templates[0].ndim == 3, "Templates should be of shape (N,1,256)"
        templates = torch.cat(self.templates,dim=0)
        assert self.labels[0].ndim == 2, "Templates should be of shape (1,N)"
        labels = torch.cat(self.labels,dim=1).squeeze(dim=0) # squeeze the extra dim to create mask
        for cls in self.class_names.keys():
            mask = labels == (cls+1)
            if torch.any(mask):
                batch_template[self.class_names[cls]] = templates[mask].mean(dim=0) # mean the acculumated templates classwise 
                self.rcnn_sh_mean[cls] = self.momentum * self.rcnn_sh_mean[cls] + (1-self.momentum) * batch_template[self.class_names[cls]]   #calculate the ema of the prototype           
                # cls is the cls index, self.class_names[cls] is the class name
                # REFER: self.class_names = {0: 'Car', 1: 'Pedestrian', 2: 'Cyclist'}        
        self.reset()
        return self.rcnn_sh_mean
        
        
            

            

        
                

