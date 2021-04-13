import numpy as np
from pcdet.datasets.carla_kitti.carla_kitti_dataset import CarlaKittiDataset


class CarlaKittiDatasetDA(CarlaKittiDataset):

    def __init__(self, dataset_cfg, class_names, **kwargs):
        super(CarlaKittiDatasetDA, self).__init__(dataset_cfg, class_names, **kwargs)
        self.real_name = dataset_cfg.get('REAL_NAME') if dataset_cfg.get('REAL_NAME', False) else 'KITTI'
        self.is_source = dataset_cfg.get('IS_SOURCE', True)

    def __getitem__(self, index):
        data_dict = super(CarlaKittiDatasetDA, self).__getitem__(index)
        data_dict['is_source'] = self.is_source

        return data_dict
