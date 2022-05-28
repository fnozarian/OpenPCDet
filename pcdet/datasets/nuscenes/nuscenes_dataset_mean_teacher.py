import copy
from collections import defaultdict
from pathlib import Path

import numpy as np

from ...utils import box_utils, common_utils
from ..nuscenes.nuscenes_dataset import NuScenesDataset

class NuScenesDatasetMeanTeacher(NuScenesDataset):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        root_path = (root_path if root_path is not None else Path(dataset_cfg.DATA_PATH)) / dataset_cfg.VERSION
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.infos = []
        self.include_nuscenes_data(self.mode)
        if self.training and self.dataset_cfg.get('BALANCED_RESAMPLING', False):
            self.infos = self.balanced_infos_resampling(self.infos)

        # if self.training and self.dataset_cfg.get('SAMPLE_FREQUENCY', 1) > 1:
        if self.dataset_cfg.get('SAMPLE_FREQUENCY', 1) > 1:
            self.infos = [x for index, x in enumerate(self.infos) if index%self.dataset_cfg.SAMPLE_FREQUENCY==0]
            print('after frequency sampling:', len(self.infos))

    

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)

        info = copy.deepcopy(self.infos[index])
        points = self.get_lidar_with_sweeps(index, max_sweeps=self.dataset_cfg.MAX_SWEEPS)

        input_dict = {
            'points': points,
            'frame_id': Path(info['lidar_path']).stem,
            'metadata': {'token': info['token']}
        }
        if self.dataset_cfg.Provide_GT:
            if 'gt_boxes' in info:
                if self.dataset_cfg.get('FILTER_MIN_POINTS_IN_GT', False):
                    mask = (info['num_lidar_pts'] > self.dataset_cfg.FILTER_MIN_POINTS_IN_GT - 1)
                else:
                    mask = None

                input_dict.update({
                    'gt_names': info['gt_names'] if mask is None else info['gt_names'][mask],
                    'gt_boxes': info['gt_boxes'] if mask is None else info['gt_boxes'][mask],
                    'num_points_in_gt': info['num_lidar_pts'] if mask is None else info['num_lidar_pts'][mask]
                })

        if self.dataset_cfg.SAME_POINT_SAMPLING:
            if self.pre_data_augmentor:
                gt_boxes_mask = np.array([n in self.class_names for n in input_dict['gt_names']], dtype=np.bool_)
                input_dict['gt_boxes_mask'] = gt_boxes_mask
                input_dict = self.pre_data_augmentor.forward(data_dict=input_dict)
            
            data_dict1 = self.process_data(input_dict)
            data_dict2 = copy.deepcopy(data_dict1)
            augment_student = self.dataset_cfg.AUGMENT_STUDENT
            augment_teacher = self.dataset_cfg.AUGMENT_TEACHER
            data_dict1 = self.prepare_data(data_dict=data_dict1, augment=augment_student, process_data=False)
            data_dict2 = self.prepare_data(data_dict=data_dict2, augment=augment_teacher, process_data=False)
            data_dict1['metadata'] = info.get('metadata', Path(info['lidar_path']).stem)
            data_dict2['metadata'] = info.get('metadata', Path(info['lidar_path']).stem)
            data_dict1.pop('num_points_in_gt', None)
            data_dict2.pop('num_points_in_gt', None)
        else:
            raise NotImplementedError
        
        output = [data_dict1, data_dict2]
        return output

    

    def prepare_data(self, data_dict, augment=True, process_data=True):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in) # lidar points
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...] # lidar boxes
                gt_names: optional, (N), string
                process_data: if False, pass in processed data
                ...

        Returns:
            data_dict:
                frame_id: string
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                use_lead_xyz: bool
                voxels: optional (num_voxels, max_points_per_voxel, 3 + C)
                voxel_coords: optional (num_voxels, 3)
                voxel_num_points: optional (num_voxels)
                ...
        """
        if self.training:
            if self.dataset_cfg.PROVIDE_GT:
                assert 'gt_boxes' in data_dict, 'gt_boxes should be provided for training'
                gt_boxes_mask = np.array([n in self.class_names for n in data_dict['gt_names']], dtype=np.bool_)
            else:
                gt_boxes_mask = None

            data_dict = self.data_augmentor.forward(
                data_dict={
                    **data_dict,
                    'gt_boxes_mask': gt_boxes_mask
                },
                augment=augment
            )

            if 'gt_boxes' not in data_dict:
                data_dict.pop('gt_boxes_mask', None)

        if self.dataset_cfg.PROVIDE_GT and data_dict.get('gt_boxes', None) is not None:
            selected = common_utils.keep_arrays_by_name(data_dict['gt_names'], self.class_names)
            data_dict['gt_boxes'] = data_dict['gt_boxes'][selected]
            data_dict['gt_names'] = data_dict['gt_names'][selected]
            gt_classes = np.array([self.class_names.index(n) + 1 for n in data_dict['gt_names']], dtype=np.int32)
            gt_boxes = np.concatenate((data_dict['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
            #data_dict['gt_boxes'] = gt_boxes
            #gt_boxes = data_dict['gt_boxes']
            gt_boxes[np.isnan(gt_boxes)] = 0 # filter nan (SET_NAN_VELOCITY_TO_ZEROS)
            data_dict['gt_boxes'] = gt_boxes

        
        if process_data:
            data_dict = self.point_feature_encoder.forward(data_dict)
            data_dict = self.data_processor.forward(
                data_dict=data_dict
            )

        data_dict.pop('gt_names', None)

        return data_dict

    def process_data(self, data_dict):
        data_dict = self.point_feature_encoder.forward(data_dict)
        data_dict = self.data_processor.forward(
            data_dict=data_dict
        )
        return data_dict


    @staticmethod
    def collate_batch(batch_list, _unused=False):
        def collate_fn(batch_list):
            data_dict = defaultdict(list)
            for cur_sample in batch_list:
                for key, val in cur_sample.items():
                    data_dict[key].append(val)
            batch_size = len(batch_list)
            ret = {}

            for key, val in data_dict.items():
                try:
                    if key in ['voxels', 'voxel_num_points']:
                        ret[key] = np.concatenate(val, axis=0)
                    elif key in ['points', 'voxel_coords']:
                        coors = []
                        for i, coor in enumerate(val):
                            coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                            coors.append(coor_pad)
                        ret[key] = np.concatenate(coors, axis=0)
                    elif key in ['gt_boxes']:
                        max_gt = max([len(x) for x in val])
                        batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                        for k in range(batch_size):
                            batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                        ret[key] = batch_gt_boxes3d
                    else:
                        ret[key] = np.stack(val, axis=0)
                except:
                    print('Error in collate_batch: key=%s' % key)
                    raise TypeError

            ret['batch_size'] = batch_size
            return ret

        if isinstance(batch_list[0], dict):
            return collate_fn(batch_list)

        else:
            assert isinstance(batch_list[0], list)
            batch_list1 = [x[0] for x in batch_list]
            batch_list2 = [x[1] for x in batch_list]
            ret1 = collate_fn(batch_list1)
            ret2 = collate_fn(batch_list2)
            return [ret1, ret2]

   