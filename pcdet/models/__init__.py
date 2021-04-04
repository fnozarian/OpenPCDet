from collections import namedtuple

import numpy as np
import torch

from .detectors import build_detector


def build_network(model_cfg, num_class, dataset):
    model = build_detector(
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )
    return model


def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        if key in ['frame_id', 'metadata', 'calib', 'image_shape']:
            continue
        batch_dict[key] = torch.from_numpy(val).float().cuda()


def model_fn_decorator():
    ModelReturn = namedtuple('ModelReturn', ['loss', 'tb_dict', 'disp_dict'])

    def model_func(model, batch_dict):
        load_data_to_gpu(batch_dict)
        ret_dict, tb_dict, disp_dict = model(batch_dict)

        loss = ret_dict['loss'].mean()
        if hasattr(model, 'update_global_step'):
            model.update_global_step()
        else:
            model.module.update_global_step()

        return ModelReturn(loss, tb_dict, disp_dict)

    return model_func


def model_fn_decorator_da():
    ModelReturn = namedtuple('ModelReturn', ['loss', 'tb_dict', 'disp_dict'])

    def model_func(model, batch_dict, batch_dict_target=None, target=False):

        # TODO(farzad) Batch combine doesn't work as pointrcnn requires samples of batch to have the same number of points.
        #  Mean num. of points in nsuc is ~ 2k while in kitti ~16k so we can't fill the rest of nusc points with resampling.
        if batch_dict_target:
            assert batch_dict.keys() == batch_dict_target.keys()

            batch_size = batch_dict['batch_size']
            combined_batch = {'batch_size': 2 * batch_size}
            for key in batch_dict.keys():
                val_s = batch_dict[key]
                val_t = batch_dict_target[key]

                if key in ['points']:
                    combined_points = np.concatenate([val_s, val_t], axis=0)
                    combined_points[len(val_s):, 0] = val_t[:, 0] + batch_size
                    combined_batch[key] = combined_points
                elif key in ['gt_boxes']:
                    max_gt = max([len(x) for x in [val_s[0], val_t[0]]])
                    combined_gt_boxes3d = np.zeros((2 * batch_size, max_gt, val_t[0].shape[-1]), dtype=np.float32)
                    for i in range(batch_size):
                        combined_gt_boxes3d[i, :val_s[i].__len__(), :] = val_s[i]
                    for i in range(batch_size):
                        combined_gt_boxes3d[i + batch_size, :val_t[i].__len__(), :] = val_t[i]
                    combined_batch[key] = combined_gt_boxes3d
                elif key in ['is_source']:
                    max_gt = max([len(x) for x in [val_s[0], val_t[0]]])
                    combined_gt_domain_labels = np.ones((2 * batch_size, max_gt), dtype=np.uint8) * -1

                    for i in range(batch_size):
                        combined_gt_domain_labels[i, :val_s[i].__len__()] = val_s[i]
                    for i in range(batch_size):
                        combined_gt_domain_labels[i + batch_size, :val_t[i].__len__()] = val_t[i]
                    combined_batch[key] = combined_gt_domain_labels
                elif isinstance(val_s, np.ndarray):
                    combined_batch[key] = np.concatenate([val_s, val_t], axis=0)

            batch_dict = combined_batch

        load_data_to_gpu(batch_dict)
        ret_dict, tb_dict, disp_dict = model(batch_dict)

        loss = ret_dict['loss']
        if target:
            if hasattr(model, 'update_global_step'):
                model.update_global_step()
            else:
                model.module.update_global_step()

        return ModelReturn(loss, tb_dict, disp_dict)

    return model_func
