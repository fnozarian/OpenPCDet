import argparse
from pcdet.datasets import build_dataloader
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets.augmentor.data_augmentor import DataAugmentor
from tools.visual_utils.visualize_utils import draw_scene_open3d

import random
import tqdm
import numpy as np
from pathlib import Path

def vis_obj_instances(cfg, instance_types='Car', num_points_in_gt_threshold=300, shuffle=True):
    data_augmentor = DataAugmentor(Path(cfg.DATA_CONFIG.DATA_PATH), cfg.DATA_CONFIG.DATA_AUGMENTOR, [instance_types])
    gt_sampler = data_augmentor.gt_sampling(cfg.DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST[0])
    objs_infos = gt_sampler.db_infos[instance_types]
    points_list = []
    for obj_infos in objs_infos:
        if obj_infos['num_points_in_gt'] > num_points_in_gt_threshold:
            gt_pc_path = Path(cfg.DATA_CONFIG.DATA_PATH) / obj_infos['path']
            points = np.fromfile(gt_pc_path, dtype=np.float32).reshape(-1, 4)
            points_list.append(points)

    if shuffle:
        random.shuffle(points_list)

    draw_scene_open3d(points_list)

def calculate_stats(data_loader):

    total_iters = len(data_loader)
    loader_iter = iter(data_loader)
    pbar = tqdm.tqdm(total=total_iters, desc='calc stats', dynamic_ncols=True)
    gt_boxes = []
    for _ in range(total_iters):
        try:
            batch = next(loader_iter)
            pbar.update()
        except StopIteration:
            print('end of loader')
        boxes = batch['gt_boxes'][0]
        gt_boxes.append(boxes)

    all_gt_boxes = np.concatenate(gt_boxes, axis=0)
    carla_kitti_anchor_sizes = np.mean(all_gt_boxes[:, 3:6], axis=0)
    kitti_anchor_sizes = np.array([[3.9, 1.6, 1.56]])
    mean_loc_z = np.mean(all_gt_boxes[:, 2])
    mean_dims_ratios = kitti_anchor_sizes / carla_kitti_anchor_sizes

    print("mean anchor_sizes: ", carla_kitti_anchor_sizes)
    print("mean anchor_bottom_heights: ", 2 * mean_loc_z)
    print("mean kitti/carla-kitti ratio: ", mean_dims_ratios)
    print("lower bound of random_world_scaling:", np.min(mean_dims_ratios, axis=1))

    return_dict = {'anchor_sizes': carla_kitti_anchor_sizes,
                   'anchor_bottom_heights': 2 * mean_loc_z }

    return return_dict

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/carla_kitti_models/'
                                                        'pointpillar_default_filter_min_points_gt_report_stats.yaml',
                        help='specify the config for training')
    # parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/pointpillar.yaml',
    #                     help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=1, required=False, help='batch size for training')
    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)

    # -----------------------create dataloader & network & optimizer---------------------------
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=['Car'],
        batch_size=args.batch_size,
        dist=False, workers=8,
        training=True)

    if cfg.get('TARGET_DATA_CONFIG', False):
        tar_train_set, tar_train_loader, sampler = build_dataloader(
            dataset_cfg=cfg.TARGET_DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            batch_size=args.batch_size,
            dist=False, workers=8,
            training=True
        )

    # calculate_stats(train_loader)
    vis_obj_instances(cfg)