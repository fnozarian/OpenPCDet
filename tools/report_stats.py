import argparse
from pcdet.datasets import build_dataloader
from pcdet.config import cfg, cfg_from_yaml_file
import tqdm
import numpy as np

parser = argparse.ArgumentParser(description='arg parser')
parser.add_argument('--cfg_file', type=str, default='cfgs/carla_kitti_models/pointpillar_default.yaml',
                    help='specify the config for training')

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

train_iter = iter(train_loader)
total_iters = len(train_loader)
pbar = tqdm.tqdm(total=total_iters, desc='calc stats', dynamic_ncols=True)

gt_boxes = []
for _ in range(len(train_iter)):
    try:
        batch = next(train_iter)
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