import copy
import os
from collections import defaultdict
from typing import Tuple, List

import numpy as np
import tqdm
from PIL import Image
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from nuscenes.utils.kitti import KittiDB
from pyquaternion import Quaternion

from pcdet.datasets import NuScenesDataset
from pcdet.utils import box_utils
from pcdet.utils import common_utils
from pcdet.datasets.kitti.kitti_object_eval_python.kitti_common import add_difficulty_to_annos

from pathlib import Path


def postprocessing(objs, height, width):
    _map = np.ones((height, width), dtype=np.int8) * -1
    objs = sorted(objs, key=lambda x: x["depth"], reverse=True)
    for i, obj in enumerate(objs):
        _map[int(round(obj["bbox"][1])):int(round(obj["bbox"][3])), int(round(obj["bbox"][0])):int(round(obj["bbox"][2]))] = i
    unique, counts = np.unique(_map, return_counts=True)
    counts = dict(zip(unique, counts))
    for i, obj in enumerate(objs):
        if i not in counts.keys():
            counts[i] = 0
        occlusion = 1.0 - counts[i] / (obj["bbox"][3] - obj["bbox"][1]) / (obj["bbox"][2] - obj["bbox"][0])
        obj["occluded"] = int(np.clip(occlusion * 4, 0, 3))

    return objs


def project_kitti_box_to_image(box, p_left, height, width):
    box = box.copy()

    # KITTI defines the box center as the bottom center of the object.
    # We use the true center, so we need to adjust half height in negative y direction.
    box.translate(np.array([0, -box.wlh[2] / 2, 0]))

    # Check that some corners are inside the image.
    corners = np.array([corner for corner in box.corners().T if corner[2] > 0]).T
    if len(corners) == 0:
        return None

    # Project corners that are in front of the camera to 2d to get bbox in pixel coords.
    imcorners = view_points(corners, p_left, normalize=True)[:2]
    bbox = (np.min(imcorners[0]), np.min(imcorners[1]), np.max(imcorners[0]), np.max(imcorners[1]))

    inside = (0 <= bbox[1] < height and 0 < bbox[3] <= height) and (0 <= bbox[0] < width and 0 < bbox[2] <= width)
    valid = (0 <= bbox[1] < height or 0 < bbox[3] <= height) and (0 <= bbox[0] < width or 0 < bbox[2] <= width)
    if not valid:
        return None

    truncated = valid and not inside
    if truncated:
        _bbox = [0] * 4
        _bbox[0] = max(0, bbox[0])
        _bbox[1] = max(0, bbox[1])
        _bbox[2] = min(width, bbox[2])
        _bbox[3] = min(height, bbox[3])

        truncated = 1.0 - ((_bbox[2] - _bbox[0]) * (_bbox[3] - _bbox[1])) / ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        bbox = _bbox
    else:
        truncated = 0.0
    return {"bbox": bbox, "truncated": truncated}


CLASS_MAP = {
#    "bicycle": "Cyclist",
#    "bus": "Misc",
    "car": "Car",
#    "construction_vehicle": "Misc",
#    "motorcycle": "Misc",
#    "pedestrian": "Pedestrian",
#    "trailer": "Misc",
#    "truck": "Misc",
#    "barrier": "Misc",
#    "traffic_cone": "Misc"
}


class NuScenesDatasetKITTIEval(NuScenesDataset):

    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(dataset_cfg, class_names, training, root_path, logger)

        self.cam_name = 'CAM_FRONT'
        self.lidar_name = 'LIDAR_TOP'
        self.image_count = 10
        self.imsize = (1600, 900)
        self.nusc_kitti_dir = './nus_kitti'
        self.split = 'training'

    def get_infos(self, num_workers=4, has_label=True, count_inside_pts=True, sample_id_list=None):
        import concurrent.futures as futures

        def process_single_scene(sample_idx):
            print('%s sample_idx: %s' % (self.split, sample_idx))
            info = {}
            pc_info = {'num_features': 4, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            image_info = {'image_idx': sample_idx, 'image_shape': self.get_image_shape(sample_idx)}
            info['image'] = image_info
            calib = self.get_calib(sample_idx)

            P2 = np.concatenate([calib.P2, np.array([[0., 0., 0., 1.]])], axis=0)
            R0_4x4 = np.zeros([4, 4], dtype=calib.R0.dtype)
            R0_4x4[3, 3] = 1.
            R0_4x4[:3, :3] = calib.R0
            V2C_4x4 = np.concatenate([calib.V2C, np.array([[0., 0., 0., 1.]])], axis=0)
            calib_info = {'P2': P2, 'R0_rect': R0_4x4, 'Tr_velo_to_cam': V2C_4x4}

            info['calib'] = calib_info

            if has_label:
                obj_list = self.get_label(sample_idx)
                if len(obj_list) == 0:
                    print(f"no gt object available for sample {sample_idx}")
                    return
                annotations = {}
                annotations['name'] = np.array([obj.cls_type for obj in obj_list])
                annotations['truncated'] = np.array([obj.truncation for obj in obj_list])
                annotations['occluded'] = np.array([obj.occlusion for obj in obj_list])
                annotations['alpha'] = np.array([obj.alpha for obj in obj_list])
                annotations['bbox'] = np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0)
                annotations['dimensions'] = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])  # lhw(camera) format
                annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
                annotations['rotation_y'] = np.array([obj.ry for obj in obj_list])
                annotations['score'] = np.array([obj.score for obj in obj_list])
                annotations['difficulty'] = np.array([obj.level for obj in obj_list], np.int32)

                num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
                num_gt = len(annotations['name'])
                index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
                annotations['index'] = np.array(index, dtype=np.int32)

                loc = annotations['location'][:num_objects]
                dims = annotations['dimensions'][:num_objects]
                rots = annotations['rotation_y'][:num_objects]
                loc_lidar = calib.rect_to_lidar(loc)
                l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
                loc_lidar[:, 2] += h[:, 0] / 2
                gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
                annotations['gt_boxes_lidar'] = gt_boxes_lidar

                info['annos'] = annotations

                if count_inside_pts:
                    points = self.get_lidar(sample_idx)
                    calib = self.get_calib(sample_idx)
                    pts_rect = calib.lidar_to_rect(points[:, 0:3])

                    fov_flag = self.get_fov_flag(pts_rect, info['image']['image_shape'], calib)
                    pts_fov = points[fov_flag]
                    corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)
                    num_points_in_gt = -np.ones(num_gt, dtype=np.int32)

                    for k in range(num_objects):
                        flag = box_utils.in_hull(pts_fov[:, 0:3], corners_lidar[k])
                        num_points_in_gt[k] = flag.sum()
                    annotations['num_points_in_gt'] = num_points_in_gt

            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)

    def get_box(self, detection_box: DetectionBox) -> Box:
        return Box(detection_box.translation, detection_box.size, Quaternion(detection_box.rotation),
                   name=detection_box.detection_name, token=detection_box.sample_token)

    def get_sample_data(self, nusc, detection_box: DetectionBox,
                        sample_data_token: str,
                        box_vis_level: BoxVisibility = BoxVisibility.ANY,
                        use_flat_vehicle_coordinates: bool = False) -> Tuple[str, Box, np.array]:
        """
        Returns the data path as well as all annotations related to that sample_data.
        Note that the boxes are transformed into the current sensor's coordinate frame.
        :param sample_data_token: Sample_data token.
        :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
        :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
                                             aligned to z-plane in the world.
        :return: (data_path, boxes, camera_intrinsic <np.array: 3, 3>)
        """

        # Retrieve sensor & pose records
        sd_record = nusc.get('sample_data', sample_data_token)
        cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        sensor_record = nusc.get('sensor', cs_record['sensor_token'])
        pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

        data_path = nusc.get_sample_data_path(sample_data_token)

        if sensor_record['modality'] == 'camera':
            cam_intrinsic = np.array(cs_record['camera_intrinsic'])
            imsize = (sd_record['width'], sd_record['height'])
        else:
            cam_intrinsic = None
            imsize = None

        box = self.get_box(detection_box)

        if use_flat_vehicle_coordinates:
            # Move box to ego vehicle coord system parallel to world z plane.
            yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
        else:
            # Move box to ego vehicle coord system.
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(pose_record['rotation']).inverse)

            #  Move box to sensor coord system.
            box.translate(-np.array(cs_record['translation']))
            box.rotate(Quaternion(cs_record['rotation']).inverse)

        if sensor_record['modality'] == 'camera' and not \
                box_in_image(box, cam_intrinsic, imsize, vis_level=box_vis_level):
            return data_path, None, cam_intrinsic

        return data_path, box, cam_intrinsic

    def nuscenes_gt_to_kitti(self, nusc, eval_boxes: EvalBoxes, pred=False, num_workers=16) -> List[dict]:
        """
        Converts nuScenes GT annotations to KITTI format.
        """
        import concurrent.futures as futures

        kitti_to_nu_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi / 2)
        kitti_to_nu_lidar_inv = kitti_to_nu_lidar.inverse

        # Create output folders.
        prefix = 'pred' if pred else 'gt'
        label_folder = os.path.join(self.nusc_kitti_dir, prefix, self.split, 'label_2')
        calib_folder = os.path.join(self.nusc_kitti_dir, prefix, self.split, 'calib')
        image_folder = os.path.join(self.nusc_kitti_dir, prefix, self.split, 'image_2')
        lidar_folder = os.path.join(self.nusc_kitti_dir, prefix, self.split, 'velodyne')
        for folder in [label_folder, calib_folder, image_folder, lidar_folder]:
            if not os.path.isdir(folder):
                os.makedirs(folder)

        conv = 'predictions' if pred else 'ground truth'
        progress_bar = tqdm.tqdm(total=len(eval_boxes.sample_tokens), desc=f'convert {conv} to kitti', dynamic_ncols=True)

        def convert_single_sample(sample_token):

            progress_bar.update()

            # Get sample data.
            sample = nusc.get('sample', sample_token)
            cam_front_token = sample['data'][self.cam_name]
            lidar_token = sample['data'][self.lidar_name]

            # Retrieve sensor records.
            sd_record_cam = nusc.get('sample_data', cam_front_token)
            sd_record_lid = nusc.get('sample_data', lidar_token)
            cs_record_cam = nusc.get('calibrated_sensor', sd_record_cam['calibrated_sensor_token'])
            cs_record_lid = nusc.get('calibrated_sensor', sd_record_lid['calibrated_sensor_token'])

            # Combine transformations and convert to KITTI format.
            # Note: cam uses same conventions in KITTI and nuScenes.
            lid_to_ego = transform_matrix(cs_record_lid['translation'], Quaternion(cs_record_lid['rotation']),
                                          inverse=False)
            ego_to_cam = transform_matrix(cs_record_cam['translation'], Quaternion(cs_record_cam['rotation']),
                                          inverse=True)
            velo_to_cam = np.dot(ego_to_cam, lid_to_ego)

            # Convert from KITTI to nuScenes LIDAR coordinates, where we apply velo_to_cam.
            velo_to_cam_kitti = np.dot(velo_to_cam, kitti_to_nu_lidar.transformation_matrix)

            # Currently not used.
            imu_to_velo_kitti = np.zeros((3, 4))  # Dummy values.
            r0_rect = Quaternion(axis=[1, 0, 0], angle=0)  # Dummy values.

            # Projection matrix.
            p_left_kitti = np.zeros((3, 4))
            p_left_kitti[:3, :3] = cs_record_cam['camera_intrinsic']  # Cameras are always rectified.

            # Create KITTI style transforms.
            velo_to_cam_rot = velo_to_cam_kitti[:3, :3]
            velo_to_cam_trans = velo_to_cam_kitti[:3, 3]

            # Check that the rotation has the same format as in KITTI.
            assert (velo_to_cam_rot.round(0) == np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])).all()
            assert (velo_to_cam_trans[1:3] < 0).all()

            # Retrieve the token from the lidar.
            # Note that this may be confusing as the filename of the camera will include the timestamp of the lidar,
            # not the camera.
            filename_cam_full = sd_record_cam['filename']
            filename_lid_full = sd_record_lid['filename']
            # token = '%06d' % token_idx # Alternative to use KITTI names.
            # token_idx += 1

            # Convert image (jpg to png).
            src_im_path = os.path.join(nusc.dataroot, filename_cam_full)
            dst_im_path = os.path.join(image_folder, sample_token + '.png')
            if not os.path.exists(dst_im_path):
                im = Image.open(src_im_path)
                im.save(dst_im_path, "PNG")

            # Convert lidar.
            # Note that we are only using a single sweep, instead of the commonly used n sweeps.
            src_lid_path = os.path.join(nusc.dataroot, filename_lid_full)
            dst_lid_path = os.path.join(lidar_folder, sample_token + '.bin')
            assert not dst_lid_path.endswith('.pcd.bin')
            pcl = LidarPointCloud.from_file(src_lid_path)
            pcl.rotate(kitti_to_nu_lidar_inv.rotation_matrix)  # In KITTI lidar frame.
            with open(dst_lid_path, "w") as lid_file:
                pcl.points.T.tofile(lid_file)

            # Add to tokens.
            # tokens.append(sample_token)

            # Create calibration file.
            kitti_transforms = dict()
            kitti_transforms['P0'] = np.zeros((3, 4))  # Dummy values.
            kitti_transforms['P1'] = np.zeros((3, 4))  # Dummy values.
            kitti_transforms['P2'] = p_left_kitti  # Left camera transform.
            kitti_transforms['P3'] = np.zeros((3, 4))  # Dummy values.
            kitti_transforms['R0_rect'] = r0_rect.rotation_matrix  # Cameras are already rectified.
            kitti_transforms['Tr_velo_to_cam'] = np.hstack((velo_to_cam_rot, velo_to_cam_trans.reshape(3, 1)))
            kitti_transforms['Tr_imu_to_velo'] = imu_to_velo_kitti
            calib_path = os.path.join(calib_folder, sample_token + '.txt')
            with open(calib_path, "w") as calib_file:
                for (key, val) in kitti_transforms.items():
                    val = val.flatten()
                    val_str = '%.12e' % val[0]
                    for v in val[1:]:
                        val_str += ' %.12e' % v
                    calib_file.write('%s: %s\n' % (key, val_str))

            sample_boxes = defaultdict(list)
            boxes = []
            for detection_box in eval_boxes[sample_token]:

                # Get box in LIDAR frame.
                # TODO Should we care about BoxVisibility here while it's being filtered in projection to 2d (image)
                _, box_lidar_nusc, _ = self.get_sample_data(nusc, detection_box, lidar_token,
                                                            box_vis_level=BoxVisibility.NONE)

                # Convert nuScenes category to nuScenes detection challenge category.
                detection_name = detection_box.detection_name

                # Skip categories that are not part of the nuScenes detection challenge.
                if detection_name is None or detection_name not in CLASS_MAP.keys():
                    continue

                # Convert to Kitti class names
                detection_name = CLASS_MAP[detection_name]

                # Convert from nuScenes to KITTI box format.
                box_cam_kitti = KittiDB.box_nuscenes_to_kitti(
                    box_lidar_nusc, Quaternion(matrix=velo_to_cam_rot), velo_to_cam_trans, r0_rect)

                # Project 3d box to 2d box in image, ignore box if it does not fall inside.
                bbox_2d = project_kitti_box_to_image(box_cam_kitti, p_left_kitti, self.imsize[1], self.imsize[0])
                if bbox_2d is None:
                    continue

                truncated = 0.0 if pred else bbox_2d["truncated"]
                box_cam_kitti.score = detection_box.detection_score if pred else -1

                # Convert quaternion to yaw and alpha angle.
                v = np.dot(box_cam_kitti.rotation_matrix, np.array([1, 0, 0]))
                yaw = -np.arctan2(v[2], v[0])
                alpha = -np.arctan2(box_cam_kitti.center[0], box_cam_kitti.center[2]) + yaw

                box = dict()
                box['name'] = detection_name
                box['truncated'] = 0. if pred else truncated
                box['occluded'] = 0  # Will be updated in postprocessing if gt sample is given
                box['bbox'] = list(bbox_2d["bbox"])
                box['dimensions'] = np.array([box_cam_kitti.wlh[2], box_cam_kitti.wlh[0], box_cam_kitti.wlh[1]])
                box['location'] = box_cam_kitti.center
                box['rotation_y'] = yaw
                box["alpha"] = alpha
                box["depth"] = np.linalg.norm(np.array(box_cam_kitti.center[:3]))
                box['score'] = box_cam_kitti.score
                box['box_cam_kitti'] = box_cam_kitti

                boxes.append(box)

            if len(boxes) == 0:
                return

            # Adding occlusion
            if not pred:
                boxes = postprocessing(boxes, self.imsize[1], self.imsize[0])

            # Write label file.
            label_path = os.path.join(label_folder, sample_token + '.txt')
            with open(label_path, "w") as label_file:
                for box in boxes:
                    # Convert box to output string format.
                    output = KittiDB.box_to_string(name=box['name'], box=box['box_cam_kitti'], bbox_2d=box['bbox'],
                                                   truncation=box['truncated'], occlusion=box['occluded'])
                    # Write to disk.
                    label_file.write(output + '\n')

            for box in boxes:
                for k, v in box.items():
                    sample_boxes[k].append(v)
            for k, v in sample_boxes.items():
                sample_boxes[k] = np.stack(v)

            metadata = dict(token=sample_token,
                            filename_cam_full=filename_cam_full,
                            filename_lid_full=filename_lid_full)

            # Adding difficulties
            info = dict(annos=sample_boxes, metadata=metadata)
            # add_difficulty_to_annos(info)
            # infos.append(info)
            return info

        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(convert_single_sample, eval_boxes.sample_tokens)

        progress_bar.close()
        return list(infos)

    def evaluation(self, det_annos, class_names, **kwargs):
        import json
        from nuscenes.nuscenes import NuScenes
        from . import nuscenes_utils
        nusc = NuScenes(version=self.dataset_cfg.VERSION, dataroot=str(self.root_path), verbose=True)
        nusc_annos = nuscenes_utils.transform_det_annos_to_nusc_annos(det_annos, nusc)
        nusc_annos['meta'] = {
            'use_camera': False,
            'use_lidar': True,
            'use_radar': False,
            'use_map': False,
            'use_external': False,
        }

        output_path = Path(kwargs['output_path'])
        output_path.mkdir(exist_ok=True, parents=True)
        res_path = str(output_path / 'results_nusc.json')
        with open(res_path, 'w') as f:
            json.dump(nusc_annos, f)

        self.logger.info(f'The predictions of NuScenes have been saved to {res_path}')

        if self.dataset_cfg.VERSION == 'v1.0-test':
            return 'No ground-truth annotations for evaluation', {}

        from nuscenes.eval.detection.config import config_factory
        from nuscenes.eval.detection.evaluate import NuScenesEval

        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
            'v1.0-test': 'test'
        }
        try:
            eval_version = 'detection_cvpr_2019'
            eval_config = config_factory(eval_version)
        except:
            eval_version = 'cvpr_2019'
            eval_config = config_factory(eval_version)

        nusc_eval = NuScenesEval(
            nusc,
            config=eval_config,
            result_path=res_path,
            eval_set=eval_set_map[self.dataset_cfg.VERSION],
            output_dir=str(output_path),
            verbose=True,
        )

        gt_boxes_in_kitti = self.nuscenes_gt_to_kitti(nusc, nusc_eval.gt_boxes)
        pred_boxes_in_kitti = self.nuscenes_gt_to_kitti(nusc, nusc_eval.pred_boxes, pred=True)
        class_names = [CLASS_MAP[class_name] for class_name in class_names if CLASS_MAP[class_name] != 'Misc']

        # Remove None gts and preds
        gt_boxes_in_kitti = [gt_box for gt_box in gt_boxes_in_kitti if gt_box is not None]
        pred_boxes_in_kitti = [det_box for det_box in pred_boxes_in_kitti if det_box is not None]

        gt_tokens = [gt_info['metadata']['token'] for gt_info in gt_boxes_in_kitti]

        # Drop det tokens that do not exist in gt tokens
        pred_boxes_filtered = [det_anno for det_anno in pred_boxes_in_kitti if det_anno['metadata']['token'] in gt_tokens]
        det_tokens = [det['metadata']['token'] for det in pred_boxes_filtered]

        assert set(det_tokens) == set(gt_tokens), \
            "Samples in split doesn't match samples in predictions."

        det_inds = np.argsort([pred_box['metadata']['token'] for pred_box in pred_boxes_filtered])
        gt_inds = np.argsort([gt_box['metadata']['token'] for gt_box in gt_boxes_in_kitti])
        assert np.all((det_inds - gt_inds) == 0), \
            "Detected samples don't have the same order as their corresponding gt samples"

        eval_det_annos = [copy.deepcopy(info['annos']) for info in pred_boxes_filtered]
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in gt_boxes_in_kitti]

        # Filter Misc class from gt and predictions
        # gt_boxes_list = []
        # pred_boxes_list = []
        # for gt_boxes, pred_boxes in zip(eval_gt_annos, eval_det_annos):
        #
        #     mask_gt = gt_boxes['name'] != 'Misc'
        #     for k, v in gt_boxes.items():
        #         if isinstance(v, np.ndarray):
        #             gt_boxes[k] = v[mask_gt, ...]
        #     gt_boxes_list.append(gt_boxes)
        #
        #     mask_pred = pred_boxes['name'] != 'Misc'
        #     for k, v in pred_boxes.items():
        #         if isinstance(v, np.ndarray):
        #             pred_boxes[k] = v[mask_pred, ...]
        #     pred_boxes_list.append(pred_boxes)

        from ..kitti.kitti_object_eval_python import eval2 as kitti_eval
        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)

        return ap_result_str, ap_dict


if __name__ == '__main__':
    import yaml
    import argparse
    from pathlib import Path
    from easydict import EasyDict

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config of dataset')
    parser.add_argument('--func', type=str, default='create_nuscenes_infos', help='')
    parser.add_argument('--version', type=str, default='v1.0-mini', help='')
    args = parser.parse_args()

    if args.func == 'create_nuscenes_infos':
        dataset_cfg = EasyDict(yaml.load(open(args.cfg_file)))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        dataset_cfg.VERSION = args.version

        nuscenes_dataset = NuScenesDatasetKITTIEval(
            dataset_cfg=dataset_cfg, class_names=None,
            root_path=ROOT_DIR / 'data' / 'nuscenes',
            logger=common_utils.create_logger(), training=False
        )
        # samples = nuscenes_dataset.nusc_gt_to_kitti()
        # pdb.set_trace()