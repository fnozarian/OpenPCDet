"""
Open3d visualization tool box
Written by Jihan YANG
All rights preserved from 2021 - present.
"""
import open3d.visualization.gui as gui
import open3d
import torch
import matplotlib
import numpy as np

box_colormap = [
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
]

classes = {0: 'Car', 1: 'Ped', 2: 'Cycle'}

def get_coor_colors(obj_labels):
    """
    Args:
        obj_labels: 1 is ground, labels > 1 indicates different instance cluster

    Returns:
        rgb: [N, 3]. color for each point.
    """
    colors = matplotlib.colors.XKCD_COLORS.values()
    max_color_num = obj_labels.max()

    color_list = list(colors)[:max_color_num+1]
    colors_rgba = [matplotlib.colors.to_rgba_array(color) for color in color_list]
    label_rgba = np.array(colors_rgba)[obj_labels]
    label_rgba = label_rgba.squeeze()[:, :3]

    return label_rgba


def draw_scenes(points, gt_boxes=None, gt_labels=None, ref_boxes=None, ref_labels=None, ref_scores=None,
                attributes=None, point_colors=None, draw_origin=True):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()

    app = gui.Application.instance
    app.initialize()
    vis = open3d.visualization.O3DVisualizer("Open3D - 3D Text", 1024, 768)

    vis.show_settings = True
    vis.show_skybox(False)
    vis.show_ground = True
    vis.show_axes = True
    vis.ground_plane = open3d.visualization.rendering.Scene.GroundPlane.XY

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry("Axis", axis_pcd)

    cloud = open3d.geometry.PointCloud()
    cloud.points = open3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry("Points", cloud)
    if point_colors is None:
        cloud.colors = open3d.utility.Vector3dVector(np.zeros((points.shape[0], 3)))
    else:
        cloud.colors = open3d.utility.Vector3dVector(point_colors)

    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (0, 0, 1), gt_labels)

    if ref_boxes is not None:
        vis = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores, attributes=attributes)

    vis.reset_camera_to_default()
    app.add_window(vis)
    app.run()

def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d


def draw_box(vis, gt_boxes, color=(0, 0, 1), ref_labels=None, score=None, attributes: dict = None):
    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])
        corners = box3d.get_box_points()

        if score is None:
            line_set.paint_uniform_color(color)
            vis.add_3d_label(corners[4], classes[ref_labels[i]])
        else:
            line_set.paint_uniform_color(box_colormap[ref_labels[i]])

        if attributes is not None and isinstance(attributes, dict):
            if 'positive' in attributes.keys() and attributes['positive'][i]:
                for corner in corners:
                    sphere = open3d.geometry.TriangleMesh.create_sphere(radius=0.05)
                    sphere.translate(corner)
                    vis.add_geometry(f"corner_{i}_{corner}", sphere)
            if 'id' in attributes.keys():
                vis.add_3d_label(corners[6], f"{attributes['id'][i]}")
        name = f"gt_box_{i}_{classes[ref_labels[i]]}" if score is None else f"pred_box_{i}_{classes[ref_labels[i]]}"
        vis.add_geometry(name, line_set)

        if score is not None:
            vis.add_3d_label(corners[5], '%.1f' % score[i])

    return vis
