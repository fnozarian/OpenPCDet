"""
Open3d visualization tool box
Written by Jihan YANG
All rights preserved from 2021 - present.
"""
import open3d.visualization.gui as gui
import open3d
import torch
from matplotlib import pyplot as plt
import numpy as np
import open3d.visualization.rendering as rendering
import cv2

box_colormap = [
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
]

classes = {0: 'Car', 1: 'Ped', 2: 'Cycle'}


class Open3DRenderer:
    """Reusable Open3D Renderer for efficient visualization and bounding box zoom-ins."""

    def __init__(self, width=1024, height=1024):
        self.width = width
        self.height = height
        self.renderer = open3d.visualization.rendering.OffscreenRenderer(width, height)
        self.scene = self.renderer.scene
        self.scene.set_background([1, 1, 1, 1])  # White background
        # Define BEV area range in meters
        self.min_x, self.max_x = -1, 20
        self.min_y, self.max_y = -20, 20

        self.bbox_material = open3d.visualization.rendering.MaterialRecord()
        self.bbox_material.shader = "unlitLine"
        self.bbox_material.line_width = 3  # Set line width

        self.point_material = open3d.visualization.rendering.MaterialRecord()

        self.fig, self.ax = plt.subplots(figsize=(10, 10))

    def dynamic_camera_position(self, points, cam_height=25, center_offset_x=6, center_offset_y=0):

        center_x = np.median(points[:, 0])
        center_y = np.median(points[:, 1])

        center_x += center_offset_x
        center_y += center_offset_y

        return [center_x, center_y, cam_height], [center_x, center_y, 0]

    def project_to_2d(self, points_3d):
        """Projects 3D points onto the 2D image plane using Open3D's camera."""
        points_2d = []
        for point in points_3d:
            screen_pos = self.renderer.scene.camera.get_view_matrix() @ np.append(point, 1)  # Transform to view space
            screen_pos = self.renderer.scene.camera.get_projection_matrix() @ screen_pos  # Transform to projection space

            # Convert to pixel coordinates
            x_ndc, y_ndc = screen_pos[0] / screen_pos[3], screen_pos[1] / screen_pos[3]
            x_pixel = int((x_ndc + 1) * 400)  # Scale to image width (800/2)
            y_pixel = int((1 - y_ndc) * 300)  # Scale to image height (600/2)
            points_2d.append((x_pixel, y_pixel))

        return points_2d

    def create_bbox(self, box, color=(1, 0, 0)):
        """Creates an Open3D LineSet for a 3D bounding box."""
        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
            [0, 4], [1, 5], [2, 6], [3, 7]  # Connecting lines
        ]
        box3d = open3d.geometry.LineSet()
        box3d.points = open3d.utility.Vector3dVector(box)
        box3d.lines = open3d.utility.Vector2iVector(lines)
        box3d.paint_uniform_color([1, 0, 0])  # Set color to red
        box3d.colors = open3d.utility.Vector3dVector([color] * len(lines))
        return box3d

    def get_corners(self, boxes):
        """Computes the 8 corner points for each 3D bounding box."""
        all_corners = []
        centers = []
        for box in boxes:
            x, y, z, dx, dy, dz, yaw = box  # Assuming [center_x, center_y, center_z, width, length, height, yaw]

            # Rotation matrix for yaw
            R = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                          [np.sin(yaw), np.cos(yaw), 0],
                          [0, 0, 1]])

            # 8 corners in local box coordinates
            l, w, h = dx / 2, dy / 2, dz / 2
            corners = np.array([
                [-l, -w, -h], [l, -w, -h], [l, w, -h], [-l, w, -h],
                [-l, -w, h], [l, -w, h], [l, w, h], [-l, w, h]
            ])

            # Rotate and translate to global position
            corners = (R @ corners.T).T + np.array([x, y, z])
            all_corners.append(corners)
            centers.append((x, y, z))  # Store center for label placement

        return all_corners, centers

    def update_camera(self, box_center, box_size, x_multiplier=2.0):
        """
        Places the camera correctly for KITTI coordinate system.

        Args:
            x_multiplier: Multiplier to adjust camera distance from the box
            box_center: Center of the bounding box (NumPy array of shape (3,))
            box_size: Size of the bounding box (NumPy array of shape (3,))
        """

        box_center = np.array(box_center, dtype=np.float32).reshape(3, 1)
        box_size = np.array(box_size, dtype=np.float32)

        # Move camera backward along X-axis to see the full box
        cam_offset = box_size[0] * x_multiplier  # Adjust this multiplier if needed
        eye = np.array([
            box_center[0] - cam_offset,  # Move back along X-axis
            box_center[1],  # Align with box Y position
            box_center[2] + box_size[2]  # Slightly above the box
        ], dtype=np.float32).reshape(3, 1)

        # Up direction: KITTI Z-axis (up)
        up = np.array([0, 0, 1], dtype=np.float32).reshape(3, 1)

        # Set camera to look at the box center
        self.scene.camera.look_at(box_center, eye, up)

    def render_scene_tb(self, points, gt_boxes=None, gt_labels=None, ref_boxes=None, ref_labels=None,
                            ref_scores=None, attributes=None, point_colors=None, draw_origin=True):
        self.scene.clear_geometry()

        # Compute dynamic camera position
        cam_position, cam_target = self.dynamic_camera_position(points)
        self.scene.camera.look_at(cam_target, cam_position, [0, 0, 1])  # Set top-down view

        # Add lighting (important for visibility)
        # scene.scene.set_lighting(scene.scene.LightingProfile.NO_SHADOWS, (0.8, 0.8, 0.8))  # Brighten scene

        # Create point cloud
        cloud = open3d.geometry.PointCloud()
        cloud.points = open3d.utility.Vector3dVector(points[:, :3])

        if point_colors is None:
            cloud.colors = open3d.utility.Vector3dVector(np.zeros((points.shape[0], 3)))
        else:
            cloud.colors = open3d.utility.Vector3dVector(point_colors)

        self.scene.add_geometry("PointCloud", cloud, self.point_material)

        # Add GT boxes (blue) and labels
        if gt_boxes is not None:
           gt_corners, gt_centers = self.get_corners(gt_boxes)
           for i, corners in enumerate(gt_corners):
               self.scene.add_geometry(f"GT_Box_{i}", self.create_bbox(corners, color=(0, 0, 1)), self.bbox_material)

        # Add Ref boxes (green)
        if ref_boxes is not None:
            ref_corners, _ = self.get_corners(ref_boxes)
            for i, corners in enumerate(ref_corners):
                self.scene.add_geometry(f"Ref_Box_{i}", self.create_bbox(corners, color=(0, 1, 0)), self.bbox_material)

        # Render the Open3D scene
        img = self.renderer.render_to_image()
        img_np = np.asarray(img)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGRA2RGB)  # Convert BGRA -> RGB

        return self.numpy_to_figure(img_np)

    def render_scene(self, points, gt_box=None):
        """Updates the scene and renders an image with correctly rotated bounding boxes."""
        self.scene.clear_geometry()

        # Add point cloud
        cloud = open3d.geometry.PointCloud()
        cloud.points = open3d.utility.Vector3dVector(points[:, :3])  # Ensure points are correctly formatted
        cloud.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))  # Default color
        self.scene.add_geometry("PointCloud", cloud, open3d.visualization.rendering.MaterialRecord())

        # Add bounding box if provided
        if gt_box is not None:
            box_center = gt_box[:3]  # [x, y, z] center of bbox
            box_size = gt_box[3:6]  # [width, length, height]
            box_yaw = gt_box[6]  # Rotation around Z-axis (yaw angle in radians)

            # Create an oriented bounding box
            bbox = open3d.geometry.OrientedBoundingBox()
            bbox.center = np.array(box_center)
            bbox.extent = np.array(box_size)

            # Compute the rotation matrix from yaw
            cos_yaw = np.cos(box_yaw)
            sin_yaw = np.sin(box_yaw)
            rot_matrix = np.array([
                [cos_yaw, -sin_yaw, 0],
                [sin_yaw, cos_yaw, 0],
                [0, 0, 1]  # KITTI Z-axis is up
            ])
            bbox.R = rot_matrix  # Set rotation matrix

            # Set color and add to scene
            bbox.color = (1, 0, 0)  # Red bounding box
            self.scene.add_geometry("GT Box", bbox, open3d.visualization.rendering.MaterialRecord())

        # Render to image
        img = self.renderer.render_to_image()
        img_np = np.asarray(img)

        return cv2.cvtColor(img_np, cv2.COLOR_BGRA2RGB) if img_np is not None else None

    def numpy_to_figure(self, image):
        """Converts a NumPy image into a Matplotlib figure."""
        self.ax.clear()
        self.ax.imshow(image)
        self.ax.axis("off")
        self.fig.canvas.draw()  # Ensure the figure updates
        return self.fig

    # Doesn't work currently
    def create_bbox_grid(self, points, gt_boxes):
        """Creates a grid of close-up views for each GT bounding box."""
        images = []
        for gt_box in gt_boxes:
            center, size = gt_box[:3], gt_box[3:6]
            self.update_camera(center, size)
            img_np = self.render_scene(points, gt_box)
            if img_np is not None:
                images.append(img_np)

        # Arrange images in a grid
        grid_size = int(np.ceil(np.sqrt(len(images))))  # Make a square grid
        img_h, img_w, _ = images[0].shape
        grid_img = np.ones((grid_size * img_h, grid_size * img_w, 3), dtype=np.uint8) * 255  # White background

        for idx, img in enumerate(images):
            row, col = divmod(idx, grid_size)
            grid_img[row * img_h:(row + 1) * img_h, col * img_w:(col + 1) * img_w, :] = img

        # Convert to figure and log to TensorBoard
        fig = self.numpy_to_figure(grid_img)
        return fig


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
            vis.add_geometry(line_set)
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
        vis.add_geometry(line_set, reset_bounding_box=False)

        if score is not None:
            vis.add_3d_label(corners[5], '%.1f' % score[i])

    return vis
