"""Visualization module for World objects."""

import colorsys
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

from embdata.sense.camera import Camera
from embdata.sense.world import World, WorldObject


class ColorScheme(Enum):
    """Color scheme options for visualization."""

    DISTINCT = auto()
    SEQUENTIAL = auto()
    PAIRED = auto()


@dataclass
class VisualizationConfig:
    """Configuration for visualization settings."""

    masks: bool = True
    labels: bool = True
    bbox2d: bool = True
    bbox3d: bool = True
    axes: bool = True
    color_scheme: ColorScheme = ColorScheme.DISTINCT
    font_scale: float = 0.5
    line_thickness: int = 2


class Visualize:
    """Class for visualizing object detections, poses, and annotations."""

    def __init__(self, config: VisualizationConfig | None = None) -> None:
        """Initialize visualization settings.

        Args:
            config: Visualization configuration settings
        """
        self.config = config or VisualizationConfig()
        self.colors = self._generate_colors(20)  # Generate 20 distinct colors

    def _generate_colors(self, n: int) -> List[Tuple[int, int, int]]:
        """Generate n visually distinct colors.

        Args:
            n: Number of colors to generate

        Returns:
            List of RGB color tuples
        """
        colors = []
        for i in range(n):
            hue = i / n
            sat = 0.9
            val = 0.9
            rgb = colorsys.hsv_to_rgb(hue, sat, val)
            colors.append(tuple(int(255 * c) for c in rgb))
        return colors

    def _get_rotation_vector(self, obj: WorldObject) -> Tuple[NDArray | None, NDArray | None]:
        """Get rotation and translation vectors from pose."""
        if obj.pose is None:
            return None, None

        # Get rotation matrix from euler angles
        rpy = [obj.pose.roll, obj.pose.pitch, obj.pose.yaw]
        rotation_matrix = R.from_euler("xyz", rpy).as_matrix()

        # Convert to OpenCV convention
        rvec, _ = cv2.Rodrigues(rotation_matrix)
        tvec = obj.pose.numpy()[:3].reshape((3, 1))

        return rvec, tvec

    def draw_mask(self, image: NDArray, obj: WorldObject, color_idx: int) -> NDArray:
        """Draw segmentation mask with gradient overlay.

        Args:
            image: Input RGB image
            obj: WorldObject containing mask
            color_idx: Index for color selection

        Returns:
            Image with mask overlay
        """
        if obj.mask is None:
            return image

        result = image.copy()
        color = self.colors[color_idx % len(self.colors)]

        # Create gradient overlay
        height, width = obj.mask.shape[:2]
        gradient = np.linspace(0.3, 0.7, height).reshape(-1, 1)
        gradient = np.tile(gradient, (1, width))

        # Apply color and gradient
        colored_mask = np.zeros_like(image)
        for i in range(3):
            colored_mask[..., i] = obj.mask * color[i]

        gradient_mask = (colored_mask * gradient[..., np.newaxis]).astype(np.uint8)
        mask_bool = obj.mask.astype(bool)
        result[mask_bool] = cv2.addWeighted(
            result[mask_bool], 0.7, gradient_mask[mask_bool], 0.3, 0,
        )

        return result

    def draw_bbox2d(self, image: NDArray, obj: WorldObject, color_idx: int) -> NDArray:
        """Draw 2D bounding box with anti-aliased lines.

        Args:
            image: Input RGB image
            obj: WorldObject containing 2D bbox
            color_idx: Index for color selection

        Returns:
            Image with 2D bbox overlay
        """
        if obj.bbox_2d is None:
            return image

        result = image.copy()
        color = self.colors[color_idx % len(self.colors)]

        # Convert bbox coordinates to integers
        x1 = int(obj.bbox_2d.x1)
        y1 = int(obj.bbox_2d.y1)
        x2 = int(obj.bbox_2d.x2)
        y2 = int(obj.bbox_2d.y2)

        # Create a larger image for anti-aliasing
        scale = 4
        h, w = result.shape[:2]
        large_img = cv2.resize(result, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)

        # Draw thick lines on larger image
        cv2.rectangle(
            large_img,
            (x1*scale, y1*scale),
            (x2*scale, y2*scale),
            color,
            thickness=self.config.line_thickness*scale,
            lineType=cv2.LINE_AA,
        )

        # Resize back to original size
        return cv2.resize(large_img, (w, h), interpolation=cv2.INTER_AREA)

    def draw_bbox3d(self, image: NDArray, obj: WorldObject, camera: Camera, color_idx: int) -> NDArray:
        """Draw 3D bounding box with occlusion handling."""
        if obj.bbox_3d is None or obj.pose is None:
            return image

        result = image.copy()
        color = self.colors[color_idx % len(self.colors)]

        # Get rotation and translation vectors
        rvec, tvec = self._get_rotation_vector(obj)
        if rvec is None or tvec is None:
            return image

        # Convert rvec back to rotation matrix to understand orientation
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        print(f"\nObject: {obj.name}")
        print(f"Rotation matrix:\n{rotation_matrix}")
        print(f"Translation: {tvec.T}")

        # Calculate dimensions in object's local frame
        length = abs(obj.bbox_3d.z2 - obj.bbox_3d.z1)  # Longest dimension (was Z)
        height = abs(obj.bbox_3d.y2 - obj.bbox_3d.y1)  # Height (Y stays the same)
        width = abs(obj.bbox_3d.x2 - obj.bbox_3d.x1)   # Width (was X)

        # Create corners in object's local frame
        # Swap width and height to make it rest on its bottom
        corners = np.float32([
            [-length/2, -width/2,  height/2],  # front top left
            [ length/2, -width/2,  height/2],  # front top right
            [ length/2, -width/2, -height/2],  # front bottom right
            [-length/2, -width/2, -height/2],  # front bottom left
            [-length/2,  width/2,  height/2],  # back top left
            [ length/2,  width/2,  height/2],  # back top right
            [ length/2,  width/2, -height/2],  # back bottom right
            [-length/2,  width/2, -height/2],  # back bottom left
        ])

        print(f"Box dimensions (l x w x h): {length:.3f} x {width:.3f} x {height:.3f}")
        print(f"Corners (local frame):\n{corners}")

        try:
            # Project corners to image plane
            corners_2d, _ = cv2.projectPoints(
                corners,
                rvec,
                tvec,
                camera.intrinsic.matrix.astype(np.float32),
                camera.distortion.numpy().astype(np.float32),
            )
            corners_2d = corners_2d.reshape(-1, 2).astype(int)

            print(f"Projected corners 2D:\n{corners_2d}")

            # Draw edges with different colors for front/back faces
            edges_front = [(0, 1), (1, 2), (2, 3), (3, 0)]  # front face
            edges_back = [(4, 5), (5, 6), (6, 7), (7, 4)]   # back face
            edges_connect = [(0, 4), (1, 5), (2, 6), (3, 7)]  # connecting edges

            # Draw front face (solid)
            for edge in edges_front:
                pt1 = corners_2d[edge[0]]
                pt2 = corners_2d[edge[1]]
                if (0 <= pt1[0] < image.shape[1] and 0 <= pt1[1] < image.shape[0] and
                    0 <= pt2[0] < image.shape[1] and 0 <= pt2[1] < image.shape[0]):
                    cv2.line(result, tuple(pt1), tuple(pt2), color,
                            thickness=self.config.line_thickness//2, lineType=cv2.LINE_AA)

            # Draw back face (dashed)
            for edge in edges_back:
                pt1 = corners_2d[edge[0]]
                pt2 = corners_2d[edge[1]]
                if (0 <= pt1[0] < image.shape[1] and 0 <= pt1[1] < image.shape[0] and
                    0 <= pt2[0] < image.shape[1] and 0 <= pt2[1] < image.shape[0]):
                    cv2.line(result, tuple(pt1), tuple(pt2), color,
                            thickness=self.config.line_thickness//2, lineType=cv2.LINE_AA)

            # Draw connecting edges
            for edge in edges_connect:
                pt1 = corners_2d[edge[0]]
                pt2 = corners_2d[edge[1]]
                if (0 <= pt1[0] < image.shape[1] and 0 <= pt1[1] < image.shape[0] and
                    0 <= pt2[0] < image.shape[1] and 0 <= pt2[1] < image.shape[0]):
                    cv2.line(result, tuple(pt1), tuple(pt2), color,
                            thickness=self.config.line_thickness//2, lineType=cv2.LINE_AA)

        except cv2.error as e:
            print(f"OpenCV Error in draw_bbox3d: {e}")
            return image

        return result

    def draw_axes(self, image: NDArray, obj: WorldObject, camera: Camera) -> NDArray:
        """Draw coordinate axes for pose visualization."""
        if obj.pose is None:
            return image

        result = image.copy()

        # Get rotation and translation vectors
        rvec, tvec = self._get_rotation_vector(obj)
        if rvec is None or tvec is None:
            return image

        # Draw main axes
        result = cv2.drawFrameAxes(
            image=result,
            cameraMatrix=camera.intrinsic.matrix,
            distCoeffs=camera.distortion.numpy(),
            rvec=rvec,
            tvec=tvec,
            length=0.1,
            thickness=self.config.line_thickness,
        )

        # Project axis endpoints for labels
        axis_points = np.float32([
            [0, 0, 0],
            [0.1, 0, 0],
            [0, 0.1, 0],
            [0, 0, 0.1],
        ]).reshape(-1, 3)

        points_2d, _ = cv2.projectPoints(
            axis_points,
            rvec,
            tvec,
            camera.intrinsic.matrix,
            camera.distortion.numpy(),
        )
        points_2d = points_2d.reshape(-1, 2)

        # Get endpoints for labels
        x_end = tuple(points_2d[1].astype(int))
        y_end = tuple(points_2d[2].astype(int))
        z_end = tuple(points_2d[3].astype(int))

        # Add labels with offset
        offset = 10
        if all(0 <= p[0] < image.shape[1] and 0 <= p[1] < image.shape[0]
               for p in [x_end, y_end, z_end]):
            # Use color indices: 0 for X (red), 1 for Y (green), 2 for Z (blue)
            result = self.draw_label(result, "X",
                                   (x_end[0] + offset, x_end[1] + offset), 0)
            result = self.draw_label(result, "Y",
                                   (y_end[0] + offset, y_end[1] + offset), 1)
            result = self.draw_label(result, "Z",
                                   (z_end[0] + offset, z_end[1] + offset), 2)

        return result

    def draw_label(self, image: NDArray, text: str, position: Tuple[int, int],
                  color_idx: int) -> NDArray:
        """Draw text label with outline and background.

        Args:
            image: Input RGB image
            text: Text to draw
            position: Position to draw text (x, y)
            color_idx: Index for color selection

        Returns:
            Image with text overlay
        """
        result = image.copy()
        color = self.colors[color_idx % len(self.colors)]

        # Get text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = self.config.font_scale
        thickness = self.config.line_thickness
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness,
        )

        x, y = position

        # Draw text outline
        cv2.putText(result, text, (x, y), font, font_scale,
                   (255, 255, 255), thickness * 3, cv2.LINE_AA)

        # Draw main text
        cv2.putText(result, text, (x, y), font, font_scale,
                   color, thickness, cv2.LINE_AA)

        return result

    def show(self, world: World) -> NDArray:
        """Main visualization function.

        Args:
            world: World object containing image, camera, and objects

        Returns:
            Annotated image with all enabled visualizations
        """
        # Convert image to numpy array and ensure BGR format for OpenCV
        # annotated_image = cv2.cvtColor(np.array(world.image.array), cv2.COLOR_RGB2BGR)
        annotated_image = world.image.array

        # Process each object in the world
        for idx, obj in enumerate(world.objects):
            # Skip special objects
            if obj.name in ["camera", "aruco", "plane", "person"]:
                continue

            # Get object pose in camera frame
            world.get_object(obj.name, reference="camera")
            print(f"Object: {obj.name}")
            print(f"Pose: {obj.pose}")
            print(f"Bbox 3D: {obj.bbox_3d}")
            print(f"Bbox 2D: {obj.bbox_2d}")

            # Draw visualizations based on config
            if self.config.masks and obj.mask is not None:
                annotated_image = self.draw_mask(annotated_image, obj, idx)

            if self.config.bbox2d and obj.bbox_2d is not None:
                annotated_image = self.draw_bbox2d(annotated_image, obj, idx)

            if self.config.bbox3d and obj.bbox_3d is not None:
                annotated_image = self.draw_bbox3d(
                    annotated_image, obj, world.camera, idx,
                )

            if self.config.axes and obj.pose is not None:
                annotated_image = self.draw_axes(annotated_image, obj, world.camera)

            if self.config.labels:
                # Project object center for label placement
                rvec, tvec = self._get_rotation_vector(obj)
                if rvec is not None and tvec is not None:
                    center_3d = obj.pose.numpy()[:3].reshape(1, 3)
                    center_2d, _ = cv2.projectPoints(
                        center_3d,
                        rvec,
                        tvec,
                        world.camera.intrinsic.matrix,
                        world.camera.distortion.numpy(),
                    )
                    center_2d = center_2d.reshape(-1, 2)[0].astype(int)
                    label_pos = (center_2d[0], center_2d[1] - 20)
                    annotated_image = self.draw_label(
                        annotated_image, obj.name, label_pos, idx,
                    )

        # Convert back to RGB for final output
        return cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)