from embdata.coordinate import BBox2D
from embdata.episode import Episode
from embdata.sample import Sample
from embdata.multi_sample import Collection
from embdata.sense.world import WorldObject
from embdata.sense.camera import Camera
from embdata.sense.image import Image as MBImage
from pydantic import Field
import json
import os
from PIL import Image
from embdata.describe import describe
from io import BytesIO
import cv2
import io
from embdata.geometry import Transform3D
from pipelearn.models.yolo6D.data.vis import Visualize, VisualizationConfig

from embdata.sense.camera import Intrinsics, Extrinsics
import numpy as np
from tqdm import tqdm


class InputFeatures(Sample):
    image: MBImage | None = None
    camera: Camera | None = Field(
        default_factory=Camera, description="Camera parameters of the scene"
    )
    depth: MBImage | None = None
    extras: dict = Field(default_factory=dict)


class Target(Sample):
    objects: Collection[WorldObject] = Field(default_factory=Collection[WorldObject])


class FeaturesTarget(Sample):
    input_features: InputFeatures = Field(default_factory=InputFeatures)
    target: Target = Field(default_factory=Target)


def draw_3d_bounding_box(image_np, objects, camera, save_path):
    """
    Draw 3D bounding box on the image using object pose and camera parameters.
    """
    # Create a copy of the image to draw on
    output_image = image_np.copy()
    
    # Define 3D bounding box corners (assuming unit cube centered at origin)
    bbox_3d = np.array([
        [-0.5, -0.5, -0.5],  # 0
        [0.5, -0.5, -0.5],   # 1
        [0.5, 0.5, -0.5],    # 2
        [-0.5, 0.5, -0.5],   # 3
        [-0.5, -0.5, 0.5],   # 4
        [0.5, -0.5, 0.5],    # 5
        [0.5, 0.5, 0.5],     # 6
        [-0.5, 0.5, 0.5]     # 7
    ])

    # Process each object
    for obj in objects:
        # Scale the bbox_3d based on object dimensions (if available)
        scale = 0.1  # Adjust this value based on your objects' actual size
        scaled_bbox_3d = bbox_3d * scale

        # Get pose values
        pose = np.array(list(obj["pose"].values()))
        trans3d = Transform3D.from_pose(pose)

        # Transform points using pose
        R = trans3d.rotation.reshape(3, 3)
        t = trans3d.translation.reshape(3, 1)
        
        # Apply transformation
        points_3d = (R @ scaled_bbox_3d.T + t).T

        # Project 3D points to 2D using camera parameters
        points_2d = []
        for point in points_3d:
            # Project point using camera intrinsics
            x = point[0] / point[2]
            y = point[1] / point[2]
            u = camera["intrinsic"]["fx"] * x + camera["intrinsic"]["cx"]
            v = camera["intrinsic"]["fy"] * y + camera["intrinsic"]["cy"]
            points_2d.append([int(u), int(v)])

        points_2d = np.array(points_2d)

        # Draw the box edges
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # top face
            (0, 4), (1, 5), (2, 6), (3, 7)   # connecting edges
        ]

        # Draw edges in blue
        for i, j in edges:
            cv2.line(output_image, tuple(points_2d[i]), tuple(points_2d[j]), (255, 0, 0), 2)

        # Add label
        label = obj["name"]
        cv2.putText(
            output_image,
            label,
            (points_2d[0][0], points_2d[0][1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 0, 0),
            2,
        )

    # save the image
    cv2.imwrite(save_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))


def draw_bounding_boxes(image_bytes, objects, camera, save_path):
    # Convert the image bytes to numpy array
    image = Image.open(io.BytesIO(image_bytes))
    image_np = np.array(image)

    # Draw 2D and 3D bounding boxes
    for obj in objects:
        # Draw 2D bounding box (green)
        x1 = int(obj["bbox_2d"]["x1"])
        y1 = int(obj["bbox_2d"]["y1"])
        x2 = int(obj["bbox_2d"]["x2"])
        y2 = int(obj["bbox_2d"]["y2"])
        cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Add label
        label = obj["name"]
        cv2.putText(
            image_np,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )

        # Draw 3D bounding box (blue)
        image_np = draw_3d_bounding_box(image_np, obj, camera)

    # Save the image
    cv2.imwrite(save_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))


def process_split(json_data, img_dir, camera, categories, visualize=False):
    """Process a single data split (train or test)."""
    images = json_data["images"]
    annotations = json_data["annotations"]

    # Create a dictionary to group annotations by image_id
    annotations_by_image = {}
    for annotation in annotations:
        image_id = annotation["image_id"]
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(annotation)

    ds_list = []
    for image in tqdm(images, desc="Processing images"):
        image_id = image["id"]
        image_folder = image["image_folder"]
        file_name = image["file_name"]
        image_path = os.path.join(img_dir, image_folder, "rgb", file_name)
        depth_path = os.path.join(
            img_dir, image_folder, "depth", file_name.split(".")[0] + ".png"
        )

        # read image using pillow
        image_data = Image.open(image_path)
        depth = Image.open(depth_path)

        input_features = InputFeatures(
            image=MBImage(array=np.array(image_data)),
            camera=camera,
            depth=MBImage(array=np.array(depth)),
            extras={"categories": categories},
        )

        objects = []
        target = Target()

        # Process annotations for this image
        for annotation in annotations_by_image.get(image_id, []):
            category_id = annotation["category_id"]
            rotation = annotation["R"]
            translation = annotation["T"]
            bbox = annotation["bbox"]

            x, y, w, h = bbox
            bbox_converted = [x, y, x + w, y + h]

            pose = Transform3D(
                rotation=np.array(rotation).reshape(3, 3),
                translation=np.array(translation).reshape(3, 1),
            ).pose()

            objects.append(
                WorldObject(
                    name=str(category_id),
                    bbox_2d=BBox2D.from_list(bbox_converted),
                    pose=pose,
                )
            )
        
        if len(objects) == 0:
            continue

        target.objects = objects
        features_target = FeaturesTarget(input_features=input_features, target=target)
        result_dict = features_target.dict()

        # Convert images to bytes
        for key in ['image', 'depth']:
            byte_arr = BytesIO()
            getattr(features_target.input_features, key).pil.save(
                byte_arr, 
                format="JPEG" if key == 'image' else "PNG"
            )
            result_dict["input_features"][key] = {"bytes": byte_arr.getvalue()}

        if visualize and len(ds_list) == 0:  # Visualize first image only
            vis = Visualize(VisualizationConfig())
            bbox2d_image = vis.draw_bbox2d(image=np.array(image_data), obj=objects[3], color_idx=0)
            Image.fromarray(bbox2d_image).save("./bbox2d.jpg")
            bbox3d_image = vis.draw_bbox3d(image=np.array(image_data), obj=objects[3], camera=camera, color_idx=0)
            Image.fromarray(bbox3d_image).save("./bbox3d.jpg")

        ds_list.append(result_dict)
    
    return ds_list

def process_train_split(train_json_path, train_img_dir, camera, visualize=False):
    """Process training data split."""
    with open(train_json_path, "r") as f:
        train_json = json.load(f)
    return process_split(train_json, train_img_dir, camera, train_json["categories"], visualize)

def process_test_split(test_json_path, test_img_dir, camera):
    """Process test data split."""
    with open(test_json_path, "r") as f:
        test_json = json.load(f)
    return process_split(test_json, test_img_dir, camera, test_json["categories"])

def main():

    # Paths
    train_json_path = "/home/tilak/projects/learning/datasets/lmo/annotations/instances_train.json"
    test_json_path = "/home/tilak/projects/learning/datasets/lmo/annotations/instances_test_bop.json"
    train_img_dir = "/home/tilak/projects/learning/datasets/lmo/train_pbr"
    test_img_dir = "/home/tilak/projects/learning/datasets/lmo/test_bop"
    camera_json_path = "/home/tilak/projects/learning/datasets/lmo/lmo/camera.json"

    # Load camera parameters
    with open(camera_json_path, "r") as f:
        camera_json = json.load(f)

    # Setup camera
    camera = Camera(
        intrinsic=Intrinsics(
            fx=camera_json["fx"],
            fy=camera_json["fy"],
            cx=camera_json["cx"],
            cy=camera_json["cy"],
        ),
        extrinsic=Extrinsics(),
    )

    from huggingface_hub import login
    login()

    # Process train split
    # train_data = process_train_split(train_json_path, train_img_dir, camera, visualize=True)
    # train_episode = Episode.from_list(
    #     train_data, observation_key="input_features", action_key="target"
    # )
    # train_episode.dataset().push_to_hub("mbodiai/bop-lmo-episode", split="train")

    # Process test split
    test_data = process_test_split(test_json_path, test_img_dir, camera)
    test_episode = Episode.from_list(
        test_data, observation_key="input_features", action_key="target"
    )
    describe(test_episode)
    test_episode.dataset().push_to_hub("mbodiai/bop-lm-episode", split="test")

if __name__ == "__main__":
    main()
