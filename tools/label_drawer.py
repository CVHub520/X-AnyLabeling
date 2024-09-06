import os
import os.path as osp
import json
import shutil

import cv2
import natsort
import numpy as np
import supervision as sv
from tqdm import tqdm


def create_video_from_images(image_folder, output_video_path, frame_rate=25):
    """
    Create a video from a sequence of images.

    This function creates a video file from a folder of images and saves it to
    the specified output path. It assumes that the images are sorted in the
    correct order for the video sequence.

    Args:
        image_folder: str
            The path to the folder containing the images.
        output_video_path: str
            The path where the output video file will be saved.
        frame_rate: int, optional
            The frame rate of the output video. Default is 25 frames per second.

    Raises:
        ValueError: If no valid image files are found in the specified folder.

    Returns:
        None: The function prints the path where the video is saved and does not
        return anything.
    """
    # define valid extension
    valid_extensions = [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]

    # get all image files in the folder
    image_files = [
        f
        for f in os.listdir(image_folder)
        if os.path.splitext(f)[1] in valid_extensions
    ]
    image_files = natsort.natsorted(image_files)
    if not image_files:
        raise ValueError("No valid image files found in the specified folder.")

    # load the first image to get the dimensions of the video
    first_image_path = os.path.join(image_folder, image_files[0])
    first_image = cv2.imread(first_image_path)
    height, width, _ = first_image.shape

    # create a video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # codec for saving the video
    video_writer = cv2.VideoWriter(
        output_video_path, fourcc, frame_rate, (width, height)
    )

    # write each image to the video
    for image_file in tqdm(image_files):
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        video_writer.write(image)

    # source release
    video_writer.release()
    print(f"Video saved at {output_video_path}")


def draw_masks_from_custom(
    save_dir,
    image_path,
    label_path=None,
    classes=[],
    save_box=True,
    save_label=True,
    keep_ori_fn=False,
):
    """
    Draws masks on images from custom dataset annotations and saves the annotated images.

    Args:
        save_dir (str): Directory path to save annotated images.
        image_path (str): Path to the directory containing input images.
        label_path (str, optional): Path to the directory containing label JSON files.
                                    If None, labels are expected alongside images.
        classes (list[str]): List of class names to consider for annotation.
        save_box (bool): Whether to draw bounding boxes around masks.
        save_label (bool): Whether to annotate masks with class labels.
        keep_ori_fn (bool): If True, keeps the original filename; otherwise, uses a frame index-based naming.

    Raises:
        FileNotFoundError: If the specified image or label file does not exist.
        ValueError: If an invalid image format is encountered.
    """
    # Correct label_path if it incorrectly points to the image_path
    if label_path == image_path:
        label_path = None

    # Ensure save directory is clean and recreated
    if osp.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    # Sort and process image files
    image_list = os.listdir(image_path)
    sorted_image_list = natsort.natsorted(image_list)
    valid_extensions = [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]

    # Initialize class ID mapping
    id_to_classes = {i: c for i, c in enumerate(classes)}

    for frame_idx, image_name in enumerate(
        tqdm(sorted_image_list, colour="green")
    ):
        image_file = osp.join(image_path, image_name)
        # Skip non-image files or JSON annotation files
        if (
            image_name.endswith(".json")
            or osp.splitext(image_name)[-1] not in valid_extensions
        ):
            print(f"Invalid image format or JSON file: {image_file}")
            continue

        # Determine label file path
        label_name = osp.splitext(image_name)[0] + ".json"
        label_file = (
            osp.join(label_path, label_name)
            if label_path
            else osp.join(image_path, label_name)
        )

        # Read image and get its dimensions
        image = cv2.imread(image_file)
        image_height, image_width = image.shape[:2]

        # Prepare output filename
        save_name = (
            image_name
            if keep_ori_fn
            else f"annotated_frame_{frame_idx:05d}.jpg"
        )

        # If no label file exists, just save the original image and move to next
        if not osp.exists(label_file):
            cv2.imwrite(osp.join(save_dir, save_name), image)
            continue

        # Load and process annotations
        with open(label_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Collect polygons, XYXY coordinates, and class indices
        xyxy_list, mask_list, cind_list = [], [], []
        for shape in data["shapes"]:
            if (
                shape["shape_type"] != "polygon"
                or shape["label"] not in classes
            ):
                continue
            label_id = classes.index(shape["label"])
            cind_list.append(label_id)
            points = np.array(shape["points"], dtype=np.int32)
            xyxy_list.append(sv.polygon_to_xyxy(polygon=points))
            mask_list.append(
                sv.polygon_to_mask(
                    polygon=points, img_shape=(image_width, image_height)
                )
            )

        # If there are no shapes to draw, save the original image and continue
        if not xyxy_list:
            cv2.imwrite(os.path.join(save_dir, save_name), image)
            continue

        # Stack coordinates, masks, and IDs for processing
        xyxy = np.stack(xyxy_list, axis=0)
        masks = np.stack(mask_list, axis=0)
        masks = masks > 0.5  # Convert to binary masks
        object_ids = np.array(cind_list, dtype=np.int32)

        # Create Detections object for annotation
        detections = sv.Detections(xyxy=xyxy, mask=masks, class_id=object_ids)

        # Annotate the image based on flags
        annotated_frame = image.copy()
        if save_box:
            box_annotator = sv.BoxAnnotator()
            annotated_frame = box_annotator.annotate(
                scene=annotated_frame, detections=detections
            )
        if save_label:
            label_annotator = sv.LabelAnnotator()
            labels = [id_to_classes[i] for i in object_ids]
            annotated_frame = label_annotator.annotate(
                annotated_frame, detections=detections, labels=labels
            )
        mask_annotator = sv.MaskAnnotator()
        annotated_frame = mask_annotator.annotate(
            scene=annotated_frame, detections=detections
        )

        # Save the annotated image
        cv2.imwrite(osp.join(save_dir, save_name), annotated_frame)


def draw_boxes_from_custom(
    save_dir,
    image_path,
    label_path=None,
    classes=[],
    save_label=True,
    keep_ori_fn=False,
):
    """
    Draws bounding boxes on images from custom rectangle annotations and saves the annotated images.

    Args:
        save_dir (str): Directory path to save annotated images.
        image_path (str): Path to the directory containing input images.
        label_path (str, optional): Path to the directory containing label JSON files.
                                    If None, labels are expected alongside images.
        classes (list[str]): List of class names to consider for annotation.
        save_label (bool): Whether to annotate boxes with class labels.
        keep_ori_fn (bool): If True, keeps the original filename; otherwise, uses a frame index-based naming.

    Raises:
        FileNotFoundError: If the specified image or label file does not exist.
        ValueError: If an invalid image format is encountered.
    """
    # Adjust label_path if incorrectly set to image_path
    if label_path == image_path:
        label_path = None

    # Prepare save directory
    if osp.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    # Retrieve and sort image files
    image_list = os.listdir(image_path)
    sorted_image_list = natsort.natsorted(image_list)
    valid_extensions = [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]

    # Map class indices to class names
    id_to_classes = {i: c for i, c in enumerate(classes)}

    for frame_idx, image_name in enumerate(
        tqdm(sorted_image_list, colour="green")
    ):
        image_file = osp.join(image_path, image_name)
        # Skip non-image files or JSON annotation files
        if (
            image_name.endswith(".json")
            or osp.splitext(image_name)[-1] not in valid_extensions
        ):
            print(f"Invalid image format or JSON file: {image_file}")
            continue

        # Determine label file path
        label_name = osp.splitext(image_name)[0] + ".json"
        label_file = (
            osp.join(label_path, label_name)
            if label_path
            else osp.join(image_path, label_name)
        )

        # Read the image
        image = cv2.imread(image_file)
        save_name = (
            image_name
            if keep_ori_fn
            else f"annotated_frame_{frame_idx:05d}.jpg"
        )

        # If no label file exists, save the original image and proceed
        if not osp.exists(label_file):
            cv2.imwrite(osp.join(save_dir, save_name), image)
            continue

        # Load and parse annotation data
        with open(label_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Collect bounding box coordinates and class indices
        xyxy_list, cind_list = [], []
        for shape in data["shapes"]:
            if (
                shape["shape_type"] != "rectangle"
                or shape["label"] not in classes
            ):
                continue
            label_id = classes.index(shape["label"])
            cind_list.append(label_id)
            points = shape["points"]
            xyxy = (
                np.array(points[:2] + points[-2:], dtype=np.int32)
                if len(points) == 4
                else np.array(points, dtype=np.int32)
            )
            xyxy_list.append(xyxy)

        # If no rectangles found, save the original image and continue
        if not xyxy_list:
            cv2.imwrite(os.path.join(save_dir, save_name), image)
            continue

        # Prepare bounding boxes and Detection object
        xyxy = np.stack(xyxy_list, axis=0)
        object_ids = np.array(cind_list, dtype=np.int32)
        detections = sv.Detections(xyxy=xyxy, mask=None, class_id=object_ids)

        # Annotate the image with boxes and optionally labels
        annotated_frame = image.copy()
        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(
            scene=annotated_frame, detections=detections
        )
        if save_label:
            label_annotator = sv.LabelAnnotator()
            labels = [id_to_classes[i] for i in object_ids]
            annotated_frame = label_annotator.annotate(
                annotated_frame, detections=detections, labels=labels
            )

        # Save the annotated image
        cv2.imwrite(osp.join(save_dir, save_name), annotated_frame)
