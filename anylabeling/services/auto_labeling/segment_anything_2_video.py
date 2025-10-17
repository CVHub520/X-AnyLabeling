import warnings

warnings.filterwarnings("ignore")

import os
import cv2
import traceback
import numpy as np

from PyQt5 import QtCore
from PyQt5.QtCore import QCoreApplication

from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.opencv import (
    get_bounding_boxes,
    qt_img_to_rgb_cv_img,
)
from anylabeling.services.auto_labeling.utils import calculate_rotation_theta

from .model import Model
from .types import AutoLabelingResult

try:
    import torch
    from sam2.build_sam import build_sam2, build_sam2_camera_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    SAM2_VIDEO_AVAILABLE = True
except ImportError:
    SAM2_VIDEO_AVAILABLE = False


class SegmentAnything2Video(Model):
    """Segmentation model using SegmentAnything2 for video processing.

    This class provides methods to perform image segmentation on video frames
    using the SegmentAnything2 model. It supports interactive marking and
    tracking of objects across frames.
    """

    class Meta:
        """Meta class to define required configurations and UI elements."""

        required_config_names = [
            "type",
            "name",
            "display_name",
            "model_cfg",
            "model_path",
        ]
        widgets = [
            "output_label",
            "output_select_combobox",
            "button_add_point",
            "button_remove_point",
            "button_add_rect",
            "button_clear",
            "button_finish_object",
            "button_auto_decode",
            "button_reset_tracker",
            "toggle_preserve_existing_annotations",
            "mask_fineness_slider",
            "mask_fineness_value_label",
        ]
        output_modes = {
            "polygon": QCoreApplication.translate("Model", "Polygon"),
            "rectangle": QCoreApplication.translate("Model", "Rectangle"),
            "rotation": QCoreApplication.translate("Model", "Rotation"),
        }
        default_output_mode = "polygon"

    def __init__(self, config_path, on_message) -> None:
        """Initialize the segmentation model with given configuration.

        Args:
            config_path (str): Path to the configuration file.
            on_message (callable): Callback for logging messages.
        """

        if not SAM2_VIDEO_AVAILABLE:
            message = "SegmentAnything2Video model will not be available. Please install related packages and try again."
            raise ImportError(message)

        super().__init__(config_path, on_message)

        device_type = self.config.get("device_type", "cuda")
        if device_type == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
        elif device_type == "mps" and torch.backends.mps.is_available():
            device = torch.device("mps")
            # if using Apple MPS, fall back to CPU for unsupported ops
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        else:
            device = torch.device("cpu")
        logger.info(f"Using device: {device}")

        if device.type == "cuda":
            apply_postprocessing = True
            # Enable automatic mixed precision for faster computations
            torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                # turn on tfloat32 for Ampere GPUs
                # (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        elif device.type == "mps":
            apply_postprocessing = True
            logger.warning(
                "Support for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
                "give numerically different outputs and sometimes degraded performance on MPS. "
                "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
            )
        elif device.type == "cpu":
            apply_postprocessing = False
            logger.warning(
                "Support for CPU devices is preliminary. SAM 2 is trained with CUDA and might "
                "give numerically different outputs and sometimes degraded performance on CPU. "
                "The post-processing step (removing small holes and sprinkles in the output masks) "
                "will be skipped, but this shouldn't affect the results in most cases."
            )

        # Load the SAM2 predictor models
        self.model_abs_path = self.get_model_abs_path(
            self.config, "model_path"
        )
        if not self.model_abs_path or not os.path.isfile(self.model_abs_path):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model",
                    "Could not download or initialize model of Segment Anything 2.",
                )
            )
        self.model_cfg = self.config["model_cfg"]
        sam2_image_model = build_sam2(
            self.model_cfg, self.model_abs_path, device=device
        )
        self.image_predictor = SAM2ImagePredictor(sam2_image_model)
        self.video_predictor = build_sam2_camera_predictor(
            self.model_cfg,
            self.model_abs_path,
            device=device,
            apply_postprocessing=apply_postprocessing,
        )
        self.is_first_init = True

        # Initialize marking and prompting structures
        self.marks = []
        self.labels = []
        self.group_ids = []
        self.prompts = []
        self.replace = True
        self.epsilon = 0.001

    def set_mask_fineness(self, epsilon):
        """Set mask fineness epsilon value"""
        self.epsilon = epsilon

    def set_auto_labeling_marks(self, marks):
        """Set marks for auto labeling.

        Args:
            marks (list): List of marks (points or rectangles).
        """
        self.marks = marks

    def set_auto_labeling_preserve_existing_annotations_state(self, state):
        """Toggle the preservation of existing annotations based on the checkbox state."""
        self.replace = not state

    def set_cache_auto_label(self, text, gid):
        """Set cache auto label"""
        self.labels.append(text)
        self.group_ids.append(gid)

    def set_auto_labeling_reset_tracker(self):
        """Reset the tracker to its initial state."""
        self.is_first_init = True
        if self.prompts:
            try:
                self.video_predictor.reset_state()
                logger.info(
                    "Successful: The tracker has been reset to its initial state."
                )
            except Exception as e:  # noqa
                pass
            self.prompts = []
            self.labels = []
            self.group_ids = []

    def set_auto_labeling_prompt(self):
        """Convert marks to prompts for the model."""
        point_coords, point_labels, box = self.marks_to_prompts()
        if box:
            promot = {
                "type": "rectangle",
                "data": np.array([[*box[:2]], [*box[2:]]], dtype=np.float32),
            }
            self.prompts.append(promot)
        elif point_coords and point_labels:
            promot = {
                "type": "point",
                "data": {
                    "point_coords": np.array(point_coords, dtype=np.float32),
                    "point_labels": np.array(point_labels, dtype=np.int32),
                },
            }
            self.prompts.append(promot)

    def marks_to_prompts(self):
        """Convert marks to prompts for the model."""
        point_coords, point_labels, box = None, None, None
        for marks in self.marks:
            if marks["type"] == "rectangle":
                box = marks["data"]
            elif marks["type"] == "point":
                if point_coords is None and point_labels is None:
                    point_coords = [marks["data"]]
                    point_labels = [marks["label"]]
                else:
                    point_coords.append(marks["data"])
                    point_labels.append(marks["label"])
        return point_coords, point_labels, box

    def post_process(self, masks, index=None):
        """Post-process the masks produced by the model.

        Args:
            masks (np.array): The masks to post-process.
            index (int, optional): The index of the mask. Defaults to None.

        Returns:
            list: A list of Shape objects representing the masks.
        """
        # Convert masks to binary format
        masks[masks > 0.0] = 255
        masks[masks <= 0.0] = 0
        masks = masks.astype(np.uint8)

        # Find contours of the masks
        contours, _ = cv2.findContours(
            masks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        # Refine and filter contours
        approx_contours = []
        for contour in contours:
            # Approximate contour using configurable epsilon
            epsilon = self.epsilon * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            approx_contours.append(approx)

        # Remove large contours (likely background)
        if len(approx_contours) > 1:
            image_size = masks.shape[0] * masks.shape[1]
            areas = [cv2.contourArea(contour) for contour in approx_contours]
            filtered_approx_contours = [
                contour
                for contour, area in zip(approx_contours, areas)
                if area < image_size * 0.9
            ]

        # Remove small contours (likely noise)
        if len(approx_contours) > 1:
            areas = [cv2.contourArea(contour) for contour in approx_contours]
            avg_area = np.mean(areas)

            filtered_approx_contours = [
                contour
                for contour, area in zip(approx_contours, areas)
                if area > avg_area * 0.2
            ]
            approx_contours = filtered_approx_contours

        if len(approx_contours) < 1:
            return []

        # Convert contours to shapes
        shapes = []
        if self.output_mode == "polygon":
            for approx in approx_contours:
                # Scale points
                points = approx.reshape(-1, 2)
                points[:, 0] = points[:, 0]
                points[:, 1] = points[:, 1]
                points = points.tolist()
                if len(points) < 3:
                    continue
                points.append(points[0])
                shape = Shape(flags={})
                for point in points:
                    point[0] = int(point[0])
                    point[1] = int(point[1])
                    shape.add_point(QtCore.QPointF(point[0], point[1]))
                # Create Polygon shape
                shape.shape_type = "polygon"
                shape.group_id = (
                    self.group_ids[index] if index is not None else None
                )
                shape.closed = True
                shape.label = (
                    "AUTOLABEL_OBJECT" if index is None else self.labels[index]
                )
                shape.selected = False
                shapes.append(shape)
        elif self.output_mode == "rectangle":
            x_min = 100000000
            y_min = 100000000
            x_max = 0
            y_max = 0
            for approx in approx_contours:
                points = approx.reshape(-1, 2)
                points[:, 0] = points[:, 0]
                points[:, 1] = points[:, 1]
                points = points.tolist()
                if len(points) < 3:
                    continue

                for point in points:
                    x_min = min(x_min, point[0])
                    y_min = min(y_min, point[1])
                    x_max = max(x_max, point[0])
                    y_max = max(y_max, point[1])

            shape = Shape(flags={})
            shape.add_point(QtCore.QPointF(x_min, y_min))
            shape.add_point(QtCore.QPointF(x_max, y_min))
            shape.add_point(QtCore.QPointF(x_max, y_max))
            shape.add_point(QtCore.QPointF(x_min, y_max))
            shape.shape_type = "rectangle"
            shape.closed = True
            shape.group_id = (
                self.group_ids[index] if index is not None else None
            )
            shape.fill_color = "#000000"
            shape.line_color = "#000000"
            shape.label = (
                "AUTOLABEL_OBJECT" if index is None else self.labels[index]
            )
            shape.selected = False
            shapes.append(shape)
        elif self.output_mode == "rotation":
            shape = Shape(flags={})
            rotation_box = get_bounding_boxes(approx_contours[0])[1]
            for point in rotation_box:
                shape.add_point(QtCore.QPointF(int(point[0]), int(point[1])))
            shape.direction = calculate_rotation_theta(rotation_box)
            shape.shape_type = self.output_mode
            shape.closed = True
            shape.fill_color = "#000000"
            shape.line_color = "#000000"
            shape.label = (
                "AUTOLABEL_OBJECT" if index is None else self.labels[index]
            )
            shape.selected = False
            shapes.append(shape)

        return shapes

    def image_process(self, rgb_image):
        """Process a single image using the SAM2 predictor.

        Args:
            rgb_image (np.array): The RGB image to process.

        Returns:
            list: A list of Shape objects representing the segmented regions.
        """
        self.image_predictor.set_image(rgb_image)

        # prompt SAM 2 image predictor to get the mask for the object
        point_coords, point_labels, box = self.marks_to_prompts()
        if not box and not (point_coords and point_labels):
            return []
        masks, _, _ = self.image_predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=False,
        )

        if len(masks.shape) == 4:
            masks = masks[0][0]
        else:
            masks = masks[0]
        shapes = self.post_process(masks)
        return shapes

    def video_process(self, cv_image, filename):
        """Process a video frame using the SAM2 predictor.

        Args:
            cv_image (np.array): The OpenCV image to process.
            filename (str): The filename of the image.

        Returns:
            tuple: A tuple containing a list of Shape objects and a boolean indicating if the frame was replaced.
        """
        if not self.prompts:
            return [], False

        if not any(
            filename.endswith(ext)
            for ext in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ):
            logger.warning(
                f"Only JPEG format is supported, but got {filename}"
            )
            return [], False

        if self.is_first_init:
            self.video_predictor.load_first_frame(cv_image)
            ann_frame_idx = 0
            for i, prompt in enumerate(self.prompts):
                ann_obj_id = (
                    i + 1
                )  # give a unique id to each object we interact with (it can be any integers)
                if prompt["type"] == "rectangle":
                    bbox = prompt["data"]
                    (
                        _,
                        out_obj_ids,
                        out_mask_logits,
                    ) = self.video_predictor.add_new_prompt(
                        frame_idx=ann_frame_idx,
                        obj_id=ann_obj_id,
                        bbox=bbox,
                    )
                elif prompt["type"] == "point":
                    points = prompt["data"]["point_coords"]
                    labels = prompt["data"]["point_labels"]
                    (
                        _,
                        out_obj_ids,
                        out_mask_logits,
                    ) = self.video_predictor.add_new_prompt(
                        frame_idx=ann_frame_idx,
                        obj_id=ann_obj_id,
                        points=points,
                        labels=labels,
                    )
            self.is_first_init = False
            return [], False
        else:
            shapes = []
            out_obj_ids, out_mask_logits = self.video_predictor.track(cv_image)
            for i in range(0, len(out_obj_ids)):
                masks = out_mask_logits[i].cpu().numpy()
                if len(masks.shape) == 4:
                    masks = masks[0][0]
                else:
                    masks = masks[0]
                shapes.extend(self.post_process(masks, i))
            return shapes, self.replace

    def predict_shapes(
        self, image, filename=None, run_tracker=False
    ) -> AutoLabelingResult:
        """Predict shapes from an image or video frame.

        Args:
            image (QtImage): The image to process.
            filename (str, optional): The filename of the image. Required for video processing. Defaults to None.
            run_tracker (bool, optional): Whether to run the tracker. Defaults to False.

        Returns:
            AutoLabelingResult: The result containing the predicted shapes and a flag indicating if the frame was replaced.
        """
        if image is None or not self.marks:
            return AutoLabelingResult([], replace=False)

        shapes = []
        cv_image = qt_img_to_rgb_cv_img(image, filename)
        try:
            if run_tracker is True:
                shapes, replace = self.video_process(cv_image, filename)
                result = AutoLabelingResult(shapes, replace=replace)
            else:
                shapes = self.image_process(cv_image)
                result = AutoLabelingResult(shapes, replace=False)
        except Exception as e:  # noqa
            logger.warning("Could not inference model")
            logger.warning(e)
            traceback.print_exc()
            return AutoLabelingResult([], replace=False)

        return result

    @staticmethod
    def get_ann_frame_idx(filename):
        """Get the annotation frame index for a given filename.

        Args:
            filename (str): The filename of the image.

        Returns:
            int: The index of the frame in the sorted list of frames, or -1 if not found.
        """
        frame_names = [
            p
            for p in os.listdir(os.path.dirname(filename))
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        if not frame_names:
            return -1
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        return frame_names.index(os.path.basename(filename))

    def unload(self):
        """Unload the model and predictors."""
        del self.image_predictor
        del self.video_predictor
