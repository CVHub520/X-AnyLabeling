import os
import traceback
from typing import List, Optional, Tuple

import cv2
import numpy as np

from PyQt6 import QtCore
from PyQt6.QtCore import QCoreApplication

from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img
from anylabeling.services.auto_labeling.utils import (
    calculate_rotation_theta,
)

from .model import Model
from .types import AutoLabelingResult

try:
    import sam3cpp

    SAM3CPP_AVAILABLE = True
except ImportError:
    SAM3CPP_AVAILABLE = False


class SegmentAnything3Video(Model):
    """Video segmentation model backed by the sam3.cpp C++/Metal engine.

    Wraps the ``sam3cpp`` Python bindings around PABannier/sam3.cpp. Supports
    the visual-tracking subset of the C++ API (point/box prompts on a single
    frame plus memory-bank propagation across subsequent frames) for SAM 3,
    SAM 3 Visual, and EdgeTAM weights distributed in GGML format.

    Build instructions for the ``sam3cpp`` module live in the
    ``examples/interactive_video_object_segmentation/sam3cpp`` tutorial.
    """

    class Meta:
        """Required configuration keys and the UI widgets to surface."""

        required_config_names = [
            "type",
            "name",
            "display_name",
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
        """Initialise the model from a config and a status callback.

        Args:
            config_path: Path to the YAML config file or a parsed config
                dict.
            on_message: Callback used to emit user-facing status messages.

        Raises:
            ImportError: If the ``sam3cpp`` Python module is not importable.
            FileNotFoundError: If the resolved model weight file cannot be
                found on disk after download.
        """
        if not SAM3CPP_AVAILABLE:
            raise ImportError(
                "SegmentAnything3Video (sam3.cpp) model will not be "
                "available. Please install the sam3cpp Python bindings "
                "from PABannier/sam3.cpp and try again."
            )

        super().__init__(config_path, on_message)

        self.model_abs_path = self.get_model_abs_path(
            self.config, "model_path"
        )
        if not self.model_abs_path or not os.path.isfile(self.model_abs_path):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model",
                    "Could not find sam3.cpp model weight file at: "
                    "{model_path}",
                ).format(model_path=self.model_abs_path)
            )

        params = sam3cpp.Params()
        params.model_path = self.model_abs_path
        params.n_threads = self.config.get("n_threads", 4)
        params.use_gpu = self.config.get("use_gpu", True)

        backend = "GPU" if params.use_gpu else "CPU"
        logger.info(f"Loading sam3.cpp model on {backend}...")
        self.sam3_model = sam3cpp.load_model(params)
        self.sam3_state = sam3cpp.create_state(self.sam3_model, params)

        # Tracker is created lazily on first prompted frame.
        self.visual_track_params = sam3cpp.VisualTrackParams()
        self.sam3_tracker = None

        self.is_first_init = True
        self.marks: List[dict] = []
        self.labels: List[str] = []
        self.group_ids: List[int] = []
        self.prompts: List[dict] = []
        self.replace = True
        self.epsilon = 0.001

    def set_mask_fineness(self, epsilon: float) -> None:
        """Set the contour-approximation epsilon used in post-processing.

        Args:
            epsilon: Multiplier of the contour arc length passed to
                ``cv2.approxPolyDP``. Smaller values produce finer
                polygons.
        """
        self.epsilon = epsilon

    def set_auto_labeling_marks(self, marks: List[dict]) -> None:
        """Cache the user's marks (points and/or rectangle) for the next call.

        Args:
            marks: List of mark dicts emitted by the canvas widget. Each
                dict has a ``type`` of ``"point"`` or ``"rectangle"`` and
                a ``data`` payload appropriate to that type.
        """
        self.marks = marks

    def set_auto_labeling_preserve_existing_annotations_state(
        self, state: bool
    ) -> None:
        """Toggle whether predictions replace or extend existing annotations.

        Args:
            state: If ``True``, predictions are appended; if ``False``,
                they replace existing shapes for the frame.
        """
        self.replace = not state

    def set_cache_auto_label(self, text: str, gid: int) -> None:
        """Queue a label and group id for the next finished object.

        Args:
            text: Label string to attach to the next prompt.
            gid: Group id to attach to the next prompt.
        """
        self.labels.append(text)
        self.group_ids.append(gid)

    def set_auto_labeling_reset_tracker(self) -> None:
        """Reset the tracker, clearing all instances and queued prompts."""
        self.is_first_init = True
        if self.sam3_tracker is not None:
            try:
                sam3cpp.tracker_reset(self.sam3_tracker)
                logger.info(
                    "Successful: The sam3.cpp visual tracker has been reset."
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(f"sam3.cpp tracker reset failed: {e}")
            self.sam3_tracker = None

        self.prompts = []
        self.labels = []
        self.group_ids = []

    def set_auto_labeling_prompt(self) -> None:
        """Convert cached marks into a sam3.cpp ``PvsParams`` prompt entry.

        Pops one entry from ``self.labels`` and ``self.group_ids`` (if
        present) and stores the resulting prompt in ``self.prompts``. Box
        prompts take precedence over point prompts when both are present.
        """
        point_coords, point_labels, box = self.marks_to_prompts()

        pvs = sam3cpp.PvsParams()
        pvs.multimask = False
        pvs.use_box = False

        if box:
            prompt_geom = {
                "type": "rectangle",
                "data": np.array([[*box[:2]], [*box[2:]]], dtype=np.float32),
            }
            pvs.box = sam3cpp.Box(box[0], box[1], box[2], box[3])
            pvs.use_box = True
        elif point_coords:
            prompt_geom = {
                "type": "points",
                "data": np.array(point_coords, dtype=np.float32),
                "label": np.array(point_labels, dtype=np.int32),
            }
            pos_points: List = []
            neg_points: List = []
            for i, p in enumerate(point_coords):
                if point_labels[i] == 1:
                    pos_points.append(sam3cpp.Point(p[0], p[1]))
                else:
                    neg_points.append(sam3cpp.Point(p[0], p[1]))
            pvs.pos_points = pos_points
            pvs.neg_points = neg_points
        else:
            return

        try:
            gid = self.group_ids.pop(0)
        except IndexError:
            gid = 0

        popped_label = self.labels.pop(0) if self.labels else "object"
        if not popped_label:
            popped_label = "object"

        self.prompts.append(
            {
                "prompt": prompt_geom,
                "pvs": pvs,
                "label": popped_label,
                "group_id": gid,
            }
        )

    def marks_to_prompts(
        self,
    ) -> Tuple[Optional[List], Optional[List], Optional[List]]:
        """Split cached marks into point coords, point labels, and a box.

        Returns:
            A ``(point_coords, point_labels, box)`` tuple. Each entry is
            ``None`` when no mark of that type is present.
        """
        point_coords: Optional[List] = None
        point_labels: Optional[List] = None
        box: Optional[List] = None
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

    def predict_shapes(
        self, image, filename: Optional[str] = None, run_tracker: bool = False
    ) -> AutoLabelingResult:
        """Run segmentation or video tracking on a single frame.

        Three execution paths share this entry point:

        - ``run_tracker=False``: single-frame segmentation from the
          currently cached marks. The image is encoded once and the PVS
          decoder is run; tracker state is not touched.
        - ``run_tracker=True`` and ``is_first_init=True``: encodes the
          first frame, creates a fresh visual tracker, and registers each
          queued prompt as an instance.
        - ``run_tracker=True`` and ``is_first_init=False``: propagates
          the existing tracker to the next frame.

        Args:
            image: ``QImage`` for the current frame.
            filename: Filename of the current frame. Unused by sam3.cpp
                but accepted for interface parity with other models.
            run_tracker: When ``True``, run the video tracker; when
                ``False``, run single-frame segmentation only.

        Returns:
            An ``AutoLabelingResult`` containing the produced shapes and
            the configured replace flag.
        """
        if not hasattr(self, "sam3_model"):
            return AutoLabelingResult([], replace=self.replace)

        cv_image = qt_img_to_rgb_cv_img(image, filename)
        if cv_image is None or cv_image.size == 0:
            logger.warning("Empty or invalid image data for sam3.cpp.")
            return AutoLabelingResult([], replace=self.replace)

        res_shapes: List[Shape] = []

        if not run_tracker:
            try:
                point_coords, point_labels, box = self.marks_to_prompts()
                if not box and not (point_coords and point_labels):
                    return AutoLabelingResult([], replace=self.replace)

                pvs = sam3cpp.PvsParams()
                pvs.multimask = False
                pvs.use_box = False

                if box:
                    pvs.box = sam3cpp.Box(box[0], box[1], box[2], box[3])
                    pvs.use_box = True
                if point_coords:
                    pos_points: List = []
                    neg_points: List = []
                    for i, p in enumerate(point_coords):
                        if point_labels[i] == 1:
                            pos_points.append(sam3cpp.Point(p[0], p[1]))
                        else:
                            neg_points.append(sam3cpp.Point(p[0], p[1]))
                    pvs.pos_points = pos_points
                    pvs.neg_points = neg_points

                sam3cpp.encode_image(
                    self.sam3_state, self.sam3_model, cv_image
                )
                result = sam3cpp.segment_pvs(
                    self.sam3_state, self.sam3_model, pvs
                )
                if result.detections:
                    det = result.detections[0]
                    mask_np = det.mask.to_numpy()
                    label = self.labels[0] if self.labels else "object"
                    gid = self.group_ids[0] if self.group_ids else 0
                    res_shapes.extend(self.post_process(mask_np, label, gid))
            except Exception as e:  # noqa: BLE001
                logger.error(f"Error during sam3.cpp static segmentation: {e}")
                traceback.print_exc()
        else:
            if self.is_first_init:
                if self.marks:
                    self.set_auto_labeling_prompt()
                try:
                    logger.info(
                        f"sam3 tracker init: {len(self.prompts)} prompts, "
                        f"marks={len(self.marks)}"
                    )
                    sam3cpp.encode_image(
                        self.sam3_state, self.sam3_model, cv_image
                    )
                    self.sam3_tracker = sam3cpp.create_visual_tracker(
                        self.sam3_model, self.visual_track_params
                    )
                    for i, prompt in enumerate(self.prompts):
                        inst_id = sam3cpp.tracker_add_instance(
                            self.sam3_tracker,
                            self.sam3_state,
                            self.sam3_model,
                            prompt["pvs"],
                        )
                        logger.info(
                            f"sam3 tracker: added instance {inst_id} "
                            f"(prompt {i+1}/{len(self.prompts)}, "
                            f"label={prompt['label']})"
                        )
                        result = sam3cpp.segment_pvs(
                            self.sam3_state,
                            self.sam3_model,
                            prompt["pvs"],
                        )
                        if result.detections:
                            det = result.detections[0]
                            mask_np = det.mask.to_numpy()
                            res_shapes.extend(
                                self.post_process(
                                    mask_np,
                                    prompt["label"],
                                    prompt["group_id"],
                                )
                            )
                    self.is_first_init = False
                except Exception as e:  # noqa: BLE001
                    logger.error(f"Error during sam3.cpp tracker init: {e}")
                    traceback.print_exc()
            else:
                if self.sam3_tracker is not None:
                    try:
                        result = sam3cpp.propagate_frame(
                            self.sam3_tracker,
                            self.sam3_state,
                            self.sam3_model,
                            cv_image,
                        )
                        idx = 0
                        for det in result.detections:
                            mask_np = det.mask.to_numpy()
                            label = (
                                self.prompts[idx]["label"]
                                if idx < len(self.prompts)
                                else f"object_{det.instance_id}"
                            )
                            group_id = (
                                self.prompts[idx]["group_id"]
                                if idx < len(self.prompts)
                                else det.instance_id
                            )
                            res_shapes.extend(
                                self.post_process(mask_np, label, group_id)
                            )
                            idx += 1
                    except Exception as e:  # noqa: BLE001
                        logger.error(f"Error during sam3.cpp track frame: {e}")
                        traceback.print_exc()

        return AutoLabelingResult(res_shapes, replace=self.replace)

    def post_process(
        self, masks: np.ndarray, label: Optional[str], group_id: int
    ) -> List[Shape]:
        """Convert a binary mask into output shapes for the current mode.

        Args:
            masks: HxW array with positive logits where the mask is set.
            label: Label string for produced shapes. ``None`` is replaced
                with ``"object"``.
            group_id: Group id assigned to all produced shapes.

        Returns:
            A list of :class:`Shape` objects in the format dictated by
            ``self.output_mode`` (polygon, rectangle, or rotation).
        """
        if label is None:
            label = "object"

        masks[masks > 0.0] = 255
        masks[masks <= 0.0] = 0
        masks = masks.astype(np.uint8)

        contours, _ = cv2.findContours(
            masks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        approx_contours = []
        for contour in contours:
            epsilon = self.epsilon * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            approx_contours.append(approx)

        if len(approx_contours) > 1:
            image_size = masks.shape[0] * masks.shape[1]
            areas = [cv2.contourArea(c) for c in approx_contours]
            approx_contours = [
                c
                for c, area in zip(approx_contours, areas)
                if area < image_size * 0.9
            ]

        if len(approx_contours) > 1:
            areas = [cv2.contourArea(c) for c in approx_contours]
            avg_area = np.mean(areas)
            approx_contours = [
                c
                for c, area in zip(approx_contours, areas)
                if area > avg_area * 0.2
            ]

        if len(approx_contours) < 1:
            return []

        shapes: List[Shape] = []
        if self.output_mode == "polygon":
            for approx in approx_contours:
                points = approx.reshape(-1, 2).tolist()
                if len(points) < 3:
                    continue
                points.append(points[0])
                shape = Shape(flags={})
                for point in points:
                    shape.add_point(
                        QtCore.QPointF(int(point[0]), int(point[1]))
                    )
                shape.shape_type = "polygon"
                shape.group_id = group_id
                shape.closed = True
                shape.label = label
                shape.selected = False
                shapes.append(shape)

        elif self.output_mode == "rectangle":
            x_min = 100000000
            y_min = 100000000
            x_max = 0
            y_max = 0
            for approx in approx_contours:
                points = approx.reshape(-1, 2).tolist()
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
            shape.group_id = group_id
            shape.fill_color = "#000000"
            shape.line_color = "#000000"
            shape.label = label
            shape.selected = False
            shapes.append(shape)

        elif self.output_mode == "rotation":
            shape = Shape(flags={})
            rotation_box = cv2.boxPoints(cv2.minAreaRect(approx_contours[0]))
            for point in rotation_box:
                shape.add_point(QtCore.QPointF(int(point[0]), int(point[1])))
            shape.direction = calculate_rotation_theta(rotation_box)
            shape.shape_type = self.output_mode
            shape.closed = True
            shape.fill_color = "#000000"
            shape.line_color = "#000000"
            shape.group_id = group_id
            shape.label = label
            shape.selected = False
            shapes.append(shape)

        return shapes

    def unload(self) -> None:
        """Release tracker, state, and model in dependency order."""
        if getattr(self, "sam3_tracker", None) is not None:
            del self.sam3_tracker
            self.sam3_tracker = None
        if hasattr(self, "sam3_state"):
            del self.sam3_state
        if hasattr(self, "sam3_model"):
            del self.sam3_model
