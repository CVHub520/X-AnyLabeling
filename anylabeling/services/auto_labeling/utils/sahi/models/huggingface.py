# OBSS SAHI Tool
# Code written by Fatih C Akyon and Devrim Cavusoglu, 2022.

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import pybboxes.functional as pbf
except Exception:
    pass

from anylabeling.services.auto_labeling.utils.sahi.models.base import (
    DetectionModel,
)
from anylabeling.services.auto_labeling.utils.sahi.prediction import (
    ObjectPrediction,
)
from anylabeling.services.auto_labeling.utils.sahi.utils.compatibility import (
    fix_full_shape_list,
    fix_shift_amount_list,
)
from anylabeling.services.auto_labeling.utils.sahi.utils.import_utils import (
    check_requirements,
    ensure_package_minimum_version,
)

logger = logging.getLogger(__name__)


class HuggingfaceDetectionModel(DetectionModel):
    def __init__(
        self,
        model_path: Optional[str] = None,
        model: Optional[Any] = None,
        processor: Optional[Any] = None,
        config_path: Optional[str] = None,
        device: Optional[str] = None,
        mask_threshold: float = 0.5,
        confidence_threshold: float = 0.3,
        category_mapping: Optional[Dict] = None,
        category_remapping: Optional[Dict] = None,
        load_at_init: bool = True,
        image_size: int = None,
    ):
        self._processor = processor
        self._image_shapes = []
        super().__init__(
            model_path,
            model,
            config_path,
            device,
            mask_threshold,
            confidence_threshold,
            category_mapping,
            category_remapping,
            load_at_init,
            image_size,
        )

    def check_dependencies(self):
        check_requirements(["torch", "transformers"])
        ensure_package_minimum_version("transformers", "4.25.1")

    @property
    def processor(self):
        return self._processor

    @property
    def image_shapes(self):
        return self._image_shapes

    @property
    def num_categories(self) -> int:
        """
        Returns number of categories
        """
        return self.model.config.num_labels

    def load_model(self):
        from transformers import AutoModelForObjectDetection, AutoProcessor

        model = AutoModelForObjectDetection.from_pretrained(self.model_path)
        if self.image_size is not None:
            processor = AutoProcessor.from_pretrained(
                self.model_path,
                size={"shortest_edge": self.image_size, "longest_edge": None},
                do_resize=True,
            )
        else:
            processor = AutoProcessor.from_pretrained(self.model_path)
        self.set_model(model, processor)

    def set_model(self, model: Any, processor: Any = None):
        processor = processor or self.processor
        if processor is None:
            raise ValueError(
                f"'processor' is required to be set, got {processor}."
            )
        elif (
            "ObjectDetection" not in model.__class__.__name__
            or "ImageProcessor" not in processor.__class__.__name__
        ):
            raise ValueError(
                "Given 'model' is not an ObjectDetectionModel or 'processor' is not a valid ImageProcessor."
            )
        self.model = model
        self.model.to(self.device)
        self._processor = processor
        self.category_mapping = self.model.config.id2label

    def perform_inference(self, image: Union[List, np.ndarray]):
        """
        Prediction is performed using self.model and the prediction result is set to self._original_predictions.
        Args:
            image: np.ndarray
                A numpy array that contains the image to be predicted. 3 channel image should be in RGB order.
        """
        import torch

        # Confirm model is loaded
        if self.model is None:
            raise RuntimeError(
                "Model is not loaded, load it by calling .load_model()"
            )

        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt")
            inputs["pixel_values"] = inputs.pixel_values.to(self.device)
            if hasattr(inputs, "pixel_mask"):
                inputs["pixel_mask"] = inputs.pixel_mask.to(self.device)
            outputs = self.model(**inputs)

        if isinstance(image, list):
            self._image_shapes = [img.shape for img in image]
        else:
            self._image_shapes = [image.shape]
        self._original_predictions = outputs

    def get_valid_predictions(self, logits, pred_boxes) -> Tuple:
        """
        Args:
            logits: torch.Tensor
            pred_boxes: torch.Tensor
        Returns:
            scores: torch.Tensor
            cat_ids: torch.Tensor
            boxes: torch.Tensor
        """
        import torch

        probs = logits.softmax(-1)
        scores = probs.max(-1).values
        cat_ids = probs.argmax(-1)
        valid_detections = torch.where(cat_ids < self.num_categories, 1, 0)
        valid_confidences = torch.where(
            scores >= self.confidence_threshold, 1, 0
        )
        valid_mask = valid_detections.logical_and(valid_confidences)
        scores = scores[valid_mask]
        cat_ids = cat_ids[valid_mask]
        boxes = pred_boxes[valid_mask]
        return scores, cat_ids, boxes

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: Optional[List[List[int]]] = [[0, 0]],
        full_shape_list: Optional[List[List[int]]] = None,
    ):
        """
        self._original_predictions is converted to a list of prediction.ObjectPrediction and set to
        self._object_prediction_list_per_image.
        Args:
            shift_amount_list: list of list
                To shift the box and mask predictions from sliced image to full sized image, should
                be in the form of List[[shift_x, shift_y],[shift_x, shift_y],...]
            full_shape_list: list of list
                Size of the full image after shifting, should be in the form of
                List[[height, width],[height, width],...]
        """
        original_predictions = self._original_predictions

        # compatilibty for sahi v0.8.15
        shift_amount_list = fix_shift_amount_list(shift_amount_list)
        full_shape_list = fix_full_shape_list(full_shape_list)

        n_image = original_predictions.logits.shape[0]
        object_prediction_list_per_image = []
        for image_ind in range(n_image):
            image_height, image_width, _ = self.image_shapes[image_ind]
            scores, cat_ids, boxes = self.get_valid_predictions(
                logits=original_predictions.logits[image_ind],
                pred_boxes=original_predictions.pred_boxes[image_ind],
            )

            # create object_prediction_list
            object_prediction_list = []

            shift_amount = shift_amount_list[image_ind]
            full_shape = (
                None if full_shape_list is None else full_shape_list[image_ind]
            )

            for ind in range(len(boxes)):
                category_id = cat_ids[ind].item()
                yolo_bbox = boxes[ind].tolist()
                bbox = list(
                    pbf.convert_bbox(
                        yolo_bbox,
                        from_type="yolo",
                        to_type="voc",
                        image_size=(image_width, image_height),
                        return_values=True,
                        strict=False,
                    )
                )

                # fix negative box coords
                bbox[0] = max(0, bbox[0])
                bbox[1] = max(0, bbox[1])
                bbox[2] = min(bbox[2], image_width)
                bbox[3] = min(bbox[3], image_height)

                object_prediction = ObjectPrediction(
                    bbox=bbox,
                    bool_mask=None,
                    category_id=category_id,
                    category_name=self.category_mapping[category_id],
                    shift_amount=shift_amount,
                    score=scores[ind].item(),
                    full_shape=full_shape,
                )
                object_prediction_list.append(object_prediction)
            object_prediction_list_per_image.append(object_prediction_list)

        self._object_prediction_list_per_image = (
            object_prediction_list_per_image
        )
