# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.

import logging
from typing import Any, Dict, List, Optional

import numpy as np

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
    check_package_minimum_version,
    check_requirements,
)

logger = logging.getLogger(__name__)


class Yolov5DetectionModel(DetectionModel):
    def check_dependencies(self) -> None:
        check_requirements(["torch", "yolov5"])

    def load_model(self):
        """
        Detection model is initialized and set to self.model.
        """

        import yolov5

        try:
            model = yolov5.load(self.model_path, device=self.device)
            self.set_model(model)
        except Exception as e:
            raise TypeError("model_path is not a valid yolov5 model path: ", e)

    def set_model(self, model: Any):
        """
        Sets the underlying YOLOv5 model.
        Args:
            model: Any
                A YOLOv5 model
        """

        if model.__class__.__module__ not in [
            "yolov5.models.common",
            "models.common",
        ]:
            raise Exception(f"Not a yolov5 model: {type(model)}")

        model.conf = self.confidence_threshold
        self.model = model

        # set category_mapping
        if not self.category_mapping:
            category_mapping = {
                str(ind): category_name
                for ind, category_name in enumerate(self.category_names)
            }
            self.category_mapping = category_mapping

    def perform_inference(self, image: np.ndarray):
        """
        Prediction is performed using self.model and the prediction result is set to self._original_predictions.
        Args:
            image: np.ndarray
                A numpy array that contains the image to be predicted. 3 channel image should be in RGB order.
        """

        # Confirm model is loaded
        if self.model is None:
            raise ValueError(
                "Model is not loaded, load it by calling .load_model()"
            )
        if self.image_size is not None:
            prediction_result = self.model(image, size=self.image_size)
        else:
            prediction_result = self.model(image)

        self._original_predictions = prediction_result

    @property
    def num_categories(self):
        """
        Returns number of categories
        """
        return len(self.model.names)

    @property
    def has_mask(self):
        """
        Returns if model output contains segmentation mask
        """
        import yolov5
        from packaging import version

        if version.parse(yolov5.__version__) < version.parse("6.2.0"):
            return False
        else:
            return False  # fix when yolov5 supports segmentation models

    @property
    def category_names(self):
        if check_package_minimum_version("yolov5", "6.2.0"):
            return list(self.model.names.values())
        else:
            return self.model.names

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

        # handle all predictions
        object_prediction_list_per_image = []
        for image_ind, image_predictions_in_xyxy_format in enumerate(
            original_predictions.xyxy
        ):
            shift_amount = shift_amount_list[image_ind]
            full_shape = (
                None if full_shape_list is None else full_shape_list[image_ind]
            )
            object_prediction_list = []

            # process predictions
            for prediction in (
                image_predictions_in_xyxy_format.cpu().detach().numpy()
            ):
                x1 = prediction[0]
                y1 = prediction[1]
                x2 = prediction[2]
                y2 = prediction[3]
                bbox = [x1, y1, x2, y2]
                score = prediction[4]
                category_id = int(prediction[5])
                category_name = self.category_mapping[str(category_id)]

                # fix negative box coords
                bbox[0] = max(0, bbox[0])
                bbox[1] = max(0, bbox[1])
                bbox[2] = max(0, bbox[2])
                bbox[3] = max(0, bbox[3])

                # fix out of image box coords
                if full_shape is not None:
                    bbox[0] = min(full_shape[1], bbox[0])
                    bbox[1] = min(full_shape[0], bbox[1])
                    bbox[2] = min(full_shape[1], bbox[2])
                    bbox[3] = min(full_shape[0], bbox[3])

                # ignore invalid predictions
                if not (bbox[0] < bbox[2]) or not (bbox[1] < bbox[3]):
                    logger.warning(
                        f"ignoring invalid prediction with bbox: {bbox}"
                    )
                    continue

                object_prediction = ObjectPrediction(
                    bbox=bbox,
                    category_id=category_id,
                    score=score,
                    bool_mask=None,
                    category_name=category_name,
                    shift_amount=shift_amount,
                    full_shape=full_shape,
                )
                object_prediction_list.append(object_prediction)
            object_prediction_list_per_image.append(object_prediction_list)

        self._object_prediction_list_per_image = (
            object_prediction_list_per_image
        )
