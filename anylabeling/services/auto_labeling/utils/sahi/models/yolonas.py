# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from yaml import safe_load

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
)

logger = logging.getLogger(__name__)


class YoloNasDetectionModel(DetectionModel):
    def __init__(
        self,
        model_name: str,
        model_path: Optional[str] = None,
        class_names_yaml_path: Optional[List[str]] = None,
        **kwargs,
    ):
        if model_name is not None and not isinstance(model_name, str):
            raise TypeError(
                f"model_name should be a string, got {model_name} with type of '{model_name.__class__.__name__}'"
            )
        if model_name not in ["yolo_nas_s", "yolo_nas_m", "yolo_nas_l"]:
            raise ValueError(f"Unsupported model type {model_name}")
        if (
            not model_path
        ):  # use pretrained models downloaded from Deci-AI remote client
            self.pretrained_weights = "coco"
            self.class_names = None
            self.num_classes = None
        else:  # use local / custom trained models
            self.pretrained_weights = None
            if not class_names_yaml_path:
                raise ValueError(
                    "'class_names_yaml_path' should be provided for the models that have custom class mapping"
                )
            with open(class_names_yaml_path, "r") as fs:
                yaml_content = safe_load(fs)
                if not isinstance(yaml_content, list):
                    raise ValueError(
                        "Invalid yaml file format, make sure your class names are given in list format in yaml"
                    )
                self.class_names = yaml_content
            self.num_classes = len(self.class_names)
        self.model_name = model_name
        super().__init__(model_path=model_path, **kwargs)

    def check_dependencies(self) -> None:
        check_requirements(["torch", "super_gradients"])

    def load_model(self):
        """
        Detection model is initialized and set to self.model.
        """
        from super_gradients.training import models

        try:
            model = models.get(
                model_name=self.model_name,
                checkpoint_path=self.model_path,
                pretrained_weights=self.pretrained_weights,
                num_classes=self.num_classes,
            ).to(device=self.device)
            self.set_model(model)
        except Exception as e:
            raise TypeError(
                "Load model failed. Provided model weights and model_name might be mismatching. ",
                e,
            )

    def set_model(self, model: Any):
        """
        Sets the underlying YoloNas model.
        Args:
            model: Any
                A YoloNas model
        """
        from super_gradients.training.processing.processing import (
            get_pretrained_processing_params,
        )

        if model.__class__.__module__.split(".")[-1] != "yolo_nas_variants":
            raise Exception(f"Not a YoloNas model: {type(model)}")

        # set default processing params for yolo_nas model
        processing_params = get_pretrained_processing_params(
            model_name=self.model_name, pretrained_weights="coco"
        )
        processing_params["conf"] = self.confidence_threshold
        if self.class_names:  # override class names for custom trained models
            processing_params["class_names"] = self.class_names
        model.set_dataset_processing_params(**processing_params)
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
        prediction_result = list(self.model.predict(image))
        self._original_predictions = prediction_result

    @property
    def num_categories(self):
        """
        Returns number of categories
        """
        return len(self.model._class_names)

    @property
    def has_mask(self):
        """
        Returns if model output contains segmentation mask
        """
        return False

    @property
    def category_names(self):
        return self.model._class_names

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
        for image_ind, image_predictions in enumerate(original_predictions):
            shift_amount = shift_amount_list[image_ind]
            full_shape = (
                None if full_shape_list is None else full_shape_list[image_ind]
            )
            object_prediction_list = []
            # process predictions
            preds = image_predictions.prediction
            for bbox_xyxy, score, category_id in zip(
                preds.bboxes_xyxy, preds.confidence, preds.labels
            ):
                bbox = bbox_xyxy
                category_name = self.category_mapping[str(int(category_id))]
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
                    category_id=int(category_id),
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
