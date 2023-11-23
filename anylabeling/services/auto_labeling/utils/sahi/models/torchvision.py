# OBSS SAHI Tool
# Code written by Fatih C Akyon and Kadir Nar, 2021.

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from anylabeling.services.auto_labeling.utils.sahi.models.base import (
    DetectionModel,
)
from anylabeling.services.auto_labeling.utils.sahi.prediction import (
    ObjectPrediction,
)
from anylabeling.services.auto_labeling.utils.sahi.utils.import_utils import (
    check_requirements,
)

logger = logging.getLogger(__name__)


class TorchVisionDetectionModel(DetectionModel):
    def check_dependencies(self) -> None:
        check_requirements(["torch", "torchvision"])

    def load_model(self):
        import torch

        from sahi.utils.torchvision import MODEL_NAME_TO_CONSTRUCTOR

        # read config params
        model_name = None
        num_classes = None
        if self.config_path is not None:
            import yaml

            with open(self.config_path, "r") as stream:
                try:
                    config = yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    raise RuntimeError(exc)

            model_name = config.get("model_name", None)
            num_classes = config.get("num_classes", None)

        # complete params if not provided in config
        if not model_name:
            model_name = "fasterrcnn_resnet50_fpn"
            logger.warning(
                f"model_name not provided in config, using default model_type: {model_name}'"
            )
        if num_classes is None:
            logger.warning(
                "num_classes not provided in config, using default num_classes: 91"
            )
            num_classes = 91
        if self.model_path is None:
            logger.warning(
                "model_path not provided in config, using pretrained weights and default num_classes: 91."
            )
            pretrained = True
            num_classes = 91
        else:
            pretrained = False

        # load model
        model = MODEL_NAME_TO_CONSTRUCTOR[model_name](
            num_classes=num_classes, pretrained=pretrained
        )
        try:
            model.load_state_dict(torch.load(self.model_path))
        except Exception as e:
            TypeError("model_path is not a valid torchvision model path: ", e)

        self.set_model(model)

    def set_model(self, model: Any):
        """
        Sets the underlying TorchVision model.
        Args:
            model: Any
                A TorchVision model
        """
        check_requirements(["torch", "torchvision"])

        model.eval()
        self.model = model.to(self.device)

        # set category_mapping
        from sahi.utils.torchvision import COCO_CLASSES

        if self.category_mapping is None:
            category_names = {
                str(i): COCO_CLASSES[i] for i in range(len(COCO_CLASSES))
            }
            self.category_mapping = category_names

    def perform_inference(self, image: np.ndarray, image_size: int = None):
        """
        Prediction is performed using self.model and the prediction result is set to self._original_predictions.
        Args:
            image: np.ndarray
                A numpy array that contains the image to be predicted. 3 channel image should be in RGB order.
            image_size: int
                Inference input size.
        """
        from sahi.utils.torch import to_float_tensor

        # arrange model input size
        if self.image_size is not None:
            # get min and max of image height and width
            min_shape, max_shape = min(image.shape[:2]), max(image.shape[:2])
            # torchvision resize transform scales the shorter dimension to the target size
            # we want to scale the longer dimension to the target size
            image_size = self.image_size * min_shape / max_shape
            self.model.transform.min_size = (image_size,)  # default is (800,)
            self.model.transform.max_size = image_size  # default is 1333

        image = to_float_tensor(image)
        image = image.to(self.device)
        prediction_result = self.model([image])

        self._original_predictions = prediction_result

    @property
    def num_categories(self):
        """
        Returns number of categories
        """
        return len(self.category_mapping)

    @property
    def has_mask(self):
        """
        Returns if model output contains segmentation mask
        """
        return self.model.with_mask

    @property
    def category_names(self):
        return list(self.category_mapping.values())

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

        # compatilibty for sahi v0.8.20
        if isinstance(shift_amount_list[0], int):
            shift_amount_list = [shift_amount_list]
        if full_shape_list is not None and isinstance(full_shape_list[0], int):
            full_shape_list = [full_shape_list]

        for image_predictions in original_predictions:
            object_prediction_list_per_image = []

            # get indices of boxes with score > confidence_threshold
            scores = image_predictions["scores"].cpu().detach().numpy()
            selected_indices = np.where(scores > self.confidence_threshold)[0]

            # parse boxes, masks, scores, category_ids from predictions
            category_ids = list(
                image_predictions["labels"][selected_indices]
                .cpu()
                .detach()
                .numpy()
            )
            boxes = list(
                image_predictions["boxes"][selected_indices]
                .cpu()
                .detach()
                .numpy()
            )
            scores = scores[selected_indices]

            # check if predictions contain mask
            masks = image_predictions.get("masks", None)
            if masks is not None:
                masks = list(
                    image_predictions["masks"][selected_indices]
                    .cpu()
                    .detach()
                    .numpy()
                )
            else:
                masks = None

            # create object_prediction_list
            object_prediction_list = []

            shift_amount = shift_amount_list[0]
            full_shape = (
                None if full_shape_list is None else full_shape_list[0]
            )

            for ind in range(len(boxes)):
                if masks is not None:
                    mask = np.array(masks[ind])
                else:
                    mask = None

                object_prediction = ObjectPrediction(
                    bbox=boxes[ind],
                    bool_mask=mask,
                    category_id=int(category_ids[ind]),
                    category_name=self.category_mapping[
                        str(int(category_ids[ind]))
                    ],
                    shift_amount=shift_amount,
                    score=scores[ind],
                    full_shape=full_shape,
                )
                object_prediction_list.append(object_prediction)
            object_prediction_list_per_image.append(object_prediction_list)

        self._object_prediction_list_per_image = (
            object_prediction_list_per_image
        )
