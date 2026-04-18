import numpy as np
from typing import List, Optional

from anylabeling.services.auto_labeling.engines.build_onnx_engine import (
    OnnxBaseModel,
)
from anylabeling.services.auto_labeling.utils import (
    letterbox,
    non_max_suppression_end2end,
    scale_boxes,
)
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
from anylabeling.views.labeling.logger import logger


class Yolo26ONNX(object):
    def __init__(
        self,
        model_path: str,
        device: str,
        conf_thres: float,
        max_det: int,
    ):
        self.net = OnnxBaseModel(model_path, device)
        self.conf_thres = conf_thres
        self.max_det = max_det
        _, _, input_height, input_width = self.net.get_input_shape()
        self.input_shape = (input_height, input_width)

    def inference(self, image):
        blob, img_size = self.preprocess(image)
        outputs = self.net.get_ort_inference(blob=blob, extract=False)
        bboxes, scores, class_ids = self.postprocess(outputs, img_size)
        return bboxes, scores, class_ids

    def preprocess(self, input_image):
        input_img = letterbox(input_image, self.input_shape)[0]
        input_img = input_img.transpose(2, 0, 1)
        input_img = input_img[np.newaxis, :, :, :].astype(np.float32)
        blob = np.ascontiguousarray(input_img) / 255.0
        return blob, input_image.shape[:2]

    def postprocess(self, outputs, img_size):
        predictions = non_max_suppression_end2end(
            outputs,
            conf_thres=self.conf_thres,
            max_det=self.max_det,
        )
        prediction = predictions[0]
        if prediction.shape[0] == 0:
            return np.array([]), np.array([]), np.array([])

        prediction[:, :4] = scale_boxes(
            self.input_shape,
            prediction[:, :4],
            img_size,
        )
        return prediction[:, :4], prediction[:, 4], prediction[:, 5]


class Yolo26OnnxDetectionModel(DetectionModel):
    def __init__(self, *args, max_det: int = 300, **kwargs):
        self.max_det = max_det
        super().__init__(*args, **kwargs)

    def check_dependencies(self) -> None:
        check_requirements(["onnxruntime"])

    def load_model(self):
        """
        Detection model is initialized and set to self.model.
        """
        self.conf_thres = self.confidence_threshold

        self.model = Yolo26ONNX(
            model_path=self.model_path,
            device=self.device,
            conf_thres=self.conf_thres,
            max_det=self.max_det,
        )

        self.category_name_list = list(self.category_mapping.values())
        self.category_name_list_len = len(self.category_name_list)

    def perform_inference(self, image: np.ndarray, image_size: int = None):
        """
        Prediction is performed using self.model and the prediction result is set to self._original_predictions.
        Args:
            image: np.ndarray
                A numpy array that contains the image to be predicted.
            image_size: int
                Inference input size.
        """

        assert (
            self.model is not None
        ), "Model is not loaded, load it by calling .load_model()"

        prediction_result = self.model.inference(image)

        self._original_predictions = [prediction_result]

    @property
    def num_categories(self):
        return self.category_name_list_len

    @property
    def has_mask(self):
        return False

    @property
    def category_names(self):
        return self.category_name_list

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

        shift_amount_list = fix_shift_amount_list(shift_amount_list)
        full_shape_list = fix_full_shape_list(full_shape_list)

        object_prediction_list_per_image = []
        for image_ind, original_prediction in enumerate(original_predictions):
            bboxes = original_prediction[0]
            scores = original_prediction[1]
            class_ids = original_prediction[2]

            shift_amount = shift_amount_list[image_ind]
            full_shape = (
                None if full_shape_list is None else full_shape_list[image_ind]
            )
            object_prediction_list = []

            for original_bbox, score, class_id in zip(
                bboxes, scores, class_ids
            ):
                x1 = int(original_bbox[0])
                y1 = int(original_bbox[1])
                x2 = int(original_bbox[2])
                y2 = int(original_bbox[3])
                bbox = [x1, y1, x2, y2]
                category_id = int(class_id)
                category_name = self.category_mapping[str(category_id)]

                bbox[0] = max(0, bbox[0])
                bbox[1] = max(0, bbox[1])
                bbox[2] = max(0, bbox[2])
                bbox[3] = max(0, bbox[3])

                if full_shape is not None:
                    bbox[0] = min(full_shape[1], bbox[0])
                    bbox[1] = min(full_shape[0], bbox[1])
                    bbox[2] = min(full_shape[1], bbox[2])
                    bbox[3] = min(full_shape[0], bbox[3])

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
