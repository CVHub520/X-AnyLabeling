import cv2
import numpy as np
from typing import List, Optional

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
from anylabeling.services.auto_labeling.engines.build_onnx_engine import (
    OnnxBaseModel,
)
from anylabeling.views.labeling.logger import logger


class Yolov5ONNX(object):
    def __init__(
        self, model_path: str, device: str, conf_thres: float, nms_thres: float
    ):
        self.net = OnnxBaseModel(model_path, device)
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres

    def inference(self, image):
        blob, img_size = self.preprocess(image)
        outputs = self.net.get_ort_inference(blob)
        bboxes, scores, class_ids = self.postprocess(outputs, img_size)
        return bboxes, scores, class_ids

    def preprocess(self, input_image):
        """
        Pre-process the input image before feeding it to the network.
        """
        # Resized
        _, _, input_height, input_width = self.net.get_input_shape()
        input_img = cv2.resize(input_image, (input_width, input_height))

        # Norm
        input_img = input_img / 255.0

        # Transposed
        input_img = input_img.transpose(2, 0, 1)

        # Processed
        blob = input_img[np.newaxis, :, :, :].astype(np.float32)

        return blob, input_image.shape[:2]

    def postprocess(self, outputs, img_size):
        """
        Post-process the network's output, to get the bounding boxes and
        their confidence scores.
        Expects output shape: (1, 25200, 85) where 85 = 4(xywh) + 1(obj_conf) + 80(class_scores)
        """
        # Lists to hold respective values while unwrapping
        _class_ids = []
        _confidences = []
        _boxes = []

        # Get number of rows (predictions)
        rows = outputs.shape[1]

        image_height, image_width = img_size
        _, _, input_height, input_width = self.net.get_input_shape()

        # Resizing factor
        x_factor = image_width / input_width
        y_factor = image_height / input_height

        # Iterate through all predictions
        for r in range(rows):
            row = outputs[0][r]
            # First 4 elements are box coordinates (cx, cy, w, h)
            # Element at index 4 is objectness score
            # Remaining elements are class scores
            classes_scores = row[5:]  # Changed from 4 to 5
            obj_conf = row[4]  # Get objectness confidence

            # Get the index of max class score and confidence
            _, confidence, _, (_, class_id) = cv2.minMaxLoc(classes_scores)
            confidence *= obj_conf  # Multiply by objectness confidence

            # Discard confidence lower than threshold
            if confidence >= self.conf_thres:
                _confidences.append(confidence)
                _class_ids.append(class_id)

                cx, cy, w, h = row[0], row[1], row[2], row[3]

                left = int((cx - w / 2) * x_factor)
                top = int((cy - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                box = np.array([left, top, width, height])
                _boxes.append(box)

        # Perform non maximum suppression to eliminate redundant
        # overlapping boxes with lower confidences.
        indices = cv2.dnn.NMSBoxes(
            _boxes,
            _confidences,
            self.conf_thres,
            self.nms_thres,
        )

        bboxes, scores, class_ids = [], [], []
        for i in indices:
            _x, _y, _w, _h = _boxes[i]
            bboxes.append([_x, _y, _x + _w, _y + _h])
            scores.append(_confidences[i])
            class_ids.append(_class_ids[i])

        return np.array(bboxes), np.array(scores), np.array(class_ids)


class Yolov5OnnxDetectionModel(DetectionModel):
    def check_dependencies(self) -> None:
        check_requirements(["onnxruntime"])

    def load_model(self):
        """
        Detection model is initialized and set to self.model.
        """
        self.conf_thres = self.confidence_threshold
        self.nms_thres = self.nms_threshold

        # set model
        self.model = Yolov5ONNX(
            model_path=self.model_path,
            device=self.device,
            conf_thres=self.conf_thres,
            nms_thres=self.nms_thres,
        )

        # set category list
        self.category_name_list = list(self.category_mapping.values())
        self.category_name_list_len = len(self.category_name_list)

    def perform_inference(self, image: np.ndarray, image_size: int = None):
        """
        Prediction is performed using self.model and the prediction result is set to self._original_predictions.
        Args:
            image: np.ndarray
                A numpy array that contains the image to be predicted. 3 channel image should be in BGR order.
            image_size: int
                Inference input size.
        """

        # Confirm model is loaded
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

        # compatilibty for sahi v0.8.15
        shift_amount_list = fix_shift_amount_list(shift_amount_list)
        full_shape_list = fix_full_shape_list(full_shape_list)

        # handle all predictions
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

            # process predictions
            for original_bbox, score, class_id in zip(
                bboxes, scores, class_ids
            ):
                x1 = int(original_bbox[0])
                y1 = int(original_bbox[1])
                x2 = int(original_bbox[2])
                y2 = int(original_bbox[3])
                bbox = [x1, y1, x2, y2]
                score = score
                category_id = int(class_id)
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
