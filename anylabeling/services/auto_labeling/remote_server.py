import base64
import cv2
import os
import requests

from PyQt5 import QtCore
from PyQt5.QtCore import QCoreApplication

from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img
from .model import Model
from .types import AutoLabelingResult


class RemoteServer(Model):

    class Meta:
        required_config_names = [
            "type",
            "display_name",
        ]
        widgets = ["remote_server_select_combobox"]
        output_modes = {
            "rectangle": QCoreApplication.translate("Model", "Rectangle"),
        }
        default_output_mode = "rectangle"

    def __init__(self, model_config, on_message) -> None:
        super().__init__(model_config, on_message)

        self.server_url = self.config.get(
            "server_url",
            os.getenv("XANYLABELING_SERVER_URL", "http://localhost:8000"),
        )
        self.predict_url = f"{self.server_url}/v1/predict"

        print(f'{self.config.get("api_key", "")}')
        self.headers = {
            "Content-Type": "application/json",
            "Token": self.config.get("api_key", ""),
        }

        self.current_model_id = None
        self.timeout = self.config.get("timeout", 30)

        self.marks = []
        self.conf_threshold = 0.0
        self.iou_threshold = 0.0
        self.epsilon_factor = 0.001
        self.replace = True
        self.reset_tracker_flag = False

    def set_model_id(self, model_id):
        self.current_model_id = model_id

    def get_available_models(self):
        """Fetch available models from remote server"""
        try:
            models_url = f"{self.server_url}/v1/models"
            response = requests.get(
                url=models_url, headers=self.headers, timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            return result.get("data", {})
        except Exception as e:
            logger.error(f"Failed to fetch available models: {e}")
            return {}

    def set_auto_labeling_marks(self, marks):
        """Set auto labeling marks"""
        self.marks = marks

    def set_auto_labeling_preserve_existing_annotations_state(self, state):
        """Toggle the preservation of existing annotations based on the checkbox state."""
        self.replace = not state

    def set_auto_labeling_conf(self, conf_thresh):
        self.conf_threshold = conf_thresh

    def set_auto_labeling_iou(self, iou_thresh):
        self.iou_threshold = iou_thresh

    def set_auto_labeling_preserve_existing_annotations_state(self, state):
        self.replace = not state

    def set_mask_fineness(self, epsilon):
        self.epsilon_factor = epsilon

    def set_auto_labeling_reset_tracker(self):
        """Reset tracker state for tracking models."""
        self.reset_tracker_flag = True

    def predict_shapes(self, image, image_path=None, text_prompt=None):
        if image is None:
            return AutoLabelingResult([], replace=self.replace)

        if self.current_model_id is None:
            logger.warning("No model selected")
            return AutoLabelingResult([], replace=self.replace)

        try:
            cv_image = qt_img_to_rgb_cv_img(image, image_path)
        except Exception as e:
            logger.warning(f"Could not process image: {e}")
            return AutoLabelingResult([], replace=self.replace)

        # Encode image to base64 as PNG
        is_success, buffer = cv2.imencode(".png", cv_image)
        if not is_success:
            raise ValueError("Failed to encode image.")
        img_base64 = base64.b64encode(buffer).decode("utf-8")
        img_data_uri = f"data:image/png;base64,{img_base64}"

        params = {}
        params["conf_threshold"] = self.conf_threshold
        params["iou_threshold"] = self.iou_threshold
        params["epsilon_factor"] = self.epsilon_factor
        if text_prompt:
            logger.debug(f"Received text prompt: {text_prompt}")
            params["text_prompt"] = text_prompt.rstrip(".")
        if self.reset_tracker_flag:
            params["reset_tracker"] = True
            self.reset_tracker_flag = False

        payload = {
            "model": self.current_model_id,
            "image": img_data_uri,
            "params": params,
        }
        logger.debug(
            f"Sending request to {self.predict_url} with payload: "
            f"model: {self.set_model_id}, "
            f"paramters: {params}"
        )

        try:
            response = requests.post(
                url=self.predict_url,
                json=payload,
                headers=self.headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            result = response.json()
            logger.debug(f"Remote server prediction result: {result}")

            data = result.get("data", {})
            shapes = []
            for shape_data in data.get("shapes", []):
                shape = Shape(
                    label=shape_data["label"],
                    shape_type=shape_data["shape_type"],
                    attributes=shape_data.get("attributes", {}),
                    description=shape_data.get("description", None),
                    difficult=shape_data.get("difficult", False),
                    direction=shape_data.get("direction", 0),
                    flags=shape_data.get("flags", None),
                    group_id=shape_data.get("group_id", None),
                    kie_linking=shape_data.get("kie_linking", []),
                    score=shape_data.get("score", None),
                )

                for point in shape_data["points"]:
                    shape.add_point(QtCore.QPointF(point[0], point[1]))

                shapes.append(shape)

            description = data.get("description", "")

            return AutoLabelingResult(
                shapes, replace=self.replace, description=description
            )

        except Exception as e:
            logger.error(f"Remote server error: {e}")
            self.on_message(f"Remote server error: {e}")
            return AutoLabelingResult([], replace=self.replace)

    def unload(self):
        """Unload the model"""
        pass
