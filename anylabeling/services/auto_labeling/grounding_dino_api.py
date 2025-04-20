import base64
import cv2
import json
import os
import re
import requests
import time

from PyQt5 import QtCore
from PyQt5.QtCore import QCoreApplication

from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img
from .model import Model
from .types import AutoLabelingResult


class Grounding_DINO_API(Model):
    """Grounding DINO API model"""

    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "conf_threshold",
            "iou_threshold",
        ]
        widgets = [
            "gd_select_combobox",
            "edit_text",
            "button_send",
            "input_conf",
            "edit_conf",
            "input_iou",
            "edit_iou",
            "toggle_preserve_existing_annotations",
            "button_set_api_token",
        ]
        output_modes = {
            "rectangle": QCoreApplication.translate("Model", "Rectangle"),
        }
        default_output_mode = "rectangle"

    def __init__(self, model_config, on_message) -> None:
        super().__init__(model_config, on_message)

        self.api_base_url = "https://api.deepdataspace.com"
        self.model_name = "GroundingDino-1.6-Pro"
        self.detection_url = (
            f"{self.api_base_url.rstrip('/')}/v2/task/grounding_dino/detection"
        )
        self.status_url_template = (
            f"{self.api_base_url.rstrip('/')}/v2/task_status/{{task_uuid}}"
        )

        self.bbox_threshold = self.config["conf_threshold"]
        self.iou_threshold = self.config["iou_threshold"]
        self.replace = True

        self.headers = {
            "Content-Type": "application/json",
            "Token": os.getenv("GROUNDING_DINO_API_TOKEN", ""),
        }

    def set_auto_labeling_api_token(self, token):
        """Set the API token for the model"""
        self.headers["Token"] = token

    def set_groundingdino_mode(self, value):
        """set model name"""
        modes = {
            "GroundingDino_1_6_Pro": "GroundingDino-1.6-Pro",
            "GroundingDino_1_6_Edge": "GroundingDino-1.6-Edge",
            "GroundingDino_1_5_Pro": "GroundingDino-1.5-Pro",
            "GroundingDino_1_5_Edge": "GroundingDino-1.5-Edge",
        }
        self.model_name = modes.get(value, value)

    def set_auto_labeling_conf(self, value):
        """set auto labeling confidence threshold"""
        if value > 0:
            self.bbox_threshold = value

    def set_auto_labeling_iou(self, value):
        """set auto labeling iou threshold"""
        if value > 0:
            self.iou_threshold = value

    def set_auto_labeling_preserve_existing_annotations_state(self, state):
        """Toggle the preservation of existing annotations based on the checkbox state."""
        self.replace = not state

    def predict_shapes(self, image, image_path=None, text_prompt=None):
        """
        Predict shapes from image using the Grounding DINO API.
        """
        if image is None:
            logger.warning("Input image is None.")
            return AutoLabelingResult([], replace=self.replace)

        if not self.headers["Token"]:
            raise ValueError(
                "API Token is not configured. Please set it before calling."
            )

        if not text_prompt:
            raise ValueError("Empty text prompt.")

        text_prompt = text_prompt.rstrip(".")
        prompt_pattern = r"^[a-zA-Z]+(\.[a-zA-Z]+)*$"
        if not re.match(prompt_pattern, text_prompt):
            raise ValueError(
                f"Invalid text prompt format. "
                f"It should be English words separated by '.' (e.g., 'cat.dog')."
            )

        cv_image = qt_img_to_rgb_cv_img(image, image_path)
        if cv_image is None:
            raise ValueError("Failed to convert input image to OpenCV format.")

        # Encode image to base64 as PNG
        is_success, buffer = cv2.imencode(".png", cv_image)
        if not is_success:
            raise ValueError("Failed to encode image.")
        img_base64 = base64.b64encode(buffer).decode("utf-8")
        img_data_uri = f"data:image/png;base64,{img_base64}"

        payload = {
            "model": self.model_name,
            "image": img_data_uri,
            "prompt": {"type": "text", "text": text_prompt},
            "targets": ["bbox"],
            "bbox_threshold": self.bbox_threshold,
            "iou_threshold": self.iou_threshold,
        }

        shapes = []
        try:
            logger.info(
                f"Sending request to {self.detection_url} with payload: "
                f"model: {self.model_name}, "
                f"prompt: {text_prompt}, "
                f"targets: {['bbox']}, "
                f"bbox_threshold: {self.bbox_threshold}, "
                f"iou_threshold: {self.iou_threshold}"
            )

            start_time = time.time()
            resp = requests.post(
                url=self.detection_url,
                json=payload,
                headers=self.headers,
                timeout=30,
            )
            resp.raise_for_status()
            json_resp = resp.json()
            logger.debug(f"Initial API response: {json_resp}")

            if (
                json_resp.get("code") != 0
                or "data" not in json_resp
                or "task_uuid" not in json_resp["data"]
            ):
                error_msg = json_resp.get(
                    "msg", "Unknown error initiating task."
                )
                logger.error(f"API Error (initiate): {error_msg}")
                raise ValueError("Unknown error initiating task.")

            task_uuid = json_resp["data"]["task_uuid"]
            logger.info(f"Task initiated with UUID: {task_uuid}")

            status_url = self.status_url_template.format(task_uuid=task_uuid)
            polling_attempts = 0
            max_polling_attempts = 30
            while polling_attempts < max_polling_attempts:
                polling_attempts += 1
                time.sleep(1)
                logger.debug(
                    f"Polling status for task {task_uuid} (Attempt {polling_attempts})"
                )
                resp = requests.get(
                    status_url, headers=self.headers, timeout=10
                )
                resp.raise_for_status()
                json_resp = resp.json()
                logger.debug(f"Polling response: {json_resp}")

                if json_resp.get("code") != 0:
                    error_msg = json_resp.get(
                        "msg", "Unknown error checking status."
                    )
                    logger.error(f"API Error (polling): {error_msg}")
                    raise ValueError("Unknown error checking status.")

                status_data = json_resp.get("data", {})
                task_status = status_data.get("status")

                if task_status not in ["waiting", "running"]:
                    break
            else:
                logger.warning(
                    f"Task {task_uuid} timed out after {max_polling_attempts} seconds."
                )
                raise ValueError("Task timed out.")

            end_time = time.time()
            logger.info(
                f"Task {task_uuid} finished with status '{task_status}' in {end_time - start_time:.2f} seconds."
            )

            if task_status == "failed":
                error_msg = status_data.get(
                    "error", "Task failed with unknown error."
                )
                logger.error(f"Task {task_uuid} failed: {error_msg}")
                raise ValueError(f"Task {task_uuid} failed: {error_msg}")
            elif task_status == "success":
                result_data = status_data.get("result", {})
                objects = result_data.get("objects", [])
                logger.info(f"Received {len(objects)} objects from API.")

                for obj in objects:
                    bbox = obj.get("bbox")
                    label = obj.get("category")
                    score = obj.get("score")

                    if bbox and label is not None and score is not None:
                        try:
                            x1, y1, x2, y2 = map(int, bbox)
                            shape = Shape(
                                label=str(label),
                                score=float(score),
                                shape_type="rectangle",
                            )
                            shape.add_point(QtCore.QPointF(x1, y1))
                            shape.add_point(QtCore.QPointF(x2, y1))
                            shape.add_point(QtCore.QPointF(x2, y2))
                            shape.add_point(QtCore.QPointF(x1, y2))
                            shapes.append(shape)
                        except (ValueError, TypeError) as coord_err:
                            logger.warning(
                                f"Skipping object due to invalid bbox format {bbox}: {coord_err}"
                            )
                    else:
                        logger.warning(
                            f"Skipping object with missing data: {obj}"
                        )

        except requests.exceptions.RequestException as req_err:
            logger.error(f"API Request failed: {req_err}")
        except json.JSONDecodeError as json_err:
            logger.error(f"Failed to decode API response: {json_err}")
        except Exception as e:
            logger.error(
                f"An unexpected error occurred during API prediction: {e}",
                exc_info=True,
            )

        result = AutoLabelingResult(shapes, replace=self.replace)
        return result

    def unload(self):
        """Unload the model"""
        pass
