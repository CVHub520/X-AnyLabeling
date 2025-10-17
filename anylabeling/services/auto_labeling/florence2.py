import warnings

warnings.filterwarnings("ignore")

import gc
from PIL import Image
from unittest.mock import patch

from PyQt5 import QtCore
from PyQt5.QtCore import QCoreApplication

from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img
from .model import Model
from .types import AutoLabelingResult, AutoLabelingMode

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor
    from transformers.dynamic_module_utils import get_imports

    FLORENCE2_AVAILABLE = True
except ImportError:
    FLORENCE2_AVAILABLE = False


class Florence2(Model):
    """Visual-Language model using Florence2"""

    # Task mapping from user-friendly names to model tokens
    TASK_MAPPING = {
        "caption": "<CAPTION>",
        "detailed_cap": "<DETAILED_CAPTION>",
        "more_detailed_cap": "<MORE_DETAILED_CAPTION>",
        "od": "<OD>",
        "region_proposal": "<REGION_PROPOSAL>",
        "dense_region_cap": "<DENSE_REGION_CAPTION>",
        "cap_to_pg": "<CAPTION_TO_PHRASE_GROUNDING>",
        "refer_exp_seg": "<REFERRING_EXPRESSION_SEGMENTATION>",
        "region_to_seg": "<REGION_TO_SEGMENTATION>",
        "ovd": "<OPEN_VOCABULARY_DETECTION>",
        "region_to_cat": "<REGION_TO_CATEGORY>",
        "region_to_desc": "<REGION_TO_DESCRIPTION>",
        "ocr": "<OCR>",
        "ocr_with_region": "<OCR_WITH_REGION>",
    }

    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "model_path",
        ]
        widgets = ["florence2_select_combobox"]
        output_modes = {
            "rectangle": QCoreApplication.translate("Model", "Rectangle"),
        }
        default_output_mode = "rectangle"

    def __init__(self, model_config, on_message) -> None:
        if not FLORENCE2_AVAILABLE:
            message = "Florence2 model will not be available. Please install related packages and try again."
            raise ImportError(message)

        # Run the parent class's init method
        super().__init__(model_config, on_message)
        model_path = self.config.get("model_path", None)
        trust_remote_code = self.config.get("trust_remote_code", True)

        device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = (
            torch.float16 if torch.cuda.is_available() else torch.float32
        )

        self.marks = []
        self.prompt_type = "caption"

        self.max_new_tokens = self.config.get("max_new_tokens", 1024)
        self.do_sample = self.config.get("do_sample", False)
        self.num_beams = self.config.get("num_beams", 3)

        # Add patch for flash attention on CPU
        def fixed_get_imports(filename):
            imports = get_imports(filename)
            if not torch.cuda.is_available() and "flash_attn" in imports:
                imports.remove("flash_attn")
            return imports

        # Load model with patched imports
        with patch(
            "transformers.dynamic_module_utils.get_imports", fixed_get_imports
        ):
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=trust_remote_code,
            )
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=trust_remote_code,
            )

        self.replace = True
        self.device = device_map

    def set_florence2_mode(self, mode):
        """Set Florence2 mode"""
        self.prompt_type = mode

    def set_auto_labeling_marks(self, marks):
        """Set auto labeling marks"""
        self.marks = marks

    def set_auto_labeling_preserve_existing_annotations_state(self, state):
        """Toggle the preservation of existing annotations based on the checkbox state."""
        self.replace = not state

    def preprocess(self, orig_img, task_token, text_prompt):
        """Preprocess input data."""

        prompt = task_token

        # Only the last mark is used for bbox prompt
        if (
            task_token
            in [
                "<REGION_TO_SEGMENTATION>",
                "<REGION_TO_CATEGORY>",
                "<REGION_TO_DESCRIPTION>",
            ]
            and len(self.marks) > 0
        ):
            # Normalize coordinates from pixel space to [0,999] range
            w, h = orig_img.size
            bbox = self.marks[-1]["data"]
            bbox_prompt = [
                int(bbox[0] * 999 / w),  # x1
                int(bbox[1] * 999 / h),  # y1
                int(bbox[2] * 999 / w),  # x2
                int(bbox[3] * 999 / h),  # y2
            ]
            bbox_prompt = (
                f"<loc_{bbox_prompt[0]}>"
                f"<loc_{bbox_prompt[1]}>"
                f"<loc_{bbox_prompt[2]}>"
                f"<loc_{bbox_prompt[3]}>"
            )
            prompt += bbox_prompt

        if (
            task_token
            in [
                "<CAPTION_TO_PHRASE_GROUNDING>",
                "<REFERRING_EXPRESSION_SEGMENTATION>",
                "<OPEN_VOCABULARY_DETECTION>",
            ]
            and text_prompt
        ):
            prompt += text_prompt

        logger.debug(f"Prompt: {prompt}")

        # Process inputs through processor
        inputs = self.processor(
            text=prompt, images=orig_img, return_tensors="pt"
        )

        # Move inputs to device and match model dtype
        model_dtype = next(self.model.parameters()).dtype
        inputs = {
            k: (
                v.to(device=self.device, dtype=model_dtype)
                if torch.is_floating_point(v)
                else v.to(self.device)
            )
            for k, v in inputs.items()
        }

        return inputs

    def _forward(self, inputs):
        """Forward pass through the model."""
        # Generate text
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            num_beams=self.num_beams,
        )

        # Decode generated text
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        return generated_text

    def postprocess(self, outputs, orig_img, task_token):
        """Postprocess model outputs."""
        outputs = self.processor.post_process_generation(
            outputs, task=task_token, image_size=orig_img.size
        )

        return outputs

    def predict_shapes(self, image, image_path=None, text_prompt=None):
        """
        Predict shapes from image
        """

        if image is None:
            return []

        try:
            image = qt_img_to_rgb_cv_img(image, image_path)
        except Exception as e:  # noqa
            logger.warning("Could not inference model")
            logger.warning(e)
            return []

        # Convert friendly task name to token
        task_token = self.TASK_MAPPING[self.prompt_type]

        if (
            task_token
            in [
                "<CAPTION_TO_PHRASE_GROUNDING>",
                "<REFERRING_EXPRESSION_SEGMENTATION>",
                "<OPEN_VOCABULARY_DETECTION>",
            ]
            and not text_prompt
        ):
            logger.warning(
                f"Could not inference model without text prompt for {task_token}"
            )
            return []

        orig_img = Image.fromarray(image).convert("RGB")

        inputs = self.preprocess(orig_img, task_token, text_prompt)
        outputs = self._forward(inputs)
        results = self.postprocess(outputs, orig_img, task_token)

        logger.debug(f"Results: {results}")

        if (
            task_token
            in [
                "<CAPTION>",
                "<DETAILED_CAPTION>",
                "<MORE_DETAILED_CAPTION>",
                "<OCR>",
            ]
            and task_token in results
        ):
            description = results[task_token]
            result = AutoLabelingResult(
                [], replace=self.replace, description=description
            )
        elif (
            task_token
            in [
                "<OD>",
                "<REGION_PROPOSAL>",
                "<DENSE_REGION_CAPTION>",
                "<CAPTION_TO_PHRASE_GROUNDING>",
            ]
            and task_token in results
        ):
            shapes = []

            # Handle bounding boxes
            bboxes = results[task_token].get("bboxes", [])
            bbox_labels = results[task_token].get("labels", [])
            for bbox, label in zip(bboxes, bbox_labels):
                x1, y1, x2, y2 = bbox
                shape = Shape(
                    label=label if label else "N/A",
                    shape_type="rectangle",
                )
                shape.add_point(QtCore.QPointF(x1, y1))
                shape.add_point(QtCore.QPointF(x2, y1))
                shape.add_point(QtCore.QPointF(x2, y2))
                shape.add_point(QtCore.QPointF(x1, y2))
                shapes.append(shape)

            # Handle polygons
            polygons = results[task_token].get("polygons", [])
            polygon_labels = results[task_token].get("labels", [])
            for polygon, label in zip(polygons, polygon_labels):
                points = polygon[0] if polygon else []
                shape = Shape(
                    label=label if label else "N/A",
                    shape_type="polygon",
                )
                for i in range(0, len(points), 2):
                    shape.add_point(QtCore.QPointF(points[i], points[i + 1]))
                shape.closed = True
                shapes.append(shape)

            result = AutoLabelingResult(shapes, replace=self.replace)
        elif (
            task_token
            in [
                "<OCR_WITH_REGION>",
                "<REFERRING_EXPRESSION_SEGMENTATION>",
            ]
            and task_token in results
        ):
            shapes = []
            if task_token == "<OCR_WITH_REGION>":
                points_list = results[task_token].get("quad_boxes", [])
                labels = results[task_token].get("labels", [])
                descriptions = []
                for label in labels:
                    if label.startswith("</s>"):
                        descriptions.append(label[4:])  # Strip "</s>" prefix
                    else:
                        descriptions.append(label)
                labels = ["text" for _ in descriptions]
            else:
                points_list = results[task_token].get("polygons", [])
                labels = [text_prompt for _ in range(len(points_list))]
                descriptions = [None for _ in labels]

            for points_data, label, description in zip(
                points_list, labels, descriptions
            ):
                shape = Shape(
                    label=label,
                    shape_type="polygon",
                    description=description if description else None,
                )
                if task_token == "<OCR_WITH_REGION>":
                    points_len = len(points_data)
                    points = [
                        (points_data[i], points_data[i + 1])
                        for i in range(0, points_len, 2)
                    ]
                else:
                    # Handle nested polygon points array
                    points_data = points_data[0] if points_data else []
                    points = [
                        (points_data[i], points_data[i + 1])
                        for i in range(0, len(points_data), 2)
                    ]

                for x, y in points:
                    shape.add_point(QtCore.QPointF(x, y))
                shape.add_point(QtCore.QPointF(points[0][0], points[0][1]))
                shape.closed = True
                shapes.append(shape)
            result = AutoLabelingResult(shapes, replace=self.replace)
        elif (
            task_token
            in [
                "<REGION_TO_CATEGORY>",
                "<REGION_TO_DESCRIPTION>",
            ]
            and task_token in results
        ):
            shapes = []
            result_str = results[task_token]

            # Extract label from string like
            # "window<loc_32><loc_162><loc_581><loc_369>"
            label = result_str.split("<")[0]

            coords = []
            for coord_str in self.marks[-1]["data"]:
                coords.append(int(coord_str))

            # Create rectangle shape
            shape = Shape(
                label=AutoLabelingMode.OBJECT,
                shape_type="rectangle",
            )
            shape.cache_label = (
                label if task_token == "<REGION_TO_CATEGORY>" else "N/A"
            )
            shape.cache_description = (
                label if task_token == "<REGION_TO_DESCRIPTION>" else None
            )
            shape.add_point(QtCore.QPointF(coords[0], coords[1]))
            shape.add_point(QtCore.QPointF(coords[2], coords[1]))
            shape.add_point(QtCore.QPointF(coords[2], coords[3]))
            shape.add_point(QtCore.QPointF(coords[0], coords[3]))
            shapes.append(shape)
            result = AutoLabelingResult(shapes, replace=self.replace)
        elif (
            task_token == "<REGION_TO_SEGMENTATION>" and task_token in results
        ):
            polygons = results[task_token].get("polygons", [[]])[0]
            shapes = []
            for polygon in polygons:
                shape = Shape(
                    label=AutoLabelingMode.OBJECT,
                    shape_type="polygon",
                )
                # [x1, y1, x2, y2, ...]
                points = [
                    (polygon[i], polygon[i + 1])
                    for i in range(0, len(polygon), 2)
                ]
                for x, y in points:
                    shape.add_point(QtCore.QPointF(x, y))
                shape.add_point(QtCore.QPointF(points[0][0], points[0][1]))
                shape.closed = True
                shapes.append(shape)
            result = AutoLabelingResult(shapes, replace=self.replace)
        elif (
            task_token == "<OPEN_VOCABULARY_DETECTION>"
            and task_token in results
        ):
            shapes = []

            bboxes = results[task_token].get("bboxes", [])
            bbox_labels = results[task_token].get("bboxes_labels", [])
            polygons = results[task_token].get("polygons", [])
            polygon_labels = results[task_token].get("polygons_labels", [])

            for bbox, label in zip(bboxes, bbox_labels):
                shape = Shape(
                    label=label if label else text_prompt,
                    shape_type="rectangle",
                )
                shape.add_point(QtCore.QPointF(bbox[0], bbox[1]))
                shape.add_point(QtCore.QPointF(bbox[2], bbox[1]))
                shape.add_point(QtCore.QPointF(bbox[2], bbox[3]))
                shape.add_point(QtCore.QPointF(bbox[0], bbox[3]))
                shapes.append(shape)

            for polygon, label in zip(polygons, polygon_labels):
                shape = Shape(
                    label=label if label else text_prompt,
                    shape_type="polygon",
                )
                points = polygon[0] if polygon else []
                for i in range(0, len(points), 2):
                    shape.add_point(QtCore.QPointF(points[i], points[i + 1]))
                shape.add_point(QtCore.QPointF(points[0], points[1]))
                shape.closed = True
                shapes.append(shape)
            result = AutoLabelingResult(shapes, replace=self.replace)

        return result

    def unload(self):
        del self.model
        del self.processor

        gc.collect()
        torch.cuda.empty_cache()
