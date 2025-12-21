import os
import numpy as np
from PIL import Image

from PyQt5 import QtCore
from PyQt5.QtCore import QCoreApplication

from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.logger import logger
from .model import Model
from .types import AutoLabelingResult

try:
    import torch
    import supervision as sv

    try:
        from supervision.detection.utils.converters import mask_to_polygons
    except ImportError:
        from supervision.detection.utils import mask_to_polygons
    from ultralytics import YOLOE as _YOLOE
    from ultralytics.models.yolo.yoloe.predict_vp import YOLOEVPSegPredictor

    YOLOE_AVAILABLE = True
except ImportError:
    YOLOE_AVAILABLE = False


class YOLOE(Model):
    """YOLOE: Real-Time Seeing Anything Model"""

    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "iou_threshold",
            "conf_threshold",
            "model_path",
            "model_pf_path",
            "embedding_model_path",
        ]
        widgets = [
            "output_select_combobox",
            "edit_text",
            "button_send",
            "input_iou",
            "edit_iou",
            "input_conf",
            "edit_conf",
            "toggle_preserve_existing_annotations",
            "button_add_rect",
            "button_clear",
        ]
        output_modes = {
            "rectangle": QCoreApplication.translate("Model", "Rectangle"),
            "polygon": QCoreApplication.translate("Model", "Polygon"),
        }
        default_output_mode = "rectangle"

    def __init__(self, model_config, on_message) -> None:
        if not YOLOE_AVAILABLE:
            message = "YOLOE model will not be available. Please install related packages and try again."
            raise ImportError(message)

        super().__init__(model_config, on_message)

        # Validate model paths
        check_model_list = [
            "model_path",
            "model_pf_path",
            "embedding_model_path",
        ]
        for model_name in check_model_list:
            model_abs_path = self.get_model_abs_path(self.config, model_name)
            if not model_abs_path or not os.path.isfile(model_abs_path):
                raise FileNotFoundError(
                    QCoreApplication.translate(
                        "Model",
                        f"Could not download or initialize {os.path.basename(self.config[model_name])} model.",
                    )
                )
            else:
                self.config[model_name] = model_abs_path

        # Visual prompting marks
        self.marks = []

        # Lazy-load model instances for different task modes
        self._text_model = None
        self._visual_model = None
        self._prompt_free_model = None

        # Cache text prompt state to avoid unnecessary model rebuilds
        self._current_text_prompt = None
        self._prompt_free_initialized = False

        # Model configuration
        input_width = self.config.get("input_width", 640)
        input_height = self.config.get("input_height", 640)
        self.input_shape = (input_height, input_width)

        self.with_mask = self.config.get("with_mask", False)
        self.max_det = self.config.get("max_det", 1000)
        self.iou_thres = self.config.get("iou_threshold", 0.70)
        self.conf_thres = self.config.get("conf_threshold", 0.25)
        self.replace = True

        self.text_prompt = None

        # Load class configurations
        classes = self.config.get("classes", None)
        if isinstance(classes, str):
            with open(classes, "r") as f:
                self.texts = [line.strip() for line in f]
        elif isinstance(classes, list):
            self.texts = classes
        elif isinstance(classes, dict):
            self.texts = list(classes.values())
        else:
            self.texts = self.load_tag_list()

        # Create symlink for embedding model if needed
        if not os.path.exists(
            os.path.basename(self.config["embedding_model_path"])
        ):
            os.symlink(
                self.config["embedding_model_path"],
                os.path.basename(self.config["embedding_model_path"]),
            )

    @staticmethod
    def build_model(model_path):
        """Build and initialize YOLOE model"""
        logger.info(f"Loading model: {model_path}...")
        model = _YOLOE(model_path)
        model.eval()
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        return model

    def set_auto_labeling_marks(self, marks):
        """Set visual prompting marks"""
        self.marks = marks

    def set_auto_labeling_iou(self, value):
        """Set IoU threshold for auto labeling"""
        if value > 0:
            self.iou_thres = value

    def set_auto_labeling_conf(self, value):
        """Set confidence threshold for auto labeling"""
        if value > 0:
            self.conf_thres = value

    def set_auto_labeling_preserve_existing_annotations_state(self, state):
        """Toggle preservation of existing annotations"""
        self.replace = not state

    def postprocess(self, results):
        """Post-process model predictions into shapes"""
        if results is None:
            return []

        detections = sv.Detections.from_ultralytics(results[0])
        logger.debug(f"detections.xyxy: {detections.xyxy}")
        logger.debug(f"detections.class_name: {detections['class_name']}")
        logger.debug(f"detections.confidence: {detections.confidence}")
        if detections.xyxy is None:
            pass

        masks = detections.mask
        bboxes = detections.xyxy
        labels = detections["class_name"]
        scores = detections.confidence
        shapes = []

        # Generate rectangle shapes
        if self.output_mode == "rectangle":
            for xyxy, label, score in zip(bboxes, labels, scores):
                shape = Shape(flags={})
                xmin, ymin, xmax, ymax = xyxy
                shape.add_point(QtCore.QPointF(float(xmin), float(ymin)))
                shape.add_point(QtCore.QPointF(float(xmax), float(ymin)))
                shape.add_point(QtCore.QPointF(float(xmax), float(ymax)))
                shape.add_point(QtCore.QPointF(float(xmin), float(ymax)))
                shape.shape_type = "rectangle"
                shape.closed = True
                shape.score = float(score)
                shape.label = str(label)
                shape.selected = False
                shapes.append(shape)

        # Generate polygon shapes from masks
        if self.output_mode == "polygon" or self.with_mask:
            for mask, label, score in zip(masks, labels, scores):
                polygons = mask_to_polygons(mask)
                points = polygons[0].tolist()
                points.append(points[0])

                shape = Shape(flags={})
                for point in points:
                    shape.add_point(QtCore.QPointF(point[0], point[1]))
                shape.shape_type = "polygon"
                shape.closed = True
                shape.label = label
                shape.selected = False
                shapes.append(shape)

        return shapes

    def _get_text_model(self, texts):
        """Get or create text prompt model instance"""
        if self._text_model is None or self._current_text_prompt != texts:
            self._text_model = self.build_model(self.config["model_path"])
            self._text_model.set_classes(
                texts, self._text_model.get_text_pe(texts)
            )
            self._current_text_prompt = texts
        return self._text_model

    def _get_visual_model(self):
        """Get or create visual prompt model instance"""
        if self._visual_model is None:
            self._visual_model = self.build_model(self.config["model_path"])
        return self._visual_model

    def _get_prompt_free_model(self):
        """Get or create prompt-free model instance"""
        if (
            self._prompt_free_model is None
            or not self._prompt_free_initialized
        ):
            self._prompt_free_model = self.build_model(
                self.config["model_pf_path"]
            )
            # Initialize prompt-free model with vocabulary
            vocab = self.build_model(self.config["model_path"]).get_vocab(
                self.texts
            )
            self._prompt_free_model.set_vocab(vocab, names=self.texts)
            self._prompt_free_model.model.model[-1].is_fused = True
            self._prompt_free_model.model.model[-1].max_det = self.max_det
            self._prompt_free_initialized = True

        # Update dynamic parameters each time
        self._prompt_free_model.model.model[-1].iou = self.iou_thres
        self._prompt_free_model.model.model[-1].conf = self.conf_thres
        return self._prompt_free_model

    def predict_shapes(self, image, image_path=None, text_prompt=None):
        """Predict shapes from image using different prompting modes"""

        if image is None:
            return []

        try:
            image = Image.open(image_path)
        except Exception as e:  # noqa
            logger.warning("Could not inference model")
            logger.warning(e)
            return []

        kwargs = {}

        # Visual prompting mode
        if self.marks:
            bboxes = []
            logger.debug(f"marks: {self.marks}")
            for mark in self.marks:
                bboxes.append(mark["data"])

            bboxes = np.array(bboxes)
            prompts = {"bboxes": bboxes, "cls": np.array([0] * len(bboxes))}

            kwargs = dict(prompts=prompts, predictor=YOLOEVPSegPredictor)
            model = self._get_visual_model()
            results = model.predict(
                source=image,
                imgsz=self.input_shape,
                conf=self.conf_thres,
                iou=self.iou_thres,
                verbose=False,
                **kwargs,
            )
            self.marks = []

        # Text prompting mode
        elif text_prompt:
            text_prompt = text_prompt.strip()
            text_prompt = text_prompt.replace(",", ".")
            while text_prompt.endswith("."):
                text_prompt = text_prompt[:-1]
            texts = [text.strip() for text in text_prompt.split(".")]

            # Reset text model if prompt changed
            if self.text_prompt is None:
                self.text_prompt = texts
            else:
                if self.text_prompt != texts:
                    self._text_model = None
                    self.text_prompt = texts
            logger.debug(f"Input texts: {texts}")

            model = self._get_text_model(texts)
            results = model.predict(
                source=image,
                imgsz=self.input_shape,
                conf=self.conf_thres,
                iou=self.iou_thres,
                verbose=False,
                **kwargs,
            )

        # Prompt-free mode
        else:
            model = self._get_prompt_free_model()
            results = model.predict(
                source=image,
                imgsz=self.input_shape,
                conf=self.conf_thres,
                iou=self.iou_thres,
                verbose=False,
                **kwargs,
            )

        shapes = self.postprocess(results)
        result = AutoLabelingResult(shapes, replace=self.replace)
        return result

    @staticmethod
    def load_tag_list():
        """Load default tag list from resources"""
        from importlib.resources import files
        from anylabeling.services.auto_labeling.configs import ram

        tag_list_resource = files(ram).joinpath("ram_tag_list.txt")
        tag_list = tag_list_resource.read_text(encoding="utf-8").splitlines()

        return tag_list

    def unload(self):
        """Clean up model instances"""
        if self._text_model:
            del self._text_model
        if self._visual_model:
            del self._visual_model
        if self._prompt_free_model:
            del self._prompt_free_model
