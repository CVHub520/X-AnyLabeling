import os
from enum import Enum

import numpy as np
from PIL import Image

from PyQt6 import QtCore
from PyQt6.QtCore import QCoreApplication

from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.logger import logger
from .model import Model
from .types import AutoLabelingResult
from .visual_prompt_profile import VisualPromptProfile

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


class _MobileCLIPTextEncoder:
    def __init__(self, checkpoint, device):
        import mobileclip

        self.model = mobileclip.create_model_and_transforms(
            "mobileclip_b", pretrained=checkpoint, device=device
        )[0]
        self.tokenizer = mobileclip.get_tokenizer("mobileclip_b")
        self.device = device

    def tokenize(self, texts):
        return self.tokenizer(texts).to(self.device)

    def encode_text(self, texts, dtype=None):
        if dtype is None:
            dtype = torch.float32
        text_features = self.model.encode_text(texts).to(dtype)
        return text_features / text_features.norm(p=2, dim=-1, keepdim=True)


class VisualPromptState(str, Enum):
    """Lifecycle states for YOLOE cross-image visual prompts."""

    NO_VISUAL_PROMPT = "NO_VISUAL_PROMPT"
    REFERENCE_MARKS_READY = "REFERENCE_MARKS_READY"
    VISUAL_PROMPT_READY = "VISUAL_PROMPT_READY"


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
            "button_generate_visual_prompt",
            "visual_prompt_status_label",
            "button_save_visual_prompt",
            "button_load_visual_prompt",
            "button_clear_visual_prompt",
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
        self._cross_image_visual_model = None
        self._prompt_free_model = None
        self._text_encoder = None

        # Cross-image visual prompt state. This is deliberately independent
        # from the one-shot marks used by the existing intra-image mode.
        self.visual_prompt_vpe = None
        self.visual_prompt_classes = []
        self.visual_prompt_reference = None
        self.visual_prompt_model_signature = None
        self.visual_prompt_profile_name = None
        self.visual_prompt_state = VisualPromptState.NO_VISUAL_PROMPT

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

    @staticmethod
    def build_model(model_path):
        """Build and initialize YOLOE model"""
        logger.info(f"Loading model: {model_path}...")
        model = _YOLOE(model_path)
        model.eval()
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        return model

    def _get_text_pe(self, model, texts):
        device = next(model.model.parameters()).device
        if self._text_encoder is None:
            self._text_encoder = _MobileCLIPTextEncoder(
                self.config["embedding_model_path"], device
            )
        model.model.clip_model = self._text_encoder
        return model.model.get_text_pe(texts, cache_clip_model=True)

    def _get_vocab(self, model, texts):
        model.model.set_classes(texts, self._get_text_pe(model, texts))
        model.model.fuse()
        head = model.model.model[-1]
        return torch.nn.ModuleList(cls_head[-1] for cls_head in head.cv3)

    def set_auto_labeling_marks(self, marks):
        """Set visual prompting marks"""
        self.marks = marks
        if marks:
            self.visual_prompt_state = VisualPromptState.REFERENCE_MARKS_READY
        elif self.has_visual_prompt():
            self.visual_prompt_state = VisualPromptState.VISUAL_PROMPT_READY
        else:
            self.visual_prompt_state = VisualPromptState.NO_VISUAL_PROMPT

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
                texts, self._get_text_pe(self._text_model, texts)
            )
            self._current_text_prompt = texts
        return self._text_model

    def _get_visual_model(self):
        """Get or create visual prompt model instance"""
        if self._visual_model is None:
            self._visual_model = self.build_model(self.config["model_path"])
        return self._visual_model

    def _get_cross_image_visual_model(self):
        """Get the model reserved for cached cross-image visual prompts."""
        if self._cross_image_visual_model is None:
            self._cross_image_visual_model = self.build_model(
                self.config["model_path"]
            )
        return self._cross_image_visual_model

    @staticmethod
    def validate_reference_boxes(boxes, image_size):
        """Validate and normalize reference boxes in xyxy pixel format."""
        boxes = np.asarray(boxes, dtype=np.float32)
        if boxes.ndim != 2 or boxes.shape[0] == 0 or boxes.shape[1] != 4:
            raise ValueError("Reference boxes must have shape [N, 4].")
        if not np.isfinite(boxes).all():
            raise ValueError("Reference boxes must contain finite values.")

        width, height = image_size
        invalid = (
            (boxes[:, 0] < 0)
            | (boxes[:, 1] < 0)
            | (boxes[:, 2] > width)
            | (boxes[:, 3] > height)
            | (boxes[:, 2] <= boxes[:, 0])
            | (boxes[:, 3] <= boxes[:, 1])
        )
        if invalid.any():
            raise ValueError(
                "Reference boxes must be valid and within image bounds."
            )
        return boxes

    @staticmethod
    def _normalize_visual_prompt_classes(classes):
        if isinstance(classes, str):
            raise TypeError(
                "Visual prompt classes must be a sequence of names."
            )
        classes = ["object"] if classes is None else list(classes)
        if not classes or any(
            not isinstance(name, str) or not name.strip() for name in classes
        ):
            raise ValueError(
                "Visual prompt classes must be non-empty strings."
            )
        return [name.strip() for name in classes]

    @staticmethod
    def _get_model_device(model):
        return next(model.model.parameters()).device

    @staticmethod
    def _get_model_embedding_dimension(model):
        head = model.model.model[-1]
        embed = getattr(head, "embed", None)
        return int(embed) if isinstance(embed, (int, np.integer)) else None

    def _make_visual_prompt_model_signature(self, model, vpe):
        model_path = os.path.abspath(self.config["model_path"])
        return {
            "model_type": self.config["type"],
            "model_name": self.config["name"],
            "model_path": model_path,
            "model_file": os.path.basename(model_path),
            "model_size": os.path.getsize(model_path),
            "architecture": type(model.model).__name__,
            "embedding_dimension": int(vpe.shape[-1]),
        }

    def _boxes_from_marks(self):
        if not self.marks:
            raise ValueError("No reference bounding boxes are available.")

        boxes = []
        for mark in self.marks:
            if mark.get("type") != "rectangle":
                raise ValueError(
                    "Cross-image visual prompts only support rectangles."
                )
            data = mark.get("data")
            if (
                not isinstance(data, (list, tuple, np.ndarray))
                or len(data) != 4
            ):
                raise ValueError("Each reference rectangle must contain xyxy.")
            boxes.append(data)
        return boxes

    def build_visual_prompt(
        self,
        reference_image,
        reference_boxes=None,
        classes=None,
    ):
        """Generate and cache a VPE from one reference image."""
        classes = self._normalize_visual_prompt_classes(classes)
        if len(classes) != 1:
            raise ValueError(
                "The current cross-image MVP supports one visual class."
            )

        reference_path = None
        if isinstance(reference_image, (str, os.PathLike)):
            reference_path = os.path.abspath(os.fspath(reference_image))
            if not os.path.isfile(reference_path):
                raise FileNotFoundError(
                    f"Reference image does not exist: {reference_path}"
                )
            with Image.open(reference_path) as opened_image:
                image = opened_image.convert("RGB")
        elif isinstance(reference_image, Image.Image):
            image = reference_image.convert("RGB")
        else:
            raise TypeError("Reference image must be a path or PIL image.")

        if reference_boxes is None:
            reference_boxes = self._boxes_from_marks()
        boxes = self.validate_reference_boxes(reference_boxes, image.size)
        prompts = {
            "bboxes": boxes,
            "cls": np.zeros(len(boxes), dtype=np.int64),
        }

        model = self._get_cross_image_visual_model()
        previous_vpe = self.visual_prompt_vpe
        previous_classes = self.visual_prompt_classes
        try:
            # Ultralytics only honors the predictor argument when no predictor
            # is already attached. A prior target prediction installs the
            # standard predictor, so generation must always reset it first.
            model.predictor = None
            model.predict(
                source=image,
                imgsz=self.input_shape,
                conf=self.conf_thres,
                iou=self.iou_thres,
                verbose=False,
                prompts=prompts,
                predictor=YOLOEVPSegPredictor,
                return_vpe=True,
            )
            predictor = model.predictor
            if predictor is None or not hasattr(predictor, "vpe"):
                raise RuntimeError("YOLOE did not return a visual embedding.")
            vpe = predictor.vpe
            reference = {
                "image_path": reference_path,
                "image_size": list(image.size),
                "boxes": boxes.tolist(),
                "instance_count": len(boxes),
                "classes": classes,
            }
            self.set_visual_prompt(vpe, classes, reference)
            self.marks = []
            logger.info(
                "Cross-image visual prompt generated: "
                f"classes={classes}, instances={len(boxes)}, "
                f"vpe_shape={tuple(self.visual_prompt_vpe.shape)}"
            )
            return self.visual_prompt_vpe
        except Exception:
            model.predictor = None
            if previous_vpe is not None and previous_classes:
                model.set_classes(previous_classes, previous_vpe)
            raise

    def set_visual_prompt(self, vpe, classes=None, reference=None):
        """Validate a VPE and bind it to the cross-image model."""
        classes = self._normalize_visual_prompt_classes(classes)
        model = self._get_cross_image_visual_model()
        device = self._get_model_device(model)
        if isinstance(vpe, np.ndarray):
            vpe = torch.from_numpy(vpe)
        if not isinstance(vpe, torch.Tensor):
            raise TypeError(
                "Visual prompt embedding must be a tensor or array."
            )
        if vpe.ndim != 3 or vpe.shape[0] != 1:
            raise ValueError(
                "Visual prompt embedding must have shape [1, C, D]."
            )
        if vpe.shape[1] != len(classes):
            raise ValueError(
                "Visual prompt class count does not match the embedding."
            )
        expected_dimension = self._get_model_embedding_dimension(model)
        if (
            expected_dimension is not None
            and vpe.shape[-1] != expected_dimension
        ):
            raise ValueError(
                "Visual prompt embedding dimension is incompatible with "
                f"the model ({vpe.shape[-1]} != {expected_dimension})."
            )
        if not torch.isfinite(vpe).all().item():
            raise ValueError(
                "Visual prompt embedding must contain finite values."
            )

        cached_vpe = (
            vpe.detach().to(device=device, dtype=torch.float32).clone()
        )
        model.set_classes(classes, cached_vpe)
        # VPE encoding is complete. Target images require the ordinary
        # segmentation predictor, which is created lazily on first inference.
        model.predictor = None

        self.visual_prompt_vpe = cached_vpe
        self.visual_prompt_classes = classes
        self.visual_prompt_reference = reference
        self.visual_prompt_model_signature = (
            self._make_visual_prompt_model_signature(model, cached_vpe)
        )
        self.visual_prompt_profile_name = None
        self.visual_prompt_state = VisualPromptState.VISUAL_PROMPT_READY

    def save_visual_prompt_profile(self, name, directory):
        """Save the cached VPE as a portable, non-pickle profile."""
        if not self.has_visual_prompt():
            raise RuntimeError("No visual prompt is ready to save.")
        profile = VisualPromptProfile.create(
            name=name,
            model_name=self.config["name"],
            model_signature=self.visual_prompt_model_signature,
            reference=self.visual_prompt_reference or {},
            classes=self.visual_prompt_classes,
            vpe=(
                self.visual_prompt_vpe.detach()
                .cpu()
                .to(dtype=torch.float32)
                .numpy()
            ),
        )
        metadata_path = profile.save(directory)
        self.visual_prompt_profile_name = profile.name.strip()
        logger.info(
            "Visual prompt profile saved: "
            f"name={self.visual_prompt_profile_name}, "
            f"path={metadata_path}"
        )
        return metadata_path

    def load_visual_prompt_profile(self, path):
        """Validate a profile and restore its VPE to the model device."""
        profile = VisualPromptProfile.load(path)
        model = self._get_cross_image_visual_model()
        current_signature = self._make_visual_prompt_model_signature(
            model, profile.vpe
        )
        profile.validate_compatibility(current_signature)
        self.set_visual_prompt(
            profile.vpe,
            classes=profile.classes,
            reference=profile.reference,
        )
        self.visual_prompt_profile_name = profile.name
        logger.info(
            "Visual prompt profile loaded: "
            f"name={profile.name}, device={self.visual_prompt_vpe.device}, "
            f"vpe_shape={tuple(self.visual_prompt_vpe.shape)}"
        )
        return profile

    def clear_visual_prompt(self):
        """Clear cached VPE state without affecting other YOLOE modes."""
        if self._cross_image_visual_model is not None:
            self._cross_image_visual_model.predictor = None
        self._cross_image_visual_model = None
        self.visual_prompt_vpe = None
        self.visual_prompt_classes = []
        self.visual_prompt_reference = None
        self.visual_prompt_model_signature = None
        self.visual_prompt_profile_name = None
        self.visual_prompt_state = VisualPromptState.NO_VISUAL_PROMPT
        self.marks = []

    def has_visual_prompt(self):
        """Return whether a validated cross-image VPE is cached."""
        return bool(
            self.visual_prompt_vpe is not None and self.visual_prompt_classes
        )

    @property
    def visual_prompt_ready(self):
        return self.has_visual_prompt()

    def get_visual_prompt_state(self):
        return self.visual_prompt_state.value

    def predict_with_visual_prompt(self, target_image):
        """Predict one target using the cached cross-image visual prompt."""
        if not self.has_visual_prompt():
            raise RuntimeError("No cross-image visual prompt is ready.")

        model = self._get_cross_image_visual_model()
        if isinstance(model.predictor, YOLOEVPSegPredictor):
            model.predictor = None
        results = model.predict(
            source=target_image,
            imgsz=self.input_shape,
            conf=self.conf_thres,
            iou=self.iou_thres,
            verbose=False,
        )
        self.visual_prompt_state = VisualPromptState.VISUAL_PROMPT_READY
        shapes = self.postprocess(results)
        return AutoLabelingResult(shapes, replace=self.replace)

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
            vocab = self._get_vocab(
                self.build_model(self.config["model_path"]), self.texts
            )
            self._prompt_free_model.set_vocab(vocab, names=self.texts)
            self._prompt_free_model.model.model[-1].is_fused = True
            self._prompt_free_model.model.model[-1].max_det = self.max_det
            self._prompt_free_initialized = True

        # Update dynamic parameters each time
        self._prompt_free_model.model.model[-1].iou = self.iou_thres
        self._prompt_free_model.model.model[-1].conf = self.conf_thres
        return self._prompt_free_model

    def predict_shapes(
        self,
        image,
        image_path=None,
        text_prompt=None,
        use_visual_prompt=False,
    ):
        """Predict shapes from image using different prompting modes"""

        if image is None:
            return []

        try:
            with Image.open(image_path) as opened_image:
                image = opened_image.convert("RGB")
        except Exception as e:  # noqa
            raise RuntimeError(
                f"Could not read target image '{image_path}': {e}"
            ) from e

        kwargs = {}

        if use_visual_prompt:
            return self.predict_with_visual_prompt(image)

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
            self.set_auto_labeling_marks([])

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

        # Cached cross-image visual prompting mode
        elif self.has_visual_prompt():
            return self.predict_with_visual_prompt(image)

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
        self.clear_visual_prompt()
        self._text_model = None
        self._visual_model = None
        self._prompt_free_model = None
        self._text_encoder = None
