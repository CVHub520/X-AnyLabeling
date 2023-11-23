from typing import Any, Dict, Optional

from anylabeling.services.auto_labeling.utils.sahi.utils.file import (
    import_model_class,
)

MODEL_TYPE_TO_MODEL_CLASS_NAME = {
    "yolov8": "Yolov8DetectionModel",
    "mmdet": "MmdetDetectionModel",
    "yolov5": "Yolov5DetectionModel",
    "detectron2": "Detectron2DetectionModel",
    "huggingface": "HuggingfaceDetectionModel",
    "yolov5sparse": "Yolov5SparseDetectionModel",
    "yolonas": "YoloNasDetectionModel",
    "yolov8_onnx": "Yolov8OnnxDetectionModel",
}


class AutoDetectionModel:
    @staticmethod
    def from_pretrained(
        model_type: str,
        model_path: Optional[str] = None,
        model: Optional[Any] = None,
        config_path: Optional[str] = None,
        device: Optional[str] = None,
        mask_threshold: float = 0.5,
        confidence_threshold: float = 0.3,
        category_mapping: Optional[Dict] = None,
        category_remapping: Optional[Dict] = None,
        load_at_init: bool = True,
        image_size: int = None,
        **kwargs,
    ):
        """
        Loads a DetectionModel from given path.

        Args:
            model_type: str
                Name of the detection framework (example: "yolov5", "mmdet", "detectron2")
            model_path: str
                Path of the detection model (ex. 'model.pt')
            config_path: str
                Path of the config file (ex. 'mmdet/configs/cascade_rcnn_r50_fpn_1x.py')
            device: str
                Device, "cpu" or "cuda:0"
            mask_threshold: float
                Value to threshold mask pixels, should be between 0 and 1
            confidence_threshold: float
                All predictions with score < confidence_threshold will be discarded
            category_mapping: dict: str to str
                Mapping from category id (str) to category name (str) e.g. {"1": "pedestrian"}
            category_remapping: dict: str to int
                Remap category ids based on category names, after performing inference e.g. {"car": 3}
            load_at_init: bool
                If True, automatically loads the model at initalization
            image_size: int
                Inference input size.
        Returns:
            Returns an instance of a DetectionModel
        Raises:
            ImportError: If given {model_type} framework is not installed
        """

        model_class_name = MODEL_TYPE_TO_MODEL_CLASS_NAME[model_type]
        DetectionModel = import_model_class(model_type, model_class_name)

        return DetectionModel(
            model_path=model_path,
            model=model,
            config_path=config_path,
            device=device,
            mask_threshold=mask_threshold,
            confidence_threshold=confidence_threshold,
            category_mapping=category_mapping,
            category_remapping=category_remapping,
            load_at_init=load_at_init,
            image_size=image_size,
            **kwargs,
        )
