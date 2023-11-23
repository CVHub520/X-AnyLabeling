# OBSS SAHI Tool
# Code written by Kadir Nar, 2022.


from packaging import version

from anylabeling.services.auto_labeling.utils.sahi.utils.import_utils import (
    get_package_info,
)


class TorchVisionTestConstants:
    FASTERRCNN_CONFIG_PATH = (
        "tests/data/models/torchvision/fasterrcnn_resnet50_fpn.yaml"
    )
    SSD300_CONFIG_PATH = "tests/data/models/torchvision/ssd300_vgg16.yaml"


_torchvision_available, _torchvision_version = get_package_info(
    "torchvision", verbose=False
)

if _torchvision_available:
    import torchvision

    MODEL_NAME_TO_CONSTRUCTOR = {
        "fasterrcnn_resnet50_fpn": torchvision.models.detection.fasterrcnn_resnet50_fpn,
        "fasterrcnn_mobilenet_v3_large_fpn": torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn,
        "fasterrcnn_mobilenet_v3_large_320_fpn": torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn,
        "retinanet_resnet50_fpn": torchvision.models.detection.retinanet_resnet50_fpn,
        "ssd300_vgg16": torchvision.models.detection.ssd300_vgg16,
        "ssdlite320_mobilenet_v3_large": torchvision.models.detection.ssdlite320_mobilenet_v3_large,
    }

    # fcos requires torchvision >= 0.12.0
    if version.parse(_torchvision_version) >= version.parse("0.12.0"):
        MODEL_NAME_TO_CONSTRUCTOR["fcos_resnet50_fpn"] = (
            torchvision.models.detection.fcos_resnet50_fpn,
        )


COCO_CLASSES = [
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]
