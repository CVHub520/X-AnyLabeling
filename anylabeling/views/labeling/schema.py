from typing import Any, Dict, List, Optional

from anylabeling.app_info import __version__


XLABEL_BASIC_FIELDS = [
    "version",
    "flags",
    "shapes",
    "imagePath",
    "imageData",
    "imageHeight",
    "imageWidth",
]


def create_xlabel_template(
    version: str = __version__,
    flags: Optional[Dict[str, Any]] = None,
    shapes: Optional[List[Dict[str, Any]]] = None,
    image_path: str = "",
    image_data: Optional[str] = None,
    image_height: int = -1,
    image_width: int = -1,
) -> Dict[str, Any]:
    return {
        "version": version,
        "flags": flags if flags is not None else {},
        "shapes": shapes if shapes is not None else [],
        "imagePath": image_path,
        "imageData": image_data,
        "imageHeight": image_height,
        "imageWidth": image_width,
    }
