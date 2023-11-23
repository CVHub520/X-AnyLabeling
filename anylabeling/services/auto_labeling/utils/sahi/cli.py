import fire

from anylabeling.services.auto_labeling.utils.sahi import (
    __version__ as sahi_version,
)
from anylabeling.services.auto_labeling.utils.sahi.predict import (
    predict,
    predict_fiftyone,
)
from anylabeling.services.auto_labeling.utils.sahi.scripts.coco2fiftyone import (
    main as coco2fiftyone,
)
from anylabeling.services.auto_labeling.utils.sahi.scripts.coco2yolov5 import (
    main as coco2yolov5,
)
from anylabeling.services.auto_labeling.utils.sahi.scripts.coco_error_analysis import (
    analyse,
)
from anylabeling.services.auto_labeling.utils.sahi.scripts.coco_evaluation import (
    evaluate,
)
from anylabeling.services.auto_labeling.utils.sahi.scripts.slice_coco import (
    slice,
)
from anylabeling.services.auto_labeling.utils.sahi.utils.import_utils import (
    print_enviroment_info,
)

coco_app = {
    "evaluate": evaluate,
    "analyse": analyse,
    "fiftyone": coco2fiftyone,
    "slice": slice,
    "yolov5": coco2yolov5,
}

sahi_app = {
    "predict": predict,
    "predict-fiftyone": predict_fiftyone,
    "coco": coco_app,
    "version": sahi_version,
    "env": print_enviroment_info,
}


def app() -> None:
    """Cli app."""
    fire.Fire(sahi_app)


if __name__ == "__main__":
    app()
