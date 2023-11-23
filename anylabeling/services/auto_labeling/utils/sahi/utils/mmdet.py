import shutil
import sys
import urllib.request
from importlib import import_module
from os import path
from pathlib import Path
from typing import Optional

from anylabeling.services.auto_labeling.utils.sahi.utils.file import (
    download_from_url,
)


def mmdet_version_as_integer():
    import mmdet

    return int(mmdet.__version__.replace(".", ""))


class MmdetTestConstants:
    MMDET_CASCADEMASKRCNN_MODEL_URL = "http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco/cascade_mask_rcnn_r50_fpn_1x_coco_20200203-9d4dcb24.pth"
    MMDET_CASCADEMASKRCNN_MODEL_PATH = "tests/data/models/mmdet/cascade_mask_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco_20200203-9d4dcb24.pth"
    MMDET_RETINANET_MODEL_URL = "http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_2x_coco/retinanet_r50_fpn_2x_coco_20200131-fdb43119.pth"
    MMDET_RETINANET_MODEL_PATH = "tests/data/models/mmdet/retinanet/retinanet_r50_fpn_2x_coco_20200131-fdb43119.pth"
    MMDET_YOLOX_TINY_MODEL_URL = "https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_tiny_8x8_300e_coco/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth"
    MMDET_YOLOX_TINY_MODEL_PATH = "tests/data/models/mmdet/yolox/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth"

    MMDET_CASCADEMASKRCNN_CONFIG_PATH = "tests/data/models/mmdet/cascade_mask_rcnn/cascade-mask-rcnn_r50_fpn_1x_coco.py"
    MMDET_RETINANET_CONFIG_PATH = (
        "tests/data/models/mmdet/retinanet/retinanet_r50_fpn_1x_coco.py"
    )
    MMDET_YOLOX_TINY_CONFIG_PATH = (
        "tests/data/models/mmdet/yolox/yolox_tiny_8xb8-300e_coco.py"
    )


def download_mmdet_cascade_mask_rcnn_model(
    destination_path: Optional[str] = None,
):
    if destination_path is None:
        destination_path = MmdetTestConstants.MMDET_CASCADEMASKRCNN_MODEL_PATH

    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)

    download_from_url(
        MmdetTestConstants.MMDET_CASCADEMASKRCNN_MODEL_URL, destination_path
    )


def download_mmdet_retinanet_model(destination_path: Optional[str] = None):
    if destination_path is None:
        destination_path = MmdetTestConstants.MMDET_RETINANET_MODEL_PATH

    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)

    download_from_url(
        MmdetTestConstants.MMDET_RETINANET_MODEL_URL, destination_path
    )


def download_mmdet_yolox_tiny_model(destination_path: Optional[str] = None):
    if destination_path is None:
        destination_path = MmdetTestConstants.MMDET_YOLOX_TINY_MODEL_PATH

    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)

    download_from_url(
        MmdetTestConstants.MMDET_YOLOX_TINY_MODEL_URL, destination_path
    )


def download_mmdet_config(
    model_name: str = "cascade_rcnn",
    config_file_name: str = "cascade_mask_rcnn_r50_fpn_1x_coco.py",
    verbose: bool = True,
) -> str:
    """
    Merges config files starting from given main config file name. Saves as single file.

    Args:
        model_name (str): mmdet model name. check https://github.com/open-mmlab/mmdetection/tree/master/configs.
        config_file_name (str): mdmet config file name.
        verbose (bool): if True, print save path.

    Returns:
        (str) abs path of the downloaded config file.
    """

    # get mmdet version
    from mmdet import __version__

    mmdet_ver = "v" + __version__

    # set main config url
    base_config_url = (
        "https://raw.githubusercontent.com/open-mmlab/mmdetection/"
        + mmdet_ver
        + "/configs/"
        + model_name
        + "/"
    )
    main_config_url = base_config_url + config_file_name

    # set final config dirs
    configs_dir = Path("mmdet_configs") / mmdet_ver
    model_config_dir = configs_dir / model_name

    # create final config dir
    configs_dir.mkdir(parents=True, exist_ok=True)
    model_config_dir.mkdir(parents=True, exist_ok=True)

    # get final config file name
    filename = Path(main_config_url).name

    # set final config file path
    final_config_path = str(model_config_dir / filename)

    if not Path(final_config_path).exists():
        # set config dirs
        temp_configs_dir = Path("temp_mmdet_configs")
        main_config_dir = temp_configs_dir / model_name

        # create config dirs
        temp_configs_dir.mkdir(parents=True, exist_ok=True)
        main_config_dir.mkdir(parents=True, exist_ok=True)

        # get main config file name
        filename = Path(main_config_url).name

        # set main config file path
        main_config_path = str(main_config_dir / filename)

        # download main config file
        urllib.request.urlretrieve(
            main_config_url,
            main_config_path,
        )

        # read main config file
        sys.path.insert(0, str(main_config_dir))
        temp_module_name = path.splitext(filename)[0]
        mod = import_module(temp_module_name)
        sys.path.pop(0)
        config_dict = {
            name: value
            for name, value in mod.__dict__.items()
            if not name.startswith("__")
        }

        # handle when config_dict["_base_"] is string
        if not isinstance(config_dict["_base_"], list):
            config_dict["_base_"] = [config_dict["_base_"]]

        # iterate over secondary config files
        for secondary_config_file_path in config_dict["_base_"]:
            # set config url
            config_url = base_config_url + secondary_config_file_path
            config_path = main_config_dir / secondary_config_file_path

            # create secondary config dir
            config_path.parent.mkdir(parents=True, exist_ok=True)

            # download secondary config files
            urllib.request.urlretrieve(
                config_url,
                str(config_path),
            )

            # read secondary config file
            secondary_config_dir = config_path.parent
            sys.path.insert(0, str(secondary_config_dir))
            temp_module_name = path.splitext(Path(config_path).name)[0]
            mod = import_module(temp_module_name)
            sys.path.pop(0)
            secondary_config_dict = {
                name: value
                for name, value in mod.__dict__.items()
                if not name.startswith("__")
            }

            # go deeper if there are more steps
            if secondary_config_dict.get("_base_") is not None:
                # handle when config_dict["_base_"] is string
                if not isinstance(secondary_config_dict["_base_"], list):
                    secondary_config_dict["_base_"] = [
                        secondary_config_dict["_base_"]
                    ]

                # iterate over third config files
                for third_config_file_path in secondary_config_dict["_base_"]:
                    # set config url
                    config_url = base_config_url + third_config_file_path
                    config_path = main_config_dir / third_config_file_path

                    # create secondary config dir
                    config_path.parent.mkdir(parents=True, exist_ok=True)
                    # download secondary config files
                    urllib.request.urlretrieve(
                        config_url,
                        str(config_path),
                    )

        # dump final config as single file
        from mmengine import Config

        config = Config.fromfile(main_config_path)
        config.dump(final_config_path)

        if verbose:
            print(
                f"mmdet config file has been downloaded to {path.abspath(final_config_path)}"
            )

        # remove temp config dir
        shutil.rmtree(temp_configs_dir)

    return path.abspath(final_config_path)
