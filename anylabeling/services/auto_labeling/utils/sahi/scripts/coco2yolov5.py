import fire

from anylabeling.services.auto_labeling.utils.sahi.utils.coco import Coco
from anylabeling.services.auto_labeling.utils.sahi.utils.file import (
    Path,
    increment_path,
)


def main(
    image_dir: str,
    dataset_json_path: str,
    train_split: str = 0.9,
    project: str = "runs/coco2yolov5",
    name: str = "exp",
    seed: str = 1,
    disable_symlink=False,
):
    """
    Args:
        images_dir (str): directory for coco images
        dataset_json_path (str): file path for the coco json file to be converted
        train_split (str): set the training split ratio
        project (str): save results to project/name
        name (str): save results to project/name"
        seed (int): fix the seed for reproducibility
        disable_symlink (bool): required in google colab env
    """

    # increment run
    save_dir = Path(increment_path(Path(project) / name, exist_ok=False))
    # load coco dict
    coco = Coco.from_coco_dict_or_path(
        coco_dict_or_path=dataset_json_path,
        image_dir=image_dir,
    )
    # export as yolov5
    coco.export_as_yolov5(
        output_dir=str(save_dir),
        train_split_rate=train_split,
        numpy_seed=seed,
        disable_symlink=disable_symlink,
    )

    print(
        f"COCO to YOLOv5 conversion results are successfully exported to {save_dir}"
    )


if __name__ == "__main__":
    fire.Fire(main)
