import os

import fire

from anylabeling.services.auto_labeling.utils.sahi.slicing import slice_coco
from anylabeling.services.auto_labeling.utils.sahi.utils.file import (
    Path,
    save_json,
)


def slice(
    image_dir: str,
    dataset_json_path: str,
    slice_size: int = 512,
    overlap_ratio: float = 0.2,
    ignore_negative_samples: bool = False,
    output_dir: str = "runs/slice_coco",
    min_area_ratio: float = 0.1,
):
    """
    Args:
        image_dir (str): directory for coco images
        dataset_json_path (str): file path for the coco dataset json file
        slice_size (int)
        overlap_ratio (float): slice overlap ratio
        ignore_negative_samples (bool): ignore images without annotation
        output_dir (str): output export dir
        min_area_ratio (float): If the cropped annotation area to original
            annotation ratio is smaller than this value, the annotation
            is filtered out. Default 0.1.
    """

    # assure slice_size is list
    slice_size_list = slice_size
    if isinstance(slice_size_list, (int, float)):
        slice_size_list = [slice_size_list]

    # slice coco dataset images and annotations
    print("Slicing step is starting...")
    for slice_size in slice_size_list:
        # in format: train_images_512_01
        output_images_folder_name = (
            Path(dataset_json_path).stem
            + f"_images_{str(slice_size)}_{str(overlap_ratio).replace('.','')}"
        )
        output_images_dir = str(Path(output_dir) / output_images_folder_name)
        sliced_coco_name = Path(dataset_json_path).name.replace(
            ".json", f"_{str(slice_size)}_{str(overlap_ratio).replace('.','')}"
        )
        coco_dict, coco_path = slice_coco(
            coco_annotation_file_path=dataset_json_path,
            image_dir=image_dir,
            output_coco_annotation_file_name="",
            output_dir=output_images_dir,
            ignore_negative_samples=ignore_negative_samples,
            slice_height=slice_size,
            slice_width=slice_size,
            min_area_ratio=min_area_ratio,
            overlap_height_ratio=overlap_ratio,
            overlap_width_ratio=overlap_ratio,
            out_ext=".jpg",
            verbose=False,
        )
        output_coco_annotation_file_path = os.path.join(
            output_dir, sliced_coco_name + ".json"
        )
        save_json(coco_dict, output_coco_annotation_file_path)
        print(
            f"Sliced dataset for 'slice_size: {slice_size}' is exported to {output_dir}"
        )


if __name__ == "__main__":
    fire.Fire(slice)
