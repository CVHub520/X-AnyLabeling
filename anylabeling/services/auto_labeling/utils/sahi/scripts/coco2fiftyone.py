import time
from pathlib import Path
from typing import List

import fire

from anylabeling.services.auto_labeling.utils.sahi.utils.file import load_json


def main(
    image_dir: str,
    dataset_json_path: str,
    *result_json_paths,
    iou_thresh: float = 0.5,
):
    """
    Args:
        image_dir (str): directory for coco images
        dataset_json_path (str): file path for the coco dataset json file
        result_json_paths (str): one or more paths for the coco result json file
        iou_thresh (float): iou threshold for coco evaluation
    """

    from fiftyone.utils.coco import add_coco_labels

    from sahi.utils.fiftyone import create_fiftyone_dataset_from_coco_file, fo

    coco_result_list = []
    result_name_list = []
    if result_json_paths:
        for result_json_path in result_json_paths:
            coco_result = load_json(result_json_path)
            coco_result_list.append(coco_result)

            # use file names as fiftyone name, create unique names if duplicate
            result_name_temp = Path(result_json_path).stem
            result_name = result_name_temp
            name_increment = 2
            while result_name in result_name_list:
                result_name = result_name_temp + "_" + str(name_increment)
                name_increment += 1
            result_name_list.append(result_name)

    dataset = create_fiftyone_dataset_from_coco_file(
        image_dir, dataset_json_path
    )

    # submit detections if coco result is given
    if result_json_paths:
        for result_name, coco_result in zip(
            result_name_list, coco_result_list
        ):
            add_coco_labels(
                dataset, result_name, coco_result, coco_id_field="gt_coco_id"
            )

    # visualize results
    session = fo.launch_app()
    session.dataset = dataset

    # order by false positives if any coco result is given
    if result_json_paths:
        # Evaluate the predictions
        first_coco_result_name = result_name_list[0]
        _ = dataset.evaluate_detections(
            first_coco_result_name,
            gt_field="gt_detections",
            eval_key=f"{first_coco_result_name}_eval",
            iou=iou_thresh,
            compute_mAP=False,
        )
        # Get the 10 most common classes in the dataset
        # counts = dataset.count_values("gt_detections.detections.label")
        # classes_top10 = sorted(counts, key=counts.get, reverse=True)[:10]
        # Print a classification report for the top-10 classes
        # results.print_report(classes=classes_top10)
        # Load the view on which we ran the `eval` evaluation
        eval_view = dataset.load_evaluation_view(
            f"{first_coco_result_name}_eval"
        )
        # Show samples with most false positives
        session.view = eval_view.sort_by(
            f"{first_coco_result_name}_eval_fp", reverse=True
        )

        print(
            "SAHI has successfully launched a Fiftyone app "
            f"at http://localhost:{fo.config.default_app_port}"
        )
    while 1:
        time.sleep(3)


if __name__ == "__main__":
    fire.Fire(main)
