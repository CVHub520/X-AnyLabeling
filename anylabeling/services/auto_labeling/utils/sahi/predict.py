# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.

import logging
import os
import time
from typing import List, Optional

from functools import cmp_to_key

import numpy as np
from tqdm import tqdm

from anylabeling.services.auto_labeling.utils.sahi.auto_model import (
    AutoDetectionModel,
)
from anylabeling.services.auto_labeling.utils.sahi.models.base import (
    DetectionModel,
)
from anylabeling.services.auto_labeling.utils.sahi.postprocess.combine import (
    GreedyNMMPostprocess,
    LSNMSPostprocess,
    NMMPostprocess,
    NMSPostprocess,
    PostprocessPredictions,
)
from anylabeling.services.auto_labeling.utils.sahi.prediction import (
    ObjectPrediction,
    PredictionResult,
)
from anylabeling.services.auto_labeling.utils.sahi.slicing import slice_image
from anylabeling.services.auto_labeling.utils.sahi.utils.coco import (
    Coco,
    CocoImage,
)
from anylabeling.services.auto_labeling.utils.sahi.utils.cv import (
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    crop_object_predictions,
    cv2,
    get_video_reader,
    read_image_as_pil,
    visualize_object_predictions,
)
from anylabeling.services.auto_labeling.utils.sahi.utils.file import (
    Path,
    increment_path,
    list_files,
    save_json,
    save_pickle,
)
from anylabeling.services.auto_labeling.utils.sahi.utils.import_utils import (
    check_requirements,
)

POSTPROCESS_NAME_TO_CLASS = {
    "GREEDYNMM": GreedyNMMPostprocess,
    "NMM": NMMPostprocess,
    "NMS": NMSPostprocess,
    "LSNMS": LSNMSPostprocess,
}

LOW_MODEL_CONFIDENCE = 0.1


logger = logging.getLogger(__name__)


def get_prediction(
    image,
    detection_model,
    shift_amount: list = [0, 0],
    full_shape=None,
    postprocess: Optional[PostprocessPredictions] = None,
    verbose: int = 0,
) -> PredictionResult:
    """
    Function for performing prediction for given image using given detection_model.

    Arguments:
        image: str or np.ndarray
            Location of image or numpy image matrix to slice
        detection_model: model.DetectionMode
        shift_amount: List
            To shift the box and mask predictions from sliced image to full
            sized image, should be in the form of [shift_x, shift_y]
        full_shape: List
            Size of the full image, should be in the form of [height, width]
        postprocess: sahi.postprocess.combine.PostprocessPredictions
        verbose: int
            0: no print (default)
            1: print prediction duration

    Returns:
        A dict with fields:
            object_prediction_list: a list of ObjectPrediction
            durations_in_seconds: a dict containing elapsed times for profiling
    """
    durations_in_seconds = dict()

    # read image as pil
    image_as_pil = read_image_as_pil(image)
    # get prediction
    time_start = time.time()
    detection_model.perform_inference(np.ascontiguousarray(image_as_pil))
    time_end = time.time() - time_start
    durations_in_seconds["prediction"] = time_end

    # process prediction
    time_start = time.time()
    # works only with 1 batch
    detection_model.convert_original_predictions(
        shift_amount=shift_amount,
        full_shape=full_shape,
    )
    object_prediction_list: List[ObjectPrediction] = (
        detection_model.object_prediction_list
    )

    # postprocess matching predictions
    if postprocess is not None:
        object_prediction_list = postprocess(object_prediction_list)

    time_end = time.time() - time_start
    durations_in_seconds["postprocess"] = time_end

    if verbose == 1:
        print(
            "Prediction performed in",
            durations_in_seconds["prediction"],
            "seconds.",
        )

    return PredictionResult(
        image=image,
        object_prediction_list=object_prediction_list,
        durations_in_seconds=durations_in_seconds,
    )


def get_sliced_prediction(
    image,
    detection_model=None,
    slice_height: int = None,
    slice_width: int = None,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2,
    perform_standard_pred: bool = True,
    postprocess_type: str = "GREEDYNMM",
    postprocess_match_metric: str = "IOS",
    postprocess_match_threshold: float = 0.5,
    postprocess_class_agnostic: bool = False,
    verbose: int = 1,
    merge_buffer_length: int = None,
    auto_slice_resolution: bool = True,
) -> PredictionResult:
    """
    Function for slice image + get predicion for each slice + combine predictions in full image.

    Args:
        image: str or np.ndarray
            Location of image or numpy image matrix to slice
        detection_model: model.DetectionModel
        slice_height: int
            Height of each slice.  Defaults to ``None``.
        slice_width: int
            Width of each slice.  Defaults to ``None``.
        overlap_height_ratio: float
            Fractional overlap in height of each window (e.g. an overlap of 0.2 for a window
            of size 512 yields an overlap of 102 pixels).
            Default to ``0.2``.
        overlap_width_ratio: float
            Fractional overlap in width of each window (e.g. an overlap of 0.2 for a window
            of size 512 yields an overlap of 102 pixels).
            Default to ``0.2``.
        perform_standard_pred: bool
            Perform a standard prediction on top of sliced predictions to increase large object
            detection accuracy. Default: True.
        postprocess_type: str
            Type of the postprocess to be used after sliced inference while merging/eliminating predictions.
            Options are 'NMM', 'GRREDYNMM' or 'NMS'. Default is 'GRREDYNMM'.
        postprocess_match_metric: str
            Metric to be used during object prediction matching after sliced prediction.
            'IOU' for intersection over union, 'IOS' for intersection over smaller area.
        postprocess_match_threshold: float
            Sliced predictions having higher iou than postprocess_match_threshold will be
            postprocessed after sliced prediction.
        postprocess_class_agnostic: bool
            If True, postprocess will ignore category ids.
        verbose: int
            0: no print
            1: print number of slices (default)
            2: print number of slices and slice/prediction durations
        merge_buffer_length: int
            The length of buffer for slices to be used during sliced prediction, which is suitable for low memory.
            It may affect the AP if it is specified. The higher the amount, the closer results to the non-buffered.
            scenario. See [the discussion](https://github.com/obss/sahi/pull/445).
        auto_slice_resolution: bool
            if slice parameters (slice_height, slice_width) are not given,
            it enables automatically calculate these params from image resolution and orientation.

    Returns:
        A Dict with fields:
            object_prediction_list: a list of sahi.prediction.ObjectPrediction
            durations_in_seconds: a dict containing elapsed times for profiling
    """

    # for profiling
    durations_in_seconds = dict()

    # currently only 1 batch supported
    num_batch = 1

    # create slices from full image
    time_start = time.time()
    slice_image_result = slice_image(
        image=image,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
        auto_slice_resolution=auto_slice_resolution,
    )
    num_slices = len(slice_image_result)
    time_end = time.time() - time_start
    durations_in_seconds["slice"] = time_end

    # init match postprocess instance
    if postprocess_type not in POSTPROCESS_NAME_TO_CLASS.keys():
        raise ValueError(
            f"postprocess_type should be one of {list(POSTPROCESS_NAME_TO_CLASS.keys())} but given as {postprocess_type}"
        )
    elif postprocess_type == "UNIONMERGE":
        # deprecated in v0.9.3
        raise ValueError(
            "'UNIONMERGE' postprocess_type is deprecated, use 'GREEDYNMM' instead."
        )
    postprocess_constructor = POSTPROCESS_NAME_TO_CLASS[postprocess_type]
    postprocess = postprocess_constructor(
        match_threshold=postprocess_match_threshold,
        match_metric=postprocess_match_metric,
        class_agnostic=postprocess_class_agnostic,
    )

    # create prediction input
    num_group = int(num_slices / num_batch)
    if verbose == 1 or verbose == 2:
        tqdm.write(f"Performing prediction on {num_slices} number of slices.")
    object_prediction_list = []
    # perform sliced prediction
    for group_ind in range(num_group):
        # prepare batch (currently supports only 1 batch)
        image_list = []
        shift_amount_list = []
        for image_ind in range(num_batch):
            image_list.append(
                slice_image_result.images[group_ind * num_batch + image_ind]
            )
            shift_amount_list.append(
                slice_image_result.starting_pixels[
                    group_ind * num_batch + image_ind
                ]
            )
        # perform batch prediction
        prediction_result = get_prediction(
            image=image_list[0],
            detection_model=detection_model,
            shift_amount=shift_amount_list[0],
            full_shape=[
                slice_image_result.original_image_height,
                slice_image_result.original_image_width,
            ],
        )
        # convert sliced predictions to full predictions
        for object_prediction in prediction_result.object_prediction_list:
            if object_prediction:  # if not empty
                object_prediction_list.append(
                    object_prediction.get_shifted_object_prediction()
                )

        # merge matching predictions during sliced prediction
        if (
            merge_buffer_length is not None
            and len(object_prediction_list) > merge_buffer_length
        ):
            object_prediction_list = postprocess(object_prediction_list)

    # perform standard prediction
    if num_slices > 1 and perform_standard_pred:
        prediction_result = get_prediction(
            image=image,
            detection_model=detection_model,
            shift_amount=[0, 0],
            full_shape=None,
            postprocess=None,
        )
        object_prediction_list.extend(prediction_result.object_prediction_list)

    # merge matching predictions
    if len(object_prediction_list) > 1:
        object_prediction_list = postprocess(object_prediction_list)

    time_end = time.time() - time_start
    durations_in_seconds["prediction"] = time_end

    if verbose == 2:
        print(
            "Slicing performed in",
            durations_in_seconds["slice"],
            "seconds.",
        )
        print(
            "Prediction performed in",
            durations_in_seconds["prediction"],
            "seconds.",
        )

    return PredictionResult(
        image=image,
        object_prediction_list=object_prediction_list,
        durations_in_seconds=durations_in_seconds,
    )


def bbox_sort(a, b, thresh):
    """
    a, b  - function receives two bounding bboxes

    thresh - the threshold takes into account how far two bounding bboxes differ in
    Y where thresh is the threshold we set for the
    minimum allowable difference in height between adjacent bboxes
    and sorts them by the X coordinate
    """

    bbox_a = a
    bbox_b = b

    if abs(bbox_a[1] - bbox_b[1]) <= thresh:
        return bbox_a[0] - bbox_b[0]

    return bbox_a[1] - bbox_b[1]


def agg_prediction(result: PredictionResult, thresh):
    coord_list = []
    res = result.to_coco_annotations()
    for ann in res:
        current_bbox = ann["bbox"]
        x = current_bbox[0]
        y = current_bbox[1]
        w = current_bbox[2]
        h = current_bbox[3]

        coord_list.append((x, y, w, h))
    cnts = sorted(
        coord_list, key=cmp_to_key(lambda a, b: bbox_sort(a, b, thresh))
    )
    for pred in range(len(res) - 1):
        res[pred]["image_id"] = cnts.index(tuple(res[pred]["bbox"]))

    return res


def predict(  # noqa: C901
    detection_model: DetectionModel = None,
    model_type: str = "mmdet",
    model_path: str = None,
    model_config_path: str = None,
    model_confidence_threshold: float = 0.25,
    model_device: str = None,
    model_category_mapping: dict = None,
    model_category_remapping: dict = None,
    source: str = None,
    no_standard_prediction: bool = False,
    no_sliced_prediction: bool = False,
    image_size: int = None,
    slice_height: int = 512,
    slice_width: int = 512,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2,
    postprocess_type: str = "GREEDYNMM",
    postprocess_match_metric: str = "IOS",
    postprocess_match_threshold: float = 0.5,
    postprocess_class_agnostic: bool = False,
    novisual: bool = False,
    view_video: bool = False,
    frame_skip_interval: int = 0,
    export_pickle: bool = False,
    export_crop: bool = False,
    dataset_json_path: bool = None,
    project: str = "runs/predict",
    name: str = "exp",
    visual_bbox_thickness: int = None,
    visual_text_size: float = None,
    visual_text_thickness: int = None,
    visual_hide_labels: bool = False,
    visual_hide_conf: bool = False,
    visual_export_format: str = "png",
    verbose: int = 1,
    return_dict: bool = False,
    force_postprocess_type: bool = False,
    **kwargs,
):
    """
    Performs prediction for all present images in given folder.

    Args:
        detection_model: sahi.model.DetectionModel
            Optionally provide custom DetectionModel to be used for inference. When provided,
            model_type, model_path, config_path, model_device, model_category_mapping, image_size
            params will be ignored
        model_type: str
            mmdet for 'MmdetDetectionModel', 'yolov5' for 'Yolov5DetectionModel'.
        model_path: str
            Path for the model weight
        model_config_path: str
            Path for the detection model config file
        model_confidence_threshold: float
            All predictions with score < model_confidence_threshold will be discarded.
        model_device: str
            device, "cpu" or "cuda"
        model_category_mapping: dict
            Mapping from category id (str) to category name (str) e.g. {"1": "pedestrian"}
        model_category_remapping: dict: str to int
            Remap category ids after performing inference
        source: str
            Folder directory that contains images or path of the image to be predicted. Also video to be predicted.
        no_standard_prediction: bool
            Dont perform standard prediction. Default: False.
        no_sliced_prediction: bool
            Dont perform sliced prediction. Default: False.
        image_size: int
            Input image size for each inference (image is scaled by preserving asp. rat.).
        slice_height: int
            Height of each slice.  Defaults to ``512``.
        slice_width: int
            Width of each slice.  Defaults to ``512``.
        overlap_height_ratio: float
            Fractional overlap in height of each window (e.g. an overlap of 0.2 for a window
            of size 512 yields an overlap of 102 pixels).
            Default to ``0.2``.
        overlap_width_ratio: float
            Fractional overlap in width of each window (e.g. an overlap of 0.2 for a window
            of size 512 yields an overlap of 102 pixels).
            Default to ``0.2``.
        postprocess_type: str
            Type of the postprocess to be used after sliced inference while merging/eliminating predictions.
            Options are 'NMM', 'GREEDYNMM', 'LSNMS' or 'NMS'. Default is 'GRREDYNMM'.
        postprocess_match_metric: str
            Metric to be used during object prediction matching after sliced prediction.
            'IOU' for intersection over union, 'IOS' for intersection over smaller area.
        postprocess_match_threshold: float
            Sliced predictions having higher iou than postprocess_match_threshold will be
            postprocessed after sliced prediction.
        postprocess_class_agnostic: bool
            If True, postprocess will ignore category ids.
        novisual: bool
            Dont export predicted video/image visuals.
        view_video: bool
            View result of prediction during video inference.
        frame_skip_interval: int
            If view_video or export_visual is slow, you can process one frames of 3(for exp: --frame_skip_interval=3).
        export_pickle: bool
            Export predictions as .pickle
        export_crop: bool
            Export predictions as cropped images.
        dataset_json_path: str
            If coco file path is provided, detection results will be exported in coco json format.
        project: str
            Save results to project/name.
        name: str
            Save results to project/name.
        visual_bbox_thickness: int
        visual_text_size: float
        visual_text_thickness: int
        visual_hide_labels: bool
        visual_hide_conf: bool
        visual_export_format: str
            Can be specified as 'jpg' or 'png'
        verbose: int
            0: no print
            1: print slice/prediction durations, number of slices
            2: print model loading/file exporting durations
        return_dict: bool
            If True, returns a dict with 'export_dir' field.
        force_postprocess_type: bool
            If True, auto postprocess check will e disabled
    """
    # assert prediction type
    if no_standard_prediction and no_sliced_prediction:
        raise ValueError(
            "'no_standard_prediction' and 'no_sliced_prediction' cannot be True at the same time."
        )

    # auto postprocess type
    if (
        not force_postprocess_type
        and model_confidence_threshold < LOW_MODEL_CONFIDENCE
        and postprocess_type != "NMS"
    ):
        logger.warning(
            f"Switching postprocess type/metric to NMS/IOU since confidence threshold is low ({model_confidence_threshold})."
        )
        postprocess_type = "NMS"
        postprocess_match_metric = "IOU"

    # for profiling
    durations_in_seconds = dict()

    # init export directories
    save_dir = Path(
        increment_path(Path(project) / name, exist_ok=False)
    )  # increment run
    crop_dir = save_dir / "crops"
    visual_dir = save_dir / "visuals"
    visual_with_gt_dir = save_dir / "visuals_with_gt"
    pickle_dir = save_dir / "pickles"
    if (
        not novisual
        or export_pickle
        or export_crop
        or dataset_json_path is not None
    ):
        save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # init image iterator
    # TODO: rewrite this as iterator class as in https://github.com/ultralytics/yolov5/blob/d059d1da03aee9a3c0059895aa4c7c14b7f25a9e/utils/datasets.py#L178
    source_is_video = False
    num_frames = None
    if dataset_json_path:
        coco: Coco = Coco.from_coco_dict_or_path(dataset_json_path)
        image_iterator = [
            str(Path(source) / Path(coco_image.file_name))
            for coco_image in coco.images
        ]
        coco_json = []
    elif os.path.isdir(source):
        image_iterator = list_files(
            directory=source,
            contains=IMAGE_EXTENSIONS,
            verbose=verbose,
        )
    elif Path(source).suffix in VIDEO_EXTENSIONS:
        source_is_video = True
        (
            read_video_frame,
            output_video_writer,
            video_file_name,
            num_frames,
        ) = get_video_reader(
            source, save_dir, frame_skip_interval, not novisual, view_video
        )
        image_iterator = read_video_frame
    else:
        image_iterator = [source]

    # init model instance
    time_start = time.time()
    if detection_model is None:
        detection_model = AutoDetectionModel.from_pretrained(
            model_type=model_type,
            model_path=model_path,
            config_path=model_config_path,
            confidence_threshold=model_confidence_threshold,
            device=model_device,
            category_mapping=model_category_mapping,
            category_remapping=model_category_remapping,
            load_at_init=False,
            image_size=image_size,
            **kwargs,
        )
        detection_model.load_model()
    time_end = time.time() - time_start
    durations_in_seconds["model_load"] = time_end

    # iterate over source images
    durations_in_seconds["prediction"] = 0
    durations_in_seconds["slice"] = 0

    input_type_str = "video frames" if source_is_video else "images"
    for ind, image_path in enumerate(
        tqdm(
            image_iterator,
            f"Performing inference on {input_type_str}",
            total=num_frames,
        )
    ):
        # get filename
        if source_is_video:
            video_name = Path(source).stem
            relative_filepath = video_name + "_frame_" + str(ind)
        elif os.path.isdir(
            source
        ):  # preserve source folder structure in export
            relative_filepath = str(Path(image_path)).split(str(Path(source)))[
                -1
            ]
            relative_filepath = (
                relative_filepath[1:]
                if relative_filepath[0] == os.sep
                else relative_filepath
            )
        else:  # no process if source is single file
            relative_filepath = Path(image_path).name

        filename_without_extension = Path(relative_filepath).stem

        # load image
        image_as_pil = read_image_as_pil(image_path)

        # perform prediction
        if not no_sliced_prediction:
            # get sliced prediction
            prediction_result = get_sliced_prediction(
                image=image_as_pil,
                detection_model=detection_model,
                slice_height=slice_height,
                slice_width=slice_width,
                overlap_height_ratio=overlap_height_ratio,
                overlap_width_ratio=overlap_width_ratio,
                perform_standard_pred=not no_standard_prediction,
                postprocess_type=postprocess_type,
                postprocess_match_metric=postprocess_match_metric,
                postprocess_match_threshold=postprocess_match_threshold,
                postprocess_class_agnostic=postprocess_class_agnostic,
                verbose=1 if verbose else 0,
            )
            object_prediction_list = prediction_result.object_prediction_list
            durations_in_seconds[
                "slice"
            ] += prediction_result.durations_in_seconds["slice"]
        else:
            # get standard prediction
            prediction_result = get_prediction(
                image=image_as_pil,
                detection_model=detection_model,
                shift_amount=[0, 0],
                full_shape=None,
                postprocess=None,
                verbose=0,
            )
            object_prediction_list = prediction_result.object_prediction_list

        durations_in_seconds[
            "prediction"
        ] += prediction_result.durations_in_seconds["prediction"]
        # Show prediction time
        if verbose:
            tqdm.write(
                "Prediction time is: {:.2f} ms".format(
                    prediction_result.durations_in_seconds["prediction"] * 1000
                )
            )

        if dataset_json_path:
            if source_is_video is True:
                raise NotImplementedError(
                    "Video input type not supported with coco formatted dataset json"
                )

            # append predictions in coco format
            for object_prediction in object_prediction_list:
                coco_prediction = object_prediction.to_coco_prediction()
                coco_prediction.image_id = coco.images[ind].id
                coco_prediction_json = coco_prediction.json
                if coco_prediction_json["bbox"]:
                    coco_json.append(coco_prediction_json)
            if not novisual:
                # convert ground truth annotations to object_prediction_list
                coco_image: CocoImage = coco.images[ind]
                object_prediction_gt_list: List[ObjectPrediction] = []
                for coco_annotation in coco_image.annotations:
                    coco_annotation_dict = coco_annotation.json
                    category_name = coco_annotation.category_name
                    full_shape = [coco_image.height, coco_image.width]
                    object_prediction_gt = (
                        ObjectPrediction.from_coco_annotation_dict(
                            annotation_dict=coco_annotation_dict,
                            category_name=category_name,
                            full_shape=full_shape,
                        )
                    )
                    object_prediction_gt_list.append(object_prediction_gt)
                # export visualizations with ground truths
                output_dir = str(
                    visual_with_gt_dir / Path(relative_filepath).parent
                )
                color = (0, 255, 0)  # original annotations in green
                result = visualize_object_predictions(
                    np.ascontiguousarray(image_as_pil),
                    object_prediction_list=object_prediction_gt_list,
                    rect_th=visual_bbox_thickness,
                    text_size=visual_text_size,
                    text_th=visual_text_thickness,
                    color=color,
                    hide_labels=visual_hide_labels,
                    hide_conf=visual_hide_conf,
                    output_dir=None,
                    file_name=None,
                    export_format=None,
                )
                color = (255, 0, 0)  # model predictions in red
                _ = visualize_object_predictions(
                    result["image"],
                    object_prediction_list=object_prediction_list,
                    rect_th=visual_bbox_thickness,
                    text_size=visual_text_size,
                    text_th=visual_text_thickness,
                    color=color,
                    hide_labels=visual_hide_labels,
                    hide_conf=visual_hide_conf,
                    output_dir=output_dir,
                    file_name=filename_without_extension,
                    export_format=visual_export_format,
                )

        time_start = time.time()
        # export prediction boxes
        if export_crop:
            output_dir = str(crop_dir / Path(relative_filepath).parent)
            crop_object_predictions(
                image=np.ascontiguousarray(image_as_pil),
                object_prediction_list=object_prediction_list,
                output_dir=output_dir,
                file_name=filename_without_extension,
                export_format=visual_export_format,
            )
        # export prediction list as pickle
        if export_pickle:
            save_path = str(
                pickle_dir
                / Path(relative_filepath).parent
                / (filename_without_extension + ".pickle")
            )
            save_pickle(data=object_prediction_list, save_path=save_path)

        # export visualization
        if not novisual or view_video:
            output_dir = str(visual_dir / Path(relative_filepath).parent)
            result = visualize_object_predictions(
                np.ascontiguousarray(image_as_pil),
                object_prediction_list=object_prediction_list,
                rect_th=visual_bbox_thickness,
                text_size=visual_text_size,
                text_th=visual_text_thickness,
                hide_labels=visual_hide_labels,
                hide_conf=visual_hide_conf,
                output_dir=output_dir if not source_is_video else None,
                file_name=filename_without_extension,
                export_format=visual_export_format,
            )
            if not novisual and source_is_video:  # export video
                output_video_writer.write(result["image"])

        # render video inference
        if view_video:
            cv2.imshow(
                "Prediction of {}".format(str(video_file_name)),
                result["image"],
            )
            cv2.waitKey(1)

        time_end = time.time() - time_start
        durations_in_seconds["export_files"] = time_end

    # export coco results
    if dataset_json_path:
        save_path = str(save_dir / "result.json")
        save_json(coco_json, save_path)

    if (
        not novisual
        or export_pickle
        or export_crop
        or dataset_json_path is not None
    ):
        print(f"Prediction results are successfully exported to {save_dir}")

    # print prediction duration
    if verbose == 2:
        print(
            "Model loaded in",
            durations_in_seconds["model_load"],
            "seconds.",
        )
        print(
            "Slicing performed in",
            durations_in_seconds["slice"],
            "seconds.",
        )
        print(
            "Prediction performed in",
            durations_in_seconds["prediction"],
            "seconds.",
        )
        if not novisual:
            print(
                "Exporting performed in",
                durations_in_seconds["export_files"],
                "seconds.",
            )

    if return_dict:
        return {"export_dir": save_dir}


def predict_fiftyone(
    model_type: str = "mmdet",
    model_path: str = None,
    model_config_path: str = None,
    model_confidence_threshold: float = 0.25,
    model_device: str = None,
    model_category_mapping: dict = None,
    model_category_remapping: dict = None,
    dataset_json_path: str = None,
    image_dir: str = None,
    no_standard_prediction: bool = False,
    no_sliced_prediction: bool = False,
    image_size: int = None,
    slice_height: int = 256,
    slice_width: int = 256,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2,
    postprocess_type: str = "GREEDYNMM",
    postprocess_match_metric: str = "IOS",
    postprocess_match_threshold: float = 0.5,
    postprocess_class_agnostic: bool = False,
    verbose: int = 1,
):
    """
    Performs prediction for all present images in given folder.

    Args:
        model_type: str
            mmdet for 'MmdetDetectionModel', 'yolov5' for 'Yolov5DetectionModel'.
        model_path: str
            Path for the model weight
        model_config_path: str
            Path for the detection model config file
        model_confidence_threshold: float
            All predictions with score < model_confidence_threshold will be discarded.
        model_device: str
            device, "cpu" or "cuda"
        model_category_mapping: dict
            Mapping from category id (str) to category name (str) e.g. {"1": "pedestrian"}
        model_category_remapping: dict: str to int
            Remap category ids after performing inference
        dataset_json_path: str
            If coco file path is provided, detection results will be exported in coco json format.
        image_dir: str
            Folder directory that contains images or path of the image to be predicted.
        no_standard_prediction: bool
            Dont perform standard prediction. Default: False.
        no_sliced_prediction: bool
            Dont perform sliced prediction. Default: False.
        image_size: int
            Input image size for each inference (image is scaled by preserving asp. rat.).
        slice_height: int
            Height of each slice.  Defaults to ``256``.
        slice_width: int
            Width of each slice.  Defaults to ``256``.
        overlap_height_ratio: float
            Fractional overlap in height of each window (e.g. an overlap of 0.2 for a window
            of size 256 yields an overlap of 51 pixels).
            Default to ``0.2``.
        overlap_width_ratio: float
            Fractional overlap in width of each window (e.g. an overlap of 0.2 for a window
            of size 256 yields an overlap of 51 pixels).
            Default to ``0.2``.
        postprocess_type: str
            Type of the postprocess to be used after sliced inference while merging/eliminating predictions.
            Options are 'NMM', 'GRREDYNMM' or 'NMS'. Default is 'GRREDYNMM'.
        postprocess_match_metric: str
            Metric to be used during object prediction matching after sliced prediction.
            'IOU' for intersection over union, 'IOS' for intersection over smaller area.
        postprocess_match_metric: str
            Metric to be used during object prediction matching after sliced prediction.
            'IOU' for intersection over union, 'IOS' for intersection over smaller area.
        postprocess_match_threshold: float
            Sliced predictions having higher iou than postprocess_match_threshold will be
            postprocessed after sliced prediction.
        postprocess_class_agnostic: bool
            If True, postprocess will ignore category ids.
        verbose: int
            0: no print
            1: print slice/prediction durations, number of slices, model loading/file exporting durations
    """
    check_requirements(["fiftyone"])

    from sahi.utils.fiftyone import create_fiftyone_dataset_from_coco_file, fo

    # assert prediction type
    if no_standard_prediction and no_sliced_prediction:
        raise ValueError(
            "'no_standard_pred' and 'no_sliced_prediction' cannot be True at the same time."
        )
    # for profiling
    durations_in_seconds = dict()

    dataset = create_fiftyone_dataset_from_coco_file(
        image_dir, dataset_json_path
    )

    # init model instance
    time_start = time.time()
    detection_model = AutoDetectionModel.from_pretrained(
        model_type=model_type,
        model_path=model_path,
        config_path=model_config_path,
        confidence_threshold=model_confidence_threshold,
        device=model_device,
        category_mapping=model_category_mapping,
        category_remapping=model_category_remapping,
        load_at_init=False,
        image_size=image_size,
    )
    detection_model.load_model()
    time_end = time.time() - time_start
    durations_in_seconds["model_load"] = time_end

    # iterate over source images
    durations_in_seconds["prediction"] = 0
    durations_in_seconds["slice"] = 0
    # Add predictions to samples
    with fo.ProgressBar() as pb:
        for sample in pb(dataset):
            # perform prediction
            if not no_sliced_prediction:
                # get sliced prediction
                prediction_result = get_sliced_prediction(
                    image=sample.filepath,
                    detection_model=detection_model,
                    slice_height=slice_height,
                    slice_width=slice_width,
                    overlap_height_ratio=overlap_height_ratio,
                    overlap_width_ratio=overlap_width_ratio,
                    perform_standard_pred=not no_standard_prediction,
                    postprocess_type=postprocess_type,
                    postprocess_match_threshold=postprocess_match_threshold,
                    postprocess_match_metric=postprocess_match_metric,
                    postprocess_class_agnostic=postprocess_class_agnostic,
                    verbose=verbose,
                )
                durations_in_seconds[
                    "slice"
                ] += prediction_result.durations_in_seconds["slice"]
            else:
                # get standard prediction
                prediction_result = get_prediction(
                    image=sample.filepath,
                    detection_model=detection_model,
                    shift_amount=[0, 0],
                    full_shape=None,
                    postprocess=None,
                    verbose=0,
                )
                durations_in_seconds[
                    "prediction"
                ] += prediction_result.durations_in_seconds["prediction"]

            # Save predictions to dataset
            sample[model_type] = fo.Detections(
                detections=prediction_result.to_fiftyone_detections()
            )
            sample.save()

    # print prediction duration
    if verbose == 1:
        print(
            "Model loaded in",
            durations_in_seconds["model_load"],
            "seconds.",
        )
        print(
            "Slicing performed in",
            durations_in_seconds["slice"],
            "seconds.",
        )
        print(
            "Prediction performed in",
            durations_in_seconds["prediction"],
            "seconds.",
        )

    # visualize results
    session = fo.launch_app()
    session.dataset = dataset
    # Evaluate the predictions
    results = dataset.evaluate_detections(
        model_type,
        gt_field="ground_truth",
        eval_key="eval",
        iou=postprocess_match_threshold,
        compute_mAP=True,
    )
    # Get the 10 most common classes in the dataset
    counts = dataset.count_values("ground_truth.detections.label")
    classes_top10 = sorted(counts, key=counts.get, reverse=True)[:10]
    # Print a classification report for the top-10 classes
    results.print_report(classes=classes_top10)
    # Load the view on which we ran the `eval` evaluation
    eval_view = dataset.load_evaluation_view("eval")
    # Show samples with most false positives
    session.view = eval_view.sort_by("eval_fp", reverse=True)
    while 1:
        time.sleep(3)
