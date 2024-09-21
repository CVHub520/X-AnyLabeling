# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2021.

import logging
from typing import List

import numpy as np
from collections import defaultdict

from anylabeling.services.auto_labeling.utils.sahi.postprocess.utils import (
    ObjectPredictionList,
    has_match,
    merge_object_prediction_pair,
)
from anylabeling.services.auto_labeling.utils.sahi.prediction import (
    ObjectPrediction,
)
from anylabeling.services.auto_labeling.utils.sahi.utils.import_utils import (
    check_requirements,
)

logger = logging.getLogger(__name__)


def batched_nms(predictions, match_metric="IOU", match_threshold=0.5):
    scores = predictions[:, 4].squeeze()
    category_ids = predictions[:, 5].squeeze()
    keep_mask = np.zeros_like(category_ids, dtype=bool)
    unique_categories = np.unique(category_ids)

    for category_id in unique_categories:
        curr_indices = np.where(category_ids == category_id)[0]
        curr_keep_indices = nms(
            predictions[curr_indices], match_metric, match_threshold
        )
        keep_mask[curr_indices[curr_keep_indices]] = True

    keep_indices = np.where(keep_mask)[0]
    sorted_indices = np.argsort(scores[keep_indices])[::-1]
    keep_indices = keep_indices[sorted_indices].tolist()

    return keep_indices


def nms(predictions, match_metric="IOU", match_threshold=0.5):
    x1 = predictions[:, 0]
    y1 = predictions[:, 1]
    x2 = predictions[:, 2]
    y2 = predictions[:, 3]
    scores = predictions[:, 4]

    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(scores)

    keep = []

    while len(order) > 0:
        idx = order[-1]
        keep.append(idx)

        order = order[:-1]

        if len(order) == 0:
            break

        xx1 = x1[order]
        xx2 = x2[order]
        yy1 = y1[order]
        yy2 = y2[order]

        xx1 = np.maximum(xx1, x1[idx])
        yy1 = np.maximum(yy1, y1[idx])
        xx2 = np.minimum(xx2, x2[idx])
        yy2 = np.minimum(yy2, y2[idx])

        w = xx2 - xx1
        h = yy2 - yy1

        w = np.maximum(w, 0.0)
        h = np.maximum(h, 0.0)

        inter = w * h

        rem_areas = areas[order]

        if match_metric == "IOU":
            union = (rem_areas - inter) + areas[idx]
            match_metric_value = inter / union
        elif match_metric == "IOS":
            smaller = np.minimum(rem_areas, areas[idx])
            match_metric_value = inter / smaller
        else:
            raise ValueError()

        mask = match_metric_value < match_threshold
        order = order[mask]

    return keep


def batched_greedy_nmm(
    object_predictions_as_tensor, match_metric="IOU", match_threshold=0.5
):
    category_ids = object_predictions_as_tensor[:, 5].squeeze()
    keep_to_merge_list = defaultdict(list)
    unique_categories = np.unique(category_ids)

    for category_id in unique_categories:
        curr_indices = np.where(category_ids == category_id)[0]
        curr_keep_to_merge_list = greedy_nmm(
            object_predictions_as_tensor[curr_indices],
            match_metric,
            match_threshold,
        )
        curr_indices_list = curr_indices.tolist()

        for curr_keep, curr_merge_list in curr_keep_to_merge_list.items():
            keep = curr_indices_list[curr_keep]
            merge_list = [
                curr_indices_list[curr_merge_ind]
                for curr_merge_ind in curr_merge_list
            ]
            keep_to_merge_list[keep] = merge_list

    return dict(keep_to_merge_list)


def greedy_nmm(
    object_predictions_as_tensor, match_metric="IOU", match_threshold=0.5
):
    keep_to_merge_list = {}

    # Extract coordinates for every prediction box present in P
    x1 = object_predictions_as_tensor[:, 0]
    y1 = object_predictions_as_tensor[:, 1]
    x2 = object_predictions_as_tensor[:, 2]
    y2 = object_predictions_as_tensor[:, 3]
    scores = object_predictions_as_tensor[:, 4]

    # Calculate area of every block in P
    areas = (x2 - x1) * (y2 - y1)

    # Sort the prediction boxes in P according to their confidence scores
    order = np.argsort(scores)

    # Initialise an empty list for filtered prediction boxes
    keep = []

    while len(order) > 0:
        # Extract the index of the prediction with the highest score (prediction S)
        idx = order[-1]

        # Push S in the filtered predictions list
        keep.append(idx)

        # Remove S from P
        order = order[:-1]

        # Sanity check
        if len(order) == 0:
            keep_to_merge_list[idx] = []
            break

        # Select coordinates of BBoxes according to the indices in order
        xx1 = x1[order]
        xx2 = x2[order]
        yy1 = y1[order]
        yy2 = y2[order]

        # Find the coordinates of the intersection boxes
        xx1 = np.maximum(xx1, x1[idx])
        yy1 = np.maximum(yy1, y1[idx])
        xx2 = np.minimum(xx2, x2[idx])
        yy2 = np.minimum(yy2, y2[idx])

        # Find height and width of the intersection boxes
        w = xx2 - xx1
        h = yy2 - yy1

        # Take max with 0.0 to avoid negative w and h due to non-overlapping boxes
        w = np.maximum(w, 0.0)
        h = np.maximum(h, 0.0)

        # Find the intersection area
        inter = w * h

        # Find the areas of BBoxes according to the indices in order
        rem_areas = areas[order]

        if match_metric == "IOU":
            # Find the union of every prediction T in P with the prediction S
            # Note that areas[idx] represents the area of S
            union = (rem_areas - inter) + areas[idx]
            # Find the IoU of every prediction in P with S
            match_metric_value = inter / union

        elif match_metric == "IOS":
            # Find the smaller area of every prediction T in P with the prediction S
            # Note that areas[idx] represents the area of S
            smaller = np.minimum(rem_areas, areas[idx])
            # Find the IoS of every prediction in P with S
            match_metric_value = inter / smaller
        else:
            raise ValueError()

        # Keep the boxes with IoU/IoS less than thresh_iou
        mask = match_metric_value < match_threshold
        matched_box_indices = order[np.where(mask == False)[0]]  # noqa: E712
        unmatched_indices = order[np.where(mask == True)[0]]  # noqa: E712

        # Update box pool
        order = unmatched_indices[np.argsort(scores[unmatched_indices])]

        # Create keep_ind to merge_ind_list mapping
        keep_to_merge_list[idx] = matched_box_indices.tolist()

    return keep_to_merge_list


def batched_nmm(
    object_predictions_as_tensor, match_metric="IOU", match_threshold=0.5
):
    category_ids = object_predictions_as_tensor[:, 5].squeeze()
    keep_to_merge_list = {}
    unique_categories = np.unique(category_ids)

    for category_id in unique_categories:
        curr_indices = np.where(category_ids == category_id)[0]
        curr_keep_to_merge_list = nmm(
            object_predictions_as_tensor[curr_indices],
            match_metric,
            match_threshold,
        )
        curr_indices_list = curr_indices.tolist()

        for curr_keep, curr_merge_list in curr_keep_to_merge_list.items():
            keep = curr_indices_list[curr_keep]
            merge_list = [
                curr_indices_list[curr_merge_ind]
                for curr_merge_ind in curr_merge_list
            ]
            keep_to_merge_list[keep] = merge_list

    return keep_to_merge_list


def nmm(object_predictions_as_tensor, match_metric="IOU", match_threshold=0.5):
    keep_to_merge_list = {}
    merge_to_keep = {}

    # Extract coordinates for every prediction box present in P
    x1 = object_predictions_as_tensor[:, 0]
    y1 = object_predictions_as_tensor[:, 1]
    x2 = object_predictions_as_tensor[:, 2]
    y2 = object_predictions_as_tensor[:, 3]
    scores = object_predictions_as_tensor[:, 4]

    # Calculate area of every block in P
    areas = (x2 - x1) * (y2 - y1)

    # Sort the prediction boxes in P according to their confidence scores
    order = np.argsort(scores)[::-1]

    for ind in range(len(object_predictions_as_tensor)):
        # Extract the index of the prediction with the highest score (prediction S)
        pred_ind = order[ind]

        # Remove selected pred
        other_pred_inds = order[order != pred_ind]

        # Select coordinates of BBoxes according to the indices in order
        xx1 = x1[other_pred_inds]
        xx2 = x2[other_pred_inds]
        yy1 = y1[other_pred_inds]
        yy2 = y2[other_pred_inds]

        # Find the coordinates of the intersection boxes
        xx1 = np.maximum(xx1, x1[pred_ind])
        yy1 = np.maximum(yy1, y1[pred_ind])
        xx2 = np.minimum(xx2, x2[pred_ind])
        yy2 = np.minimum(yy2, y2[pred_ind])

        # Find height and width of the intersection boxes
        w = xx2 - xx1
        h = yy2 - yy1

        # Take max with 0.0 to avoid negative w and h due to non-overlapping boxes
        w = np.maximum(w, 0.0)
        h = np.maximum(h, 0.0)

        # Find the intersection area
        inter = w * h

        # Find the areas of BBoxes according to the indices in order
        rem_areas = areas[other_pred_inds]

        if match_metric == "IOU":
            # Find the union of every prediction T in P with the prediction S
            # Note that areas[idx] represents the area of S
            union = (rem_areas - inter) + areas[pred_ind]
            # Find the IoU of every prediction in P with S
            match_metric_value = inter / union

        elif match_metric == "IOS":
            # Find the smaller area of every prediction T in P with the prediction S
            # Note that areas[idx] represents the area of S
            smaller = np.minimum(rem_areas, areas[pred_ind])
            # Find the IoS of every prediction in P with S
            match_metric_value = inter / smaller
        else:
            raise ValueError()

        # Keep the boxes with IoU/IoS less than thresh_iou
        mask = match_metric_value < match_threshold
        matched_box_indices = other_pred_inds[~mask]

        # Create keep_ind to merge_ind_list mapping
        if pred_ind not in merge_to_keep:
            keep_to_merge_list[pred_ind] = []

            for matched_box_ind in matched_box_indices.tolist():
                if matched_box_ind not in merge_to_keep:
                    keep_to_merge_list[pred_ind].append(matched_box_ind)
                    merge_to_keep[matched_box_ind] = pred_ind

        else:
            keep = merge_to_keep[pred_ind]
            for matched_box_ind in matched_box_indices.tolist():
                if (
                    matched_box_ind not in keep_to_merge_list
                    and matched_box_ind not in merge_to_keep
                ):
                    keep_to_merge_list[keep].append(matched_box_ind)
                    merge_to_keep[matched_box_ind] = keep

    return keep_to_merge_list


class PostprocessPredictions:
    """Utilities for calculating IOU/IOS based match for given ObjectPredictions"""

    def __init__(
        self,
        match_threshold: float = 0.5,
        match_metric: str = "IOU",
        class_agnostic: bool = True,
    ):
        self.match_threshold = match_threshold
        self.class_agnostic = class_agnostic
        self.match_metric = match_metric

    def __call__(self):
        raise NotImplementedError()


class NMSPostprocess(PostprocessPredictions):
    def __call__(
        self,
        object_predictions: List[ObjectPrediction],
    ):
        object_prediction_list = ObjectPredictionList(object_predictions)
        object_predictions_as_torch = object_prediction_list.totensor()
        if self.class_agnostic:
            keep = nms(
                object_predictions_as_torch,
                match_threshold=self.match_threshold,
                match_metric=self.match_metric,
            )
        else:
            keep = batched_nms(
                object_predictions_as_torch,
                match_threshold=self.match_threshold,
                match_metric=self.match_metric,
            )

        selected_object_predictions = object_prediction_list[keep].tolist()
        if not isinstance(selected_object_predictions, list):
            selected_object_predictions = [selected_object_predictions]

        return selected_object_predictions


class NMMPostprocess(PostprocessPredictions):
    def __call__(
        self,
        object_predictions: List[ObjectPrediction],
    ):
        object_prediction_list = ObjectPredictionList(object_predictions)
        object_predictions_as_torch = object_prediction_list.totensor()
        if self.class_agnostic:
            keep_to_merge_list = nmm(
                object_predictions_as_torch,
                match_threshold=self.match_threshold,
                match_metric=self.match_metric,
            )
        else:
            keep_to_merge_list = batched_nmm(
                object_predictions_as_torch,
                match_threshold=self.match_threshold,
                match_metric=self.match_metric,
            )

        selected_object_predictions = []
        for keep_ind, merge_ind_list in keep_to_merge_list.items():
            for merge_ind in merge_ind_list:
                if has_match(
                    object_prediction_list[keep_ind].tolist(),
                    object_prediction_list[merge_ind].tolist(),
                    self.match_metric,
                    self.match_threshold,
                ):
                    object_prediction_list[keep_ind] = (
                        merge_object_prediction_pair(
                            object_prediction_list[keep_ind].tolist(),
                            object_prediction_list[merge_ind].tolist(),
                        )
                    )
            selected_object_predictions.append(
                object_prediction_list[keep_ind].tolist()
            )

        return selected_object_predictions


class GreedyNMMPostprocess(PostprocessPredictions):
    def __call__(
        self,
        object_predictions: List[ObjectPrediction],
    ):
        object_prediction_list = ObjectPredictionList(object_predictions)
        object_predictions_as_numpy = object_prediction_list.tonumpy()
        if self.class_agnostic:
            keep_to_merge_list = greedy_nmm(
                object_predictions_as_numpy,
                match_threshold=self.match_threshold,
                match_metric=self.match_metric,
            )
        else:
            keep_to_merge_list = batched_greedy_nmm(
                object_predictions_as_numpy,
                match_threshold=self.match_threshold,
                match_metric=self.match_metric,
            )

        selected_object_predictions = []
        for keep_ind, merge_ind_list in keep_to_merge_list.items():
            for merge_ind in merge_ind_list:
                if has_match(
                    object_prediction_list[keep_ind].tolist(),
                    object_prediction_list[merge_ind].tolist(),
                    self.match_metric,
                    self.match_threshold,
                ):
                    object_prediction_list[keep_ind] = (
                        merge_object_prediction_pair(
                            object_prediction_list[keep_ind].tolist(),
                            object_prediction_list[merge_ind].tolist(),
                        )
                    )
            selected_object_predictions.append(
                object_prediction_list[keep_ind].tolist()
            )

        return selected_object_predictions


class LSNMSPostprocess(PostprocessPredictions):
    # https://github.com/remydubois/lsnms/blob/10b8165893db5bfea4a7cb23e268a502b35883cf/lsnms/nms.py#L62
    def __call__(
        self,
        object_predictions: List[ObjectPrediction],
    ):
        try:
            from lsnms import nms
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                'Please run "pip install lsnms>0.3.1" to install lsnms first for lsnms utilities.'
            )

        if self.match_metric == "IOS":
            NotImplementedError(
                f"match_metric={self.match_metric} is not supported for LSNMSPostprocess"
            )

        logger.warning(
            "LSNMSPostprocess is experimental and not recommended to use."
        )

        object_prediction_list = ObjectPredictionList(object_predictions)
        object_predictions_as_numpy = object_prediction_list.tonumpy()

        boxes = object_predictions_as_numpy[:, :4]
        scores = object_predictions_as_numpy[:, 4]
        class_ids = object_predictions_as_numpy[:, 5].astype("uint8")

        keep = nms(
            boxes,
            scores,
            iou_threshold=self.match_threshold,
            class_ids=None if self.class_agnostic else class_ids,
        )

        selected_object_predictions = object_prediction_list[keep].tolist()
        if not isinstance(selected_object_predictions, list):
            selected_object_predictions = [selected_object_predictions]

        return selected_object_predictions
