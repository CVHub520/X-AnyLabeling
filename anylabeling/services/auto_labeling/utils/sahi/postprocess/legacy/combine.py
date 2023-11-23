# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2021.

import copy
from typing import List

import numpy as np

from anylabeling.services.auto_labeling.utils.sahi.annotation import (
    BoundingBox,
    Category,
    Mask,
)
from anylabeling.services.auto_labeling.utils.sahi.postprocess.utils import (
    calculate_area,
    calculate_box_union,
    calculate_intersection_area,
)
from anylabeling.services.auto_labeling.utils.sahi.prediction import (
    ObjectPrediction,
)


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
        if match_metric == "IOU":
            self.calculate_match = self.calculate_bbox_iou
        elif match_metric == "IOS":
            self.calculate_match = self.calculate_bbox_ios
        else:
            raise ValueError(
                f"'match_metric' should be one of ['IOU', 'IOS'] but given as {match_metric}"
            )

    def _has_match(
        self, pred1: ObjectPrediction, pred2: ObjectPrediction
    ) -> bool:
        threshold_condition = (
            self.calculate_match(pred1, pred2) > self.match_threshold
        )
        category_condition = (
            self.has_same_category_id(pred1, pred2) or self.class_agnostic
        )
        return threshold_condition and category_condition

    @staticmethod
    def get_score_func(object_prediction: ObjectPrediction):
        """Used for sorting predictions"""
        return object_prediction.score.value

    @staticmethod
    def has_same_category_id(
        pred1: ObjectPrediction, pred2: ObjectPrediction
    ) -> bool:
        return pred1.category.id == pred2.category.id

    @staticmethod
    def calculate_bbox_iou(
        pred1: ObjectPrediction, pred2: ObjectPrediction
    ) -> float:
        """Returns the ratio of intersection area to the union"""
        box1 = np.array(pred1.bbox.to_xyxy())
        box2 = np.array(pred2.bbox.to_xyxy())
        area1 = calculate_area(box1)
        area2 = calculate_area(box2)
        intersect = calculate_intersection_area(box1, box2)
        return intersect / (area1 + area2 - intersect)

    @staticmethod
    def calculate_bbox_ios(
        pred1: ObjectPrediction, pred2: ObjectPrediction
    ) -> float:
        """Returns the ratio of intersection area to the smaller box's area"""
        box1 = np.array(pred1.bbox.to_xyxy())
        box2 = np.array(pred2.bbox.to_xyxy())
        area1 = calculate_area(box1)
        area2 = calculate_area(box2)
        intersect = calculate_intersection_area(box1, box2)
        smaller_area = np.minimum(area1, area2)
        return intersect / smaller_area

    def __call__(self):
        raise NotImplementedError()


class NMSPostprocess(PostprocessPredictions):
    def __call__(
        self,
        object_predictions: List[ObjectPrediction],
    ):
        source_object_predictions: List[ObjectPrediction] = copy.deepcopy(
            object_predictions
        )
        selected_object_predictions: List[ObjectPrediction] = []
        while len(source_object_predictions) > 0:
            # select object prediction with highest score
            source_object_predictions.sort(
                reverse=True, key=self.get_score_func
            )
            selected_object_prediction = source_object_predictions[0]
            # remove selected prediction from source list
            del source_object_predictions[0]
            # if any element from remaining source prediction list matches, remove it
            new_source_object_predictions: List[ObjectPrediction] = []
            for candidate_object_prediction in source_object_predictions:
                if self._has_match(
                    selected_object_prediction, candidate_object_prediction
                ):
                    pass
                else:
                    new_source_object_predictions.append(
                        candidate_object_prediction
                    )
            source_object_predictions = new_source_object_predictions
            # append selected prediction to selected list
            selected_object_predictions.append(selected_object_prediction)
        return selected_object_predictions


class UnionMergePostprocess(PostprocessPredictions):
    def __call__(
        self,
        object_predictions: List[ObjectPrediction],
    ):
        source_object_predictions: List[ObjectPrediction] = copy.deepcopy(
            object_predictions
        )
        selected_object_predictions: List[ObjectPrediction] = []
        while len(source_object_predictions) > 0:
            # select object prediction with highest score
            source_object_predictions.sort(
                reverse=True, key=self.get_score_func
            )
            selected_object_prediction = source_object_predictions[0]
            # remove selected prediction from source list
            del source_object_predictions[0]
            # if any element from remaining source prediction list matches, remove it and merge with selected prediction
            new_source_object_predictions: List[ObjectPrediction] = []
            for ind, candidate_object_prediction in enumerate(
                source_object_predictions
            ):
                if self._has_match(
                    selected_object_prediction, candidate_object_prediction
                ):
                    selected_object_prediction = (
                        self._merge_object_prediction_pair(
                            selected_object_prediction,
                            candidate_object_prediction,
                        )
                    )
                else:
                    new_source_object_predictions.append(
                        candidate_object_prediction
                    )
            source_object_predictions = new_source_object_predictions
            # append selected prediction to selected list
            selected_object_predictions.append(selected_object_prediction)
        return selected_object_predictions

    def _merge_object_prediction_pair(
        self,
        pred1: ObjectPrediction,
        pred2: ObjectPrediction,
    ) -> ObjectPrediction:
        shift_amount = pred1.bbox.shift_amount
        merged_bbox: BoundingBox = self._get_merged_bbox(pred1, pred2)
        merged_score: float = self._get_merged_score(pred1, pred2)
        merged_category: Category = self._get_merged_category(pred1, pred2)
        if pred1.mask and pred2.mask:
            merged_mask: Mask = self._get_merged_mask(pred1, pred2)
            bool_mask = merged_mask.bool_mask
            full_shape = merged_mask.full_shape
        else:
            bool_mask = None
            full_shape = None
        return ObjectPrediction(
            bbox=merged_bbox.to_xyxy(),
            score=merged_score,
            category_id=merged_category.id,
            category_name=merged_category.name,
            bool_mask=bool_mask,
            shift_amount=shift_amount,
            full_shape=full_shape,
        )

    @staticmethod
    def _get_merged_category(
        pred1: ObjectPrediction, pred2: ObjectPrediction
    ) -> Category:
        if pred1.score.value > pred2.score.value:
            return pred1.category
        else:
            return pred2.category

    @staticmethod
    def _get_merged_bbox(
        pred1: ObjectPrediction, pred2: ObjectPrediction
    ) -> BoundingBox:
        box1: List[int] = pred1.bbox.to_xyxy()
        box2: List[int] = pred2.bbox.to_xyxy()
        bbox = BoundingBox(box=calculate_box_union(box1, box2))
        return bbox

    @staticmethod
    def _get_merged_score(
        pred1: ObjectPrediction,
        pred2: ObjectPrediction,
    ) -> float:
        scores: List[float] = [pred.score.value for pred in (pred1, pred2)]
        return max(scores)

    @staticmethod
    def _get_merged_mask(
        pred1: ObjectPrediction, pred2: ObjectPrediction
    ) -> Mask:
        mask1 = pred1.mask
        mask2 = pred2.mask
        union_mask = np.logical_or(mask1.bool_mask, mask2.bool_mask)
        return Mask(
            bool_mask=union_mask,
            full_shape=mask1.full_shape,
            shift_amount=mask1.shift_amount,
        )
