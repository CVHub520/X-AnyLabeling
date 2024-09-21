# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.
# Modified by Sinan O Altinuc, 2020.

import copy
import logging
import os
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from threading import Lock, Thread
from typing import Dict, List, Optional, Set, Union

import numpy as np
from tqdm import tqdm

from anylabeling.services.auto_labeling.utils.sahi.utils.file import (
    is_colab,
    load_json,
    save_json,
)
from anylabeling.services.auto_labeling.utils.sahi.utils.shapely import (
    ShapelyAnnotation,
    box,
    get_shapely_multipolygon,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
)


class CocoCategory:
    """
    COCO formatted category.
    """

    def __init__(self, id=None, name=None, supercategory=None):
        self.id = int(id)
        self.name = name
        self.supercategory = supercategory if supercategory else name

    @classmethod
    def from_coco_category(cls, category):
        """
        Creates CocoCategory object using coco category.

        Args:
            category: Dict
                {"supercategory": "person", "id": 1, "name": "person"},
        """
        return cls(
            id=category["id"],
            name=category["name"],
            supercategory=(
                category["supercategory"]
                if "supercategory" in category
                else category["name"]
            ),
        )

    @property
    def json(self):
        return {
            "id": self.id,
            "name": self.name,
            "supercategory": self.supercategory,
        }

    def __repr__(self):
        return f"""CocoCategory<
    id: {self.id},
    name: {self.name},
    supercategory: {self.supercategory}>"""


class CocoAnnotation:
    """
    COCO formatted annotation.
    """

    @classmethod
    def from_coco_segmentation(
        cls, segmentation, category_id, category_name, iscrowd=0
    ):
        """
        Creates CocoAnnotation object using coco segmentation.

        Args:
            segmentation: List[List]
                [[1, 1, 325, 125, 250, 200, 5, 200]]
            category_id: int
                Category id of the annotation
            category_name: str
                Category name of the annotation
            iscrowd: int
                0 or 1
        """
        return cls(
            segmentation=segmentation,
            category_id=category_id,
            category_name=category_name,
            iscrowd=iscrowd,
        )

    @classmethod
    def from_coco_bbox(cls, bbox, category_id, category_name, iscrowd=0):
        """
        Creates CocoAnnotation object using coco bbox

        Args:
            bbox: List
                [xmin, ymin, width, height]
            category_id: int
                Category id of the annotation
            category_name: str
                Category name of the annotation
            iscrowd: int
                0 or 1
        """
        return cls(
            bbox=bbox,
            category_id=category_id,
            category_name=category_name,
            iscrowd=iscrowd,
        )

    @classmethod
    def from_coco_annotation_dict(
        cls, annotation_dict: Dict, category_name: Optional[str] = None
    ):
        """
        Creates CocoAnnotation object from category name and COCO formatted
        annotation dict (with fields "bbox", "segmentation", "category_id").

        Args:
            category_name: str
                Category name of the annotation
            annotation_dict: dict
                COCO formatted annotation dict (with fields "bbox", "segmentation", "category_id")
        """
        if annotation_dict.__contains__("segmentation") and not isinstance(
            annotation_dict["segmentation"], list
        ):
            has_rle_segmentation = True
            logger.warning(
                f"Segmentation annotation for id {annotation_dict['id']} is skipped since RLE segmentation format is not supported."
            )
        else:
            has_rle_segmentation = False

        if (
            annotation_dict.__contains__("segmentation")
            and annotation_dict["segmentation"]
            and not has_rle_segmentation
        ):
            return cls(
                segmentation=annotation_dict["segmentation"],
                category_id=annotation_dict["category_id"],
                category_name=category_name,
            )
        else:
            return cls(
                bbox=annotation_dict["bbox"],
                category_id=annotation_dict["category_id"],
                category_name=category_name,
            )

    @classmethod
    def from_shapely_annotation(
        cls,
        shapely_annotation: ShapelyAnnotation,
        category_id: int,
        category_name: str,
        iscrowd: int,
    ):
        """
        Creates CocoAnnotation object from ShapelyAnnotation object.

        Args:
            shapely_annotation (ShapelyAnnotation)
            category_id (int): Category id of the annotation
            category_name (str): Category name of the annotation
            iscrowd (int): 0 or 1
        """
        coco_annotation = cls(
            bbox=[0, 0, 0, 0],
            category_id=category_id,
            category_name=category_name,
            iscrowd=iscrowd,
        )
        coco_annotation._segmentation = (
            shapely_annotation.to_coco_segmentation()
        )
        coco_annotation._shapely_annotation = shapely_annotation
        return coco_annotation

    def __init__(
        self,
        segmentation=None,
        bbox=None,
        category_id=None,
        category_name=None,
        image_id=None,
        iscrowd=0,
    ):
        """
        Creates coco annotation object using bbox or segmentation

        Args:
            segmentation: List[List]
                [[1, 1, 325, 125, 250, 200, 5, 200]]
            bbox: List
                [xmin, ymin, width, height]
            category_id: int
                Category id of the annotation
            category_name: str
                Category name of the annotation
            image_id: int
                Image ID of the annotation
            iscrowd: int
                0 or 1
        """
        if bbox is None and segmentation is None:
            raise ValueError("you must provide a bbox or polygon")

        self._segmentation = segmentation
        self._category_id = category_id
        self._category_name = category_name
        self._image_id = image_id
        self._iscrowd = iscrowd

        if self._segmentation:
            shapely_annotation = ShapelyAnnotation.from_coco_segmentation(
                segmentation=self._segmentation
            )
        else:
            shapely_annotation = ShapelyAnnotation.from_coco_bbox(bbox=bbox)
        self._shapely_annotation = shapely_annotation

    def get_sliced_coco_annotation(self, slice_bbox: List[int]):
        shapely_polygon = box(
            slice_bbox[0], slice_bbox[1], slice_bbox[2], slice_bbox[3]
        )
        intersection_shapely_annotation = (
            self._shapely_annotation.get_intersection(shapely_polygon)
        )
        return CocoAnnotation.from_shapely_annotation(
            intersection_shapely_annotation,
            category_id=self.category_id,
            category_name=self.category_name,
            iscrowd=self.iscrowd,
        )

    @property
    def area(self):
        """
        Returns area of annotation polygon (or bbox if no polygon available)
        """
        return self._shapely_annotation.area

    @property
    def bbox(self):
        """
        Returns coco formatted bbox of the annotation as [xmin, ymin, width, height]
        """
        return self._shapely_annotation.to_xywh()

    @property
    def segmentation(self):
        """
        Returns coco formatted segmentation of the annotation as [[1, 1, 325, 125, 250, 200, 5, 200]]
        """
        if self._segmentation:
            return self._shapely_annotation.to_coco_segmentation()
        else:
            return []

    @property
    def category_id(self):
        """
        Returns category id of the annotation as int
        """
        return self._category_id

    @category_id.setter
    def category_id(self, i):
        if not isinstance(i, int):
            raise Exception("category_id must be an integer")
        self._category_id = i

    @property
    def image_id(self):
        """
        Returns image id of the annotation as int
        """
        return self._image_id

    @image_id.setter
    def image_id(self, i):
        if not isinstance(i, int):
            raise Exception("image_id must be an integer")
        self._image_id = i

    @property
    def category_name(self):
        """
        Returns category name of the annotation as str
        """
        return self._category_name

    @category_name.setter
    def category_name(self, n):
        if not isinstance(n, str):
            raise Exception("category_name must be a string")
        self._category_name = n

    @property
    def iscrowd(self):
        """
        Returns iscrowd info of the annotation
        """
        return self._iscrowd

    @property
    def json(self):
        return {
            "image_id": self.image_id,
            "bbox": self.bbox,
            "category_id": self.category_id,
            "segmentation": self.segmentation,
            "iscrowd": self.iscrowd,
            "area": self.area,
        }

    def serialize(self):
        print(".serialize() is deprectaed, use .json instead")

    def __repr__(self):
        return f"""CocoAnnotation<
    image_id: {self.image_id},
    bbox: {self.bbox},
    segmentation: {self.segmentation},
    category_id: {self.category_id},
    category_name: {self.category_name},
    iscrowd: {self.iscrowd},
    area: {self.area}>"""


class CocoPrediction(CocoAnnotation):
    """
    Class for handling predictions in coco format.
    """

    @classmethod
    def from_coco_segmentation(
        cls,
        segmentation,
        category_id,
        category_name,
        score,
        iscrowd=0,
        image_id=None,
    ):
        """
        Creates CocoAnnotation object using coco segmentation.

        Args:
            segmentation: List[List]
                [[1, 1, 325, 125, 250, 200, 5, 200]]
            category_id: int
                Category id of the annotation
            category_name: str
                Category name of the annotation
            score: float
                Prediction score between 0 and 1
            iscrowd: int
                0 or 1
        """
        return cls(
            segmentation=segmentation,
            category_id=category_id,
            category_name=category_name,
            score=score,
            iscrowd=iscrowd,
            image_id=image_id,
        )

    @classmethod
    def from_coco_bbox(
        cls, bbox, category_id, category_name, score, iscrowd=0, image_id=None
    ):
        """
        Creates CocoAnnotation object using coco bbox

        Args:
            bbox: List
                [xmin, ymin, width, height]
            category_id: int
                Category id of the annotation
            category_name: str
                Category name of the annotation
            score: float
                Prediction score between 0 and 1
            iscrowd: int
                0 or 1
        """
        return cls(
            bbox=bbox,
            category_id=category_id,
            category_name=category_name,
            score=score,
            iscrowd=iscrowd,
            image_id=image_id,
        )

    @classmethod
    def from_coco_annotation_dict(
        cls, category_name, annotation_dict, score, image_id=None
    ):
        """
        Creates CocoAnnotation object from category name and COCO formatted
        annotation dict (with fields "bbox", "segmentation", "category_id").

        Args:
            category_name: str
                Category name of the annotation
            annotation_dict: dict
                COCO formatted annotation dict (with fields "bbox", "segmentation", "category_id")
            score: float
                Prediction score between 0 and 1
        """
        if annotation_dict["segmentation"]:
            return cls(
                segmentation=annotation_dict["segmentation"],
                category_id=annotation_dict["category_id"],
                category_name=category_name,
                score=score,
                image_id=image_id,
            )
        else:
            return cls(
                bbox=annotation_dict["bbox"],
                category_id=annotation_dict["category_id"],
                category_name=category_name,
                image_id=image_id,
            )

    def __init__(
        self,
        segmentation=None,
        bbox=None,
        category_id=None,
        category_name=None,
        image_id=None,
        score=None,
        iscrowd=0,
    ):
        """

        Args:
            segmentation: List[List]
                [[1, 1, 325, 125, 250, 200, 5, 200]]
            bbox: List
                [xmin, ymin, width, height]
            category_id: int
                Category id of the annotation
            category_name: str
                Category name of the annotation
            image_id: int
                Image ID of the annotation
            score: float
                Prediction score between 0 and 1
            iscrowd: int
                0 or 1
        """
        self.score = score
        super().__init__(
            segmentation=segmentation,
            bbox=bbox,
            category_id=category_id,
            category_name=category_name,
            image_id=image_id,
            iscrowd=iscrowd,
        )

    @property
    def json(self):
        return {
            "image_id": self.image_id,
            "bbox": self.bbox,
            "score": self.score,
            "category_id": self.category_id,
            "category_name": self.category_name,
            "segmentation": self.segmentation,
            "iscrowd": self.iscrowd,
            "area": self.area,
        }

    def serialize(self):
        print(".serialize() is deprectaed, use .json instead")

    def __repr__(self):
        return f"""CocoPrediction<
    image_id: {self.image_id},
    bbox: {self.bbox},
    segmentation: {self.segmentation},
    score: {self.score},
    category_id: {self.category_id},
    category_name: {self.category_name},
    iscrowd: {self.iscrowd},
    area: {self.area}>"""


class CocoVidAnnotation(CocoAnnotation):
    """
    COCOVid formatted annotation.
    https://github.com/open-mmlab/mmtracking/blob/master/docs/tutorials/customize_dataset.md#the-cocovid-annotation-file
    """

    def __init__(
        self,
        bbox=None,
        category_id=None,
        category_name=None,
        image_id=None,
        instance_id=None,
        iscrowd=0,
        id=None,
    ):
        """
        Args:
            bbox: List
                [xmin, ymin, width, height]
            category_id: int
                Category id of the annotation
            category_name: str
                Category name of the annotation
            image_id: int
                Image ID of the annotation
            instance_id: int
                Used for tracking
            iscrowd: int
                0 or 1
            id: int
                Annotation id
        """
        super(CocoVidAnnotation, self).__init__(
            bbox=bbox,
            category_id=category_id,
            category_name=category_name,
            image_id=image_id,
            iscrowd=iscrowd,
        )
        self.instance_id = instance_id
        self.id = id

    @property
    def json(self):
        return {
            "id": self.id,
            "image_id": self.image_id,
            "bbox": self.bbox,
            "segmentation": self.segmentation,
            "category_id": self.category_id,
            "category_name": self.category_name,
            "instance_id": self.instance_id,
            "iscrowd": self.iscrowd,
            "area": self.area,
        }

    def __repr__(self):
        return f"""CocoAnnotation<
    id: {self.id},
    image_id: {self.image_id},
    bbox: {self.bbox},
    segmentation: {self.segmentation},
    category_id: {self.category_id},
    category_name: {self.category_name},
    instance_id: {self.instance_id},
    iscrowd: {self.iscrowd},
    area: {self.area}>"""


class CocoImage:
    @classmethod
    def from_coco_image_dict(cls, image_dict):
        """
        Creates CocoImage object from COCO formatted image dict (with fields "id", "file_name", "height" and "weight").

        Args:
            image_dict: dict
                COCO formatted image dict (with fields "id", "file_name", "height" and "weight")
        """
        return cls(
            id=image_dict["id"],
            file_name=image_dict["file_name"],
            height=image_dict["height"],
            width=image_dict["width"],
        )

    def __init__(
        self, file_name: str, height: int, width: int, id: int = None
    ):
        """
        Creates CocoImage object

        Args:
            id : int
                Image id
            file_name : str
                Image path
            height : int
                Image height in pixels
            width : int
                Image width in pixels
        """
        self.id = int(id) if id else id
        self.file_name = file_name
        self.height = int(height)
        self.width = int(width)
        self.annotations = (
            []
        )  # list of CocoAnnotation that belong to this image
        self.predictions = (
            []
        )  # list of CocoPrediction that belong to this image

    def add_annotation(self, annotation):
        """
        Adds annotation to this CocoImage instance

        annotation : CocoAnnotation
        """

        if not isinstance(annotation, CocoAnnotation):
            raise TypeError("annotation must be a CocoAnnotation instance")
        self.annotations.append(annotation)

    def add_prediction(self, prediction):
        """
        Adds prediction to this CocoImage instance

        prediction : CocoPrediction
        """

        if not isinstance(prediction, CocoPrediction):
            raise TypeError("prediction must be a CocoPrediction instance")
        self.predictions.append(prediction)

    @property
    def json(self):
        return {
            "id": self.id,
            "file_name": self.file_name,
            "height": self.height,
            "width": self.width,
        }

    def __repr__(self):
        return f"""CocoImage<
    id: {self.id},
    file_name: {self.file_name},
    height: {self.height},
    width: {self.width},
    annotations: List[CocoAnnotation],
    predictions: List[CocoPrediction]>"""


class CocoVidImage(CocoImage):
    """
    COCOVid formatted image.
    https://github.com/open-mmlab/mmtracking/blob/master/docs/tutorials/customize_dataset.md#the-cocovid-annotation-file
    """

    def __init__(
        self,
        file_name,
        height,
        width,
        video_id=None,
        frame_id=None,
        id=None,
    ):
        """
        Creates CocoVidImage object

        Args:
            id: int
                Image id
            file_name: str
                Image path
            height: int
                Image height in pixels
            width: int
                Image width in pixels
            frame_id: int
                0-indexed frame id
            video_id: int
                Video id
        """
        super(CocoVidImage, self).__init__(
            file_name=file_name, height=height, width=width, id=id
        )
        self.frame_id = frame_id
        self.video_id = video_id

    @classmethod
    def from_coco_image(cls, coco_image, video_id=None, frame_id=None):
        """
        Creates CocoVidImage object using CocoImage object.
        Args:
            coco_image: CocoImage
            frame_id: int
                0-indexed frame id
            video_id: int
                Video id

        """
        return cls(
            file_name=coco_image.file_name,
            height=coco_image.height,
            width=coco_image.width,
            id=coco_image.id,
            video_id=video_id,
            frame_id=frame_id,
        )

    def add_annotation(self, annotation):
        """
        Adds annotation to this CocoImage instance
        annotation : CocoVidAnnotation
        """

        if not isinstance(annotation, CocoVidAnnotation):
            raise TypeError("annotation must be a CocoVidAnnotation instance")
        self.annotations.append(annotation)

    @property
    def json(self):
        return {
            "file_name": self.file_name,
            "height": self.height,
            "width": self.width,
            "id": self.id,
            "video_id": self.video_id,
            "frame_id": self.frame_id,
        }

    def __repr__(self):
        return f"""CocoVidImage<
    file_name: {self.file_name},
    height: {self.height},
    width: {self.width},
    id: {self.id},
    video_id: {self.video_id},
    frame_id: {self.frame_id},
    annotations: List[CocoVidAnnotation]>"""


class CocoVideo:
    """
    COCO formatted video.
    https://github.com/open-mmlab/mmtracking/blob/master/docs/tutorials/customize_dataset.md#the-cocovid-annotation-file
    """

    def __init__(
        self,
        name: str,
        id: int = None,
        fps: float = None,
        height: int = None,
        width: int = None,
    ):
        """
        Creates CocoVideo object

        Args:
            name: str
                Video name
            id: int
                Video id
            fps: float
                Video fps
            height: int
                Video height in pixels
            width: int
                Video width in pixels
        """
        self.name = name
        self.id = id
        self.fps = fps
        self.height = height
        self.width = width
        self.images = []  # list of CocoImage that belong to this video

    def add_image(self, image):
        """
        Adds image to this CocoVideo instance
        Args:
            image: CocoImage
        """

        if not isinstance(image, CocoImage):
            raise TypeError("image must be a CocoImage instance")
        self.images.append(CocoVidImage.from_coco_image(image))

    def add_cocovidimage(self, cocovidimage):
        """
        Adds CocoVidImage to this CocoVideo instance
        Args:
            cocovidimage: CocoVidImage
        """

        if not isinstance(cocovidimage, CocoVidImage):
            raise TypeError("cocovidimage must be a CocoVidImage instance")
        self.images.append(cocovidimage)

    @property
    def json(self):
        return {
            "name": self.name,
            "id": self.id,
            "fps": self.fps,
            "height": self.height,
            "width": self.width,
        }

    def __repr__(self):
        return f"""CocoVideo<
    id: {self.id},
    name: {self.name},
    fps: {self.fps},
    height: {self.height},
    width: {self.width},
    images: List[CocoVidImage]>"""


class Coco:
    def __init__(
        self,
        name=None,
        image_dir=None,
        remapping_dict=None,
        ignore_negative_samples=False,
        clip_bboxes_to_img_dims=False,
        image_id_setting="auto",
    ):
        """
        Creates Coco object.

        Args:
            name: str
                Name of the Coco dataset, it determines exported json name.
            image_dir: str
                Base file directory that contains dataset images. Required for dataset merging.
            remapping_dict: dict
                {1:0, 2:1} maps category id 1 to 0 and category id 2 to 1
            ignore_negative_samples: bool
                If True ignores images without annotations in all operations.
            image_id_setting: str
                how to assign image ids while exporting can be
                    auto --> will assign id from scratch (<CocoImage>.id will be ignored)
                    manual --> you will need to provide image ids in <CocoImage> instances (<CocoImage>.id can not be None)
        """
        if image_id_setting not in ["auto", "manual"]:
            raise ValueError(
                "image_id_setting must be either 'auto' or 'manual'"
            )
        self.name = name
        self.image_dir = image_dir
        self.remapping_dict = remapping_dict
        self.ignore_negative_samples = ignore_negative_samples
        self.categories = []
        self.images = []
        self._stats = None
        self.clip_bboxes_to_img_dims = clip_bboxes_to_img_dims
        self.image_id_setting = image_id_setting

    def add_categories_from_coco_category_list(self, coco_category_list):
        """
        Creates CocoCategory object using coco category list.

        Args:
            coco_category_list: List[Dict]
                [
                    {"supercategory": "person", "id": 1, "name": "person"},
                    {"supercategory": "vehicle", "id": 2, "name": "bicycle"}
                ]
        """

        for coco_category in coco_category_list:
            if self.remapping_dict is not None:
                for source_id in self.remapping_dict.keys():
                    if coco_category["id"] == source_id:
                        target_id = self.remapping_dict[source_id]
                        coco_category["id"] = target_id

            self.add_category(CocoCategory.from_coco_category(coco_category))

    def add_category(self, category):
        """
        Adds category to this Coco instance

        Args:
            category: CocoCategory
        """

        # assert type(category) == CocoCategory, "category must be a CocoCategory instance"
        if not isinstance(category, CocoCategory):
            raise TypeError("category must be a CocoCategory instance")
        self.categories.append(category)

    def add_image(self, image):
        """
        Adds image to this Coco instance

        Args:
            image: CocoImage
        """

        if self.image_id_setting == "manual" and image.id is None:
            raise ValueError(
                "image id should be manually set for image_id_setting='manual'"
            )
        self.images.append(image)

    def update_categories(self, desired_name2id, update_image_filenames=False):
        """
        Rearranges category mapping of given COCO object based on given desired_name2id.
        Can also be used to filter some of the categories.

        Args:
            desired_name2id: dict
                {"big_vehicle": 1, "car": 2, "human": 3}
            update_image_filenames: bool
                If True, updates coco image file_names with absolute file paths.
        """
        # init vars
        currentid2desiredid_mapping = {}
        updated_coco = Coco(
            name=self.name,
            image_dir=self.image_dir,
            remapping_dict=self.remapping_dict,
            ignore_negative_samples=self.ignore_negative_samples,
        )
        # create category id mapping (currentid2desiredid_mapping)
        for coco_category in copy.deepcopy(self.categories):
            current_category_id = coco_category.id
            current_category_name = coco_category.name
            if current_category_name in desired_name2id.keys():
                currentid2desiredid_mapping[current_category_id] = (
                    desired_name2id[current_category_name]
                )
            else:
                # ignore categories that are not included in desired_name2id
                currentid2desiredid_mapping[current_category_id] = None

        # add updated categories
        for name in desired_name2id.keys():
            updated_coco_category = CocoCategory(
                id=desired_name2id[name], name=name, supercategory=name
            )
            updated_coco.add_category(updated_coco_category)

        # add updated images & annotations
        for coco_image in copy.deepcopy(self.images):
            updated_coco_image = CocoImage.from_coco_image_dict(
                coco_image.json
            )
            # update filename to abspath
            file_name_is_abspath = (
                True
                if os.path.abspath(coco_image.file_name)
                == coco_image.file_name
                else False
            )
            if update_image_filenames and not file_name_is_abspath:
                updated_coco_image.file_name = str(
                    Path(os.path.abspath(self.image_dir))
                    / coco_image.file_name
                )
            # update annotations
            for coco_annotation in coco_image.annotations:
                current_category_id = coco_annotation.category_id
                desired_category_id = currentid2desiredid_mapping[
                    current_category_id
                ]
                # append annotations with category id present in desired_name2id
                if desired_category_id is not None:
                    # update cetegory id
                    coco_annotation.category_id = desired_category_id
                    # append updated annotation to target coco dict
                    updated_coco_image.add_annotation(coco_annotation)
            updated_coco.add_image(updated_coco_image)

        # overwrite instance
        self.__class__ = updated_coco.__class__
        self.__dict__ = updated_coco.__dict__

    def merge(self, coco, desired_name2id=None, verbose=1):
        """
        Combines the images/annotations/categories of given coco object with current one.

        Args:
            coco : sahi.utils.coco.Coco instance
                A COCO dataset object
            desired_name2id : dict
                {"human": 1, "car": 2, "big_vehicle": 3}
            verbose: bool
                If True, merging info is printed
        """
        if self.image_dir is None or coco.image_dir is None:
            raise ValueError("image_dir should be provided for merging.")
        if verbose:
            if not desired_name2id:
                print(
                    "'desired_name2id' is not specified, combining all categories."
                )

        # create desired_name2id by combining all categories, if desired_name2id is not specified
        coco1 = self
        coco2 = coco
        category_ind = 0
        if desired_name2id is None:
            desired_name2id = {}
            for coco in [coco1, coco2]:
                temp_categories = copy.deepcopy(coco.json_categories)
                for temp_category in temp_categories:
                    if temp_category["name"] not in desired_name2id:
                        desired_name2id[temp_category["name"]] = category_ind
                        category_ind += 1
                    else:
                        continue

        # update categories and image paths
        for coco in [coco1, coco2]:
            coco.update_categories(
                desired_name2id=desired_name2id, update_image_filenames=True
            )

        # combine images and categories
        coco1.images.extend(coco2.images)
        self.images: List[CocoImage] = coco1.images
        self.categories = coco1.categories

        # print categories
        if verbose:
            print(
                "Categories are formed as:\n",
                self.json_categories,
            )

    @classmethod
    def from_coco_dict_or_path(
        cls,
        coco_dict_or_path: Union[Dict, str],
        image_dir: Optional[str] = None,
        remapping_dict: Optional[Dict] = None,
        ignore_negative_samples: bool = False,
        clip_bboxes_to_img_dims: bool = False,
        use_threads: bool = False,
        num_threads: int = 10,
    ):
        """
        Creates coco object from COCO formatted dict or COCO dataset file path.

        Args:
            coco_dict_or_path: dict/str or List[dict/str]
                COCO formatted dict or COCO dataset file path
                List of COCO formatted dict or COCO dataset file path
            image_dir: str
                Base file directory that contains dataset images. Required for merging and yolov5 conversion.
            remapping_dict: dict
                {1:0, 2:1} maps category id 1 to 0 and category id 2 to 1
            ignore_negative_samples: bool
                If True ignores images without annotations in all operations.
            clip_bboxes_to_img_dims: bool = False
                Limits bounding boxes to image dimensions.
            use_threads: bool = False
                Use threads when processing the json image list, defaults to False
            num_threads: int = 10
                Slice the image list to given number of chunks, defaults to 10

        Properties:
            images: list of CocoImage
            category_mapping: dict
        """
        # init coco object
        coco = cls(
            image_dir=image_dir,
            remapping_dict=remapping_dict,
            ignore_negative_samples=ignore_negative_samples,
            clip_bboxes_to_img_dims=clip_bboxes_to_img_dims,
        )

        if type(coco_dict_or_path) not in [str, dict]:
            raise TypeError("coco_dict_or_path should be a dict or str")

        # load coco dict if path is given
        if isinstance(coco_dict_or_path, str):
            coco_dict = load_json(coco_dict_or_path)
        else:
            coco_dict = coco_dict_or_path

        dict_size = len(coco_dict["images"])

        # arrange image id to annotation id mapping
        coco.add_categories_from_coco_category_list(coco_dict["categories"])
        image_id_to_annotation_list = get_imageid2annotationlist_mapping(
            coco_dict
        )
        category_mapping = coco.category_mapping

        # https://github.com/obss/sahi/issues/98
        image_id_set: Set = set()

        lock = Lock()

        def fill_image_id_set(
            start,
            finish,
            image_list,
            _image_id_set,
            _image_id_to_annotation_list,
            _coco,
            lock,
        ):
            for coco_image_dict in tqdm(
                image_list[start:finish],
                f"Loading coco annotations between {start} and {finish}",
            ):
                coco_image = CocoImage.from_coco_image_dict(coco_image_dict)
                image_id = coco_image_dict["id"]
                # https://github.com/obss/sahi/issues/98
                if image_id in _image_id_set:
                    print(f"duplicate image_id: {image_id}, will be ignored.")
                    continue
                else:
                    lock.acquire()
                    _image_id_set.add(image_id)
                    lock.release()

                # select annotations of the image
                annotation_list = _image_id_to_annotation_list[image_id]
                for coco_annotation_dict in annotation_list:
                    # apply category remapping if remapping_dict is provided
                    if _coco.remapping_dict is not None:
                        # apply category remapping (id:id)
                        category_id = _coco.remapping_dict[
                            coco_annotation_dict["category_id"]
                        ]
                        # update category id
                        coco_annotation_dict["category_id"] = category_id
                    else:
                        category_id = coco_annotation_dict["category_id"]
                    # get category name (id:name)
                    category_name = category_mapping[category_id]
                    coco_annotation = CocoAnnotation.from_coco_annotation_dict(
                        category_name=category_name,
                        annotation_dict=coco_annotation_dict,
                    )
                    coco_image.add_annotation(coco_annotation)
                _coco.add_image(coco_image)

        chunk_size = dict_size / num_threads

        if use_threads is True:
            for i in range(num_threads):
                start = i * chunk_size
                finish = start + chunk_size
                if finish > dict_size:
                    finish = dict_size
                t = Thread(
                    target=fill_image_id_set,
                    args=(
                        start,
                        finish,
                        coco_dict["images"],
                        image_id_set,
                        image_id_to_annotation_list,
                        coco,
                        lock,
                    ),
                )
                t.start()

            main_thread = threading.currentThread()
            for t in threading.enumerate():
                if t is not main_thread:
                    t.join()

        else:
            for coco_image_dict in tqdm(
                coco_dict["images"], "Loading coco annotations"
            ):
                coco_image = CocoImage.from_coco_image_dict(coco_image_dict)
                image_id = coco_image_dict["id"]
                # https://github.com/obss/sahi/issues/98
                if image_id in image_id_set:
                    print(f"duplicate image_id: {image_id}, will be ignored.")
                    continue
                else:
                    image_id_set.add(image_id)
                # select annotations of the image
                annotation_list = image_id_to_annotation_list[image_id]
                for coco_annotation_dict in annotation_list:
                    # apply category remapping if remapping_dict is provided
                    if coco.remapping_dict is not None:
                        # apply category remapping (id:id)
                        category_id = coco.remapping_dict[
                            coco_annotation_dict["category_id"]
                        ]
                        # update category id
                        coco_annotation_dict["category_id"] = category_id
                    else:
                        category_id = coco_annotation_dict["category_id"]
                    # get category name (id:name)
                    category_name = category_mapping[category_id]
                    coco_annotation = CocoAnnotation.from_coco_annotation_dict(
                        category_name=category_name,
                        annotation_dict=coco_annotation_dict,
                    )
                    coco_image.add_annotation(coco_annotation)
                coco.add_image(coco_image)

        if clip_bboxes_to_img_dims:
            coco = coco.get_coco_with_clipped_bboxes()
        return coco

    @property
    def json_categories(self):
        categories = []
        for category in self.categories:
            categories.append(category.json)
        return categories

    @property
    def category_mapping(self):
        category_mapping = {}
        for category in self.categories:
            category_mapping[category.id] = category.name
        return category_mapping

    @property
    def json(self):
        return create_coco_dict(
            images=self.images,
            categories=self.json_categories,
            ignore_negative_samples=self.ignore_negative_samples,
            image_id_setting=self.image_id_setting,
        )

    @property
    def prediction_array(self):
        return create_coco_prediction_array(
            images=self.images,
            ignore_negative_samples=self.ignore_negative_samples,
            image_id_setting=self.image_id_setting,
        )

    @property
    def stats(self):
        if not self._stats:
            self.calculate_stats()
        return self._stats

    def calculate_stats(self):
        """
        Iterates over all annotations and calculates total number of
        """
        # init all stats
        num_annotations = 0
        num_images = len(self.images)
        num_negative_images = 0
        num_categories = len(self.json_categories)
        category_name_to_zero = {
            category["name"]: 0 for category in self.json_categories
        }
        category_name_to_inf = {
            category["name"]: float("inf") for category in self.json_categories
        }
        num_images_per_category = copy.deepcopy(category_name_to_zero)
        num_annotations_per_category = copy.deepcopy(category_name_to_zero)
        min_annotation_area_per_category = copy.deepcopy(category_name_to_inf)
        max_annotation_area_per_category = copy.deepcopy(category_name_to_zero)
        min_num_annotations_in_image = float("inf")
        max_num_annotations_in_image = 0
        total_annotation_area = 0
        min_annotation_area = 1e10
        max_annotation_area = 0
        for image in self.images:
            image_contains_category = {}
            for annotation in image.annotations:
                annotation_area = annotation.area
                total_annotation_area += annotation_area
                num_annotations_per_category[annotation.category_name] += 1
                image_contains_category[annotation.category_name] = 1
                # update min&max annotation area
                if annotation_area > max_annotation_area:
                    max_annotation_area = annotation_area
                if annotation_area < min_annotation_area:
                    min_annotation_area = annotation_area
                if (
                    annotation_area
                    > max_annotation_area_per_category[
                        annotation.category_name
                    ]
                ):
                    max_annotation_area_per_category[
                        annotation.category_name
                    ] = annotation_area
                if (
                    annotation_area
                    < min_annotation_area_per_category[
                        annotation.category_name
                    ]
                ):
                    min_annotation_area_per_category[
                        annotation.category_name
                    ] = annotation_area
            # update num_negative_images
            if len(image.annotations) == 0:
                num_negative_images += 1
            # update num_annotations
            num_annotations += len(image.annotations)
            # update num_images_per_category
            num_images_per_category = dict(
                Counter(num_images_per_category)
                + Counter(image_contains_category)
            )
            # update min&max_num_annotations_in_image
            num_annotations_in_image = len(image.annotations)
            if num_annotations_in_image > max_num_annotations_in_image:
                max_num_annotations_in_image = num_annotations_in_image
            if num_annotations_in_image < min_num_annotations_in_image:
                min_num_annotations_in_image = num_annotations_in_image
        if (num_images - num_negative_images) > 0:
            avg_num_annotations_in_image = num_annotations / (
                num_images - num_negative_images
            )
            avg_annotation_area = total_annotation_area / num_annotations
        else:
            avg_num_annotations_in_image = 0
            avg_annotation_area = 0

        self._stats = {
            "num_images": num_images,
            "num_annotations": num_annotations,
            "num_categories": num_categories,
            "num_negative_images": num_negative_images,
            "num_images_per_category": num_images_per_category,
            "num_annotations_per_category": num_annotations_per_category,
            "min_num_annotations_in_image": min_num_annotations_in_image,
            "max_num_annotations_in_image": max_num_annotations_in_image,
            "avg_num_annotations_in_image": avg_num_annotations_in_image,
            "min_annotation_area": min_annotation_area,
            "max_annotation_area": max_annotation_area,
            "avg_annotation_area": avg_annotation_area,
            "min_annotation_area_per_category": min_annotation_area_per_category,
            "max_annotation_area_per_category": max_annotation_area_per_category,
        }

    def split_coco_as_train_val(self, train_split_rate=0.9, numpy_seed=0):
        """
        Split images into train-val and returns them as sahi.utils.coco.Coco objects.

        Args:
            train_split_rate: float
            numpy_seed: int
                To fix the numpy seed.

        Returns:
            result : dict
                {
                    "train_coco": "",
                    "val_coco": "",
                }
        """
        # fix numpy numpy seed
        np.random.seed(numpy_seed)

        # divide images
        num_images = len(self.images)
        shuffled_images = copy.deepcopy(self.images)
        np.random.shuffle(shuffled_images)
        num_train = int(num_images * train_split_rate)
        train_images = shuffled_images[:num_train]
        val_images = shuffled_images[num_train:]

        # form train val coco objects
        train_coco = Coco(
            name=self.name if self.name else "split" + "_train",
            image_dir=self.image_dir,
        )
        train_coco.images = train_images
        train_coco.categories = self.categories

        val_coco = Coco(
            name=self.name if self.name else "split" + "_val",
            image_dir=self.image_dir,
        )
        val_coco.images = val_images
        val_coco.categories = self.categories

        # return result
        return {
            "train_coco": train_coco,
            "val_coco": val_coco,
        }

    def export_as_yolov5(
        self,
        output_dir,
        train_split_rate=1,
        numpy_seed=0,
        mp=False,
        disable_symlink=False,
    ):
        """
        Exports current COCO dataset in ultralytics/yolov5 format.
        Creates train val folders with image symlinks and txt files and a data yaml file.

        Args:
            output_dir: str
                Export directory.
            train_split_rate: float
                If given 1, will be exported as train split.
                If given 0, will be exported as val split.
                If in between 0-1, both train/val splits will be calculated and exported.
            numpy_seed: int
                To fix the numpy seed.
            mp: bool
                If True, multiprocess mode is on.
                Should be called in 'if __name__ == __main__:' block.
            disable_symlink: bool
                If True, symlinks will not be created. Instead, images will be copied.
        """
        try:
            import yaml
        except ImportError:
            raise ImportError(
                'Please run "pip install -U pyyaml" '
                "to install yaml first for yolov5 formatted exporting."
            )

        # set split_mode
        if 0 < train_split_rate and train_split_rate < 1:
            split_mode = "TRAINVAL"
        elif train_split_rate == 0:
            split_mode = "VAL"
        elif train_split_rate == 1:
            split_mode = "TRAIN"
        else:
            raise ValueError("train_split_rate cannot be <0 or >1")

        # split dataset
        if split_mode == "TRAINVAL":
            result = self.split_coco_as_train_val(
                train_split_rate=train_split_rate,
                numpy_seed=numpy_seed,
            )
            train_coco = result["train_coco"]
            val_coco = result["val_coco"]
        elif split_mode == "TRAIN":
            train_coco = self
            val_coco = None
        elif split_mode == "VAL":
            train_coco = None
            val_coco = self

        # create train val image dirs
        train_dir = ""
        val_dir = ""
        if split_mode in ["TRAINVAL", "TRAIN"]:
            train_dir = Path(os.path.abspath(output_dir)) / "train/"
            train_dir.mkdir(parents=True, exist_ok=True)  # create dir
        if split_mode in ["TRAINVAL", "VAL"]:
            val_dir = Path(os.path.abspath(output_dir)) / "val/"
            val_dir.mkdir(parents=True, exist_ok=True)  # create dir

        # create image symlinks and annotation txts
        if split_mode in ["TRAINVAL", "TRAIN"]:
            export_yolov5_images_and_txts_from_coco_object(
                output_dir=train_dir,
                coco=train_coco,
                ignore_negative_samples=self.ignore_negative_samples,
                mp=mp,
                disable_symlink=disable_symlink,
            )
        if split_mode in ["TRAINVAL", "VAL"]:
            export_yolov5_images_and_txts_from_coco_object(
                output_dir=val_dir,
                coco=val_coco,
                ignore_negative_samples=self.ignore_negative_samples,
                mp=mp,
                disable_symlink=disable_symlink,
            )

        # create yolov5 data yaml
        data = {
            "train": str(train_dir),
            "val": str(val_dir),
            "nc": len(self.category_mapping),
            "names": list(self.category_mapping.values()),
        }
        yaml_path = str(Path(output_dir) / "data.yml")
        with open(yaml_path, "w") as outfile:
            yaml.dump(data, outfile, default_flow_style=None)

    def get_subsampled_coco(
        self, subsample_ratio: int = 2, category_id: int = None
    ):
        """
        Subsamples images with subsample_ratio and returns as sahi.utils.coco.Coco object.

        Args:
            subsample_ratio: int
                10 means take every 10th image with its annotations
            category_id: int
                subsample only images containing given category_id, if -1 then subsamples negative samples
        Returns:
            subsampled_coco: sahi.utils.coco.Coco
        """
        subsampled_coco = Coco(
            name=self.name,
            image_dir=self.image_dir,
            remapping_dict=self.remapping_dict,
            ignore_negative_samples=self.ignore_negative_samples,
        )
        subsampled_coco.add_categories_from_coco_category_list(
            self.json_categories
        )

        if category_id is not None:
            # get images that contain given category id
            images_that_contain_category: List[CocoImage] = []
            for image in self.images:
                category_id_to_contains = defaultdict(lambda: 0)
                annotation: CocoAnnotation
                for annotation in image.annotations:
                    category_id_to_contains[annotation.category_id] = 1
                if category_id_to_contains[category_id]:
                    add_this_image = True
                elif category_id == -1 and len(image.annotations) == 0:
                    # if category_id is given as -1, select negative samples
                    add_this_image = True
                else:
                    add_this_image = False

                if add_this_image:
                    images_that_contain_category.append(image)

            # get images that does not contain given category id
            images_that_doesnt_contain_category: List[CocoImage] = []
            for image in self.images:
                category_id_to_contains = defaultdict(lambda: 0)
                annotation: CocoAnnotation
                for annotation in image.annotations:
                    category_id_to_contains[annotation.category_id] = 1
                if category_id_to_contains[category_id]:
                    add_this_image = False
                elif category_id == -1 and len(image.annotations) == 0:
                    # if category_id is given as -1, dont select negative samples
                    add_this_image = False
                else:
                    add_this_image = True

                if add_this_image:
                    images_that_doesnt_contain_category.append(image)

        if category_id:
            selected_images = images_that_contain_category
            # add images that does not contain given category without subsampling
            for image_ind in range(len(images_that_doesnt_contain_category)):
                subsampled_coco.add_image(
                    images_that_doesnt_contain_category[image_ind]
                )
        else:
            selected_images = self.images
        for image_ind in range(0, len(selected_images), subsample_ratio):
            subsampled_coco.add_image(selected_images[image_ind])

        return subsampled_coco

    def get_upsampled_coco(
        self, upsample_ratio: int = 2, category_id: int = None
    ):
        """
        Upsamples images with upsample_ratio and returns as sahi.utils.coco.Coco object.

        Args:
            upsample_ratio: int
                10 means copy each sample 10 times
            category_id: int
                upsample only images containing given category_id, if -1 then upsamples negative samples
        Returns:
            upsampled_coco: sahi.utils.coco.Coco
        """
        upsampled_coco = Coco(
            name=self.name,
            image_dir=self.image_dir,
            remapping_dict=self.remapping_dict,
            ignore_negative_samples=self.ignore_negative_samples,
        )
        upsampled_coco.add_categories_from_coco_category_list(
            self.json_categories
        )
        for ind in range(upsample_ratio):
            for image_ind in range(len(self.images)):
                # calculate add_this_image
                if category_id is not None:
                    category_id_to_contains = defaultdict(lambda: 0)
                    annotation: CocoAnnotation
                    for annotation in self.images[image_ind].annotations:
                        category_id_to_contains[annotation.category_id] = 1
                    if category_id_to_contains[category_id]:
                        add_this_image = True
                    elif (
                        category_id == -1
                        and len(self.images[image_ind].annotations) == 0
                    ):
                        # if category_id is given as -1, select negative samples
                        add_this_image = True
                    elif ind == 0:
                        # in first iteration add all images
                        add_this_image = True
                    else:
                        add_this_image = False
                else:
                    add_this_image = True

                if add_this_image:
                    upsampled_coco.add_image(self.images[image_ind])

        return upsampled_coco

    def get_area_filtered_coco(
        self, min=0, max=float("inf"), intervals_per_category=None
    ):
        """
        Filters annotation areas with given min and max values and returns remaining
        images as sahi.utils.coco.Coco object.

        Args:
            min: int
                minimum allowed area
            max: int
                maximum allowed area
            intervals_per_category: dict of dicts
                {
                    "human": {"min": 20, "max": 10000},
                    "vehicle": {"min": 50, "max": 15000},
                }
        Returns:
            area_filtered_coco: sahi.utils.coco.Coco
        """
        area_filtered_coco = Coco(
            name=self.name,
            image_dir=self.image_dir,
            remapping_dict=self.remapping_dict,
            ignore_negative_samples=self.ignore_negative_samples,
        )
        area_filtered_coco.add_categories_from_coco_category_list(
            self.json_categories
        )
        for image in self.images:
            is_valid_image = True
            for annotation in image.annotations:
                if (
                    intervals_per_category is not None
                    and annotation.category_name
                    in intervals_per_category.keys()
                ):
                    category_based_min = intervals_per_category[
                        annotation.category_name
                    ]["min"]
                    category_based_max = intervals_per_category[
                        annotation.category_name
                    ]["max"]
                    if (
                        annotation.area < category_based_min
                        or annotation.area > category_based_max
                    ):
                        is_valid_image = False
                if annotation.area < min or annotation.area > max:
                    is_valid_image = False
            if is_valid_image:
                area_filtered_coco.add_image(image)

        return area_filtered_coco

    def get_coco_with_clipped_bboxes(self):
        """
        Limits overflowing bounding boxes to image dimensions.
        """
        from sahi.slicing import annotation_inside_slice

        coco = Coco(
            name=self.name,
            image_dir=self.image_dir,
            remapping_dict=self.remapping_dict,
            ignore_negative_samples=self.ignore_negative_samples,
        )
        coco.add_categories_from_coco_category_list(self.json_categories)

        for coco_img in self.images:
            img_dims = [0, 0, coco_img.width, coco_img.height]
            coco_image = CocoImage(
                file_name=coco_img.file_name,
                height=coco_img.height,
                width=coco_img.width,
                id=coco_img.id,
            )
            for coco_ann in coco_img.annotations:
                ann_dict: Dict = coco_ann.json
                if annotation_inside_slice(
                    annotation=ann_dict, slice_bbox=img_dims
                ):
                    shapely_ann = coco_ann.get_sliced_coco_annotation(img_dims)
                    bbox = ShapelyAnnotation.to_xywh(
                        shapely_ann._shapely_annotation
                    )
                    coco_ann_from_shapely = CocoAnnotation(
                        bbox=bbox,
                        category_id=coco_ann.category_id,
                        category_name=coco_ann.category_name,
                        image_id=coco_ann.image_id,
                    )
                    coco_image.add_annotation(coco_ann_from_shapely)
                else:
                    continue
            coco.add_image(coco_image)
        return coco


def export_yolov5_images_and_txts_from_coco_object(
    output_dir,
    coco,
    ignore_negative_samples=False,
    mp=False,
    disable_symlink=False,
):
    """
    Creates image symlinks and annotation txts in yolo format from coco dataset.

    Args:
        output_dir: str
            Export directory.
        coco: sahi.utils.coco.Coco
            Initialized Coco object that contains images and categories.
        ignore_negative_samples: bool
            If True ignores images without annotations in all operations.
        mp: bool
            If True, multiprocess mode is on.
            Should be called in 'if __name__ == __main__:' block.
        disable_symlink: bool
            If True, symlinks are not created. Instead images are copied.
    """
    logger.info(
        "generating image symlinks and annotation files for yolov5..."
    ),
    # symlink is not supported in colab
    if is_colab() and not disable_symlink:
        logger.warning("symlink is not supported in colab, disabling it...")
        disable_symlink = True
    if mp:
        with Pool(processes=48) as pool:
            args = [
                (
                    coco_image,
                    coco.image_dir,
                    output_dir,
                    ignore_negative_samples,
                    disable_symlink,
                )
                for coco_image in coco.images
            ]
            pool.starmap(
                export_single_yolov5_image_and_corresponding_txt,
                tqdm(args, total=len(args)),
            )
    else:
        for coco_image in tqdm(coco.images):
            export_single_yolov5_image_and_corresponding_txt(
                coco_image,
                coco.image_dir,
                output_dir,
                ignore_negative_samples,
                disable_symlink,
            )


def export_single_yolov5_image_and_corresponding_txt(
    coco_image,
    coco_image_dir,
    output_dir,
    ignore_negative_samples=False,
    disable_symlink=False,
):
    """
    Generates yolov5 formatted image symlink and annotation txt file.

    Args:
        coco_image: sahi.utils.coco.CocoImage
        coco_image_dir: str
        output_dir: str
            Export directory.
        ignore_negative_samples: bool
            If True ignores images without annotations in all operations.
    """
    # if coco_image contains any invalid annotations, skip it
    contains_invalid_annotations = False
    for coco_annotation in coco_image.annotations:
        if len(coco_annotation.bbox) != 4:
            contains_invalid_annotations = True
            break
    if contains_invalid_annotations:
        return
    # skip images without annotations
    if len(coco_image.annotations) == 0 and ignore_negative_samples:
        return
    # skip images without suffix
    # https://github.com/obss/sahi/issues/114
    if Path(coco_image.file_name).suffix == "":
        print(
            f"image file has no suffix, skipping it: '{coco_image.file_name}'"
        )
        return
    elif Path(coco_image.file_name).suffix in [
        ".txt"
    ]:  # TODO: extend this list
        print(
            f"image file has incorrect suffix, skipping it: '{coco_image.file_name}'"
        )
        return
    # set coco and yolo image paths
    if Path(coco_image.file_name).is_file():
        coco_image_path = os.path.abspath(coco_image.file_name)
    else:
        if coco_image_dir is None:
            raise ValueError(
                "You have to specify image_dir of Coco object for yolov5 conversion."
            )

        coco_image_path = os.path.abspath(
            str(Path(coco_image_dir) / coco_image.file_name)
        )

    yolo_image_path_temp = str(
        Path(output_dir) / Path(coco_image.file_name).name
    )
    # increment target file name if already present
    yolo_image_path = copy.deepcopy(yolo_image_path_temp)
    name_increment = 2
    while Path(yolo_image_path).is_file():
        parent_dir = Path(yolo_image_path_temp).parent
        filename = Path(yolo_image_path_temp).stem
        filesuffix = Path(yolo_image_path_temp).suffix
        filename = filename + "_" + str(name_increment)
        yolo_image_path = str(parent_dir / (filename + filesuffix))
        name_increment += 1
    # create a symbolic link pointing to coco_image_path named yolo_image_path
    if disable_symlink:
        import shutil

        shutil.copy(coco_image_path, yolo_image_path)
    else:
        os.symlink(coco_image_path, yolo_image_path)
    # calculate annotation normalization ratios
    width = coco_image.width
    height = coco_image.height
    dw = 1.0 / (width)
    dh = 1.0 / (height)
    # set annotation filepath
    image_file_suffix = Path(yolo_image_path).suffix
    yolo_annotation_path = yolo_image_path.replace(image_file_suffix, ".txt")
    # create annotation file
    annotations = coco_image.annotations
    with open(yolo_annotation_path, "w") as outfile:
        for annotation in annotations:
            # convert coco bbox to yolo bbox
            x_center = annotation.bbox[0] + annotation.bbox[2] / 2.0
            y_center = annotation.bbox[1] + annotation.bbox[3] / 2.0
            bbox_width = annotation.bbox[2]
            bbox_height = annotation.bbox[3]
            x_center = x_center * dw
            y_center = y_center * dh
            bbox_width = bbox_width * dw
            bbox_height = bbox_height * dh
            category_id = annotation.category_id
            yolo_bbox = (x_center, y_center, bbox_width, bbox_height)
            # save yolo annotation
            outfile.write(
                str(category_id)
                + " "
                + " ".join([str(value) for value in yolo_bbox])
                + "\n"
            )


def update_categories(desired_name2id: dict, coco_dict: dict) -> dict:
    """
    Rearranges category mapping of given COCO dictionary based on given category_mapping.
    Can also be used to filter some of the categories.

    Arguments:
    ---------
        desired_name2id : dict
            {"big_vehicle": 1, "car": 2, "human": 3}
        coco_dict : dict
            COCO formatted dictionary.
    Returns:
    ---------
        coco_target : dict
            COCO dict with updated/filtred categories.
    """
    # so that original variable doesnt get affected
    coco_source = copy.deepcopy(coco_dict)

    # init target coco dict
    coco_target = {"images": [], "annotations": [], "categories": []}

    # init vars
    currentid2desiredid_mapping = {}
    # create category id mapping (currentid2desiredid_mapping)
    for category in coco_source["categories"]:
        current_category_id = category["id"]
        current_category_name = category["name"]
        if current_category_name in desired_name2id.keys():
            currentid2desiredid_mapping[current_category_id] = desired_name2id[
                current_category_name
            ]
        else:
            # ignore categories that are not included in desired_name2id
            currentid2desiredid_mapping[current_category_id] = -1

    # update annotations
    for annotation in coco_source["annotations"]:
        current_category_id = annotation["category_id"]
        desired_category_id = currentid2desiredid_mapping[current_category_id]
        # append annotations with category id present in desired_name2id
        if desired_category_id != -1:
            # update cetegory id
            annotation["category_id"] = desired_category_id
            # append updated annotation to target coco dict
            coco_target["annotations"].append(annotation)

    # create desired categories
    categories = []
    for name in desired_name2id.keys():
        category = {}
        category["name"] = category["supercategory"] = name
        category["id"] = desired_name2id[name]
        categories.append(category)

    # update categories
    coco_target["categories"] = categories

    # update images
    coco_target["images"] = coco_source["images"]

    return coco_target


def update_categories_from_file(
    desired_name2id: dict, coco_path: str, save_path: str
) -> None:
    """
    Rearranges category mapping of a COCO dictionary in coco_path based on given category_mapping.
    Can also be used to filter some of the categories.
    Arguments:
    ---------
        desired_name2id : dict
            {"human": 1, "car": 2, "big_vehicle": 3}
        coco_path : str
            "dirname/coco.json"
    """
    # load source coco dict
    coco_source = load_json(coco_path)

    # update categories
    coco_target = update_categories(desired_name2id, coco_source)

    # save modified coco file
    save_json(coco_target, save_path)


def merge(
    coco_dict1: dict, coco_dict2: dict, desired_name2id: dict = None
) -> dict:
    """
    Combines 2 coco formatted annotations dicts, and returns the combined coco dict.

    Arguments:
    ---------
        coco_dict1 : dict
            First coco dictionary.
        coco_dict2 : dict
            Second coco dictionary.
        desired_name2id : dict
            {"human": 1, "car": 2, "big_vehicle": 3}
    Returns:
    ---------
        merged_coco_dict : dict
            Merged COCO dict.
    """

    # copy input dicts so that original dicts are not affected
    temp_coco_dict1 = copy.deepcopy(coco_dict1)
    temp_coco_dict2 = copy.deepcopy(coco_dict2)

    # rearrange categories if any desired_name2id mapping is given
    if desired_name2id is not None:
        temp_coco_dict1 = update_categories(desired_name2id, temp_coco_dict1)
        temp_coco_dict2 = update_categories(desired_name2id, temp_coco_dict2)

    # rearrange categories of the second coco based on first, if their categories are not the same
    if temp_coco_dict1["categories"] != temp_coco_dict2["categories"]:
        desired_name2id = {
            category["name"]: category["id"]
            for category in temp_coco_dict1["categories"]
        }
        temp_coco_dict2 = update_categories(desired_name2id, temp_coco_dict2)

    # calculate first image and annotation index of the second coco file
    max_image_id = np.array(
        [image["id"] for image in coco_dict1["images"]]
    ).max()
    max_annotation_id = np.array(
        [annotation["id"] for annotation in coco_dict1["annotations"]]
    ).max()

    merged_coco_dict = temp_coco_dict1

    for image in temp_coco_dict2["images"]:
        image["id"] += max_image_id + 1
        merged_coco_dict["images"].append(image)

    for annotation in temp_coco_dict2["annotations"]:
        annotation["image_id"] += max_image_id + 1
        annotation["id"] += max_annotation_id + 1
        merged_coco_dict["annotations"].append(annotation)

    return merged_coco_dict


def merge_from_list(coco_dict_list, desired_name2id=None, verbose=1):
    """
    Combines a list of coco formatted annotations dicts, and returns the combined coco dict.

    Arguments:
    ---------
        coco_dict_list: list of dict
            A list of coco dicts
        desired_name2id: dict
            {"human": 1, "car": 2, "big_vehicle": 3}
        verbose: bool
            If True, merging info is printed
    Returns:
    ---------
        merged_coco_dict: dict
            Merged COCO dict.
    """
    if verbose:
        if not desired_name2id:
            print(
                "'desired_name2id' is not specified, combining all categories."
            )

    # create desired_name2id by combinin all categories, if desired_name2id is not specified
    if desired_name2id is None:
        desired_name2id = {}
        ind = 0
        for coco_dict in coco_dict_list:
            temp_categories = copy.deepcopy(coco_dict["categories"])
            for temp_category in temp_categories:
                if temp_category["name"] not in desired_name2id:
                    desired_name2id[temp_category["name"]] = ind
                    ind += 1
                else:
                    continue

    for ind, coco_dict in enumerate(coco_dict_list):
        if ind == 0:
            merged_coco_dict = copy.deepcopy(coco_dict)
        else:
            merged_coco_dict = merge(
                merged_coco_dict, coco_dict, desired_name2id
            )

    # print categories
    if verbose:
        print(
            "Categories are formed as:\n",
            merged_coco_dict["categories"],
        )

    return merged_coco_dict


def merge_from_file(coco_path1: str, coco_path2: str, save_path: str):
    """
    Combines 2 coco formatted annotations files given their paths, and saves the combined file to save_path.

    Arguments:
    ---------
        coco_path1 : str
            Path for the first coco file.
        coco_path2 : str
            Path for the second coco file.
        save_path : str
            "dirname/coco.json"
    """

    # load coco files to be combined
    coco_dict1 = load_json(coco_path1)
    coco_dict2 = load_json(coco_path2)

    # merge coco dicts
    merged_coco_dict = merge(coco_dict1, coco_dict2)

    # save merged coco dict
    save_json(merged_coco_dict, save_path)


def get_imageid2annotationlist_mapping(
    coco_dict: dict,
) -> Dict[int, List[CocoAnnotation]]:
    """
    Get image_id to annotationlist mapping for faster indexing.

    Arguments
    ---------
        coco_dict : dict
            coco dict with fields "images", "annotations", "categories"
    Returns
    -------
        image_id_to_annotation_list : dict
        {
            1: [CocoAnnotation, CocoAnnotation, CocoAnnotation],
            2: [CocoAnnotation]
        }

        where
        CocoAnnotation = {
            'area': 2795520,
            'bbox': [491.0, 1035.0, 153.0, 182.0],
            'category_id': 1,
            'id': 1,
            'image_id': 1,
            'iscrowd': 0,
            'segmentation': [[491.0, 1035.0, 644.0, 1035.0, 644.0, 1217.0, 491.0, 1217.0]]
        }
    """
    image_id_to_annotation_list: Dict = defaultdict(list)
    print("indexing coco dataset annotations...")
    for annotation in coco_dict["annotations"]:
        image_id = annotation["image_id"]
        image_id_to_annotation_list[image_id].append(annotation)

    return image_id_to_annotation_list


def create_coco_dict(
    images, categories, ignore_negative_samples=False, image_id_setting="auto"
):
    """
    Creates COCO dict with fields "images", "annotations", "categories".

    Arguments
    ---------
        images : List of CocoImage containing a list of CocoAnnotation
        categories : List of Dict
            COCO categories
        ignore_negative_samples : Bool
            If True, images without annotations are ignored
        image_id_setting: str
            how to assign image ids while exporting can be
                auto --> will assign id from scratch (<CocoImage>.id will be ignored)
                manual --> you will need to provide image ids in <CocoImage> instances (<CocoImage>.id can not be None)
    Returns
    -------
        coco_dict : Dict
            COCO dict with fields "images", "annotations", "categories"
    """
    # assertion of parameters
    if image_id_setting not in ["auto", "manual"]:
        raise ValueError(
            "'image_id_setting' should be one of ['auto', 'manual']"
        )

    # define accumulators
    image_index = 1
    annotation_id = 1
    coco_dict = dict(images=[], annotations=[], categories=categories)
    for coco_image in images:
        # get coco annotations
        coco_annotations = coco_image.annotations
        # get num annotations
        num_annotations = len(coco_annotations)
        # if ignore_negative_samples is True and no annotations, skip image
        if ignore_negative_samples and num_annotations == 0:
            continue
        else:
            # get image_id
            if image_id_setting == "auto":
                image_id = image_index
                image_index += 1
            elif image_id_setting == "manual":
                if coco_image.id is None:
                    raise ValueError(
                        "'coco_image.id' should be set manually when image_id_setting == 'manual'"
                    )
                image_id = coco_image.id

            # create coco image object
            out_image = {
                "height": coco_image.height,
                "width": coco_image.width,
                "id": image_id,
                "file_name": coco_image.file_name,
            }
            coco_dict["images"].append(out_image)

            # do the same for image annotations
            for coco_annotation in coco_annotations:
                # create coco annotation object
                out_annotation = {
                    "iscrowd": 0,
                    "image_id": image_id,
                    "bbox": coco_annotation.bbox,
                    "segmentation": coco_annotation.segmentation,
                    "category_id": coco_annotation.category_id,
                    "id": annotation_id,
                    "area": coco_annotation.area,
                }
                coco_dict["annotations"].append(out_annotation)
                # increment annotation id
                annotation_id += 1

    # return coco dict
    return coco_dict


def create_coco_prediction_array(
    images, ignore_negative_samples=False, image_id_setting="auto"
):
    """
    Creates COCO prediction array which is list of predictions

    Arguments
    ---------
        images : List of CocoImage containing a list of CocoAnnotation
        ignore_negative_samples : Bool
            If True, images without predictions are ignored
        image_id_setting: str
            how to assign image ids while exporting can be
                auto --> will assign id from scratch (<CocoImage>.id will be ignored)
                manual --> you will need to provide image ids in <CocoImage> instances (<CocoImage>.id can not be None)
    Returns
    -------
        coco_prediction_array : List
            COCO predictions array
    """
    # assertion of parameters
    if image_id_setting not in ["auto", "manual"]:
        raise ValueError(
            "'image_id_setting' should be one of ['auto', 'manual']"
        )
    # define accumulators
    image_index = 1
    prediction_id = 1
    predictions_array = []
    for coco_image in images:
        # get coco predictions
        coco_predictions = coco_image.predictions
        # get num predictions
        num_predictions = len(coco_predictions)
        # if ignore_negative_samples is True and no annotations, skip image
        if ignore_negative_samples and num_predictions == 0:
            continue
        else:
            # get image_id
            if image_id_setting == "auto":
                image_id = image_index
                image_index += 1
            elif image_id_setting == "manual":
                if coco_image.id is None:
                    raise ValueError(
                        "'coco_image.id' should be set manually when image_id_setting == 'manual'"
                    )
                image_id = coco_image.id

            # create coco prediction object
            for prediction_index, coco_prediction in enumerate(
                coco_predictions
            ):
                # create coco prediction object
                out_prediction = {
                    "id": prediction_id,
                    "image_id": image_id,
                    "bbox": coco_prediction.bbox,
                    "score": coco_prediction.score,
                    "category_id": coco_prediction.category_id,
                    "segmentation": coco_prediction.segmentation,
                    "iscrowd": coco_prediction.iscrowd,
                    "area": coco_prediction.area,
                }
                predictions_array.append(out_prediction)

                # increment prediction id
                prediction_id += 1

    # return predictions array
    return predictions_array


def add_bbox_and_area_to_coco(
    source_coco_path: str = "",
    target_coco_path: str = "",
    add_bbox: bool = True,
    add_area: bool = True,
) -> dict:
    """
    Takes single coco dataset file path, calculates and fills bbox and area fields of the annotations
    and exports the updated coco dict.
    Returns:
    coco_dict : dict
        Updated coco dict
    """
    coco_dict = load_json(source_coco_path)
    coco_dict = copy.deepcopy(coco_dict)

    annotations = coco_dict["annotations"]
    for ind, annotation in enumerate(annotations):
        # assign annotation bbox
        if add_bbox:
            coco_polygons = []
            [
                coco_polygons.extend(coco_polygon)
                for coco_polygon in annotation["segmentation"]
            ]
            minx, miny, maxx, maxy = list(
                [
                    min(coco_polygons[0::2]),
                    min(coco_polygons[1::2]),
                    max(coco_polygons[0::2]),
                    max(coco_polygons[1::2]),
                ]
            )
            x, y, width, height = (
                minx,
                miny,
                maxx - minx,
                maxy - miny,
            )
            annotations[ind]["bbox"] = [x, y, width, height]

        # assign annotation area
        if add_area:
            shapely_multipolygon = get_shapely_multipolygon(
                coco_segmentation=annotation["segmentation"]
            )
            annotations[ind]["area"] = shapely_multipolygon.area

    coco_dict["annotations"] = annotations
    save_json(coco_dict, target_coco_path)
    return coco_dict


@dataclass
class DatasetClassCounts:
    """Stores the number of images that include each category in a dataset"""

    counts: dict
    total_images: int

    def frequencies(self):
        """calculates the frequenct of images that contain each category"""
        return {
            cid: count / self.total_images
            for cid, count in self.counts.items()
        }

    def __add__(self, o):
        total = self.total_images + o.total_images
        exclusive_keys = set(o.counts.keys()) - set(self.counts.keys())
        counts = {}
        for k, v in self.counts.items():
            counts[k] = v + o.counts.get(k, 0)
        for k in exclusive_keys:
            counts[k] = o.counts[k]
        return DatasetClassCounts(counts, total)


def count_images_with_category(coco_file_path):
    """Reads a coco dataset file and returns an DatasetClassCounts object
     that stores the number of images that include each category in a dataset
    Returns: DatasetClassCounts object
    coco_file_path : str
        path to coco dataset file
    """

    image_id_2_category_2_count = defaultdict(lambda: defaultdict(lambda: 0))
    coco = load_json(coco_file_path)
    for annotation in coco["annotations"]:
        image_id = annotation["image_id"]
        cid = annotation["category_id"]
        image_id_2_category_2_count[image_id][cid] = (
            image_id_2_category_2_count[image_id][cid] + 1
        )

    category_2_count = defaultdict(lambda: 0)
    for (
        image_id,
        image_category_2_count,
    ) in image_id_2_category_2_count.items():
        for cid, count in image_category_2_count.items():
            if count > 0:
                category_2_count[cid] = category_2_count[cid] + 1

    category_2_count = dict(category_2_count)
    total_images = len(image_id_2_category_2_count.keys())
    return DatasetClassCounts(category_2_count, total_images)


class CocoVid:
    def __init__(self, name=None, remapping_dict=None):
        """
        Creates CocoVid object.

        Args:
            name: str
                Name of the CocoVid dataset, it determines exported json name.
            remapping_dict: dict
                {1:0, 2:1} maps category id 1 to 0 and category id 2 to 1
        """
        self.name = name
        self.remapping_dict = remapping_dict
        self.categories = []
        self.videos = []

    def add_categories_from_coco_category_list(self, coco_category_list):
        """
        Creates CocoCategory object using coco category list.

        Args:
            coco_category_list: List[Dict]
                [
                    {"supercategory": "person", "id": 1, "name": "person"},
                    {"supercategory": "vehicle", "id": 2, "name": "bicycle"}
                ]
        """

        for coco_category in coco_category_list:
            if self.remapping_dict is not None:
                for source_id in self.remapping_dict.keys():
                    if coco_category["id"] == source_id:
                        target_id = self.remapping_dict[source_id]
                        coco_category["id"] = target_id

            self.add_category(CocoCategory.from_coco_category(coco_category))

    def add_category(self, category):
        """
        Adds category to this CocoVid instance

        Args:
            category: CocoCategory
        """

        if not isinstance(category, CocoCategory):
            raise TypeError("category must be a CocoCategory instance")
        self.categories.append(category)

    @property
    def json_categories(self):
        categories = []
        for category in self.categories:
            categories.append(category.json)
        return categories

    @property
    def category_mapping(self):
        category_mapping = {}
        for category in self.categories:
            category_mapping[category.id] = category.name
        return category_mapping

    def add_video(self, video):
        """
        Adds video to this CocoVid instance

        Args:
            video: CocoVideo
        """

        if not isinstance(video, CocoVideo):
            raise TypeError("video must be a CocoVideo instance")
        self.videos.append(video)

    @property
    def json(self):
        coco_dict = {
            "videos": [],
            "images": [],
            "annotations": [],
            "categories": self.json_categories,
        }
        annotation_id = 1
        image_id = 1
        video_id = 1
        global_instance_id = 1
        for coco_video in self.videos:
            coco_video.id = video_id
            coco_dict["videos"].append(coco_video.json)

            frame_id = 0
            instance_id_set = set()
            for cocovid_image in coco_video.images:
                cocovid_image.id = image_id
                cocovid_image.frame_id = frame_id
                cocovid_image.video_id = coco_video.id
                coco_dict["images"].append(cocovid_image.json)

                for cocovid_annotation in cocovid_image.annotations:
                    instance_id_set.add(cocovid_annotation.instance_id)
                    cocovid_annotation.instance_id += global_instance_id

                    cocovid_annotation.id = annotation_id
                    cocovid_annotation.image_id = cocovid_image.id
                    coco_dict["annotations"].append(cocovid_annotation.json)

                    # increment annotation_id
                    annotation_id = copy.deepcopy(annotation_id + 1)
                # increment image_id and frame_id
                image_id = copy.deepcopy(image_id + 1)
                frame_id = copy.deepcopy(frame_id + 1)
            # increment video_id and global_instance_id
            video_id = copy.deepcopy(video_id + 1)
            global_instance_id += len(instance_id_set)

        return coco_dict


def remove_invalid_coco_results(
    result_list_or_path: Union[List, str],
    dataset_dict_or_path: Union[Dict, str] = None,
):
    """
    Removes invalid predictions from coco result such as:
        - negative bbox value
        - extreme bbox value

    Args:
        result_list_or_path: path or list for coco result json
        dataset_dict_or_path (optional): path or dict for coco dataset json
    """

    # prepare coco results
    if isinstance(result_list_or_path, str):
        result_list = load_json(result_list_or_path)
    elif isinstance(result_list_or_path, list):
        result_list = result_list_or_path
    else:
        raise TypeError('incorrect type for "result_list_or_path"')

    # prepare image info from coco dataset
    if dataset_dict_or_path is not None:
        if isinstance(dataset_dict_or_path, str):
            dataset_dict = load_json(dataset_dict_or_path)
        elif isinstance(dataset_dict_or_path, dict):
            dataset_dict = dataset_dict_or_path
        else:
            raise TypeError('incorrect type for "dataset_dict"')
        image_id_to_height = {}
        image_id_to_width = {}
        for coco_image in dataset_dict["images"]:
            image_id_to_height[coco_image["id"]] = coco_image["height"]
            image_id_to_width[coco_image["id"]] = coco_image["width"]

    # remove invalid predictions
    fixed_result_list = []
    for coco_result in result_list:
        bbox = coco_result["bbox"]
        # ignore invalid predictions
        if not bbox:
            print("ignoring invalid prediction with empty bbox")
            continue
        if bbox[0] < 0 or bbox[1] < 0 or bbox[2] < 0 or bbox[3] < 0:
            print(f"ignoring invalid prediction with bbox: {bbox}")
            continue
        if dataset_dict_or_path is not None:
            if (
                bbox[1] > image_id_to_height[coco_result["image_id"]]
                or bbox[3] > image_id_to_height[coco_result["image_id"]]
                or bbox[0] > image_id_to_width[coco_result["image_id"]]
                or bbox[2] > image_id_to_width[coco_result["image_id"]]
            ):
                print(f"ignoring invalid prediction with bbox: {bbox}")
                continue
        fixed_result_list.append(coco_result)
    return fixed_result_list


def export_coco_as_yolov5(
    output_dir: str,
    train_coco: Coco = None,
    val_coco: Coco = None,
    train_split_rate: float = 0.9,
    numpy_seed=0,
    disable_symlink=False,
):
    """
    Exports current COCO dataset in ultralytics/yolov5 format.
    Creates train val folders with image symlinks and txt files and a data yaml file.

    Args:
        output_dir: str
            Export directory.
        train_coco: Coco
            coco object for training
        val_coco: Coco
            coco object for val
        train_split_rate: float
            train split rate between 0 and 1. will be used when val_coco is None.
        numpy_seed: int
            To fix the numpy seed.
        disable_symlink: bool
            If True, copy images instead of creating symlinks.

    Returns:
        yaml_path: str
            Path for the exported yolov5 data.yml
    """
    try:
        import yaml
    except ImportError:
        raise ImportError(
            'Please run "pip install -U pyyaml" '
            "to install yaml first for yolov5 formatted exporting."
        )

    # set split_mode
    if train_coco and not val_coco:
        split_mode = True
    elif train_coco and val_coco:
        split_mode = False
    else:
        raise ValueError("'train_coco' have to be provided")

    # check train_split_rate
    if split_mode and not (0 < train_split_rate < 1):
        raise ValueError("train_split_rate cannot be <0 or >1")

    # split dataset
    if split_mode:
        result = train_coco.split_coco_as_train_val(
            train_split_rate=train_split_rate,
            numpy_seed=numpy_seed,
        )
        train_coco = result["train_coco"]
        val_coco = result["val_coco"]

    # create train val image dirs
    train_dir = Path(os.path.abspath(output_dir)) / "train/"
    train_dir.mkdir(parents=True, exist_ok=True)  # create dir
    val_dir = Path(os.path.abspath(output_dir)) / "val/"
    val_dir.mkdir(parents=True, exist_ok=True)  # create dir

    # create image symlinks and annotation txts
    export_yolov5_images_and_txts_from_coco_object(
        output_dir=train_dir,
        coco=train_coco,
        ignore_negative_samples=train_coco.ignore_negative_samples,
        mp=False,
        disable_symlink=disable_symlink,
    )
    export_yolov5_images_and_txts_from_coco_object(
        output_dir=val_dir,
        coco=val_coco,
        ignore_negative_samples=val_coco.ignore_negative_samples,
        mp=False,
        disable_symlink=disable_symlink,
    )

    # create yolov5 data yaml
    data = {
        "train": str(train_dir).replace("\\", "/"),
        "val": str(val_dir).replace("\\", "/"),
        "nc": len(train_coco.category_mapping),
        "names": list(train_coco.category_mapping.values()),
    }
    yaml_path = str(Path(output_dir) / "data.yml")
    with open(yaml_path, "w") as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

    return yaml_path


def export_coco_as_yolov5_via_yml(
    yml_path: str,
    output_dir: str,
    train_split_rate: float = 0.9,
    numpy_seed=0,
    disable_symlink=False,
):
    """
    Exports current COCO dataset in ultralytics/yolov5 format.
    Creates train val folders with image symlinks and txt files and a data yaml file.
    Uses a yml file as input.

    Args:
        yml_path: str
            file should contain these fields:
                train_json_path: str
                train_image_dir: str
                val_json_path: str
                val_image_dir: str
        output_dir: str
            Export directory.
        train_split_rate: float
            train split rate between 0 and 1. will be used when val_json_path is None.
        numpy_seed: int
            To fix the numpy seed.
        disable_symlink: bool
            If True, copy images instead of creating symlinks.

    Returns:
        yaml_path: str
            Path for the exported yolov5 data.yml
    """
    try:
        import yaml
    except ImportError:
        raise ImportError(
            'Please run "pip install -U pyyaml" '
            "to install yaml first for yolov5 formatted exporting."
        )

    with open(yml_path, "r") as stream:
        config_dict = yaml.safe_load(stream)

    if config_dict["train_json_path"]:
        if not config_dict["train_image_dir"]:
            raise ValueError(f"{yml_path} is missing `train_image_dir`")
        train_coco = Coco.from_coco_dict_or_path(
            config_dict["train_json_path"],
            image_dir=config_dict["train_image_dir"],
        )
    else:
        train_coco = None

    if config_dict["val_json_path"]:
        if not config_dict["val_image_dir"]:
            raise ValueError(f"{yml_path} is missing `val_image_dir`")
        val_coco = Coco.from_coco_dict_or_path(
            config_dict["val_json_path"],
            image_dir=config_dict["val_image_dir"],
        )
    else:
        val_coco = None

    yaml_path = export_coco_as_yolov5(
        output_dir=output_dir,
        train_coco=train_coco,
        val_coco=val_coco,
        train_split_rate=train_split_rate,
        numpy_seed=numpy_seed,
        disable_symlink=disable_symlink,
    )

    return yaml_path
