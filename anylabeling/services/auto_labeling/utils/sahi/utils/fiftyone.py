import os
import subprocess
import sys

from anylabeling.services.auto_labeling.utils.sahi.utils.import_utils import (
    is_available,
)

if is_available("fiftyone"):
    # to fix https://github.com/voxel51/fiftyone/issues/845
    if sys.platform == "win32":
        _ = subprocess.run("tskill mongod", stderr=subprocess.DEVNULL)
    else:
        _ = subprocess.run(["pkill", "mongod"], stderr=subprocess.DEVNULL)

    # import fo utilities
    import fiftyone as fo
    from fiftyone.utils.coco import (
        COCODetectionDatasetImporter as BaseCOCODetectionDatasetImporter,
    )
    from fiftyone.utils.coco import (
        _get_matching_image_ids,
        load_coco_detection_annotations,
    )

    class COCODetectionDatasetImporter(BaseCOCODetectionDatasetImporter):
        def setup(self):
            if self.labels_path is not None and os.path.isfile(
                self.labels_path
            ):
                (
                    info,
                    classes,
                    supercategory_map,
                    images,
                    annotations,
                ) = load_coco_detection_annotations(
                    self.labels_path, extra_attrs=self.extra_attrs
                )

                if classes is not None:
                    info["classes"] = classes

                image_ids = _get_matching_image_ids(
                    classes,
                    images,
                    annotations,
                    image_ids=self.image_ids,
                    classes=self.classes,
                    shuffle=self.shuffle,
                    seed=self.seed,
                    max_samples=self.max_samples,
                )

                filenames = [images[_id]["file_name"] for _id in image_ids]

                _image_ids = set(image_ids)
                image_dicts_map = {
                    i["file_name"]: i
                    for _id, i in images.items()
                    if _id in _image_ids
                }
            else:
                info = {}
                classes = None
                supercategory_map = None
                image_dicts_map = {}
                annotations = None
                filenames = []

            self._image_paths_map = {
                image["file_name"]: os.path.join(
                    self.data_path, image["file_name"]
                )
                for image in images.values()
            }

            self._info = info
            self._classes = classes
            self._supercategory_map = supercategory_map
            self._image_dicts_map = image_dicts_map
            self._annotations = annotations
            self._filenames = filenames

    def create_fiftyone_dataset_from_coco_file(
        coco_image_dir: str, coco_json_path: str
    ):
        coco_importer = COCODetectionDatasetImporter(
            data_path=coco_image_dir,
            labels_path=coco_json_path,
            include_id=True,
        )
        dataset = fo.Dataset.from_importer(coco_importer, label_field="gt")
        return dataset

    def launch_fiftyone_app(coco_image_dir: str, coco_json_path: str):
        dataset = create_fiftyone_dataset_from_coco_file(
            coco_image_dir, coco_json_path
        )
        session = fo.launch_app()
        session.dataset = dataset
        return session
