import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PIL import Image
from PyQt6 import QtCore, QtWidgets

from anylabeling.services.auto_labeling.internimage_cls import InternImage_CLS
from anylabeling.services.auto_labeling.types import AutoLabelingResult
from anylabeling.services.auto_labeling.yolo11_cls import YOLO11_CLS
from anylabeling.services.auto_labeling.yolov5_cls import YOLOv5_CLS
from anylabeling.services.auto_labeling.yolov8_cls import YOLOv8_CLS
from anylabeling.views.labeling.classifier.utils import (
    get_first_true_flag,
    load_flags_from_json,
)
from anylabeling.views.labeling.label_widget import LabelingWidget
from anylabeling.views.labeling.utils.batch import save_auto_labeling_result
from anylabeling.views.labeling.widgets.classifier_dialog import (
    ClassifierDialog,
)


class ClassificationFlagsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(
            []
        )

    def test_classification_models_return_complete_flags_without_description(
        self,
    ):
        for model_class in (
            YOLOv5_CLS,
            YOLOv8_CLS,
            YOLO11_CLS,
            InternImage_CLS,
        ):
            with self.subTest(model=model_class.__name__):
                model = model_class.__new__(model_class)
                model.classes = ["cat", "dog", "bird"]
                if model_class is InternImage_CLS:
                    model.classes = dict(enumerate(model.classes))
                model.preprocess = mock.Mock(
                    return_value=np.zeros((1, 3, 8, 8))
                )
                model.net = mock.Mock()
                model.net.get_ort_inference.return_value = np.array(
                    [[0.1, 0.8, 0.1]]
                )
                module = (
                    "internimage_cls"
                    if model_class is InternImage_CLS
                    else "yolov5_cls"
                )
                with mock.patch(
                    f"anylabeling.services.auto_labeling.{module}.qt_img_to_rgb_cv_img"
                ):
                    result = model.predict_shapes(object())

                self.assertEqual(
                    result.flags, {"cat": False, "dog": True, "bird": False}
                )
                self.assertEqual(result.description, "")
                self.assertEqual(result.shapes, [])
                self.assertFalse(result.replace)

    def test_live_prediction_is_available_to_classifier_before_saving(self):
        flag_widget = QtWidgets.QListWidget()
        widget = SimpleNamespace(
            image=True,
            image_path="/tmp/classification.png",
            filename="/tmp/classification.png",
            image_list=["/tmp/classification.png"],
            flag_widget=flag_widget,
            canvas=SimpleNamespace(shapes=[]),
            load_shapes=mock.Mock(),
            other_data={"description": "keep", "checked": True},
            set_dirty=mock.Mock(),
            _sync_annotation_checked_state=mock.Mock(),
        )
        widget.load_flags = lambda flags: LabelingWidget.load_flags(
            widget, flags
        )
        widget.load_flags({"cat": True, "dog": False, "unrelated": True})
        flag_widget.itemChanged.connect(widget.set_dirty)
        widget.set_dirty.side_effect = lambda: self.assertFalse(
            widget.other_data["checked"]
        )

        LabelingWidget.new_shapes_from_auto_labeling(
            widget,
            AutoLabelingResult(
                [], replace=False, flags={"cat": False, "dog": True}
            ),
        )

        self.assertEqual(widget.other_data["description"], "keep")
        widget.set_dirty.assert_called_once_with()
        widget._sync_annotation_checked_state.assert_called_once_with()
        self.assertEqual(
            [flag_widget.item(i).checkState() for i in range(3)],
            [
                QtCore.Qt.CheckState.Unchecked,
                QtCore.Qt.CheckState.Checked,
                QtCore.Qt.CheckState.Checked,
            ],
        )
        dialog = SimpleNamespace(
            parent=lambda: widget,
            update_image_display=mock.Mock(),
            update_navigation_state=mock.Mock(),
            page_input=mock.Mock(),
            create_checkbox_group=mock.Mock(),
            load_current_flags=mock.Mock(),
        )
        ClassifierDialog.load_initial_data(dialog)
        self.assertEqual(dialog.labels, ["cat", "dog", "unrelated"])
        self.assertEqual(widget.image_flags, dialog.labels)
        dialog.create_checkbox_group.assert_called_once_with()
        dialog.load_current_flags.assert_called_once_with()

    def test_batch_flags_round_trip_and_reclassification(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = os.path.join(directory, "image.png")
            output_dir = os.path.join(directory, "labels")
            os.mkdir(output_dir)
            label_path = os.path.join(output_dir, "image.json")
            Image.new("RGB", (8, 6)).save(image_path)
            widget = SimpleNamespace(
                output_dir=output_dir, _config={"store_data": False}
            )

            save_auto_labeling_result(
                widget,
                image_path,
                AutoLabelingResult(
                    [], replace=False, flags={"cat": True, "dog": False}
                ),
            )
            self.assertEqual(
                load_flags_from_json(label_path), {"cat": True, "dog": False}
            )
            with open(label_path, encoding="utf-8") as stream:
                data = json.load(stream)
            data.update(
                description="keep", checked=True, tags=["keep"], custom=1
            )
            data["flags"]["unrelated"] = False
            data["shapes"] = [{"label": "existing"}]
            with open(label_path, "w", encoding="utf-8") as stream:
                json.dump(data, stream)

            save_auto_labeling_result(
                widget,
                image_path,
                AutoLabelingResult(
                    [], replace=False, flags={"cat": False, "dog": True}
                ),
            )
            flags = load_flags_from_json(label_path)
            self.assertEqual(
                flags, {"cat": False, "dog": True, "unrelated": False}
            )
            self.assertEqual(get_first_true_flag(flags), "dog")
            with open(label_path, encoding="utf-8") as stream:
                saved = json.load(stream)
            self.assertFalse(saved["checked"])
            for key in ("description", "shapes", "tags", "custom"):
                self.assertEqual(saved[key], data[key])

            save_auto_labeling_result(
                widget, image_path, AutoLabelingResult([], replace=False)
            )
            self.assertEqual(load_flags_from_json(label_path), flags)


if __name__ == "__main__":
    unittest.main()
