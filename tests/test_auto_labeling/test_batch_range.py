import json
import tempfile
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtWidgets

from anylabeling.services.auto_labeling.types import AutoLabelingResult
from anylabeling.views.labeling.utils import batch


class TestBatchRange(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(
            []
        )

    def test_range_inputs_keep_inclusive_bounds(self):
        dialog = batch.BatchRangeDialog(10, 4)
        self.addCleanup(dialog.close)
        self.assertEqual(dialog.from_input.value(), 4)
        self.assertEqual(dialog.to_input.value(), 10)
        dialog.to_input.setValue(5)
        dialog.from_input.setValue(8)
        self.assertEqual(dialog.to_input.value(), 8)
        dialog.from_input.setValue(1)
        dialog.to_input.setValue(1)
        self.assertEqual(dialog.to_input.value(), 1)
        dialog.to_input.setValue(100)
        self.assertEqual(dialog.to_input.value(), 10)

    def test_skip_status_combinations_and_alignment(self):
        from itertools import combinations

        for count in range(4):
            for selected in combinations((False, True, "failed"), count):
                dialog = batch.BatchRangeDialog(
                    3, statuses=[False, True, "failed"], skip_statuses=selected
                )
                try:
                    dialog.show()
                    self.app.processEvents()
                    self.assertEqual(
                        dialog.selected_skip_statuses(), list(selected)
                    )
                    self.assertEqual(
                        dialog.status_checkboxes["failed"].geometry().right(),
                        dialog.to_input.geometry().right(),
                    )
                    label = next(
                        item
                        for item in dialog.findChildren(QtWidgets.QLabel)
                        if item.text() == "Skip statuses:"
                    )
                    self.assertEqual(label.x(), dialog.from_input.x())
                finally:
                    dialog.close()
        dialog = batch.BatchRangeDialog(
            3, skip_statuses=[True], allow_skip=False
        )
        try:
            self.assertEqual(dialog.selected_skip_statuses(), [])
            self.assertTrue(
                all(
                    not box.isEnabled()
                    for box in dialog.status_checkboxes.values()
                )
            )
        finally:
            dialog.close()

    def test_skipped_images_never_reach_inference_or_save(self):
        from itertools import combinations

        with tempfile.TemporaryDirectory() as directory:
            images = [os.path.join(directory, f"{i}.png") for i in range(4)]
            for index, status in enumerate((False, True, "failed")):
                with open(
                    os.path.splitext(images[index])[0] + ".json", "w"
                ) as stream:
                    json.dump({"checked": status}, stream)
            for count in range(4):
                for selected in combinations((False, True, "failed"), count):
                    manager = Mock()
                    manager.loaded_model_config = {
                        "type": "yolo",
                        "model": Mock(),
                    }
                    manager.predict_shapes.return_value = AutoLabelingResult(
                        []
                    )
                    widget = SimpleNamespace(
                        may_continue=lambda: True,
                        file_list_widget=Mock(),
                        settings=Mock(value=lambda *args: []),
                        image_list=images,
                        filename=images[0],
                        fn_to_index={path: i for i, path in enumerate(images)},
                        image=object(),
                        output_dir=directory,
                        auto_labeling_widget=SimpleNamespace(
                            model_manager=manager,
                            button_skip_detection=Mock(
                                isChecked=lambda: False
                            ),
                        ),
                        cancel_processing=False,
                        tr=lambda text: text,
                    )
                    dialog = Mock()
                    dialog.exec.return_value = (
                        QtWidgets.QDialog.DialogCode.Accepted
                    )
                    dialog.from_input.value.return_value = 1
                    dialog.to_input.value.return_value = 4
                    dialog.selected_skip_statuses.return_value = list(selected)
                    with (
                        patch.object(
                            batch, "BatchRangeDialog", return_value=dialog
                        ),
                        patch.object(
                            batch,
                            "show_progress_dialog_and_process",
                            side_effect=lambda app: batch.process_next_image(
                                app, Mock()
                            ),
                        ),
                        patch.object(
                            batch.BatchProcessingThread,
                            "start",
                            new=batch.BatchProcessingThread.run,
                        ),
                        patch.object(
                            batch, "save_auto_labeling_result"
                        ) as save,
                        patch.object(batch, "finish_processing"),
                        patch.object(
                            batch.QtWidgets.QMessageBox, "information"
                        ),
                    ):
                        batch.run_all_images(widget)
                    expected = [
                        path
                        for path, status in zip(
                            images, (False, True, "failed", False)
                        )
                        if status not in selected
                    ]
                    self.assertEqual(
                        [
                            call.args[1]
                            for call in manager.predict_shapes.call_args_list
                        ],
                        expected,
                    )
                    self.assertEqual(
                        [call.args[1] for call in save.call_args_list],
                        expected,
                    )

    def test_auto_run_processes_only_selected_images(self):
        for model_type in ("yolo", batch._BATCH_PROCESSING_VIDEO_MODELS[0]):
            for first, last in ((2, 4), (3, 3), (1, 5)):
                with self.subTest(model=model_type, first=first, last=last):
                    image_list = [f"/tmp/{index}.png" for index in range(5)]
                    manager = Mock()
                    manager.loaded_model_config = {
                        "type": model_type,
                        "model": Mock(),
                    }
                    manager.predict_shapes.return_value = AutoLabelingResult(
                        []
                    )
                    widget = SimpleNamespace(
                        may_continue=lambda: True,
                        file_list_widget=Mock(),
                        settings=Mock(value=lambda *args: []),
                        image_list=image_list,
                        filename=image_list[2],
                        fn_to_index={
                            path: i for i, path in enumerate(image_list)
                        },
                        image=object(),
                        auto_labeling_widget=SimpleNamespace(
                            model_manager=manager,
                            button_skip_detection=Mock(
                                isChecked=lambda: False
                            ),
                        ),
                        cancel_processing=False,
                        load_file=Mock(),
                        tr=lambda text: text,
                    )
                    dialog = Mock()
                    dialog.selected_skip_statuses.return_value = []
                    dialog.exec.return_value = (
                        QtWidgets.QDialog.DialogCode.Accepted
                    )
                    dialog.from_input.value.return_value = first
                    dialog.to_input.value.return_value = last
                    progress = Mock()
                    with (
                        patch.object(
                            batch, "BatchRangeDialog", return_value=dialog
                        ),
                        patch.object(
                            batch,
                            "show_progress_dialog_and_process",
                            side_effect=lambda app: batch.process_next_image(
                                app, progress
                            ),
                        ),
                        patch.object(
                            batch.BatchProcessingThread,
                            "start",
                            new=batch.BatchProcessingThread.run,
                        ),
                        patch.object(
                            batch, "save_auto_labeling_result"
                        ) as save,
                        patch.object(batch, "finish_processing"),
                    ):
                        batch.run_all_images(widget)

                    expected = image_list[first - 1 : last]
                    self.assertEqual(
                        [
                            call.args[1]
                            for call in manager.predict_shapes.call_args_list
                        ],
                        expected,
                    )
                    self.assertEqual(widget.current_index, 2)
                    self.assertEqual(widget.image_index, last)
                    progress.setValue.assert_called_with(len(expected))
                    if model_type == "yolo":
                        self.assertEqual(
                            [call.args[1] for call in save.call_args_list],
                            expected,
                        )
                    else:
                        self.assertEqual(
                            [
                                call.args[0]
                                for call in widget.load_file.call_args_list
                            ],
                            expected,
                        )


if __name__ == "__main__":
    unittest.main()
