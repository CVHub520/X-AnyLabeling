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
