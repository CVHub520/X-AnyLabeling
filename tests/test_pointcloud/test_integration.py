import os
import subprocess
import sys
import textwrap
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtGui, QtWidgets

from anylabeling.views.labeling import label_widget

POINTCLOUD_MODULE = "anylabeling.views.labeling.widgets.pointcloud_dialog"


class TestPointCloudIntegration(unittest.TestCase):
    def make_widget(self):
        widget = SimpleNamespace(
            pointcloud_window=None,
            on_pointcloud_window_destroyed=Mock(),
            may_continue=Mock(return_value=True),
            training_dialog=None,
            settings=Mock(),
            filename="image.png",
            recent_files=["image.png"],
            last_open_dir=None,
            _settings_controller=Mock(),
            _config={},
            dirty=True,
            canvas=object(),
            tr=lambda text: text,
            error_message=Mock(),
        )
        return widget

    def test_main_starts_with_entry_enabled_and_pointcloud_unloaded(self):
        script = textwrap.dedent("""\
            import sys
            import tempfile
            from pathlib import Path
            from PyQt6 import QtCore, QtWidgets
            from anylabeling import config
            import anylabeling.resources.resources
            from anylabeling.views.mainwindow import MainWindow

            with tempfile.TemporaryDirectory() as directory:
                config.set_work_directory(directory)
                config.current_config_file = str(
                    Path(directory) / '.xanylabelingrc'
                )
                QtCore.QSettings.setDefaultFormat(
                    QtCore.QSettings.Format.IniFormat
                )
                QtCore.QSettings.setPath(
                    QtCore.QSettings.Format.IniFormat,
                    QtCore.QSettings.Scope.UserScope,
                    directory,
                )
                app = QtWidgets.QApplication([])
                window = MainWindow(app, config=config.get_default_config())
                view = window.labeling_widget.view
                assert view.filename is None
                assert view.actions.open_pointcloud.isEnabled()
                actions = view.actions.tool
                assert actions.index(view.actions.open_pointcloud) == (
                    actions.index(view.actions.open_vqa) + 1
                )
                assert not any('pointcloud' in name for name in sys.modules)
                window.close()
            """)
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_open_without_image_reuses_and_restores_window(self):
        widget = self.make_widget()
        widget.filename = None
        canvas = widget.canvas
        window = Mock()
        window.isMinimized.side_effect = [False, True]
        constructor = Mock(return_value=window)
        module = SimpleNamespace(PointCloudDialog=constructor)
        with patch.dict(
            sys.modules,
            {POINTCLOUD_MODULE: module},
        ):
            label_widget.LabelingWidget.open_pointcloud(widget)
            label_widget.LabelingWidget.open_pointcloud(widget)

        constructor.assert_called_once_with(widget)
        window.show.assert_called_once_with()
        window.showNormal.assert_called_once_with()
        self.assertEqual(window.raise_.call_count, 2)
        self.assertEqual(window.activateWindow.call_count, 2)
        window.destroyed.connect.assert_called_once_with(
            widget.on_pointcloud_window_destroyed
        )
        self.assertIsNone(widget.filename)
        self.assertIs(widget.canvas, canvas)
        self.assertTrue(widget.dirty)
        widget.may_continue.assert_not_called()

    def test_destroyed_workspace_clears_session_reference(self):
        widget = self.make_widget()
        widget.pointcloud_window = Mock()
        label_widget.LabelingWidget.on_pointcloud_window_destroyed(widget)
        self.assertIsNone(widget.pointcloud_window)

    def test_initialization_failure_stays_inside_workspace_boundary(self):
        for error in (
            ImportError("missing renderer"),
            RuntimeError("display"),
        ):
            with self.subTest(error=error):
                widget = self.make_widget()
                module = SimpleNamespace(
                    PointCloudDialog=Mock(side_effect=error)
                )
                with patch.dict(
                    sys.modules,
                    {POINTCLOUD_MODULE: module},
                ):
                    label_widget.LabelingWidget.open_pointcloud(widget)
                self.assertIsNone(widget.pointcloud_window)
                widget.error_message.assert_called_once()
                self.assertEqual(widget.filename, "image.png")
                self.assertTrue(widget.dirty)

    def test_image_cancellation_preserves_pointcloud_and_settings_session(
        self,
    ):
        widget = self.make_widget()
        widget.may_continue.return_value = False
        window = widget.pointcloud_window = Mock()
        event = QtGui.QCloseEvent()

        label_widget.LabelingWidget.closeEvent(widget, event)

        self.assertFalse(event.isAccepted())
        window.can_close.assert_not_called()
        window.close_after_approval.assert_not_called()
        widget._settings_controller.close_session.assert_not_called()
        widget.settings.setValue.assert_not_called()

    def test_pointcloud_cancellation_preserves_image_task(self):
        widget = self.make_widget()
        window = widget.pointcloud_window = Mock()
        window.can_close.return_value = False
        event = QtGui.QCloseEvent()

        label_widget.LabelingWidget.closeEvent(widget, event)

        self.assertFalse(event.isAccepted())
        window.close_after_approval.assert_not_called()
        self.assertEqual(widget.filename, "image.png")
        self.assertTrue(widget.dirty)
        widget._settings_controller.close_session.assert_not_called()

    def test_workspace_closes_only_after_both_tasks_approve(self):
        widget = self.make_widget()
        window = widget.pointcloud_window = Mock()
        calls = Mock()
        calls.attach_mock(widget.may_continue, "image_check")
        calls.attach_mock(window.can_close, "pointcloud_check")
        calls.attach_mock(window.close_after_approval, "pointcloud_close")
        event = QtGui.QCloseEvent()

        with patch.object(label_widget, "save_config"):
            label_widget.LabelingWidget.closeEvent(widget, event)

        self.assertTrue(event.isAccepted())
        self.assertEqual(
            [call[0] for call in calls.mock_calls],
            ["image_check", "pointcloud_check", "pointcloud_close"],
        )
        widget._settings_controller.close_session.assert_called_once_with()

    def test_another_tool_veto_keeps_pointcloud_alive(self):
        for tool in ("training", "video"):
            with self.subTest(tool=tool):
                widget = self.make_widget()
                window = widget.pointcloud_window = Mock()
                if tool == "training":
                    widget.training_dialog = Mock(
                        prepare_for_application_close=Mock(return_value=False)
                    )
                else:
                    widget.video_classifier_window = Mock(
                        isVisible=Mock(return_value=True)
                    )
                event = QtGui.QCloseEvent()

                label_widget.LabelingWidget.closeEvent(widget, event)

                self.assertFalse(event.isAccepted())
                window.close_after_approval.assert_not_called()

    def test_failed_or_cancelled_image_save_blocks_exit(self):
        widget = self.make_widget()
        widget.image_tags_widget = Mock()
        widget.save_file = Mock()
        with patch.object(
            QtWidgets.QMessageBox,
            "question",
            return_value=QtWidgets.QMessageBox.StandardButton.Save,
        ):
            self.assertFalse(label_widget.LabelingWidget.may_continue(widget))
        self.assertTrue(widget.dirty)

    def test_successful_image_save_allows_exit(self):
        widget = self.make_widget()
        widget.image_tags_widget = Mock()
        widget.save_file = Mock(
            side_effect=lambda: setattr(widget, "dirty", False)
        )
        with patch.object(
            QtWidgets.QMessageBox,
            "question",
            return_value=QtWidgets.QMessageBox.StandardButton.Save,
        ):
            self.assertTrue(label_widget.LabelingWidget.may_continue(widget))


if __name__ == "__main__":
    unittest.main()
