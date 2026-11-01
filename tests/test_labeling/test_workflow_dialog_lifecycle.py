import os
import threading
import unittest
from contextlib import contextmanager, ExitStack
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtCore, QtTest, QtWidgets, sip

from anylabeling.views.labeling.widgets.chatbot_dialog import ChatbotDialog
from anylabeling.views.labeling.widgets.ppocr_dialog import PPOCRDialog
from anylabeling.views.labeling.widgets.video_classifier_dialog import (
    VideoClassifierDialog,
)
from anylabeling.views.labeling.widgets.vqa_dialog import VQADialog


class WorkflowDialogLifecycleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(
            []
        )

    @contextmanager
    def dialog(self, dialog_class):
        with ExitStack() as stack:
            for name in ("showEvent", "resizeEvent"):
                stack.enter_context(
                    mock.patch.object(
                        dialog_class, name, lambda self, event: None
                    )
                )
            stack.enter_context(
                mock.patch.object(
                    dialog_class, "eventFilter", lambda self, obj, event: False
                )
            )
            dialog = dialog_class.__new__(dialog_class)
            QtWidgets.QDialog.__init__(dialog)
            dialog._save_dirty_sidecar = mock.Mock(return_value=True)
            dialog.release_resources = mock.Mock()
            dialog.save_config = mock.Mock()
            dialog.cancel_event = threading.Event()
            dialog.stop_worker = mock.Mock()
            dialog.hideAllTooltips = mock.Mock()
            dialog.loading_timer = QtCore.QTimer(dialog)
            dialog.loading_timer.start(10000)
            dialog.show()
            self.app.processEvents()
            try:
                yield dialog
            finally:
                if not sip.isdeleted(dialog):
                    sip.delete(dialog)

    def exit_dialog(self, dialog, action):
        if action == "escape":
            QtTest.QTest.keyClick(dialog, QtCore.Qt.Key.Key_Escape)
        else:
            getattr(dialog, action)()

    def test_exit_runs_cleanup_once(self):
        for dialog_class in (
            VideoClassifierDialog,
            VQADialog,
            PPOCRDialog,
            ChatbotDialog,
        ):
            for action in ("escape", "close", "reject"):
                with (
                    self.subTest(dialog=dialog_class.__name__, action=action),
                    self.dialog(dialog_class) as dialog,
                ):
                    rejected = mock.Mock()
                    dialog.rejected.connect(rejected)
                    self.exit_dialog(dialog, action)
                    self.assertFalse(dialog.isVisible())
                    if dialog_class is VideoClassifierDialog:
                        dialog._save_dirty_sidecar.assert_called_once_with()
                        dialog.release_resources.assert_called_once_with()
                    elif dialog_class is VQADialog:
                        dialog.save_config.assert_called_once_with()
                    elif dialog_class is PPOCRDialog:
                        self.assertTrue(dialog.cancel_event.is_set())
                        dialog.stop_worker.assert_called_once_with(wait=False)
                    else:
                        dialog.hideAllTooltips.assert_called_once_with()
                        self.assertFalse(dialog.loading_timer.isActive())
                    if dialog_class is not VQADialog:
                        rejected.assert_called_once_with()

    def test_chatbot_escape_finishes_modal_execution(self):
        with self.dialog(ChatbotDialog) as dialog:
            QtCore.QTimer.singleShot(
                0, lambda: self.exit_dialog(dialog, "escape")
            )
            self.assertEqual(
                dialog.exec(), QtWidgets.QDialog.DialogCode.Rejected
            )
            self.assertFalse(dialog.isVisible())
            dialog.hideAllTooltips.assert_called_once_with()
            self.assertFalse(dialog.loading_timer.isActive())

    def test_video_save_failure_blocks_exit_and_allows_retry(self):
        for action in ("escape", "close", "reject"):
            with (
                self.subTest(action=action),
                self.dialog(VideoClassifierDialog) as dialog,
            ):
                dialog.setAttribute(
                    QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True
                )
                dialog._save_dirty_sidecar.return_value = False
                rejected = mock.Mock()
                dialog.rejected.connect(rejected)
                self.exit_dialog(dialog, action)
                self.app.sendPostedEvents(
                    None, QtCore.QEvent.Type.DeferredDelete
                )
                self.assertFalse(sip.isdeleted(dialog))
                self.assertTrue(dialog.isVisible())
                dialog.release_resources.assert_not_called()
                rejected.assert_not_called()
                dialog._save_dirty_sidecar.return_value = True
                self.exit_dialog(dialog, action)
                dialog.release_resources.assert_called_once_with()
                rejected.assert_called_once_with()
                self.app.sendPostedEvents(
                    None, QtCore.QEvent.Type.DeferredDelete
                )
                self.assertTrue(sip.isdeleted(dialog))

    def test_cached_dialogs_can_reopen_after_escape(self):
        for dialog_class in (VQADialog, PPOCRDialog):
            with (
                self.subTest(dialog=dialog_class.__name__),
                self.dialog(dialog_class) as dialog,
            ):
                self.exit_dialog(dialog, "escape")
                dialog.show()
                self.app.processEvents()
                self.assertTrue(dialog.isVisible())
                self.exit_dialog(dialog, "escape")
                self.assertFalse(dialog.isVisible())
                callback = (
                    dialog.save_config
                    if dialog_class is VQADialog
                    else dialog.stop_worker
                )
                self.assertEqual(callback.call_count, 2)
