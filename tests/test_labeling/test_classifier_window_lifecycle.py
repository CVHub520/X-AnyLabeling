import os
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtWidgets

from anylabeling.views.labeling.label_widget import LabelingWidget
from anylabeling.views.labeling.widgets.classifier_dialog import (
    ClassifierDialog,
)


class ClassifierWindowLifecycleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(
            []
        )

    def setUp(self):
        self.main = QtWidgets.QWidget()
        self.parent = QtWidgets.QWidget(self.main)
        self.parent.image_list = ["sample.png"]
        self.main.show()
        self.app.processEvents()
        self.addCleanup(self.main.close)
        self.addCleanup(self.main.deleteLater)

    def test_classifier_keeps_main_window_visible_and_is_reused(self):
        with (
            mock.patch.object(ClassifierDialog, "init_ui", lambda self: None),
            mock.patch.object(
                ClassifierDialog, "load_initial_data", lambda self: None
            ),
            mock.patch.object(
                ClassifierDialog, "setup_shortcuts", lambda self: None
            ),
            mock.patch.object(
                ClassifierDialog, "showEvent", lambda self, event: None
            ),
        ):
            LabelingWidget.open_image_classifier(self.parent)
            dialog = self.parent.image_classifier_window
            self.assertTrue(self.main.isVisible())
            self.assertTrue(dialog.isVisible())
            self.assertFalse(dialog.isModal())
            self.assertIs(dialog.parent(), self.parent)
            LabelingWidget.open_image_classifier(self.parent)
            self.assertIs(dialog, self.parent.image_classifier_window)
            dialog.close()
            self.assertTrue(self.main.isVisible())
            LabelingWidget.open_image_classifier(self.parent)
            self.assertIs(dialog, self.parent.image_classifier_window)
            self.assertTrue(dialog.isVisible())
            dialog.close()

    def test_constructor_failure_keeps_main_window_visible(self):
        with mock.patch(
            "anylabeling.views.labeling.label_widget.ClassifierDialog",
            side_effect=RuntimeError("initialization failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "initialization failed"):
                LabelingWidget.open_image_classifier(self.parent)
        self.assertTrue(self.main.isVisible())
