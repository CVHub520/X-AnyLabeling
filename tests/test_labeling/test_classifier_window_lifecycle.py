import os
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtCore, QtTest, QtWidgets

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

    def test_dialog_exit_restores_main_window(self):
        for action in ("escape", "close", "reject", "accept"):
            with self.subTest(action=action):
                observed = []
                dialogs = []

                def exit_dialog():
                    dialog = self.parent.findChild(ClassifierDialog)
                    dialogs.append(dialog)
                    observed.append(
                        (self.main.isVisible(), dialog.isVisible())
                    )
                    if action == "escape":
                        QtTest.QTest.keyClick(dialog, QtCore.Qt.Key.Key_Escape)
                    else:
                        getattr(dialog, action)()

                with (
                    mock.patch.object(
                        ClassifierDialog, "init_ui", lambda self: None
                    ),
                    mock.patch.object(
                        ClassifierDialog,
                        "load_initial_data",
                        lambda self: None,
                    ),
                    mock.patch.object(
                        ClassifierDialog, "setup_shortcuts", lambda self: None
                    ),
                    mock.patch.object(
                        ClassifierDialog, "showEvent", lambda self, event: None
                    ),
                ):
                    QtCore.QTimer.singleShot(0, exit_dialog)
                    LabelingWidget.open_image_classifier(self.parent)

                self.assertEqual(observed, [(False, True)])
                self.assertTrue(self.main.isVisible())
                dialog = dialogs[0]
                self.assertFalse(dialog.isVisible())
                dialog.setParent(None)
                dialog.deleteLater()

    def test_constructor_failure_restores_main_window(self):
        with mock.patch(
            "anylabeling.views.labeling.label_widget.ClassifierDialog",
            side_effect=RuntimeError("initialization failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "initialization failed"):
                LabelingWidget.open_image_classifier(self.parent)
        self.assertTrue(self.main.isVisible())

    def test_execution_failure_restores_main_window(self):
        with mock.patch(
            "anylabeling.views.labeling.label_widget.ClassifierDialog"
        ) as factory:
            factory.return_value.exec.side_effect = RuntimeError(
                "execution failed"
            )
            with self.assertRaisesRegex(RuntimeError, "execution failed"):
                LabelingWidget.open_image_classifier(self.parent)
        self.assertTrue(self.main.isVisible())
