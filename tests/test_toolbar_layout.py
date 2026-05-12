import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6 import QtCore, QtGui, QtWidgets

    import anylabeling.resources.resources  # noqa: F401
    from anylabeling.views.labeling.utils.qt import new_icon
    from anylabeling.views.labeling.utils.theme import init_theme
    from anylabeling.views.labeling.widgets.toolbar import ToolBar

    PYQT_AVAILABLE = True
except Exception:
    PYQT_AVAILABLE = False


@unittest.skipUnless(PYQT_AVAILABLE, "PyQt6 is required for toolbar tests")
class TestToolBarLayout(unittest.TestCase):

    def setUp(self):
        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication([])
        self._widgets = []

    def tearDown(self):
        for widget in self._widgets:
            widget.close()
        init_theme("light")
        self.app.processEvents()

    def test_first_tool_button_stays_inside_vertical_toolbar(self):
        init_theme("dark")
        toolbar = ToolBar("Tools")
        self._widgets.append(toolbar)
        toolbar.setOrientation(QtCore.Qt.Orientation.Vertical)
        toolbar.setToolButtonStyle(
            QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly
        )
        toolbar.setIconSize(QtCore.QSize(24, 24))
        toolbar.setMaximumWidth(40)

        action = QtGui.QAction("Open Dir", toolbar)
        action.setIcon(new_icon("open"))
        toolbar.addAction(action)

        toolbar.resize(40, 80)
        toolbar.show()
        self.app.processEvents()

        button = toolbar.widgetForAction(action)
        self.assertIsNotNone(button)
        self.assertGreaterEqual(button.height(), toolbar.iconSize().height() + 4)
        self.assertGreater(button.geometry().top(), 0)
        self.assertGreaterEqual(button.geometry().left(), 0)
        self.assertLessEqual(button.geometry().right(), toolbar.width())
