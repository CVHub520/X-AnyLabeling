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

        button = toolbar.findChild(QtWidgets.QToolButton)
        self.assertIsNotNone(button)
        self.assertGreaterEqual(button.height(), button.iconSize().height() + 4)
        position = button.mapTo(toolbar, QtCore.QPoint())
        self.assertGreater(position.y(), 0)
        self.assertGreaterEqual(position.x(), 0)
        self.assertLessEqual(position.x() + button.width(), toolbar.width())

    def test_scaled_buttons_fit_after_actions_are_rebuilt(self):
        for mode in ("light", "dark"):
            init_theme(mode)
            toolbar = ToolBar("Tools")
            self._widgets.append(toolbar)
            toolbar.addAction(QtGui.QAction("Open", toolbar))
            toolbar.setUiScale(1.5)
            toolbar.clear()
            toolbar.addAction(QtGui.QAction("Save", toolbar))
            toolbar.show()
            self.app.processEvents()
            button = toolbar.findChildren(QtWidgets.QToolButton)[-1]
            self.assertEqual(button.size(), QtCore.QSize(42, 42))
            self.assertEqual(button.iconSize(), QtCore.QSize(36, 36))
            rect = QtCore.QRect(
                button.mapTo(toolbar, QtCore.QPoint()), button.size()
            )
            self.assertTrue(toolbar.rect().contains(rect))
            toolbar.setUiScale(1.0)
            self.app.processEvents()
            self.assertEqual(button.size(), QtCore.QSize(28, 28))
