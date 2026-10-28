import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    import cv2
    import numpy as np
    from PyQt6 import QtCore, QtGui, QtWidgets

    from anylabeling.views.labeling.widgets.canvas import Canvas

    PYQT_AVAILABLE = True
except Exception:
    PYQT_AVAILABLE = False


@unittest.skipUnless(
    PYQT_AVAILABLE, "PyQt6 is required for canvas magic wand tests"
)
class TestCanvasMagicWand(unittest.TestCase):

    def setUp(self):
        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication([])
        self.canvas = Canvas(parent=None)
        image = QtGui.QImage(40, 30, QtGui.QImage.Format.Format_RGB32)
        image.fill(QtGui.QColor(0, 0, 0))
        painter = QtGui.QPainter(image)
        painter.fillRect(1, 1, 14, 14, QtGui.QColor(100, 120, 100))
        painter.fillRect(2, 2, 12, 12, QtGui.QColor(100, 100, 100))
        painter.fillRect(25, 2, 10, 10, QtGui.QColor(100, 100, 100))
        painter.end()
        self.canvas.load_pixmap(QtGui.QPixmap.fromImage(image))
        self.canvas.set_editing(False)
        self.canvas.set_magic_wand_mode(True)

    def tearDown(self):
        self.canvas.close()
        self.app.processEvents()

    @staticmethod
    def mouse_event(event_type, pos, button, buttons):
        return QtGui.QMouseEvent(
            event_type,
            QtCore.QPointF(pos),
            QtCore.QPointF(pos),
            button,
            buttons,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )

    def test_flood_fill_selects_only_seed_connected_pixels(self):
        image = np.zeros((5, 8, 3), dtype=np.uint8)
        image[1:4, 1:3] = 100
        image[1:4, 5:7] = 100

        mask = self.canvas._compute_magic_wand_mask(image, (1, 2), 0)

        self.assertEqual(mask[2, 1], 255)
        self.assertEqual(mask[2, 5], 0)

    def test_perceptual_distance_tolerates_shading_not_color(self):
        image = np.zeros((3, 4, 3), dtype=np.uint8)
        image[:, 0] = (110, 110, 110)
        image[:, 1] = (180, 180, 180)
        image[:, 2] = (240, 240, 240)
        image[:, 3] = (120, 80, 40)

        mask = self.canvas._compute_magic_wand_mask(image, (1, 1), 15)

        self.assertEqual(mask[1, 0], 255)
        self.assertEqual(mask[1, 1], 255)
        self.assertEqual(mask[1, 2], 255)
        self.assertEqual(mask[1, 3], 0)

    def test_distance_and_masks_match_original_algorithm(self):
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, (60, 80, 3), dtype=np.uint8)
        for seed in ((0, 0), (40, 30), (79, 59)):
            for weight in (0.0, 0.5, 1.0):
                lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
                lab -= lab[seed[1], seed[0]].copy()
                lab[..., 0] *= (100.0 / 255.0) * weight
                np.square(lab, out=lab)
                expected_distance = np.sqrt(
                    np.sum(lab, axis=2, dtype=np.float32)
                )
                distance = self.canvas._compute_magic_wand_distance(
                    image, seed, weight
                )
                np.testing.assert_array_equal(distance, expected_distance)
                for threshold in (0, 15, 50, 100, 255):
                    candidates = (distance <= threshold).astype(np.uint8)
                    expected = np.zeros((62, 82), dtype=np.uint8)
                    cv2.floodFill(
                        candidates,
                        expected,
                        seed,
                        0,
                        0,
                        0,
                        4
                        | cv2.FLOODFILL_FIXED_RANGE
                        | cv2.FLOODFILL_MASK_ONLY
                        | (255 << 8),
                    )
                    actual = self.canvas._connected_magic_wand_mask(
                        distance, seed, threshold
                    )
                    np.testing.assert_array_equal(actual, expected[1:-1, 1:-1])
                np.testing.assert_array_equal(distance, expected_distance)

    def test_cropped_preview_preserves_image_coordinates(self):
        self.canvas._start_magic_wand(
            QtCore.QPointF(28, 5), QtCore.QPointF(28, 5)
        )
        expected = self.canvas._polylines_to_painter_path(
            self.canvas._mask_to_polylines(self.canvas._magic_wand_mask)
        )
        self.assertEqual(self.canvas._magic_wand_path, expected)

    def test_drag_increases_threshold_in_all_directions(self):
        self.canvas._start_magic_wand(
            QtCore.QPointF(5, 5), QtCore.QPointF(10, 10)
        )
        initial_size = np.count_nonzero(self.canvas._magic_wand_mask)

        self.canvas._drag_magic_wand(QtCore.QPointF(40, 10))
        self.assertEqual(self.canvas._magic_wand_threshold, 25)
        distance = self.canvas._magic_wand_distance
        self.assertGreater(
            np.count_nonzero(self.canvas._magic_wand_mask), initial_size
        )

        self.canvas._drag_magic_wand(QtCore.QPointF(-50, 10))
        self.assertEqual(self.canvas._magic_wand_threshold, 35)
        self.assertIs(self.canvas._magic_wand_distance, distance)

    def test_zero_threshold_changes_only_after_sensitivity_distance(self):
        self.canvas.magic_wand_default_threshold = 0
        self.canvas._start_magic_wand(
            QtCore.QPointF(5, 5), QtCore.QPointF(10, 10)
        )

        self.canvas._drag_magic_wand(QtCore.QPointF(12, 10))
        self.assertEqual(self.canvas._magic_wand_threshold, 0)

        self.canvas._drag_magic_wand(QtCore.QPointF(13, 10))
        self.assertEqual(self.canvas._magic_wand_threshold, 1)

    def test_cached_image_owns_its_memory(self):
        source = self.canvas._magic_wand_image()
        expected = source.copy()

        images = []
        for _ in range(4):
            image = QtGui.QImage(40, 30, QtGui.QImage.Format.Format_RGB888)
            image.fill(QtGui.QColor(255, 0, 0))
            images.append(image)

        self.assertTrue(source.flags.owndata)
        np.testing.assert_array_equal(source, expected)

    def test_finished_selection_does_not_leak_into_next_press(self):
        self.canvas._start_magic_wand(
            QtCore.QPointF(5, 5), QtCore.QPointF(5, 5)
        )
        self.assertEqual(np.count_nonzero(self.canvas._magic_wand_mask), 144)
        self.assertTrue(self.canvas._finish_magic_wand())

        images = []
        for _ in range(4):
            image = QtGui.QImage(40, 30, QtGui.QImage.Format.Format_RGB888)
            image.fill(QtGui.QColor(255, 0, 0))
            images.append(image)

        self.canvas._start_magic_wand(
            QtCore.QPointF(28, 5), QtCore.QPointF(28, 5)
        )

        self.assertTrue(self.canvas._magic_wand_active)
        self.assertEqual(self.canvas._magic_wand_seed, (28, 5))
        self.assertEqual(np.count_nonzero(self.canvas._magic_wand_mask), 100)
        self.assertEqual(self.canvas._magic_wand_mask[5, 5], 0)

    def test_finishing_selection_creates_polygon(self):
        created = []
        self.canvas.new_shape.connect(lambda: created.append(True))
        self.canvas._start_magic_wand(
            QtCore.QPointF(5, 5), QtCore.QPointF(5, 5)
        )

        finished = self.canvas._finish_magic_wand()

        self.assertTrue(finished)
        self.assertEqual(created, [True])
        self.assertEqual(len(self.canvas.shapes), 1)
        shape = self.canvas.shapes[0]
        self.assertEqual(shape.shape_type, "polygon")
        self.assertGreaterEqual(len(shape.points), 3)
        self.assertIsNone(shape.mask)

    def test_mouse_drag_requires_right_click_to_create_polygon(self):
        self.canvas.resize(40, 30)
        created = []
        mode_changes = []
        self.canvas.new_shape.connect(lambda: created.append(True))
        self.canvas.mode_changed.connect(lambda: mode_changes.append(True))

        self.canvas.mousePressEvent(
            self.mouse_event(
                QtCore.QEvent.Type.MouseButtonPress,
                QtCore.QPointF(5, 5),
                QtCore.Qt.MouseButton.LeftButton,
                QtCore.Qt.MouseButton.LeftButton,
            )
        )
        self.canvas.mouseMoveEvent(
            self.mouse_event(
                QtCore.QEvent.Type.MouseMove,
                QtCore.QPointF(20, 5),
                QtCore.Qt.MouseButton.NoButton,
                QtCore.Qt.MouseButton.LeftButton,
            )
        )
        self.assertEqual(self.canvas._magic_wand_threshold, 20)
        self.assertEqual(np.count_nonzero(self.canvas._magic_wand_mask), 196)
        self.canvas.mouseReleaseEvent(
            self.mouse_event(
                QtCore.QEvent.Type.MouseButtonRelease,
                QtCore.QPointF(20, 5),
                QtCore.Qt.MouseButton.LeftButton,
                QtCore.Qt.MouseButton.NoButton,
            )
        )

        self.assertTrue(self.canvas._magic_wand_active)
        self.assertEqual(len(self.canvas.shapes), 0)
        self.assertEqual(created, [])
        self.assertEqual(mode_changes, [])

        self.canvas.mouseMoveEvent(
            self.mouse_event(
                QtCore.QEvent.Type.MouseMove,
                QtCore.QPointF(35, 5),
                QtCore.Qt.MouseButton.NoButton,
                QtCore.Qt.MouseButton.NoButton,
            )
        )
        self.assertEqual(self.canvas._magic_wand_threshold, 20)

        self.canvas.mousePressEvent(
            self.mouse_event(
                QtCore.QEvent.Type.MouseButtonPress,
                QtCore.QPointF(35, 5),
                QtCore.Qt.MouseButton.RightButton,
                QtCore.Qt.MouseButton.RightButton,
            )
        )
        self.assertEqual(created, [])
        self.canvas.mouseReleaseEvent(
            self.mouse_event(
                QtCore.QEvent.Type.MouseButtonRelease,
                QtCore.QPointF(35, 5),
                QtCore.Qt.MouseButton.RightButton,
                QtCore.Qt.MouseButton.NoButton,
            )
        )

        self.assertFalse(self.canvas._magic_wand_active)
        self.assertEqual(len(self.canvas.shapes), 1)
        self.assertEqual(created, [True])
        self.assertEqual(mode_changes, [])
        self.assertTrue(self.canvas.drawing())
        self.assertTrue(self.canvas.is_magic_wand_mode)

        for button in (
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.MouseButton.RightButton,
        ):
            self.canvas.mousePressEvent(
                self.mouse_event(
                    QtCore.QEvent.Type.MouseButtonPress,
                    QtCore.QPointF(28, 5),
                    button,
                    button,
                )
            )
            self.canvas.mouseReleaseEvent(
                self.mouse_event(
                    QtCore.QEvent.Type.MouseButtonRelease,
                    QtCore.QPointF(28, 5),
                    button,
                    QtCore.Qt.MouseButton.NoButton,
                )
            )

        self.assertEqual(len(self.canvas.shapes), 2)
        self.assertEqual(created, [True, True])
        self.assertEqual(mode_changes, [])
        self.assertTrue(self.canvas.drawing())
        self.assertTrue(self.canvas.is_magic_wand_mode)
        self.assertFalse(self.canvas._magic_wand_active)
        self.assertEqual(self.canvas._magic_wand_threshold, 15)
        self.assertGreaterEqual(
            min(point.x() for point in self.canvas.shapes[-1].points), 25
        )

    def test_escape_cancels_preview(self):
        self.canvas.resize(40, 30)
        self.canvas.mousePressEvent(
            self.mouse_event(
                QtCore.QEvent.Type.MouseButtonPress,
                QtCore.QPointF(5, 5),
                QtCore.Qt.MouseButton.LeftButton,
                QtCore.Qt.MouseButton.LeftButton,
            )
        )
        self.canvas.mouseReleaseEvent(
            self.mouse_event(
                QtCore.QEvent.Type.MouseButtonRelease,
                QtCore.QPointF(5, 5),
                QtCore.Qt.MouseButton.LeftButton,
                QtCore.Qt.MouseButton.NoButton,
            )
        )
        self.assertIsNotNone(self.canvas._magic_wand_distance)
        self.assertEqual(self.canvas.shapes, [])
        event = QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress,
            QtCore.Qt.Key.Key_Escape,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )

        self.canvas.keyPressEvent(event)

        self.assertTrue(event.isAccepted())
        self.assertFalse(self.canvas._magic_wand_active)
        self.assertIsNone(self.canvas._magic_wand_distance)
        self.assertIsNone(self.canvas._magic_wand_mask)
        self.assertEqual(self.canvas.shapes, [])


if __name__ == "__main__":
    unittest.main()
