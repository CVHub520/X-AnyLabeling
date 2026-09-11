import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6 import QtWidgets

    from anylabeling.views.labeling.widgets.pixel_edge_widget import (
        PixelEdgeWidget,
    )

    PYQT_AVAILABLE = True
except Exception:
    PYQT_AVAILABLE = False


@unittest.skipUnless(PYQT_AVAILABLE, "PyQt6 is required")
class TestPixelEdgeWidget(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(
            []
        )

    def test_threshold_mode_exposes_only_the_relevant_editor(self):
        panel = PixelEdgeWidget({"threshold_mode": "auto"})

        self.assertFalse(panel.threshold.isEnabled())
        self.assertTrue(panel.adjustment.isEnabled())

        panel.threshold_mode.setCurrentIndex(
            panel.threshold_mode.findData("manual")
        )

        self.assertTrue(panel.threshold.isEnabled())
        self.assertFalse(panel.adjustment.isEnabled())

    def test_candidate_confirmation_is_explicit(self):
        panel = PixelEdgeWidget()

        self.assertFalse(panel.confirm_button.isEnabled())
        panel.set_pending(True)
        self.assertTrue(panel.confirm_button.isEnabled())
        self.assertTrue(panel.cancel_button.isEnabled())

    def test_ctrl_enter_emits_preview_confirmation(self):
        panel = PixelEdgeWidget()
        confirmed = []
        panel.confirm_requested.connect(lambda: confirmed.append(True))

        panel.update_confirm_shortcut("Alt+Return")
        panel.set_pending(True)
        panel.confirm_button.click()

        self.assertEqual(confirmed, [True])
        self.assertIn("Alt+", panel.confirm_button.text())

    def test_close_discards_preview_and_emits_close(self):
        panel = PixelEdgeWidget()
        cancelled = []
        closed = []
        panel.cancel_requested.connect(lambda: cancelled.append(True))
        panel.close_requested.connect(lambda: closed.append(True))
        panel.show()

        panel.close_button.click()

        self.assertEqual(cancelled, [True])
        self.assertEqual(closed, [True])
        self.assertFalse(panel.isVisible())

    def test_panel_settings_do_not_include_target_polarity(self):
        panel = PixelEdgeWidget()

        values = panel.settings()

        self.assertNotIn("polarity", values)
        self.assertIn("threshold", values)
        self.assertIn("threshold_adjustment", values)

    def test_distance_controls_explicitly_use_original_image_pixels(self):
        panel = PixelEdgeWidget()
        labels = [
            label.text() for label in panel.findChildren(QtWidgets.QLabel)
        ]

        self.assertIn("点间距(px)", labels)
        self.assertIn("搜索半径(px)", labels)
        self.assertIn("像素格直角走线", labels)
        self.assertIn("原图像素", panel.point_spacing.toolTip())
        self.assertIn("原图像素", panel.search_radius.toolTip())

    def test_model_postprocess_explains_automatic_half_pixel_validation(self):
        panel = PixelEdgeWidget()

        self.assertIn("像素直角贴边", panel.model_refine.text())
        self.assertIn("关闭", panel.model_refine.toolTip())
        self.assertIn("原有结果接收流程", panel.model_refine.toolTip())

    def test_multi_object_continuous_and_gap_controls_are_persisted(self):
        panel = PixelEdgeWidget(
            {
                "annotate_all_in_box": True,
                "continuous_box": True,
                "gap_bridge_max": 5,
                "gap_bridge_ratio": 0.3,
            }
        )

        values = panel.settings()

        self.assertTrue(values["annotate_all_in_box"])
        self.assertTrue(values["continuous_box"])
        self.assertEqual(values["gap_bridge_max"], 5)
        self.assertAlmostEqual(values["gap_bridge_ratio"], 0.3)
        self.assertIn(
            "修补面积(%)",
            labels := [
                label.text() for label in panel.findChildren(QtWidgets.QLabel)
            ],
        )
        self.assertIn("四角", panel.rectangle_button.text())


if __name__ == "__main__":
    unittest.main()
