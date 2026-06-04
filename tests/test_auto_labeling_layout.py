import os
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6 import QtCore, QtWidgets, uic

    import anylabeling.resources.resources  # noqa: F401
    from anylabeling import config
    from anylabeling.config import get_config
    from anylabeling.views.labeling.utils.style import (
        get_model_selection_scroll_area_style,
    )
    from anylabeling.views.labeling.widgets.auto_labeling.auto_labeling import (
        AutoLabelingWidget,
        update_model_selection_scroll_area_height,
    )
    from anylabeling.views.labeling.widgets.searchable_model_dropdown import (
        SearchableModelDropdownPopup,
    )

    PYQT_AVAILABLE = True
except Exception:
    PYQT_AVAILABLE = False


@unittest.skipUnless(
    PYQT_AVAILABLE, "PyQt6 is required for auto labeling layout tests"
)
class TestAutoLabelingLayout(unittest.TestCase):
    def setUp(self):
        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication([])
        self._widgets = []

    def tearDown(self):
        for widget in self._widgets:
            widget.close()
        self.app.processEvents()

    def test_model_selection_uses_horizontal_scroll_area(self):
        form = QtWidgets.QWidget()
        self._widgets.append(form)
        ui_path = (
            Path(__file__).resolve().parents[1]
            / "anylabeling/views/labeling/widgets/auto_labeling/auto_labeling.ui"
        )

        uic.loadUi(str(ui_path), form)

        scroll_area = form.findChild(
            QtWidgets.QScrollArea, "model_selection_scroll_area"
        )
        scroll_area.setStyleSheet(get_model_selection_scroll_area_style())
        container = form.findChild(
            QtWidgets.QWidget, "model_selection_container"
        )

        self.assertIsNotNone(scroll_area)
        self.assertIs(scroll_area.widget(), container)
        self.assertTrue(scroll_area.widgetResizable())
        self.assertEqual(
            scroll_area.horizontalScrollBarPolicy(),
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded,
        )
        self.assertEqual(
            scroll_area.verticalScrollBarPolicy(),
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
        )
        self.assertIsNotNone(container.layout())
        spacer = next(
            (
                container.layout().itemAt(index).spacerItem()
                for index in range(container.layout().count())
                if container.layout().itemAt(index).spacerItem() is not None
            ),
            None,
        )
        self.assertIsNotNone(spacer)
        self.assertEqual(
            spacer.sizePolicy().horizontalPolicy(),
            QtWidgets.QSizePolicy.Policy.Expanding,
        )

        slider = form.findChild(QtWidgets.QSlider, "mask_fineness_slider")
        self.assertGreaterEqual(slider.minimumWidth(), 120)

        form.resize(5000, 100)
        form.show()
        self.app.processEvents()
        update_model_selection_scroll_area_height(scroll_area)

        self.assertEqual(scroll_area.horizontalScrollBar().maximum(), 0)
        self.assertEqual(scroll_area.height(), container.sizeHint().height())

        form.resize(320, 100)
        self.app.processEvents()
        update_model_selection_scroll_area_height(scroll_area)

        self.assertGreater(scroll_area.horizontalScrollBar().maximum(), 0)
        self.assertEqual(
            scroll_area.horizontalScrollBar().sizeHint().height(), 16
        )
        self.assertEqual(
            scroll_area.height(),
            container.sizeHint().height()
            + scroll_area.horizontalScrollBar().sizeHint().height(),
        )

    def test_default_hidden_controls_do_not_stretch_buttons(self):
        form = QtWidgets.QWidget()
        self._widgets.append(form)
        ui_path = (
            Path(__file__).resolve().parents[1]
            / "anylabeling/views/labeling/widgets/auto_labeling/auto_labeling.ui"
        )
        uic.loadUi(str(ui_path), form)

        hidden_widget_names = (
            "button_run",
            "button_add_point",
            "button_remove_point",
            "button_add_rect",
            "add_pos_rect",
            "add_neg_rect",
            "button_run_rect",
            "button_clear",
            "button_finish_object",
            "button_send",
            "edit_text",
            "edit_conf",
            "edit_iou",
            "input_box_thres",
            "input_conf",
            "input_iou",
            "output_label",
            "output_select_combobox",
            "toggle_preserve_existing_annotations",
            "button_set_api_token",
            "button_classes_filter",
            "button_reset_tracker",
            "upn_select_combobox",
            "gd_select_combobox",
            "florence2_select_combobox",
            "remote_server_select_combobox",
            "remote_task_select_combobox",
            "button_auto_decode",
            "button_cropping",
            "button_skip_detection",
            "mask_fineness_slider",
            "mask_fineness_value_label",
        )
        for widget_name in hidden_widget_names:
            getattr(form, widget_name).hide()

        form.resize(1600, 100)
        form.show()
        self.app.processEvents()

        self.assertLessEqual(
            form.model_selection_button.width(),
            form.model_selection_button.sizeHint().width() + 2,
        )
        self.assertLessEqual(
            form.button_close.width(), form.button_close.sizeHint().width() + 2
        )

    def test_initial_show_reflows_model_selection_row(self):
        config.current_config_file = "anylabeling/configs/xanylabeling_config.yaml"
        parent = type(
            "Parent",
            (),
            {
                "_config": get_config(),
                "new_shapes_from_auto_labeling": lambda _self, _result: None,
            },
        )()
        root = QtWidgets.QWidget()
        self._widgets.append(root)
        layout = QtWidgets.QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QtWidgets.QLabel("Mode"))
        layout.addSpacing(5)
        widget = AutoLabelingWidget(parent)
        widget.hide()
        layout.addWidget(widget)
        layout.addWidget(QtWidgets.QFrame(), 1)

        root.resize(1600, 300)
        root.show()
        self.app.processEvents()
        widget.show()
        self.app.processEvents()
        self.app.processEvents()

        button_top = widget.model_selection_button.mapTo(
            widget, widget.model_selection_button.rect().topLeft()
        ).y()
        scroll_top = widget.model_selection_scroll_area.geometry().top()
        button_bottom = (
            button_top + widget.model_selection_button.geometry().height()
        )
        status_top = widget.model_status_label.geometry().top()

        self.assertEqual(button_top, scroll_top)
        self.assertLessEqual(button_bottom, status_top)

    def test_model_dropdown_search_matches_display_names(self):
        dropdown = SearchableModelDropdownPopup(
            {
                "Meta": {
                    "sam2_hiera_base_video-r20240901": {
                        "display_name": "Segment Anything 2 Video (Base)"
                    },
                    "sam2_hiera_base-r20240801": {
                        "display_name": "Segment Anything 2.1 (Base)"
                    },
                    "sam_hq_vit_b-r20231111": {
                        "display_name": "SAM-HQ (ViT-Base)"
                    },
                }
            }
        )
        self._widgets.append(dropdown)
        dropdown.show()
        self.app.processEvents()

        dropdown.filter_models("seg")
        self.app.processEvents()

        visible_names = [
            item.display_name
            for item in dropdown.model_items.values()
            if item.isVisible()
        ]

        self.assertIn("Segment Anything 2 Video (Base)", visible_names)
        self.assertIn("Segment Anything 2.1 (Base)", visible_names)
        self.assertNotIn("SAM-HQ (ViT-Base)", visible_names)
