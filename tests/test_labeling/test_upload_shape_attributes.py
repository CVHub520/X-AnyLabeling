import json
import os
import tempfile
import unittest
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6 import QtCore, QtWidgets

    from anylabeling.views.labeling.settings.runtime_applier import (
        SettingsRuntimeApplier,
    )
    from anylabeling.views.labeling.label_widget import LabelingWidget
    from anylabeling.views.labeling.utils.upload import (
        upload_shape_attrs_file,
        validate_shape_attributes_config,
    )

    PYQT_AVAILABLE = True
except Exception:
    PYQT_AVAILABLE = False


class _Canvas(QtCore.QObject if PYQT_AVAILABLE else object):
    if PYQT_AVAILABLE:
        mode_changed = QtCore.pyqtSignal()


@unittest.skipUnless(
    PYQT_AVAILABLE, "PyQt6 is required for shape attribute upload tests"
)
class TestUploadShapeAttributes(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance()
        if cls.app is None:
            cls.app = QtWidgets.QApplication([])

    def test_validate_shape_attributes_config_accepts_supported_widgets(self):
        config = {
            "car": {
                "color": ["red", "blue"],
                "visibility": ["low", "high"],
                "vehicle_id": "",
                "occluded_by": [],
            },
            "__widget_types__": {
                "car": {
                    "visibility": "radiobutton",
                    "vehicle_id": "lineedit",
                    "occluded_by": "group_id",
                }
            },
        }

        attributes, widget_types = validate_shape_attributes_config(config)

        self.assertEqual(attributes, {"car": config["car"]})
        self.assertEqual(widget_types, config["__widget_types__"])

    def test_validate_shape_attributes_config_rejects_invalid_structures(self):
        cases = [
            ("root", [], ["label='<root>'", "actual_type='list'"]),
            (
                "label mapping",
                {"car": []},
                ["label='car'", "property='<all>'"],
            ),
            (
                "widget mapping",
                {"car": {}, "__widget_types__": []},
                ["label='__widget_types__'", "actual_type='list'"],
            ),
            (
                "unknown widget label",
                {"car": {}, "__widget_types__": {"truck": {}}},
                ["label='truck'", "expected='an existing attribute label'"],
            ),
            (
                "unknown widget property",
                {
                    "car": {"color": ["red"]},
                    "__widget_types__": {
                        "car": {"visibility": "radiobutton"}
                    },
                },
                [
                    "label='car'",
                    "property='visibility'",
                    "widget_type='radiobutton'",
                ],
            ),
            (
                "unknown widget type",
                {
                    "car": {"color": ["red"]},
                    "__widget_types__": {"car": {"color": "slider"}},
                },
                ["property='color'", "widget_type='slider'"],
            ),
            (
                "empty combobox",
                {"car": {"color": []}},
                ["widget_type='combobox'", "actual_type='list'"],
            ),
            (
                "non-string combobox option",
                {"car": {"color": ["red", 1]}},
                ["property='color'", "expected='a non-empty list of strings'"],
            ),
            (
                "empty radiobutton",
                {
                    "car": {"visibility": []},
                    "__widget_types__": {
                        "car": {"visibility": "radiobutton"}
                    },
                },
                ["property='visibility'", "widget_type='radiobutton'"],
            ),
            (
                "invalid lineedit",
                {
                    "car": {"vehicle_id": []},
                    "__widget_types__": {
                        "car": {"vehicle_id": "lineedit"}
                    },
                },
                ["widget_type='lineedit'", "expected='a string'"],
            ),
            (
                "invalid group_id",
                {
                    "car": {"occluded_by": ["1"]},
                    "__widget_types__": {
                        "car": {"occluded_by": "group_id"}
                    },
                },
                ["widget_type='group_id'", "expected='an empty list'"],
            ),
        ]

        for name, config, expected_fragments in cases:
            with self.subTest(name=name):
                with self.assertRaises(ValueError) as context:
                    validate_shape_attributes_config(config)

                message = str(context.exception)
                for fragment in expected_fragments:
                    self.assertIn(fragment, message)

    def test_invalid_upload_preserves_active_configuration(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", encoding="utf-8"
        ) as file:
            json.dump({"car": {"color": []}}, file)
            file.flush()
            attributes = {"person": {"age": ["adult", "child"]}}
            widget_types = {"person": {"age": "combobox"}}
            unique_label_list = SimpleNamespace(
                find_items_by_label=Mock(),
                create_item_from_label=Mock(),
                addItem=Mock(),
                set_item_label=Mock(),
            )
            canvas = SimpleNamespace(h_shape_is_hovered=True)
            runtime_applier = SimpleNamespace(
                set_auto_switch_to_edit_mode=Mock()
            )
            widget = SimpleNamespace(
                tr=lambda text: text,
                attributes=attributes,
                attribute_widget_types=widget_types,
                unique_label_list=unique_label_list,
                shape_attributes=SimpleNamespace(show=Mock()),
                scroll_area=SimpleNamespace(show=Mock()),
                canvas=canvas,
                _settings_runtime_applier=runtime_applier,
            )

            with patch.object(
                QtWidgets.QFileDialog,
                "getOpenFileName",
                return_value=(file.name, ""),
            ), patch(
                "anylabeling.views.labeling.utils.upload.Popup"
            ) as popup, patch(
                "anylabeling.views.labeling.utils.upload.logger.error"
            ) as log_error:
                upload_shape_attrs_file(widget, 128)

            self.assertIs(widget.attributes, attributes)
            self.assertIs(widget.attribute_widget_types, widget_types)
            unique_label_list.find_items_by_label.assert_not_called()
            widget.shape_attributes.show.assert_not_called()
            widget.scroll_area.show.assert_not_called()
            self.assertTrue(canvas.h_shape_is_hovered)
            runtime_applier.set_auto_switch_to_edit_mode.assert_not_called()
            log_error.assert_called_once()
            popup.return_value.show_popup.assert_called_once()
            self.assertNotIn(
                "popup_height",
                popup.return_value.show_popup.call_args.kwargs,
            )

    def test_valid_upload_rebuilds_open_attribute_panel(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", encoding="utf-8"
        ) as file:
            json.dump({"car": {"size": ["small", "large"]}}, file)
            file.flush()
            shape = SimpleNamespace(
                label="car",
                group_id=None,
                attributes={"color": "red"},
            )
            canvas = SimpleNamespace(
                shapes=[shape],
                selected_shapes=[shape],
                current=None,
                h_shape_is_hovered=True,
                update=Mock(),
            )
            widget = SimpleNamespace(
                tr=lambda text: text,
                attributes={"car": {"color": ["red", "blue"]}},
                attribute_widget_types={},
                unique_label_list=SimpleNamespace(
                    find_items_by_label=Mock(return_value=[object()])
                ),
                shape_attributes=SimpleNamespace(show=Mock()),
                scroll_area=QtWidgets.QScrollArea(),
                canvas=canvas,
                attribute_selection_changed=Mock(),
                attribute_radio_changed=Mock(),
                attribute_line_changed=Mock(),
                save_attributes=Mock(),
                show_attributes_panel=Mock(),
                hide_attributes_panel=Mock(),
                _settings_runtime_applier=SimpleNamespace(
                    set_auto_switch_to_edit_mode=Mock()
                ),
            )
            widget.update_attributes = MethodType(
                LabelingWidget.update_attributes, widget
            )
            LabelingWidget.update_attributes(widget, 0)
            old_container = widget.grid_layout_container

            with patch.object(
                QtWidgets.QFileDialog,
                "getOpenFileName",
                return_value=(file.name, ""),
            ), patch("anylabeling.views.labeling.utils.upload.Popup"):
                upload_shape_attrs_file(widget, 128)

            self.assertIsNot(widget.grid_layout_container, old_container)
            combo = widget.grid_layout_container.findChild(
                QtWidgets.QComboBox
            )
            self.assertEqual(
                [combo.itemText(index) for index in range(combo.count())],
                ["small", "large"],
            )
            self.assertEqual(shape.attributes["size"], "small")
            widget.save_attributes.assert_called_once_with([shape])
            widget.hide_attributes_panel.assert_not_called()

    def test_repeated_upload_keeps_auto_switch_signal_state_consistent(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", encoding="utf-8"
        ) as file:
            json.dump({"car": {"color": ["red", "blue"]}}, file)
            file.flush()

            for initially_enabled in (False, True):
                with self.subTest(initially_enabled=initially_enabled):
                    canvas = _Canvas()
                    canvas.shapes = []
                    canvas.selected_shapes = []
                    set_edit_mode = Mock()
                    widget = SimpleNamespace(
                        tr=lambda text: text,
                        canvas=canvas,
                        set_edit_mode=set_edit_mode,
                        _auto_switch_signal_connected=False,
                        _config={
                            "auto_switch_to_edit_mode": initially_enabled
                        },
                        unique_label_list=SimpleNamespace(
                            find_items_by_label=Mock(
                                return_value=[object()]
                            )
                        ),
                        shape_attributes=SimpleNamespace(show=Mock()),
                        scroll_area=SimpleNamespace(show=Mock()),
                        hide_attributes_panel=Mock(),
                    )
                    applier = SettingsRuntimeApplier(widget)
                    widget._settings_runtime_applier = applier
                    applier.set_auto_switch_to_edit_mode(initially_enabled)

                    with patch.object(
                        QtWidgets.QFileDialog,
                        "getOpenFileName",
                        return_value=(file.name, ""),
                    ), patch(
                        "anylabeling.views.labeling.utils.upload.Popup"
                    ) as popup:
                        upload_shape_attrs_file(widget, 128)
                        upload_shape_attrs_file(widget, 128)

                    calls = popup.return_value.show_popup.call_args_list
                    self.assertEqual(len(calls), 2)
                    self.assertTrue(
                        all(
                            call.kwargs.get("popup_height") == 65
                            for call in calls
                        )
                    )
                    self.assertFalse(widget._auto_switch_signal_connected)
                    self.assertEqual(
                        widget._config["auto_switch_to_edit_mode"],
                        initially_enabled,
                    )
                    canvas.mode_changed.emit()
                    set_edit_mode.assert_not_called()

                    applier.set_auto_switch_to_edit_mode(True)
                    canvas.mode_changed.emit()

                    set_edit_mode.assert_called_once_with()
