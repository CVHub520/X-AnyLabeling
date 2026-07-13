import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6 import QtCore, QtWidgets

    from anylabeling.views.labeling.settings.runtime_applier import (
        SettingsRuntimeApplier,
    )
    from anylabeling.views.labeling.utils.upload import (
        upload_shape_attrs_file,
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

    def test_repeated_upload_keeps_auto_switch_signal_state_consistent(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", encoding="utf-8"
        ) as file:
            json.dump({"car": {"color": ["red", "blue"]}}, file)
            file.flush()

            for initially_enabled in (False, True):
                with self.subTest(initially_enabled=initially_enabled):
                    canvas = _Canvas()
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
