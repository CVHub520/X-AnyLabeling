"""Locale behavior of the newly added ellipse setting."""

from pathlib import Path
from types import SimpleNamespace

from PyQt6.QtCore import QTranslator

from anylabeling.views.labeling.settings.dialog import SettingsDialog
from anylabeling.views.labeling.settings.schema import SETTING_FIELD_MAP


def test_new_setting_title_uses_compiled_chinese_translation() -> None:
    """Localize the new ellipse title and retain existing title semantics."""
    translator = QTranslator()
    resource = (
        Path(__file__).parents[2]
        / "anylabeling/resources/translations/zh_CN.qm"
    )
    assert translator.load(str(resource))
    widget = SimpleNamespace(
        tr=lambda text: translator.translate("SettingsDialog", text) or text
    )
    title = SettingsDialog._display_field_title(
        widget, SETTING_FIELD_MAP["canvas.ellipse.polygon_points"], None
    )
    assert title == "椭圆拟合边数"
    legacy = SETTING_FIELD_MAP["canvas.epsilon"]
    assert (
        SettingsDialog._display_field_title(widget, legacy, None)
        == legacy.label
    )
