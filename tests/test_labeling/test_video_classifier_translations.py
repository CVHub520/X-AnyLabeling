from PyQt6.QtCore import QTranslator

from anylabeling.resources import resources  # noqa: F401


def test_video_classifier_embedded_translations():
    sources = [
        "Show shortcuts",
        "Open tutorial",
        "Play/Pause",
        "Previous/Next frame",
        "Back/Forward 1 second",
        "Previous/Next segment",
        "Mark in/out",
        "Create segment from marks",
        "Delete selected segment",
        "Save annotations",
        "Undo",
        "Redo",
        "Select label by number",
        "Shortcuts",
        "Key",
        "Action",
        "Segment {n}",
    ]
    for language in ("en_US", "zh_CN", "ja_JP", "ko_KR"):
        translator = QTranslator()
        assert translator.load(f":/languages/translations/{language}.qm")
        for source in sources:
            translated = translator.translate("VideoClassifierDialog", source)
            assert translated, (language, source)
            if "{n}" in source:
                assert "{n}" in translated
