import os

import pytest
from PyQt6.QtWidgets import QApplication

from anylabeling.views.labeling.widgets.navigator_widget import NavigatorWidget


def test_navigator_preserves_error_fallback_and_propagates_interrupts():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    widget = NavigatorWidget()

    class BrokenPath:
        def __str__(self):
            raise RuntimeError("unavailable image path")

    class InterruptedPath:
        def __str__(self):
            raise KeyboardInterrupt()

    widget.set_image(BrokenPath())
    with pytest.raises(KeyboardInterrupt):
        widget.set_image(InterruptedPath())
    widget.close()
    assert app is not None
