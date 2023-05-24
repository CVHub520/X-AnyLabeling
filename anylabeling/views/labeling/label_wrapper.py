"""This module defines labeling wrapper and related functions"""

from PyQt5.QtWidgets import QVBoxLayout, QWidget

from .label_widget import LabelingWidget


class LabelingWrapper(QWidget):
    """Wrapper widget for labeling module"""

    def __init__(
        self,
        parent,
        config=None,
        filename=None,
        output=None,
        output_file=None,
        output_dir=None,
    ):
        super().__init__()
        self.parent = parent

        # Create a labeling widget
        view = LabelingWidget(
            self,
            config=config,
            filename=filename,
            output=output,
            output_file=output_file,
            output_dir=output_dir,
        )

        # Create the main layout and put labeling into
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(view)
        self.setLayout(main_layout)
