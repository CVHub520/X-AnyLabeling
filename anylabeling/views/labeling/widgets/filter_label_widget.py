from PyQt5.QtWidgets import QWidget, QHBoxLayout, QComboBox


class LabelFilterComboBox(QWidget):
    def __init__(self, parent=None, items=[]):
        super(LabelFilterComboBox, self).__init__(parent)
        self.items = items
        self.combo_box = QComboBox()
        self.combo_box.addItems(self.items)
        self.combo_box.currentIndexChanged.connect(
            parent.combo_selection_changed
        )

        layout = QHBoxLayout()
        layout.addWidget(self.combo_box)
        self.setLayout(layout)

    def update_items(self, items):
        self.items = items
        self.combo_box.clear()
        self.combo_box.addItems(self.items)
