from types import SimpleNamespace
from unittest.mock import Mock, patch

from PyQt6.QtWidgets import QMessageBox

from anylabeling.views.labeling.widgets.classifier_dialog import (
    ClassifierDialog,
)


def test_ai_suggestions_validate_and_overwrite_existing_values():
    shape = SimpleNamespace(label="car", attributes={"color": "red"})
    owner = SimpleNamespace(
        filename="image.jpg",
        canvas=SimpleNamespace(shapes=[shape]),
        attributes={
            "car": {"color": ["red", "blue"], "type": ["car"], "note": ""}
        },
        attribute_widget_types={"car": {"note": "lineedit"}},
    )
    form = SimpleNamespace(
        set_value=lambda s, n, v: s.attributes.update({n: v})
    )
    dialog = SimpleNamespace(
        loading_msg=Mock(),
        label_widget=owner,
        multitask_form=form,
        object_combo=Mock(currentIndex=Mock(return_value=1)),
        select_object=Mock(),
        tr=lambda s: s,
    )
    schema = {"color": ["red", "blue"], "type": ["car"], "note": "text"}
    with patch.object(
        QMessageBox,
        "information",
    ) as information:
        ClassifierDialog.handle_multitask_ai_result(
            dialog,
            '{"color":"blue","type":"truck","note":"visible","extra":"x"}',
            True,
            "",
            "image.jpg",
            shape,
            schema,
        )
    assert shape.attributes == {"color": "blue", "note": "visible"}
    assert all(
        name in information.call_args.args[2] for name in ("type", "extra")
    )
    dialog.select_object.assert_called_once()


def test_ai_suggestions_reject_stale_object():
    shape = SimpleNamespace(label="car", attributes={})
    dialog = SimpleNamespace(
        loading_msg=Mock(),
        tr=lambda s: s,
        label_widget=SimpleNamespace(
            filename="other.jpg", canvas=SimpleNamespace(shapes=[shape])
        ),
    )
    with patch.object(QMessageBox, "warning") as warning:
        ClassifierDialog.handle_multitask_ai_result(
            dialog,
            '{"color":"red"}',
            True,
            "",
            "image.jpg",
            shape,
            {"color": ["red"]},
        )
    warning.assert_called_once()
    assert not shape.attributes


def test_ai_worker_reports_unreachable_service():
    import requests

    from anylabeling.views.labeling.vqa.utils import AIWorkerThread

    worker = AIWorkerThread("test", "", {})
    worker.load_models_config = Mock(
        return_value={"settings": {"provider": "ollama", "model_id": "vl"}}
    )
    worker.load_providers_config = Mock(
        return_value={
            "ollama": {
                "api_address": "http://localhost:11434/v1",
                "api_key": "ollama",
            }
        }
    )
    worker.call_openai_api = Mock(side_effect=requests.ConnectionError())
    result = []
    worker.finished.connect(lambda *args: result.append(args))
    worker.run()
    assert not result[0][1]
    assert "localhost:11434" in result[0][2]
    assert "Chatbot" in result[0][2]


def test_ai_loading_dialog_uses_normal_window():
    from PyQt6 import QtCore, QtWidgets
    from anylabeling.views.labeling.vqa.dialogs import AILoadingDialog

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    dialog = AILoadingDialog()
    assert dialog.graphicsEffect() is None
    assert not dialog.windowFlags() & QtCore.Qt.WindowType.FramelessWindowHint
    assert "background-color" in dialog.styleSheet()
    dialog.close()


def test_multitask_ai_dialog_model_selection_and_prompt():
    from PyQt6 import QtWidgets
    from anylabeling.views.labeling.widgets.classifier_dialog import (
        MultiTaskAIDialog,
    )
    from anylabeling.views.labeling.vqa.utils import AIWorkerThread

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    parent = QtWidgets.QWidget()
    config = {
        "settings": {"provider": "ollama", "model_id": "first"},
        "models_data": {"ollama": {"first": {}, "second": {}}},
    }
    with (
        patch.object(
            AIWorkerThread, "load_models_config", return_value=config
        ),
        patch(
            "anylabeling.views.labeling.widgets.classifier_dialog.save_model_selection"
        ) as save,
    ):
        dialog = MultiTaskAIDialog(parent, "@image describe")
        assert dialog.models.currentData() == ("ollama", "first")
        assert dialog.generate.isEnabled()
        dialog.model_search.setText("second")
        assert dialog.models.view().isRowHidden(1)
        assert not dialog.models.view().isRowHidden(2)
        assert dialog.models.currentData() == ("ollama", "first")
        dialog.model_search.clear()
        dialog.models.setCurrentIndex(2)
        save.assert_called_once_with("ollama", "second")
        dialog.prompt_input.clear()
        assert not dialog.generate.isEnabled()
        dialog.close()
    parent.close()


def test_multitask_ai_dialog_without_model():
    from PyQt6 import QtWidgets
    from anylabeling.views.labeling.widgets.classifier_dialog import (
        MultiTaskAIDialog,
    )
    from anylabeling.views.labeling.vqa.utils import AIWorkerThread

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    parent = QtWidgets.QWidget()
    with patch.object(AIWorkerThread, "load_models_config", return_value=None):
        dialog = MultiTaskAIDialog(parent, "@image describe")
        assert dialog.models.currentData() is None
        assert not dialog.generate.isEnabled()
        dialog.close()
    parent.close()


def test_generate_warning_remembers_only_confirmed_choice():
    from PyQt6 import QtWidgets
    from anylabeling.views.labeling.widgets.classifier_dialog import (
        MultiTaskAIDialog,
        OverwriteConfirmationDialog,
    )
    from anylabeling.views.labeling.vqa.utils import AIWorkerThread

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    parent = QtWidgets.QWidget()
    settings = Mock()
    settings.value.return_value = False
    parent.label_widget = SimpleNamespace(settings=settings)
    with patch.object(AIWorkerThread, "load_models_config", return_value=None):
        dialog = MultiTaskAIDialog(parent, "@image")
        dialog.accept = Mock()

        def cancel(message):
            message.checkbox.setChecked(True)
            return QtWidgets.QDialog.DialogCode.Rejected

        with patch.object(OverwriteConfirmationDialog, "exec", cancel):
            dialog.confirm_generate()
        settings.setValue.assert_not_called()
        dialog.accept.assert_not_called()

        def confirm(message):
            message.checkbox.setChecked(True)
            return QtWidgets.QDialog.DialogCode.Accepted

        with patch.object(OverwriteConfirmationDialog, "exec", confirm):
            dialog.confirm_generate()
        settings.setValue.assert_called_once_with(
            "classifier/multitask_ai_hide_overwrite_warning", True
        )
        dialog.accept.assert_called_once()
        dialog.close()
    parent.close()


def test_auto_run_image_range_is_inclusive_and_uses_current_image():
    from PyQt6.QtWidgets import QDialog
    from anylabeling.views.labeling.widgets.classifier_dialog import (
        ClassifierDialog,
    )

    owner = SimpleNamespace(
        image_files=["one.jpg", "two.jpg", "three.jpg", "four.jpg"],
        label_widget=SimpleNamespace(filename="two.jpg"),
    )
    range_dialog = SimpleNamespace(
        exec=Mock(return_value=QDialog.DialogCode.Accepted),
        from_input=Mock(value=Mock(return_value=2)),
        to_input=Mock(value=Mock(return_value=3)),
    )
    with patch(
        "anylabeling.views.labeling.utils.batch.BatchRangeDialog",
        return_value=range_dialog,
    ) as factory:
        paths = ClassifierDialog.select_batch_image_range(owner)
    factory.assert_called_once_with(4, 2, parent=owner)
    assert paths == ["two.jpg", "three.jpg"]
    range_dialog.exec.return_value = QDialog.DialogCode.Rejected
    with patch(
        "anylabeling.views.labeling.utils.batch.BatchRangeDialog",
        return_value=range_dialog,
    ):
        assert ClassifierDialog.select_batch_image_range(owner) is None


def test_batch_ai_dialog_scrollbar_and_button_spacing():
    from PyQt6 import QtWidgets
    from anylabeling.views.labeling.vqa.utils import AIWorkerThread
    from anylabeling.views.labeling.widgets.classifier_dialog import (
        MultiTaskAIDialog,
    )

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    parent = QtWidgets.QWidget()
    with patch.object(AIWorkerThread, "load_models_config", return_value=None):
        dialog = MultiTaskAIDialog(parent, "long prompt\n" * 30, batch=True)
        dialog.resize(420, 300)
        dialog.show()
        app.processEvents()
        cancel = next(
            button
            for button in dialog.findChildren(QtWidgets.QPushButton)
            if button.text() == "Cancel"
        )
        assert cancel.geometry().right() < dialog.generate.geometry().left()
        assert (
            "QScrollBar::handle:vertical"
            in dialog.prompt_input.verticalScrollBar().styleSheet()
        )
        dialog.close()
    parent.close()
