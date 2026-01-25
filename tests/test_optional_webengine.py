from anylabeling.views.labeling.chatbot import chat as chat_module
from anylabeling.views.labeling.widgets import about_dialog as about_module


def test_webengine_optional_import_paths():
    assert hasattr(chat_module, "QWebEngineView")
    assert hasattr(about_module, "QWebEngineView")
    assert about_module.QWebEngineView is None or about_module.QWebEngineView
    assert chat_module.QWebEngineView is None or chat_module.QWebEngineView

