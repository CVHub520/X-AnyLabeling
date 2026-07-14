import ssl

import pytest

from anylabeling.services.auto_labeling import model as auto_model
from anylabeling.services.auto_labeling import model_manager as manager_module
from anylabeling.services.auto_labeling.model import Model
from anylabeling.services.auto_labeling.model_manager import ModelManager


class DummyModel(Model):
    def predict_shapes(self, image, filename=None):
        return None

    def unload(self):
        pass


class FakeResponse:
    headers = {"Content-Length": "6"}

    def __init__(self):
        self.chunks = [b"secure", b""]

    def read(self, _):
        return self.chunks.pop(0)


def test_download_with_retry_keeps_tls_certificate_verification(
    tmp_path, monkeypatch
):
    captured_contexts = []

    def fake_urlopen(req, timeout, context=None):
        captured_contexts.append(context)
        if context is not None:
            assert context.check_hostname
            assert context.verify_mode == ssl.CERT_REQUIRED
        return FakeResponse()

    monkeypatch.setattr(auto_model.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(auto_model, "get_config", lambda: {})

    model = DummyModel({}, lambda _: None)
    model.MAX_RETRIES = 1
    dest_path = tmp_path / "model.onnx"

    assert model.download_with_retry(
        "https://example.com/model.onnx", str(dest_path), None
    )
    assert dest_path.read_bytes() == b"secure"
    assert captured_contexts


def test_download_with_retry_rejects_part_path_symlink(
    tmp_path, monkeypatch
):
    model_directory = tmp_path / "models"
    model_directory.mkdir()
    outside_file = tmp_path / "outside.part"
    outside_file.write_bytes(b"keep")
    dest_path = model_directory / "model.onnx"
    dest_path.with_suffix(".onnx.part").symlink_to(outside_file)
    monkeypatch.setattr(auto_model, "get_config", lambda: {})
    model = DummyModel({}, lambda _: None)

    with pytest.raises(ValueError, match="model directory"):
        model.download_with_retry(
            "https://example.com/model.onnx",
            str(dest_path),
            None,
            model_directory=str(model_directory),
        )

    assert outside_file.read_bytes() == b"keep"


@pytest.mark.parametrize(
    "name",
    [
        "../../escape",
        "nested/model",
        r"nested\model",
        "/absolute/path",
        "model name",
        "..",
    ],
)
def test_load_custom_model_rejects_invalid_name(
    tmp_path, monkeypatch, name
):
    config_file = tmp_path / "model.yaml"
    config_file.write_text(
        f"type: yolov8\nname: {name!r}\ndisplay_name: Test\n",
        encoding="utf-8",
    )
    saved_configs = []
    monkeypatch.setattr(
        ModelManager, "load_model_configs", lambda self: None
    )
    monkeypatch.setattr(manager_module, "get_config", lambda: {})
    monkeypatch.setattr(manager_module, "save_config", saved_configs.append)

    manager = ModelManager()

    assert not manager.load_custom_model(str(config_file))
    assert not saved_configs


def test_get_model_abs_path_rejects_path_traversal(
    tmp_path, monkeypatch
):
    outside_file = tmp_path / "outside" / "model.onnx"
    outside_file.parent.mkdir()
    outside_file.write_bytes(b"keep")
    monkeypatch.setattr(auto_model, "get_work_directory", lambda: str(tmp_path))
    monkeypatch.setattr(auto_model, "get_config", lambda: {})
    model = DummyModel({}, lambda _: None)
    download_called = False

    def download(*args, **kwargs):
        nonlocal download_called
        download_called = True

    monkeypatch.setattr(model, "download_with_retry", download)

    with pytest.raises(ValueError, match="model directory"):
        model.get_model_abs_path(
            {
                "name": "../../../outside",
                "model_path": "https://example.com/model.onnx",
            },
            "model_path",
        )

    assert outside_file.read_bytes() == b"keep"
    assert not download_called
