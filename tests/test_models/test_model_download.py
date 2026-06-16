import ssl

from anylabeling.services.auto_labeling import model as auto_model
from anylabeling.services.auto_labeling.model import Model


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
