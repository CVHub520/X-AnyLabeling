import sys
import unittest
from types import SimpleNamespace
from unittest import mock

from anylabeling.services.auto_labeling import yoloe


class TestYoloeEmbeddingModel(unittest.TestCase):
    def test_text_encoder_uses_explicit_checkpoint_path(self):
        clip_model = mock.Mock()
        mobileclip = SimpleNamespace(
            create_model_and_transforms=mock.Mock(
                return_value=(clip_model, None, None)
            ),
            get_tokenizer=mock.Mock(return_value=mock.Mock()),
        )
        checkpoint = "/models/mobileclip_blt.pt"

        with mock.patch.dict(sys.modules, {"mobileclip": mobileclip}):
            yoloe._MobileCLIPTextEncoder(checkpoint, "cpu")

        mobileclip.create_model_and_transforms.assert_called_once_with(
            "mobileclip_b", pretrained=checkpoint, device="cpu"
        )

    def test_text_embeddings_use_instance_encoder(self):
        parameter = SimpleNamespace(device="cpu")
        inner_model = mock.Mock()
        inner_model.parameters.return_value = iter([parameter])
        inner_model.get_text_pe.return_value = "embeddings"
        model = SimpleNamespace(model=inner_model)
        instance = SimpleNamespace(
            _text_encoder=None,
            config={"embedding_model_path": "/models/mobileclip_blt.pt"},
        )
        encoder = mock.Mock()

        with mock.patch.object(
            yoloe, "_MobileCLIPTextEncoder", return_value=encoder
        ) as encoder_cls:
            result = yoloe.YOLOE._get_text_pe(instance, model, ["cat"])

        encoder_cls.assert_called_once_with("/models/mobileclip_blt.pt", "cpu")
        self.assertIs(inner_model.clip_model, encoder)
        inner_model.get_text_pe.assert_called_once_with(
            ["cat"], cache_clip_model=True
        )
        self.assertEqual(result, "embeddings")

    def test_prompt_free_vocab_uses_instance_encoder(self):
        head = SimpleNamespace(cv3=[["layer", "first"], ["layer", "second"]])
        inner_model = SimpleNamespace(
            model=[head],
            set_classes=mock.Mock(),
            fuse=mock.Mock(),
        )
        model = SimpleNamespace(model=inner_model)
        instance = SimpleNamespace(
            _get_text_pe=mock.Mock(return_value="embeddings")
        )

        torch = SimpleNamespace(
            nn=SimpleNamespace(ModuleList=lambda modules: list(modules))
        )
        with mock.patch.object(
            yoloe,
            "torch",
            torch,
            create=True,
        ):
            vocab = yoloe.YOLOE._get_vocab(instance, model, ["cat"])

        instance._get_text_pe.assert_called_once_with(model, ["cat"])
        inner_model.set_classes.assert_called_once_with(["cat"], "embeddings")
        inner_model.fuse.assert_called_once_with()
        self.assertEqual(vocab, ["first", "second"])


if __name__ == "__main__":
    unittest.main()
