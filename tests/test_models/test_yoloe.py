import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

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


class TestYoloeTracking(unittest.TestCase):
    def test_update_tracker_uses_detection_indices(self):
        tracker = mock.Mock()
        tracker.update.return_value = np.array(
            [[10, 20, 30, 40, 7, 0.9, 1, 0]], dtype=np.float32
        )
        instance = SimpleNamespace(tracker=tracker)
        bboxes = np.array([[10, 20, 30, 40]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)
        class_ids = np.array([1], dtype=np.float32)
        image = mock.Mock()

        with mock.patch.object(
            yoloe, "xyxy2xywh", return_value="xywh"
        ) as convert:
            with mock.patch.object(yoloe.np, "asarray", return_value="frame"):
                result = yoloe.YOLOE._update_tracker(
                    instance, bboxes, scores, class_ids, image
                )

        convert.assert_called_once_with(bboxes)
        tracker.update.assert_called_once()
        args = tracker.update.call_args.args
        np.testing.assert_array_equal(args[0], scores)
        self.assertEqual(args[1], "xywh")
        np.testing.assert_array_equal(args[2], class_ids)
        self.assertEqual(args[3], "frame")
        np.testing.assert_array_equal(result, tracker.update.return_value)

    def test_reset_tracker_clears_temporal_identity(self):
        tracker = mock.Mock()
        instance = SimpleNamespace(tracker=tracker)

        yoloe.YOLOE.set_auto_labeling_reset_tracker(instance)

        tracker.reset.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
