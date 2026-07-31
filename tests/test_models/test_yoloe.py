import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from anylabeling.services.auto_labeling import (
    _AUTO_LABELING_RESET_TRACKER_MODELS,
)
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
    def test_yoloe_is_registered_for_tracker_reset(self):
        self.assertIn("yoloe", _AUTO_LABELING_RESET_TRACKER_MODELS)

    def test_tracker_resets_when_prompt_context_changes(self):
        tracker = mock.Mock()
        instance = SimpleNamespace(
            tracker=tracker,
            _active_tracker_context=None,
        )
        instance.set_auto_labeling_reset_tracker = lambda: (
            yoloe.YOLOE.set_auto_labeling_reset_tracker(instance)
        )

        yoloe.YOLOE._activate_tracker_context(instance, "text", ["person"])
        yoloe.YOLOE._activate_tracker_context(instance, "text", ["person"])
        tracker.reset.assert_not_called()

        yoloe.YOLOE._activate_tracker_context(instance, "text", ["vehicle"])
        yoloe.YOLOE._activate_tracker_context(instance, "prompt_free")
        yoloe.YOLOE._activate_tracker_context(instance, "visual")

        self.assertEqual(tracker.reset.call_count, 3)

    def test_postprocess_preserves_mask_detection_index_and_group_id(self):
        class FakeDetections:
            """Two detections whose tracker order is deliberately reversed."""

            xyxy = np.asarray(
                [[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32
            )
            class_id = np.asarray([0, 1], dtype=np.int64)
            confidence = np.asarray([0.6, 0.9], dtype=np.float32)
            mask = np.asarray(
                [
                    [[1, 0], [0, 0]],
                    [[0, 0], [0, 1]],
                ],
                dtype=np.uint8,
            )

            def __getitem__(self, key):
                """Return class names for the supervision data field."""
                if key != "class_name":
                    raise KeyError(key)
                return np.asarray(["person", "vehicle"])

        tracks = np.asarray(
            [
                [20, 20, 30, 30, 42, 0.9, 1, 1],
                [0, 0, 10, 10, 17, 0.6, 0, 0],
            ],
            dtype=np.float32,
        )
        instance = SimpleNamespace(
            tracker=mock.Mock(),
            output_mode="polygon",
            with_mask=True,
            _update_tracker=mock.Mock(return_value=tracks),
        )
        supervision = SimpleNamespace(
            Detections=SimpleNamespace(
                from_ultralytics=mock.Mock(return_value=FakeDetections())
            )
        )
        converted_masks = []

        def fake_mask_to_polygons(mask):
            converted_masks.append(mask.copy())
            return [np.asarray([[0, 0], [1, 0], [1, 1]])]

        with (
            mock.patch.object(yoloe, "sv", supervision, create=True),
            mock.patch.object(
                yoloe,
                "mask_to_polygons",
                side_effect=fake_mask_to_polygons,
                create=True,
            ),
        ):
            shapes = yoloe.YOLOE.postprocess(
                instance, [object()], image=object()
            )

        self.assertEqual(
            [shape.label for shape in shapes], ["vehicle", "person"]
        )
        self.assertEqual([shape.group_id for shape in shapes], [42, 17])
        np.testing.assert_array_equal(
            converted_masks[0], FakeDetections.mask[1]
        )
        np.testing.assert_array_equal(
            converted_masks[1], FakeDetections.mask[0]
        )

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
