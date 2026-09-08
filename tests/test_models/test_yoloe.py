import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image

from anylabeling.services.auto_labeling import (
    _AUTO_LABELING_RESET_TRACKER_MODELS,
)
from anylabeling.services.auto_labeling import yoloe


class _FakeDetections:
    """Provide deterministic YOLOE detections for post-processing tests."""

    xyxy = np.asarray([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
    class_id = np.asarray([0, 1], dtype=np.int64)
    confidence = np.asarray([0.6, 0.9], dtype=np.float32)
    mask = np.asarray(
        [
            [[1, 0], [0, 0]],
            [[0, 0], [0, 1]],
        ],
        dtype=np.uint8,
    )

    def __getitem__(self, key: str) -> np.ndarray:
        """Return a named supervision data field.

        Args:
            key (str): Requested supervision field name.

        Returns:
            np.ndarray: Class names for the two fake detections.

        Raises:
            KeyError: If a field other than ``class_name`` is requested.
        """
        if key != "class_name":
            raise KeyError(key)
        return np.asarray(["person", "vehicle"])


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
    """Verify YOLOE tracking integration without loading model weights."""

    def test_yoloe_is_registered_for_tracker_reset(self) -> None:
        """Verify model-manager reset dispatch includes YOLOE."""
        self.assertIn("yoloe", _AUTO_LABELING_RESET_TRACKER_MODELS)

    def test_tracker_widgets_are_only_shown_when_configured(self) -> None:
        """Verify legacy YOLOE configurations retain their existing UI."""
        without_tracker = SimpleNamespace(
            Meta=yoloe.YOLOE.Meta,
            tracker=None,
        )
        with_tracker = SimpleNamespace(
            Meta=yoloe.YOLOE.Meta,
            tracker=mock.Mock(),
        )

        legacy_widgets = yoloe.YOLOE.get_required_widgets(without_tracker)
        tracking_widgets = yoloe.YOLOE.get_required_widgets(with_tracker)

        self.assertNotIn("button_reset_tracker", legacy_widgets)
        self.assertIn("button_reset_tracker", tracking_widgets)

    def test_tracker_resets_when_prompt_context_changes(self) -> None:
        """Verify identities reset whenever the active vocabulary changes."""
        tracker = mock.Mock()
        instance = SimpleNamespace(
            tracker=tracker,
            _active_tracker_context=None,
        )

        def reset_tracker() -> None:
            """Dispatch the reset method through the lightweight fixture."""
            yoloe.YOLOE.set_auto_labeling_reset_tracker(instance)

        instance.set_auto_labeling_reset_tracker = reset_tracker

        yoloe.YOLOE._activate_tracker_context(instance, "text", ["person"])
        yoloe.YOLOE._activate_tracker_context(instance, "text", ["person"])
        tracker.reset.assert_not_called()

        yoloe.YOLOE._activate_tracker_context(instance, "text", ["vehicle"])
        yoloe.YOLOE._activate_tracker_context(instance, "prompt_free")
        yoloe.YOLOE._activate_tracker_context(instance, "visual")

        self.assertEqual(tracker.reset.call_count, 3)

    def test_postprocess_preserves_mask_detection_index_and_group_id(
        self,
    ) -> None:
        """Verify write-back preserves masks and IDs in detection order."""
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
                from_ultralytics=mock.Mock(return_value=_FakeDetections())
            )
        )
        converted_masks: list[np.ndarray] = []

        def fake_mask_to_polygons(mask: np.ndarray) -> list[np.ndarray]:
            """Record a mask and return a minimal valid polygon.

            Args:
                mask (np.ndarray): Instance mask selected by detection index.

            Returns:
                list[np.ndarray]: One triangular polygon.
            """
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
            [shape.label for shape in shapes], ["person", "vehicle"]
        )
        self.assertEqual([shape.group_id for shape in shapes], [17, 42])
        np.testing.assert_array_equal(
            converted_masks[0], _FakeDetections.mask[0]
        )
        np.testing.assert_array_equal(
            converted_masks[1], _FakeDetections.mask[1]
        )

    def test_partial_tracker_output_keeps_unconfirmed_new_targets(
        self,
    ) -> None:
        """One confirmed track plus one new target must both stay visible."""
        tracks = np.asarray(
            [[0, 0, 10, 10, 17, 0.6, 0, 0]],
            dtype=np.float32,
        )
        instance = SimpleNamespace(
            tracker=mock.Mock(),
            output_mode="rectangle",
            with_mask=True,
            _update_tracker=mock.Mock(return_value=tracks),
        )
        supervision = SimpleNamespace(
            Detections=SimpleNamespace(
                from_ultralytics=mock.Mock(return_value=_FakeDetections())
            )
        )
        converted_masks: list[np.ndarray] = []

        def fake_mask_to_polygons(mask: np.ndarray) -> list[np.ndarray]:
            """Record a mask and return a minimal valid polygon.

            Args:
                mask (np.ndarray): Instance mask selected by detection index.

            Returns:
                list[np.ndarray]: One triangular polygon.
            """
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

        rectangles = shapes[:2]
        polygons = shapes[2:]
        self.assertEqual(
            [shape.label for shape in rectangles], ["person", "vehicle"]
        )
        self.assertEqual(
            [shape.group_id for shape in shapes], [17, None, 17, None]
        )
        self.assertEqual(
            [shape.label for shape in polygons], ["person", "vehicle"]
        )
        np.testing.assert_array_equal(
            converted_masks[0], _FakeDetections.mask[0]
        )
        np.testing.assert_array_equal(
            converted_masks[1], _FakeDetections.mask[1]
        )

    def test_empty_tracker_output_preserves_raw_detections(self) -> None:
        """Verify an empty tracker result never drops detector annotations."""
        instance = SimpleNamespace(
            tracker=mock.Mock(),
            output_mode="rectangle",
            with_mask=True,
            _update_tracker=mock.Mock(
                return_value=np.empty((0, 8), dtype=np.float32)
            ),
        )
        supervision = SimpleNamespace(
            Detections=SimpleNamespace(
                from_ultralytics=mock.Mock(return_value=_FakeDetections())
            )
        )
        converted_masks: list[np.ndarray] = []

        def fake_mask_to_polygons(mask: np.ndarray) -> list[np.ndarray]:
            """Record a raw mask and return a minimal valid polygon.

            Args:
                mask (np.ndarray): Raw detector mask.

            Returns:
                list[np.ndarray]: One triangular polygon.
            """
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

        rectangles = shapes[:2]
        polygons = shapes[2:]
        self.assertEqual(
            [shape.label for shape in rectangles], ["person", "vehicle"]
        )
        self.assertEqual([shape.group_id for shape in shapes], [None] * 4)
        self.assertEqual(
            [(point.x(), point.y()) for point in rectangles[0].points],
            [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
        )
        self.assertEqual(
            [shape.label for shape in polygons], ["person", "vehicle"]
        )
        np.testing.assert_array_equal(
            converted_masks[0], _FakeDetections.mask[0]
        )
        np.testing.assert_array_equal(
            converted_masks[1], _FakeDetections.mask[1]
        )

    def test_update_tracker_converts_pil_rgb_to_bgr(self) -> None:
        """Verify GMC receives the BGR channel order expected by OpenCV."""
        tracker = mock.Mock()
        tracker.update.return_value = np.array(
            [[10, 20, 30, 40, 7, 0.9, 1, 0]], dtype=np.float32
        )
        instance = SimpleNamespace(tracker=tracker)
        bboxes = np.array([[10, 20, 30, 40]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)
        class_ids = np.array([1], dtype=np.float32)
        image = Image.new("RGB", (2, 2), color=(10, 20, 30))

        with mock.patch.object(
            yoloe, "xyxy2xywh", return_value="xywh"
        ) as convert:
            result = yoloe.YOLOE._update_tracker(
                instance, bboxes, scores, class_ids, image
            )

        convert.assert_called_once_with(bboxes)
        tracker.update.assert_called_once()
        args = tracker.update.call_args.args
        np.testing.assert_array_equal(args[0], scores)
        self.assertEqual(args[1], "xywh")
        np.testing.assert_array_equal(args[2], class_ids)
        np.testing.assert_array_equal(
            args[3], np.full((2, 2, 3), [30, 20, 10], dtype=np.uint8)
        )
        self.assertTrue(args[3].flags.c_contiguous)
        np.testing.assert_array_equal(result, tracker.update.return_value)

    def test_tracktrack_keeps_id_across_two_cpu_frames(self) -> None:
        """Verify real TrackTrack association keeps one ID for two frames."""
        tracker = yoloe.YOLOE._build_tracker(
            {
                "tracker_type": "tracktrack",
                "track_high_thresh": 0.5,
                "track_low_thresh": 0.1,
                "new_track_thresh": 0.6,
                "track_buffer": 30,
                "match_thresh": 0.7,
                "lost_match_thr": 0.0,
                "iou_weight": 0.5,
                "conf_weight": 0.1,
                "angle_weight": 0.05,
                "penalty_p": 0.2,
                "reduce_step": 0.05,
                "tai_thr": 0.55,
                "min_track_len": 1,
                "gmc_method": "none",
            }
        )
        instance = SimpleNamespace(tracker=tracker)
        scores = np.asarray([0.9], dtype=np.float32)
        class_ids = np.asarray([0], dtype=np.int64)
        frame = Image.new("RGB", (32, 32), color=(0, 0, 0))

        first = yoloe.YOLOE._update_tracker(
            instance,
            np.asarray([[10, 10, 20, 20]], dtype=np.float32),
            scores,
            class_ids,
            frame,
        )
        second = yoloe.YOLOE._update_tracker(
            instance,
            np.asarray([[11, 10, 21, 20]], dtype=np.float32),
            scores,
            class_ids,
            frame,
        )

        self.assertEqual(first.shape, (1, 8))
        self.assertEqual(second.shape, (1, 8))
        self.assertEqual(int(first[0, 4]), int(second[0, 4]))
        self.assertEqual(int(first[0, 7]), 0)
        self.assertEqual(int(second[0, 7]), 0)

    def test_reset_tracker_clears_temporal_identity(self) -> None:
        """Verify the reset control clears all tracker state."""
        tracker = mock.Mock()
        instance = SimpleNamespace(tracker=tracker)

        yoloe.YOLOE.set_auto_labeling_reset_tracker(instance)

        tracker.reset.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
