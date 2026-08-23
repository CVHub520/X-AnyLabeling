import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from PIL import Image

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


class TestYoloeCrossImageVisualPrompt(unittest.TestCase):
    @staticmethod
    def make_model(embedding_dimension=512):
        parameter = torch.nn.Parameter(torch.zeros(1))
        inner_model = mock.Mock()
        inner_model.parameters.side_effect = lambda: iter([parameter])
        inner_model.model = [SimpleNamespace(embed=embedding_dimension)]
        return SimpleNamespace(
            model=inner_model,
            predictor=object(),
            predict=mock.Mock(return_value=["prediction"]),
            set_classes=mock.Mock(),
        )

    @staticmethod
    def make_instance(model=None):
        instance = yoloe.YOLOE.__new__(yoloe.YOLOE)
        instance.config = {
            "type": "yoloe",
            "name": "unit-test-yoloe",
            "model_path": __file__,
        }
        instance.input_shape = (640, 640)
        instance.conf_thres = 0.25
        instance.iou_thres = 0.70
        instance.replace = True
        instance.marks = []
        instance._cross_image_visual_model = model
        instance.visual_prompt_vpe = None
        instance.visual_prompt_classes = []
        instance.visual_prompt_reference = None
        instance.visual_prompt_model_signature = None
        instance.visual_prompt_profile_name = None
        instance.visual_prompt_state = yoloe.VisualPromptState.NO_VISUAL_PROMPT
        return instance

    def test_reference_box_validation(self):
        boxes = yoloe.YOLOE.validate_reference_boxes(
            [[1, 2, 10, 20]], (100, 80)
        )
        self.assertEqual(boxes.dtype, np.float32)
        np.testing.assert_array_equal(boxes, [[1, 2, 10, 20]])

        invalid_boxes = (
            [],
            [[1, 2, 1, 20]],
            [[-1, 2, 10, 20]],
            [[1, 2, 101, 20]],
            [[1, 2, np.nan, 20]],
        )
        for invalid in invalid_boxes:
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    yoloe.YOLOE.validate_reference_boxes(invalid, (100, 80))

    def test_marks_update_visual_prompt_state(self):
        instance = self.make_instance()

        instance.set_auto_labeling_marks(
            [{"type": "rectangle", "data": [1, 2, 10, 20]}]
        )
        self.assertEqual(
            instance.visual_prompt_state,
            yoloe.VisualPromptState.REFERENCE_MARKS_READY,
        )

        instance.set_auto_labeling_marks([])
        self.assertEqual(
            instance.visual_prompt_state,
            yoloe.VisualPromptState.NO_VISUAL_PROMPT,
        )

    def test_set_visual_prompt_validates_and_resets_predictor(self):
        model = self.make_model()
        instance = self.make_instance(model)
        source_vpe = torch.ones(1, 1, 512, requires_grad=True)

        instance.set_visual_prompt(
            source_vpe,
            ["object"],
            {"image_path": "reference.jpg"},
        )

        self.assertTrue(instance.has_visual_prompt())
        self.assertTrue(instance.visual_prompt_ready)
        self.assertIsNone(model.predictor)
        self.assertFalse(instance.visual_prompt_vpe.requires_grad)
        self.assertIsNot(instance.visual_prompt_vpe, source_vpe)
        model.set_classes.assert_called_once_with(
            ["object"], instance.visual_prompt_vpe
        )
        self.assertEqual(
            instance.get_visual_prompt_state(), "VISUAL_PROMPT_READY"
        )
        self.assertEqual(
            instance.visual_prompt_model_signature["embedding_dimension"],
            512,
        )

    def test_set_visual_prompt_rejects_incompatible_embeddings(self):
        instance = self.make_instance(self.make_model())

        invalid_embeddings = (
            torch.ones(1, 512),
            torch.ones(2, 1, 512),
            torch.ones(1, 2, 512),
            torch.ones(1, 1, 256),
            torch.full((1, 1, 512), torch.nan),
        )
        for embedding in invalid_embeddings:
            with self.subTest(shape=embedding.shape):
                with self.assertRaises(ValueError):
                    instance.set_visual_prompt(embedding, ["object"])

        with self.assertRaises(TypeError):
            instance.set_visual_prompt(torch.ones(1, 1, 512), "object")

    def test_build_visual_prompt_resets_existing_predictor(self):
        model = self.make_model()
        instance = self.make_instance(model)
        instance.marks = [
            {"type": "rectangle", "data": [1, 2, 10, 20], "label": 1}
        ]
        generated_vpe = torch.ones(1, 1, 512)

        def generate_vpe(**kwargs):
            self.assertIsNone(model.predictor)
            self.assertIs(kwargs["predictor"], yoloe.YOLOEVPSegPredictor)
            self.assertTrue(kwargs["return_vpe"])
            model.predictor = SimpleNamespace(vpe=generated_vpe)
            return ["reference prediction"]

        model.predict.side_effect = generate_vpe
        vpe = instance.build_visual_prompt(Image.new("RGB", (100, 80)))

        self.assertTrue(torch.equal(vpe, generated_vpe))
        self.assertIsNone(model.predictor)
        self.assertEqual(instance.marks, [])
        self.assertEqual(
            instance.visual_prompt_reference["boxes"],
            [[1.0, 2.0, 10.0, 20.0]],
        )

    def test_target_prediction_keeps_cached_prompt_ready(self):
        model = self.make_model()
        instance = self.make_instance(model)
        instance.postprocess = mock.Mock(return_value=["shape"])
        instance.set_visual_prompt(torch.ones(1, 1, 512), ["object"])
        cached_vpe = instance.visual_prompt_vpe

        result = instance.predict_with_visual_prompt(
            Image.new("RGB", (100, 80))
        )

        self.assertEqual(result.shapes, ["shape"])
        self.assertIs(instance.visual_prompt_vpe, cached_vpe)
        self.assertTrue(instance.has_visual_prompt())
        self.assertEqual(
            instance.visual_prompt_state,
            yoloe.VisualPromptState.VISUAL_PROMPT_READY,
        )

    def test_clear_visual_prompt_does_not_clear_other_mode_models(self):
        cross_model = self.make_model()
        instance = self.make_instance(cross_model)
        instance._text_model = object()
        instance._visual_model = object()
        instance._prompt_free_model = object()
        instance.set_visual_prompt(torch.ones(1, 1, 512), ["object"])

        instance.clear_visual_prompt()

        self.assertFalse(instance.has_visual_prompt())
        self.assertIsNone(instance._cross_image_visual_model)
        self.assertIsNotNone(instance._text_model)
        self.assertIsNotNone(instance._visual_model)
        self.assertIsNotNone(instance._prompt_free_model)
        self.assertEqual(
            instance.visual_prompt_state,
            yoloe.VisualPromptState.NO_VISUAL_PROMPT,
        )

    def test_profile_round_trip_restores_embedding_to_model_device(self):
        model = self.make_model()
        instance = self.make_instance(model)
        instance.set_visual_prompt(
            torch.ones(1, 1, 512),
            ["object"],
            {
                "image_path": "reference.jpg",
                "image_size": [100, 80],
                "boxes": [[1.0, 2.0, 10.0, 20.0]],
                "instance_count": 1,
                "classes": ["object"],
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            instance.save_visual_prompt_profile("product_A", directory)
            instance.clear_visual_prompt()
            instance._cross_image_visual_model = model

            profile = instance.load_visual_prompt_profile(directory)

        self.assertEqual(profile.name, "product_A")
        self.assertEqual(instance.visual_prompt_profile_name, "product_A")
        self.assertEqual(instance.visual_prompt_vpe.device.type, "cpu")
        self.assertEqual(instance.visual_prompt_vpe.dtype, torch.float32)
        self.assertTrue(instance.has_visual_prompt())

    def test_incompatible_profile_does_not_replace_cached_prompt(self):
        model = self.make_model()
        instance = self.make_instance(model)
        instance.set_visual_prompt(
            torch.ones(1, 1, 512),
            ["object"],
            {"boxes": [[1.0, 2.0, 10.0, 20.0]], "instance_count": 1},
        )
        cached_vpe = instance.visual_prompt_vpe
        profile = SimpleNamespace(
            vpe=np.ones((1, 1, 512), dtype=np.float32),
            validate_compatibility=mock.Mock(
                side_effect=ValueError("incompatible model")
            ),
        )

        with mock.patch.object(
            yoloe.VisualPromptProfile, "load", return_value=profile
        ):
            with self.assertRaisesRegex(ValueError, "incompatible model"):
                instance.load_visual_prompt_profile("metadata.json")

        self.assertIs(instance.visual_prompt_vpe, cached_vpe)
        self.assertTrue(instance.has_visual_prompt())
        model.set_classes.assert_called_once()


if __name__ == "__main__":
    unittest.main()
