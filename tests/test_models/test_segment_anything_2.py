from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import yaml

from anylabeling.services.auto_labeling.__base__.sam2 import (
    AutomaticMaskGeneration,
    SAM2ImageEncoder,
    SAM2ImageDecoder,
    SegmentAnything2ONNX,
    iter_point_batches,
)
from anylabeling.services.auto_labeling.segment_anything_2 import (
    SegmentAnything2,
)


def test_sam2_amg_defaults_are_configured():
    config_dir = (
        Path(__file__).resolve().parents[2]
        / "anylabeling/configs/auto_labeling"
    )
    for model_size in ("tiny", "small", "base", "large"):
        with open(
            config_dir / f"sam2_hiera_{model_size}.yaml",
            "r",
            encoding="utf-8",
        ) as config_file:
            config = yaml.safe_load(config_file)
        assert config["amg_points_per_side"] == 32
        assert config["amg_min_area"] == 100


def test_iter_point_batches_splits_evenly():
    pts = np.arange(20).reshape(10, 2)
    batches = list(iter_point_batches(pts, 4))
    assert [len(b) for b in batches] == [4, 4, 2]
    assert np.array_equal(np.concatenate(batches, axis=0), pts)


def test_iter_point_batches_single_batch():
    pts = np.arange(6).reshape(3, 2)
    batches = list(iter_point_batches(pts, 64))
    assert len(batches) == 1
    assert np.array_equal(batches[0], pts)


def test_encoder_preprocessing_matches_official_sam2():
    image = np.array(
        [
            [
                [0, 10, 20],
                [30, 40, 50],
                [60, 70, 80],
                [90, 100, 110],
                [120, 130, 140],
            ],
            [
                [15, 25, 35],
                [45, 55, 65],
                [75, 85, 95],
                [105, 115, 125],
                [135, 145, 155],
            ],
            [
                [20, 30, 40],
                [50, 60, 70],
                [80, 90, 100],
                [110, 120, 130],
                [140, 150, 160],
            ],
        ],
        dtype=np.uint8,
    )
    encoder = SAM2ImageEncoder.__new__(SAM2ImageEncoder)
    encoder.input_height = 4
    encoder.input_width = 4

    actual = encoder.prepare_input(image)

    assert actual.shape == (1, 3, 4, 4)
    assert actual.dtype == np.float32
    np.testing.assert_allclose(
        actual[0, :, 0, 0],
        [-1.9894683, -1.7293417, -1.3251417],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        actual[0, :, -1, -1],
        [0.15112588, 0.45903358, 0.8535077],
        atol=1e-6,
    )
    np.testing.assert_allclose(actual.sum(), -27.340752, atol=1e-5)


class _FakeDecoder:
    def set_image_size(self, original_size):
        self.original_size = original_size

    def predict_batch(
        self,
        image_embedding,
        high_res_feats_0,
        high_res_feats_1,
        coordinates,
        labels,
    ):
        batch_size = len(coordinates)
        masks = np.zeros((batch_size, 3, 8, 8), dtype=np.float32)
        scores = np.tile(
            np.array([[0.9, 0.8, 0.7]], dtype=np.float32),
            (batch_size, 1),
        )
        return masks, scores


def test_predict_masks_batch_preserves_all_candidates():
    model = SegmentAnything2ONNX.__new__(SegmentAnything2ONNX)
    model.decoder = _FakeDecoder()
    embedding = {
        "image_embedding": None,
        "high_res_feats_0": None,
        "high_res_feats_1": None,
        "original_size": (80, 120),
    }
    points = np.arange(10, dtype=np.float32).reshape(5, 2)

    masks, scores = model.predict_masks_batch(
        embedding, points, points_per_batch=2
    )

    assert masks.shape == (5, 3, 8, 8)
    assert scores.shape == (5, 3)
    assert model.decoder.original_size == (80, 120)


def test_amg_shapes_receive_incrementing_object_labels():
    first_mask = np.zeros((32, 32), dtype=np.uint8)
    first_mask[2:10, 2:10] = 1
    first_mask[20:30, 20:30] = 1
    second_mask = np.zeros((32, 32), dtype=np.uint8)
    second_mask[8:24, 8:24] = 1

    model = SegmentAnything2.__new__(SegmentAnything2)
    model.image_embedding_cache = Mock()
    model.image_embedding_cache.get.return_value = {}
    model.model = Mock()
    model.stop_inference = False
    model.amg = AutomaticMaskGeneration(
        points_per_side=32,
        pred_iou_thresh=0.8,
        stability_score_thresh=0.95,
        stability_score_offset=1.0,
        box_nms_thresh=0.7,
        points_per_batch=64,
        min_mask_region_area=10,
    )
    model.output_mode = "polygon"
    model.epsilon = 0.001

    module_path = "anylabeling.services.auto_labeling.segment_anything_2"
    with (
        patch(
            f"{module_path}.qt_img_to_rgb_cv_img",
            return_value=np.zeros((32, 32, 3), dtype=np.uint8),
        ),
        patch.object(
            model.amg,
            "generate",
            return_value=[
                AutomaticMaskGeneration.mask_to_rle(first_mask),
                AutomaticMaskGeneration.mask_to_rle(second_mask),
            ],
        ) as generate,
    ):
        result = model._predict_auto_grid(Mock(), "image.jpg")

    assert [shape.label for shape in result.shapes] == [
        "object1",
        "object2",
        "object3",
    ]
    generate.assert_called_once()
    assert model.amg.min_mask_region_area == 10


def test_prepare_inputs_preserves_broadcast_mask_flag():
    decoder = SAM2ImageDecoder.__new__(SAM2ImageDecoder)
    decoder.encoder_input_size = (1024, 1024)
    decoder.orig_im_size = (80, 120)
    decoder.scale_factor = 4
    coordinates = [
        np.array([[10.0, 20.0]], dtype=np.float32),
        np.array([[30.0, 40.0]], dtype=np.float32),
        np.array([[50.0, 60.0]], dtype=np.float32),
    ]
    labels = [np.ones(1, dtype=np.float32) for _ in coordinates]

    inputs = decoder.prepare_inputs(
        np.zeros((1, 1, 1, 1), dtype=np.float32),
        np.zeros((1, 1, 1, 1), dtype=np.float32),
        np.zeros((1, 1, 1, 1), dtype=np.float32),
        coordinates,
        labels,
    )

    assert inputs[3].shape == (3, 1, 2)
    assert inputs[4].shape == (3, 1)
    assert inputs[5].shape == (3, 1, 256, 256)
    assert inputs[6].shape == (1,)
