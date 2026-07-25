import cv2
import numpy as np

from anylabeling.services.auto_labeling.__base__.sam2 import (
    AutomaticMaskGeneration,
)


def _generate(decode_batch, image_hw, **kwargs):
    should_stop = kwargs.pop("should_stop", None)
    points_per_chunk = kwargs.pop("points_per_chunk", None)
    if points_per_chunk is not None:
        kwargs["points_per_batch"] = points_per_chunk
    generator = AutomaticMaskGeneration(**kwargs)
    return generator.generate(
        decode_batch,
        image_hw,
        should_stop=should_stop,
    )


def _stub_decode_factory(h=32, w=32):
    """Two objects at grid-normalized regions A and B in a 32x32 low-res mask.
    A: rows/cols 4..12 ; B: rows/cols 20..28. A point 'hits' an object if it
    falls inside that object's pixel box in a 320x320 image."""

    def decode_batch(points_xy):
        batch_size = len(points_xy)
        logits = np.full((batch_size, h, w), -10.0, dtype=np.float32)
        ious = np.full((batch_size,), 0.5, dtype=np.float32)
        for k, (x, y) in enumerate(points_xy):
            if 40 <= x <= 120 and 40 <= y <= 120:
                logits[k, 4:12, 4:12] = 10.0
                ious[k] = 0.95
            elif 200 <= x <= 280 and 200 <= y <= 280:
                logits[k, 20:28, 20:28] = 10.0
                ious[k] = 0.95
        return logits, ious

    return decode_batch


def _decode_rles(rles):
    return [AutomaticMaskGeneration.rle_to_mask(rle) for rle in rles]


def test_build_point_grid_shape_and_range():
    grid = AutomaticMaskGeneration.build_point_grid(4, 320, 320)
    assert grid.shape == (16, 2)
    assert grid[:, 0].min() > 0 and grid[:, 0].max() < 320


def test_generate_returns_two_deduped_masks():
    decode = _stub_decode_factory()
    masks = _decode_rles(_generate(decode, (320, 320), points_per_side=32))
    assert len(masks) == 2
    for mask in masks:
        assert mask.shape == (320, 320)
        assert mask.dtype == np.bool_
        assert mask.sum() > 0


def test_generate_filters_low_iou_only_background():
    def decode(points_xy):
        batch_size = len(points_xy)
        logits = np.full((batch_size, 32, 32), -10.0, dtype=np.float32)
        ious = np.full((batch_size,), 0.5, dtype=np.float32)
        return logits, ious

    masks = _generate(decode, (320, 320), points_per_side=16)
    assert masks == []


def test_generate_should_stop_returns_empty():
    decode = _stub_decode_factory()
    masks = _generate(
        decode,
        (320, 320),
        points_per_side=32,
        should_stop=lambda: True,
    )
    assert masks == []


def test_generate_chunking_matches_single_pass():
    decode = _stub_decode_factory()
    single_chunk = _decode_rles(
        _generate(
            decode,
            (320, 320),
            points_per_side=32,
            points_per_chunk=1024,
        )
    )
    small_chunks = _decode_rles(
        _generate(
            decode,
            (320, 320),
            points_per_side=32,
            points_per_chunk=4,
        )
    )
    assert len(small_chunks) == 2  # chunk size must not change the result
    assert all(
        np.array_equal(actual, expected)
        for actual, expected in zip(small_chunks, single_chunk)
    )


def test_generate_keeps_all_multimask_candidates():
    def decode(points_xy):
        logits = np.full((len(points_xy), 3, 16, 16), -10.0, dtype=np.float32)
        logits[:, 0, 1:5, 1:5] = 10.0
        logits[:, 1, 6:10, 6:10] = 10.0
        logits[:, 2, 11:15, 11:15] = 10.0
        ious = np.tile(
            np.array([[0.95, 0.94, 0.93]], dtype=np.float32),
            (len(points_xy), 1),
        )
        return logits, ious

    masks = _decode_rles(
        _generate(
            decode,
            (64, 64),
            points_per_side=1,
            stability_score_thresh=0.0,
        )
    )
    assert len(masks) == 3
    assert all(mask.sum() > 0 for mask in masks)


def test_generate_resizes_logits_before_thresholding():
    low_res_logits = np.array(
        [[10.0, -10.0], [-10.0, -10.0]], dtype=np.float32
    )

    def decode(points_xy):
        logits = np.repeat(low_res_logits[None], len(points_xy), axis=0)
        return logits, np.full((len(points_xy),), 0.99, np.float32)

    masks = _decode_rles(
        _generate(
            decode,
            (5, 5),
            points_per_side=1,
            stability_score_thresh=0.0,
        )
    )
    expected = cv2.resize(
        low_res_logits, (5, 5), interpolation=cv2.INTER_LINEAR
    )
    assert len(masks) == 1
    assert np.array_equal(masks[0], expected > 0)


def test_generate_reruns_nms_after_removing_small_regions():
    def decode(points_xy):
        logits = np.full((len(points_xy), 2, 32, 32), -10.0, dtype=np.float32)
        logits[:, :, 2:12, 2:12] = 10.0
        logits[:, 1, 30, 30] = 10.0
        ious = np.tile(
            np.array([[0.99, 0.98]], dtype=np.float32),
            (len(points_xy), 1),
        )
        return logits, ious

    unprocessed = _generate(
        decode,
        (32, 32),
        points_per_side=1,
        stability_score_thresh=0.0,
    )
    processed = _decode_rles(
        _generate(
            decode,
            (32, 32),
            points_per_side=1,
            stability_score_thresh=0.0,
            min_mask_region_area=2,
        )
    )

    assert len(unprocessed) == 2
    assert len(processed) == 1
    assert processed[0].sum() == 100
    assert not processed[0][30, 30]


def test_remove_small_regions_handles_holes_and_islands():
    mask = np.zeros((24, 24), dtype=np.bool_)
    mask[2:12, 2:12] = True
    mask[5, 5] = False
    mask[20, 20] = True

    mask, holes_changed = AutomaticMaskGeneration._remove_small_regions(
        mask, 2, mode="holes"
    )
    mask, islands_changed = AutomaticMaskGeneration._remove_small_regions(
        mask, 2, mode="islands"
    )

    assert holes_changed
    assert islands_changed
    assert mask.sum() == 100
    assert mask[5, 5]
    assert not mask[20, 20]


def test_generate_filters_stability_after_resizing():
    low_res_logits = np.array(
        [[10.0, -10.0], [-10.0, -10.0]], dtype=np.float32
    )

    def decode(points_xy):
        logits = np.repeat(low_res_logits[None], len(points_xy), axis=0)
        return logits, np.full((len(points_xy),), 0.99, np.float32)

    assert _generate(decode, (5, 5), points_per_side=1) == []


def test_rle_round_trip():
    mask = np.zeros((7, 9), dtype=np.bool_)
    mask[:3, :4] = True
    mask[5:, 7:] = True
    assert np.array_equal(
        AutomaticMaskGeneration.rle_to_mask(
            AutomaticMaskGeneration.mask_to_rle(mask)
        ),
        mask,
    )


def test_box_nms_honors_cancellation():
    calls = 0

    def should_stop():
        nonlocal calls
        calls += 1
        return calls > 1

    boxes = np.array(
        [[0, 0, 4, 4], [5, 5, 9, 9], [10, 10, 14, 14]],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    assert (
        AutomaticMaskGeneration._box_nms(boxes, scores, 0.7, should_stop) == []
    )


def test_generate_decodes_each_point_once():
    batch_sizes = []

    def decode(points_xy):
        batch_sizes.append(len(points_xy))
        logits = np.full((len(points_xy), 1, 8, 8), -10.0, dtype=np.float32)
        logits[:, :, 1:7, 1:7] = 10.0
        return logits, np.full((len(points_xy), 1), 0.99, np.float32)

    masks = _generate(
        decode,
        (32, 32),
        points_per_side=2,
        points_per_chunk=4,
    )

    assert len(masks) == 1
    assert batch_sizes == [4]


def _square_mask(size=64, lo=16, hi=48):
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[lo:hi, lo:hi] = 255
    return mask


def test_polygon_mode_produces_closed_polygon():
    shapes = AutomaticMaskGeneration.masks_to_shapes(
        _square_mask(),
        "polygon",
        min_area=10,
    )
    assert len(shapes) == 1
    assert shapes[0].shape_type == "polygon"
    assert shapes[0].closed is True
    assert len(shapes[0].points) >= 4


def test_contour_mode_produces_open_linestrip():
    shapes = AutomaticMaskGeneration.masks_to_shapes(
        _square_mask(),
        "contour",
        min_area=10,
    )
    assert len(shapes) == 1
    assert shapes[0].shape_type == "linestrip"
    assert shapes[0].closed is False


def test_min_area_filters_small_regions():
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:13, 10:13] = 255
    shapes = AutomaticMaskGeneration.masks_to_shapes(
        mask,
        "polygon",
        min_area=100,
    )
    assert shapes == []
