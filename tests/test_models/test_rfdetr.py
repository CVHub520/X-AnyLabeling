import numpy as np
import pytest

from anylabeling.services.auto_labeling.rfdetr import RFDETR
from tools.onnx_exporter.export_rfdetr_onnx import (
    postprocess as detection_postprocess,
)
from tools.onnx_exporter.export_rfdetr_seg_onnx import (
    postprocess as segmentation_postprocess,
)


@pytest.mark.parametrize("num_select", [0, 1, 5, 6, 7, 300])
@pytest.mark.parametrize("conf_threshold", [0.0, 0.65, 0.99])
@pytest.mark.parametrize(
    "backend,with_masks",
    [
        ("model", False),
        ("model", True),
        ("detection", False),
        ("segmentation", False),
        ("segmentation", True),
    ],
)
def test_postprocess_topk(num_select, conf_threshold, backend, with_masks):
    outputs = [
        np.array(
            [
                [
                    [0.25, 0.25, 0.5, 0.5],
                    [0.5, 0.5, 0.2, 0.2],
                    [0.75, 0.75, 0.1, 0.1],
                ]
            ],
            dtype=np.float32,
        ),
        np.array([[[0.1, 0.9], [0.8, 0.2], [0.6, 0.4]]], dtype=np.float32),
    ]
    if with_masks:
        outputs.append(
            np.array(
                [[[[-1, -1], [-1, -1]], [[1, 1], [1, 1]], [[-1, 1], [1, -1]]]],
                dtype=np.float32,
            )
        )

    if backend == "model":
        model = RFDETR.__new__(RFDETR)
        model.num_select = num_select
        model.conf_thres = conf_threshold
        boxes, scores, labels, masks = model.postprocess(outputs, (100, 200))
    elif backend == "detection":
        boxes, scores, labels = detection_postprocess(
            outputs, conf_threshold, num_select, (100, 200)
        )
        masks = None
    else:
        boxes, scores, labels, masks = segmentation_postprocess(
            outputs, conf_threshold, num_select, (100, 200)
        )

    indices = np.array([1, 2, 4, 5, 3, 0])[:num_select]
    probabilities = 1 / (1 + np.exp(-outputs[1].reshape(-1)))
    indices = indices[probabilities[indices] > conf_threshold]
    expected_boxes = np.array(
        [[0, 0, 100, 50], [80, 40, 120, 60], [140, 70, 160, 80]],
        dtype=np.float32,
    )

    np.testing.assert_allclose(boxes, expected_boxes[indices // 2])
    np.testing.assert_allclose(scores, probabilities[indices])
    np.testing.assert_array_equal(labels, indices % 2)
    if with_masks:
        assert masks.shape == (len(indices), 100, 200)
        assert masks.dtype == np.uint8
        expected_corners = np.array(
            [[[0, 0], [0, 0]], [[255, 255], [255, 255]], [[0, 255], [255, 0]]],
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(
            masks[:, [0, -1]][:, :, [0, -1]],
            expected_corners[indices // 2],
        )
    else:
        assert masks is None
