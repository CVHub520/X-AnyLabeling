import numpy as np

from anylabeling.services.auto_labeling.dfine_seg import DFINESeg
from anylabeling.services.auto_labeling.rfdetr import RFDETR


def make_model():
    model = DFINESeg.__new__(DFINESeg)
    model.classes = ["person", "car"]
    model.input_height = 100
    model.input_width = 100
    model.conf_thres = 0.5
    model.iou_thres = 0.7
    model.mask_thres = 0.5
    model.epsilon = 0.001
    model.filter_classes = None
    return model


def test_instance_segmentation_scales_boxes_and_applies_nms():
    model = make_model()
    outputs = [
        np.array([[0, 0, 1, 1]], dtype=np.int64),
        np.array(
            [
                [
                    [10, 10, 30, 30],
                    [11, 11, 31, 31],
                    [40, 20, 60, 40],
                    [10, 10, 30, 30],
                ]
            ],
            dtype=np.float32,
        ),
        np.array([[0.9, 0.8, 0.7, 0.85]], dtype=np.float32),
        np.ones((1, 4, 4, 4), dtype=np.float32),
    ]

    boxes, scores, labels, masks = model.postprocess(outputs, (100, 200))

    np.testing.assert_allclose(
        boxes,
        [[20, 10, 60, 30], [20, 10, 60, 30], [80, 20, 120, 40]],
    )
    np.testing.assert_allclose(scores, [0.9, 0.85, 0.7])
    np.testing.assert_array_equal(labels, [0, 1, 1])
    assert masks.shape == (3, 100, 200)


def test_instance_segmentation_postprocess_resizes_and_crops_masks():
    model = make_model()
    outputs = [
        np.array([[1]], dtype=np.int64),
        np.array([[[25, 25, 75, 75]]], dtype=np.float32),
        np.array([[0.8]], dtype=np.float32),
        np.ones((1, 1, 4, 4), dtype=np.float32),
    ]

    boxes, _, _, masks = model.postprocess(outputs, (20, 40))

    np.testing.assert_allclose(boxes, [[10, 5, 30, 15]])
    assert masks.shape == (1, 20, 40)
    assert not masks[0, 4, 20]
    assert masks[0, 5, 10]
    assert not masks[0, 15, 20]


def test_rfdetr_postprocess_clamps_num_select_to_available_candidates():
    model = RFDETR.__new__(RFDETR)
    model.conf_thres = 0.0
    model.num_select = 300
    model.filter_classes = None

    outputs = [
        np.array([[[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]]], dtype=np.float32),
        np.array([[[0.1, 0.9], [0.8, 0.2], [0.6, 0.4]]], dtype=np.float32),
    ]

    boxes, scores, labels, masks = model.postprocess(outputs, (100, 100))

    assert boxes.shape[0] == 6
    assert scores.shape[0] == 6
    assert labels.shape[0] == 6
    assert masks is None
