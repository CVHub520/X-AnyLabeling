import numpy as np
import yaml

from anylabeling.services.auto_labeling.segment_anything_3 import (
    SegmentAnything3,
)


def test_sam3_config_is_registered():
    with open("anylabeling/configs/models.yaml", "r", encoding="utf-8") as f:
        models = yaml.safe_load(f)

    assert {
        "model_name": "sam3_vit_h-r20260426",
        "config_file": ":/sam3_vit_h.yaml",
    } in models

    with open(
        "anylabeling/configs/auto_labeling/sam3_vit_h.yaml",
        "r",
        encoding="utf-8",
    ) as f:
        config = yaml.safe_load(f)

    assert config["type"] == "segment_anything_3"
    assert config["encoder_model_path"].endswith("sam3_image_encoder.onnx")
    assert config["encoder_model_data_path"].endswith(
        "sam3_image_encoder.onnx.data"
    )
    assert config["decoder_model_path"].endswith("sam3_decoder.onnx")
    assert config["decoder_model_data_path"].endswith(
        "sam3_decoder.onnx.data"
    )
    assert config["language_encoder_path"].endswith(
        "sam3_language_encoder.onnx"
    )
    assert config["language_encoder_data_path"].endswith(
        "sam3_language_encoder.onnx.data"
    )


def test_sam3_post_process_polygon_from_multiple_masks():
    model = SegmentAnything3.__new__(SegmentAnything3)
    model.output_mode = "polygon"
    model.epsilon = 0.001
    masks = np.zeros((2, 1, 20, 20), dtype=np.bool_)
    masks[0, 0, 2:8, 3:10] = True
    masks[1, 0, 10:16, 11:18] = True

    shapes = model.post_process(masks, "truck")

    assert len(shapes) == 2
    assert all(shape.label == "truck" for shape in shapes)
    assert all(shape.shape_type == "polygon" for shape in shapes)
    assert all(shape.closed for shape in shapes)


def test_sam3_post_process_rectangle_output_mode():
    model = SegmentAnything3.__new__(SegmentAnything3)
    model.output_mode = "rectangle"
    model.epsilon = 0.001
    masks = np.zeros((1, 1, 20, 20), dtype=np.bool_)
    masks[0, 0, 4:9, 5:12] = True
    scores = np.array([0.88], dtype=np.float32)

    shapes = model.post_process(masks, "car", scores)

    assert len(shapes) == 1
    assert shapes[0].label == "car"
    assert shapes[0].score == float(scores[0])
    assert shapes[0].shape_type == "rectangle"


def test_sam3_splits_and_deduplicates_text_prompts():
    assert SegmentAnything3.split_text_prompts("person.car.person") == [
        "person",
        "car",
    ]
    assert SegmentAnything3.split_text_prompts(" person, car , ") == [
        "person",
        "car",
    ]
