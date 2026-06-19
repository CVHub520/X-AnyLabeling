import os

from anylabeling.services.auto_labeling.ppocr_v6 import PPOCRv6


def test_rec_char_dict_path_uses_inference_yml_character_dict(tmp_path):
    rec_model = tmp_path / "rec_model.onnx"
    rec_model.touch()
    inference_yml = tmp_path / "inference.yml"
    inference_yml.write_text(
        "\n".join(
            [
                "PostProcess:",
                "  character_dict:",
                "  - A",
                "  - B",
                "  - ' '",
            ]
        ),
        encoding="utf-8",
    )

    dict_path = PPOCRv6.get_rec_char_dict_path(
        {"rec_model_path": str(rec_model)}, "/unused"
    )

    assert os.path.isfile(dict_path)
    with open(dict_path, "r", encoding="utf-8") as f:
        assert f.read().splitlines() == ["A", "B", " "]


def test_rec_char_dict_path_uses_packaged_config_value():
    current_dir = os.path.join(
        os.getcwd(), "anylabeling", "services", "auto_labeling"
    )

    dict_path = PPOCRv6.get_rec_char_dict_path(
        {"rec_char_dict_path": "ppocrv6_tiny_dict.txt"},
        current_dir,
    )

    assert dict_path.endswith(
        os.path.join("configs", "ppocr", "ppocrv6_tiny_dict.txt")
    )
    assert os.path.isfile(dict_path)


def test_parse_det_db_params_reads_official_inference_yml(tmp_path):
    det_model = tmp_path / "det_model.onnx"
    det_model.touch()
    (tmp_path / "inference.yml").write_text(
        "\n".join(
            [
                "PostProcess:",
                "  thresh: 0.2",
                "  box_thresh: 0.45",
                "  unclip_ratio: 1.4",
                "  max_candidates: 3000",
            ]
        ),
        encoding="utf-8",
    )

    assert PPOCRv6.get_det_db_params(
        {"det_model_path": str(det_model)}
    ) == {
        "det_db_thresh": 0.2,
        "det_db_box_thresh": 0.45,
        "det_db_unclip_ratio": 1.4,
        "det_db_max_candidates": 3000,
    }


def test_parse_det_db_params_falls_back_to_config_values():
    assert PPOCRv6.get_det_db_params(
        {
            "det_db_thresh": 0.2,
            "det_db_box_thresh": 0.45,
            "det_db_unclip_ratio": 1.4,
            "det_db_max_candidates": 3000,
        }
    ) == {
        "det_db_thresh": 0.2,
        "det_db_box_thresh": 0.45,
        "det_db_unclip_ratio": 1.4,
        "det_db_max_candidates": 3000,
    }
