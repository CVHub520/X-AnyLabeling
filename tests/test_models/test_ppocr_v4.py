import os

from anylabeling.services.auto_labeling.ppocr_v4 import PPOCRv4


def test_rec_char_dict_path_uses_existing_relative_path(tmp_path, monkeypatch):
    dict_path = tmp_path / "dict.txt"
    dict_path.write_text("a\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert PPOCRv4.get_rec_char_dict_path(
        {"rec_char_dict_path": "dict.txt"}, "/unused"
    ) == str(dict_path)


def test_rec_char_dict_path_uses_config_relative_path(tmp_path, monkeypatch):
    cwd = tmp_path / "cwd"
    config_dir = tmp_path / "config"
    cwd.mkdir()
    config_dir.mkdir()
    dict_path = config_dir / "dict.txt"
    dict_path.write_text("a\n", encoding="utf-8")
    monkeypatch.chdir(cwd)

    assert PPOCRv4.get_rec_char_dict_path(
        {
            "config_file": str(config_dir / "model.yaml"),
            "rec_char_dict_path": "dict.txt",
        },
        "/unused",
    ) == str(dict_path)


def test_rec_char_dict_path_keeps_lang_default():
    assert PPOCRv4.get_rec_char_dict_path(
        {"lang": "japan"}, "/x/auto_labeling"
    ) == os.path.join(
        "/x/auto_labeling", "configs/ppocr/japan_dict.txt"
    )
