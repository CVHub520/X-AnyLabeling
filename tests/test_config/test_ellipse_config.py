"""Shared early CLI and GUI ellipse configuration semantics."""

from pathlib import Path

import pytest

from anylabeling import config


@pytest.mark.parametrize(
    "override, expected", [(None, 64), (12, 12), (128, 128), (512, 512)]
)
def test_precedence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    override: int | None,
    expected: int,
) -> None:
    """An explicit count overrides the selected file; omission preserves it."""
    monkeypatch.setattr(config, "_work_directory", str(tmp_path))
    selected = tmp_path / "chosen.yaml"
    selected.write_text(
        "canvas:\n  ellipse:\n    polygon_points: 64\n", encoding="utf-8"
    )
    result = config.get_ellipse_config(str(selected), override)
    assert result["canvas"]["ellipse"]["polygon_points"] == expected
    assert selected.read_text(encoding="utf-8").endswith("64\n")


@pytest.mark.parametrize(
    "value", ["11", "513", "12.5", "true", "'128'", "garbage"]
)
def test_selected_file_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """Early conversion rejects the same invalid values as the GUI loader."""
    monkeypatch.setattr(config, "_work_directory", str(tmp_path))
    selected = tmp_path / "invalid.yaml"
    selected.write_text(
        f"canvas:\n  ellipse:\n    polygon_points: {value}\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="integer"):
        config.get_ellipse_config(str(selected))


def test_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A fresh isolated work directory uses the common factory value."""
    monkeypatch.setattr(config, "_work_directory", str(tmp_path))
    monkeypatch.setattr(config, "current_config_file", None)
    assert (
        config.get_ellipse_config()["canvas"]["ellipse"]["polygon_points"]
        == 128
    )
