from unittest.mock import patch

import numpy as np
import pytest
from PyQt6 import QtCore

from anylabeling.views.labeling.pointcloud.model import ClassDefinition

from . import test_dialog as dialog_tests
from .test_dialog import open_cloud, select_class, wait_load

app = dialog_tests.app
window = dialog_tests.window


@pytest.fixture
def display_window(window, app, tmp_path):
    labels = np.array(
        [
            0,
            10,
            (1 << 16) | 10,
            (1 << 16) | 10,
            (1 << 16) | 30,
            7 << 16,
            0xFFFFFFFF,
            30,
            (2 << 16) | 70,
            0,
            (2 << 16) | 10,
            70,
        ],
        dtype="<u4",
    )
    points = np.arange(len(labels) * 4, dtype=np.float32).reshape(-1, 4)
    points[:, 3] = [np.nan, np.inf, -np.inf, 0, 4, 10, 5, 6, 8, 2, 1, 3]
    source = tmp_path / "display.bin"
    dialog_tests.cloud(source, count=len(labels))
    points.tofile(source)
    labels.tofile(source.with_suffix(".label"))
    open_cloud(window, app, source)
    return window


def select_instance(window, key):
    if key is None:
        window.instance_list.setCurrentRow(-1)
        return
    for row in range(window.instance_list.count()):
        item = window.instance_list.item(row)
        if item.data(QtCore.Qt.ItemDataRole.UserRole) == key:
            window.instance_list.setCurrentRow(row)
            return
    raise AssertionError(key)


def test_bulk_visibility_preserves_labels_and_refresh(display_window):
    window = display_window
    original = window.document.labels.copy()
    window.classes_visibility_action.trigger()
    assert not window._visible.any()
    window.classes_visibility_action.trigger()
    assert window._visible.all()
    window.instances_visibility_action.trigger()
    np.testing.assert_array_equal(window._visible, original >> 16 == 0)
    window._refresh()
    np.testing.assert_array_equal(window._visible, original >> 16 == 0)
    window.instances_visibility_action.trigger()
    np.testing.assert_array_equal(window._visible, original >> 16 != 0)
    window._restore_all()
    assert window._visible.all()
    np.testing.assert_array_equal(window.document.labels, original)


def test_multiple_instances_highlight_and_filter_together(display_window):
    window = display_window
    baseline = window.viewport._colors.copy()
    original = window.document.labels.copy()
    keys = {(10, 1), (30, 1)}
    for row in range(window.instance_list.count()):
        item = window.instance_list.item(row)
        if item.data(QtCore.Qt.ItemDataRole.UserRole) in keys:
            item.setSelected(True)
    expected = np.isin(original, [(1 << 16) | 10, (1 << 16) | 30])
    np.testing.assert_allclose(
        window.viewport._colors[expected, :3],
        baseline[expected, :3] * 0.4 + np.array([1, 0.85, 0.2]) * 0.6,
        atol=1e-7,
    )
    np.testing.assert_array_equal(
        window.viewport._colors[~expected], baseline[~expected]
    )
    for item in window.instance_list.selectedItems():
        item.setCheckState(QtCore.Qt.CheckState.Checked)
    np.testing.assert_array_equal(window._visible, expected)
    window._refresh()
    assert len(window.instance_list.selectedItems()) == 2
    np.testing.assert_array_equal(window._visible, expected)
    hide_class(window, 30)
    np.testing.assert_array_equal(window._visible, original == ((1 << 16) | 10))
    np.testing.assert_array_equal(window.document.labels, original)
    assert_matches_full_refresh(window)


def hide_class(window, semantic_id):
    for row in range(window.class_list.count()):
        item = window.class_list.item(row)
        if item.data(QtCore.Qt.ItemDataRole.UserRole) == semantic_id:
            item.setCheckState(QtCore.Qt.CheckState.Unchecked)
            return
    raise AssertionError(semantic_id)


def assert_matches_full_refresh(window):
    colors = window.viewport._colors.copy()
    visible = window._visible.copy()
    summary = window.summary.text()
    legend = window.legend.text()
    np.testing.assert_array_equal(window.viewport._visible, visible)
    assert window._visible_count == np.count_nonzero(visible)
    window._display_signature = None
    window._refresh_display()
    np.testing.assert_array_equal(window.viewport._colors, colors)
    np.testing.assert_array_equal(window._visible, visible)
    np.testing.assert_array_equal(window.viewport._visible, visible)
    assert window.summary.text() == summary
    assert window.legend.text() == legend


@pytest.mark.parametrize("mode", ["semantic", "instance", "intensity"])
@pytest.mark.parametrize(
    "filter_mode", ["all", "hidden", "focus"]
)
def test_incremental_edits_match_full_refresh_with_modes_and_filters(
    display_window, mode, filter_mode
):
    window = display_window
    doc = window.document
    window.color_mode.setCurrentIndex(window.color_mode.findData(mode))
    select_instance(window, (10, 1))
    if filter_mode == "hidden":
        hide_class(window, 10)
    elif filter_mode == "focus":
        window.instance_list.currentItem().setCheckState(QtCore.Qt.CheckState.Checked)
    assert_matches_full_refresh(window)
    with patch.object(
        window.viewport, "set_colors", wraps=window.viewport.set_colors
    ) as set_colors:
        doc.assign_semantic([0, 1, 7, 9], 30, overwrite=True)
        window._refresh()
        set_colors.assert_called_once()
        np.testing.assert_array_equal(
            set_colors.call_args.kwargs["indices"], doc.last_changed_indices
        )
    assert_matches_full_refresh(window)
    doc.add_to_instance([1, 7], (30, 1))
    window._refresh()
    assert_matches_full_refresh(window)
    window.undo()
    assert_matches_full_refresh(window)
    window.redo()
    assert_matches_full_refresh(window)
    doc.clear([0, 4, 6])
    window._refresh()
    assert_matches_full_refresh(window)


def test_target_noops_and_save_state_do_not_recolor(display_window, tmp_path):
    window = display_window
    doc = window.document
    original = window.viewport._colors.copy()
    with patch.object(
        window.viewport, "set_colors", wraps=window.viewport.set_colors
    ) as set_colors:
        select_class(window, 30)
        window._refresh_display()
        assert doc.assign_semantic([1], 10, overwrite=True) == 0
        window._refresh()
        set_colors.assert_not_called()
    np.testing.assert_array_equal(window.viewport._colors, original)
    doc.assign_semantic([0], 10)
    window._refresh()
    assert "Modified" in window.summary.text()
    revision = doc.revision
    target = tmp_path / "exported.label"
    doc.labels.astype("<u4").tofile(target)
    with patch.object(
        window.viewport, "set_colors", wraps=window.viewport.set_colors
    ) as set_colors:
        doc.mark_saved(target)
        window._refresh()
        assert doc.revision == revision
        assert "Saved" in window.summary.text()
        assert str(target) in window.file_list.currentItem().toolTip()
        assert not window.windowTitle().endswith(" *")
        target.unlink()
        window._refresh_display()
        assert "No result file" in window.summary.text()
        set_colors.assert_not_called()
    assert_matches_full_refresh(window)


def test_revision_jump_requires_full_refresh(display_window):
    window = display_window
    doc = window.document
    doc.assign_semantic([0], 10)
    doc.assign_semantic([9], 30)
    with patch.object(
        window.viewport, "set_colors", wraps=window.viewport.set_colors
    ) as set_colors:
        window._refresh()
        set_colors.assert_called_once()
        assert set_colors.call_args.kwargs["indices"] is None
    assert_matches_full_refresh(window)
    window.undo()
    assert_matches_full_refresh(window)
    revision = doc.revision
    assert doc.assign_semantic([0], 10, overwrite=True) == 0
    assert doc.can_redo
    with patch.object(
        window.viewport, "set_colors", wraps=window.viewport.set_colors
    ) as set_colors:
        window._refresh()
        assert doc.revision == revision
        set_colors.assert_not_called()
    window.redo()
    assert_matches_full_refresh(window)


def test_instance_filter_and_palette_changes_invalidate_display(
    display_window,
):
    window = display_window
    for mode in ("instance", "intensity", "semantic"):
        window.color_mode.setCurrentIndex(window.color_mode.findData(mode))
        for key in ((10, 1), (30, 1), (0, 7), None):
            select_instance(window, key)
            assert_matches_full_refresh(window)
        window.instance_list.item(0).setCheckState(QtCore.Qt.CheckState.Checked)
        assert_matches_full_refresh(window)
        window.instance_list.item(1).setCheckState(QtCore.Qt.CheckState.Checked)
        assert_matches_full_refresh(window)
        hide_class(window, 10)
        assert_matches_full_refresh(window)
        window._restore_all()
        assert_matches_full_refresh(window)
    before = window.viewport._colors.copy()
    definition = ClassDefinition(10, "Updated vehicle", "#010203")
    window.classes = [
        definition if item.id == 10 else item for item in window.classes
    ]
    window._refresh()
    np.testing.assert_allclose(
        window.viewport._colors[1, :3], np.array([1, 2, 3]) / 255
    )
    assert not np.array_equal(before, window.viewport._colors)
    assert_matches_full_refresh(window)


def test_replacement_document_resets_intensity_range_at_same_revision(
    display_window, app, tmp_path
):
    window = display_window
    window.color_mode.setCurrentIndex(window.color_mode.findData("intensity"))
    original_document = window.document
    assert original_document.revision == 0
    assert window._intensity_range == (0, 10)
    source = tmp_path / "replacement.bin"
    points = np.arange(48, dtype=np.float32).reshape(-1, 4)
    points[:, 3] = np.linspace(100, 200, len(points))
    points.tofile(source)
    open_cloud(window, app, source)
    assert window.document is not original_document
    assert window.document.revision == 0
    assert window._intensity_range == (100, 200)
    np.testing.assert_allclose(window.viewport._colors[0, :3], [0, 0.25, 1])
    np.testing.assert_allclose(window.viewport._colors[-1, :3], [1, 0.9, 0.2])
    assert_matches_full_refresh(window)


@pytest.mark.parametrize("mode", ["semantic", "instance", "intensity"])
def test_failed_render_preparation_restores_previous_colors_and_filters(
    display_window, app, tmp_path, mode
):
    window = display_window
    doc = window.document
    window.color_mode.setCurrentIndex(window.color_mode.findData(mode))
    doc.assign_semantic([0], 10)
    window._refresh()
    select_instance(window, (30, 1))
    hide_class(window, 10)
    colors = window.viewport._colors.copy()
    visible = window.viewport._visible.copy()
    summary = window.summary.text()
    revision = doc.revision
    source = tmp_path / "failed.bin"
    np.arange(20, dtype=np.float32).reshape(-1, 4).tofile(source)
    window._leave_decision = lambda: "discard"
    window._errors.clear()
    with patch.object(
        window.viewport,
        "validate_rendering",
        side_effect=RuntimeError("injected render failure"),
    ):
        window.open_paths([source])
        wait_load(window, app)
    assert window.document is doc
    assert doc.revision == revision
    assert doc.dirty and doc.can_undo
    assert window._current_instance() == (30, 1)
    assert window._errors == ["injected render failure"]
    np.testing.assert_array_equal(window.viewport._points, doc.frame.points)
    np.testing.assert_array_equal(window.viewport._colors, colors)
    np.testing.assert_array_equal(window.viewport._visible, visible)
    assert window.summary.text() == summary
    assert_matches_full_refresh(window)


def test_color_mode_stays_selected_when_switching_tools(display_window):
    window = display_window
    assert window.color_mode.currentData() == "semantic"
    for mode in ("semantic", "intensity", "instance"):
        window.color_mode.setCurrentIndex(window.color_mode.findData(mode))
        for tool in ("brush", "polygon", "browse"):
            window._select_tool(tool)
            assert window.color_mode.currentData() == mode


def test_rgb_display_and_unavailable_mode_fallback(window, app, tmp_path):
    path = tmp_path / "colors.ply"
    path.write_text(
        "ply\nformat ascii 1.0\nelement vertex 2\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n0 0 0 255 0 0\n1 1 1 0 128 255\n"
    )
    open_cloud(window, app, path)
    modes = window.color_mode
    assert modes.currentData() == "semantic"
    assert not modes.model().item(modes.findData("intensity")).isEnabled()
    assert modes.model().item(modes.findData("rgb")).isEnabled()
    modes.setCurrentIndex(modes.findData("rgb"))
    before = window.document.labels.copy()
    np.testing.assert_allclose(
        window._display_colors(slice(None), None),
        [[1, 0, 0], [0, 128 / 255, 1]],
    )
    window._select_tool("brush")
    assert modes.currentData() == "rgb"
    np.testing.assert_array_equal(window.document.labels, before)
    open_cloud(window, app, dialog_tests.cloud(tmp_path / "other.bin"))
    assert modes.currentData() == "semantic"
    assert not modes.model().item(modes.findData("rgb")).isEnabled()
