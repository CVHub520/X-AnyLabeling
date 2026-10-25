import copy
import os
from types import SimpleNamespace
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6 import QtCore, QtGui, QtTest, QtWidgets

from anylabeling.resources import resources
from anylabeling.views.labeling.label_widget import LabelingWidget
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.widgets.canvas import Canvas
from anylabeling.views.labeling.widgets.navigator_widget import NavigatorDialog
from anylabeling.views.labeling.widgets.zoom_widget import ZoomWidget


class NavigationView(QtWidgets.QWidget):
    MANUAL_ZOOM = LabelingWidget.MANUAL_ZOOM
    set_zoom = LabelingWidget.set_zoom
    zoom_request = LabelingWidget.zoom_request
    _animate_zoom = LabelingWidget._animate_zoom
    _apply_zoom = LabelingWidget._apply_zoom
    pan_canvas = LabelingWidget.pan_canvas
    on_navigator_request = LabelingWidget.on_navigator_request

    def __init__(self):
        super().__init__()
        self.image = QtGui.QImage(800, 600, QtGui.QImage.Format.Format_RGB32)
        self.image.fill(QtGui.QColor("white"))
        self.canvas = Canvas(parent=None)
        self.canvas.pixmap = QtGui.QPixmap.fromImage(self.image)
        self._canvas_scroll_area = QtWidgets.QScrollArea(self)
        self._canvas_scroll_area.setWidget(self.canvas)
        self._canvas_scroll_area.setWidgetResizable(True)
        self._canvas_scroll_area.resize(600, 400)
        self.resize(600, 400)
        self.zoom_widget = ZoomWidget()
        self._zoom_animation = QtCore.QVariantAnimation(self)
        self._zoom_animation.setDuration(90)
        self._zoom_animation.valueChanged.connect(self._animate_zoom)
        self.canvas.zoom_request.connect(self.zoom_request)
        self.canvas.pan_request.connect(self.pan_canvas)
        self.canvas.interaction_started.connect(self._zoom_animation.stop)
        self.actions = SimpleNamespace(
            fit_width=QtGui.QAction(self), fit_window=QtGui.QAction(self)
        )
        self.filename = "image.png"
        self.zoom_mode = self.MANUAL_ZOOM
        self.zoom_values = {}
        self.pan_values = {}
        self.update_navigator_viewport = Mock()
        self.canvas.menus = (Mock(), Mock())
        self.show()


@pytest.fixture(scope="session")
def navigation_app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def view(navigation_app):
    app = navigation_app
    widget = NavigationView()
    app.processEvents()
    yield widget
    widget._zoom_animation.stop()
    widget.canvas.restore_cursor()
    widget.zoom_widget.close()
    widget.close()
    widget.deleteLater()
    app.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)


def image_position(view, viewport_pos):
    return view.canvas.transform_pos(
        viewport_pos - QtCore.QPointF(view.canvas.pos())
    )


def assert_point(actual, expected):
    assert actual.x() == pytest.approx(expected.x(), abs=1e-8)
    assert actual.y() == pytest.approx(expected.y(), abs=1e-8)


def wheel(
    view,
    delta=120,
    pixel=0,
    modifiers=QtCore.Qt.KeyboardModifier.NoModifier,
    point=None,
):
    if point is None:
        point = QtCore.QPointF(110.5, 83.25)
    event = QtGui.QWheelEvent(
        point,
        point,
        QtCore.QPoint(0, pixel),
        QtCore.QPoint(0, delta),
        QtCore.Qt.MouseButton.NoButton,
        modifiers,
        QtCore.Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    view.canvas.wheelEvent(event)


def mouse(
    view,
    kind,
    position,
    button=QtCore.Qt.MouseButton.RightButton,
    modifiers=QtCore.Qt.KeyboardModifier.NoModifier,
):
    local = position - QtCore.QPointF(view.canvas.pos())
    buttons = (
        button
        if kind != QtCore.QEvent.Type.MouseButtonRelease
        else QtCore.Qt.MouseButton.NoButton
    )
    event = QtGui.QMouseEvent(
        kind, local, position, button, buttons, modifiers
    )
    method = {
        QtCore.QEvent.Type.MouseButtonPress: view.canvas.mousePressEvent,
        QtCore.QEvent.Type.MouseMove: view.canvas.mouseMoveEvent,
        QtCore.QEvent.Type.MouseButtonRelease: view.canvas.mouseReleaseEvent,
        QtCore.QEvent.Type.MouseButtonDblClick: view.canvas.mouseDoubleClickEvent,
    }[kind]
    method(event)


@pytest.mark.parametrize(
    "start,end", [(100, 50), (50, 100), (100, 210), (210, 10)]
)
def test_zoom_keeps_cursor_anchor_across_scrollbar_boundaries(
    view, start, end
):
    view.set_zoom(start)
    for bar in (
        view._canvas_scroll_area.horizontalScrollBar(),
        view._canvas_scroll_area.verticalScrollBar(),
    ):
        bar.setValue(bar.maximum() // 2)
    view.pan_canvas(QtCore.QPointF(133.25, -47.5))
    point = QtCore.QPointF(120.5, 90.25)
    anchor = image_position(view, point)
    view.set_zoom(end, pos=point - QtCore.QPointF(view.canvas.pos()))
    QtWidgets.QApplication.processEvents()
    assert_point(image_position(view, point), anchor)


@pytest.mark.parametrize("delta,pixel", [(120, 0), (15, 0), (0, 3)])
def test_wheel_animates_fractional_zoom_with_fixed_anchor(view, delta, pixel):
    point = QtCore.QPointF(110.5, 83.25) + QtCore.QPointF(view.canvas.pos())
    anchor = image_position(view, point)
    wheel(view, delta, pixel)
    target = 1.1 ** ((delta or pixel) / 120)
    for elapsed in (15, 30, 60, 90):
        view._zoom_animation.setCurrentTime(elapsed)
        QtWidgets.QApplication.processEvents()
        assert_point(image_position(view, point), anchor)
        if elapsed < 90:
            assert 1 < view.canvas.scale < target
    assert view.canvas.scale == pytest.approx(target)
    assert view.zoom_values[view.filename][1] == pytest.approx(target * 100)


def test_rapid_wheel_events_accumulate_and_reverse_without_rounding_drift(
    view,
):
    for delta in (15,) * 8 + (-15,) * 8:
        wheel(view, delta)
    view._zoom_animation.setCurrentTime(90)
    assert view.canvas.scale == pytest.approx(1.0)
    wheel(view, 0)
    assert (
        view._zoom_animation.state() == QtCore.QAbstractAnimation.State.Stopped
    )


@pytest.mark.parametrize(
    "value,delta,expected", [(999, 1200, 10), (10.01, -1200, 0.1)]
)
def test_wheel_obeys_zoom_limits(view, value, delta, expected):
    view.set_zoom(value)
    wheel(view, delta)
    view._zoom_animation.setCurrentTime(90)
    assert view.canvas.scale == expected


@pytest.mark.parametrize("zoom", [30, 300, 1000])
@pytest.mark.parametrize("corner", [(0, 0), (800, 0), (0, 600), (800, 600)])
def test_right_drag_can_place_every_image_corner_anywhere(view, zoom, corner):
    view.set_zoom(zoom)
    image_corner = QtCore.QPointF(*corner)
    start = QtCore.QPointF(200, 150)
    target = QtCore.QPointF(325.25, 215.75)
    current = (
        image_corner + view.canvas.offset_to_center()
    ) * view.canvas.scale + QtCore.QPointF(view.canvas.pos())
    delta = target - current
    mouse(view, QtCore.QEvent.Type.MouseButtonPress, start)
    mouse(view, QtCore.QEvent.Type.MouseMove, start + delta / 2)
    mouse(view, QtCore.QEvent.Type.MouseMove, start + delta)
    mouse(view, QtCore.QEvent.Type.MouseButtonRelease, start + delta)
    assert_point(image_position(view, target), image_corner)
    view.canvas.menus[0].exec.assert_not_called()
    assert not view.canvas.selected_shapes_copy


def test_right_click_keeps_context_menu_and_does_not_pan(view):
    point = QtCore.QPointF(150, 100)
    mouse(view, QtCore.QEvent.Type.MouseButtonPress, point)
    mouse(view, QtCore.QEvent.Type.MouseMove, point + QtCore.QPointF(1, 1))
    mouse(view, QtCore.QEvent.Type.MouseButtonRelease, point)
    view.canvas.menus[0].exec.assert_called_once()
    assert view.canvas.pan_offset.isNull()


def test_right_drag_preserves_selection_and_ctrl_right_drag_copies(view):
    shape = Shape(label="object", shape_type="rectangle")
    shape.points = [
        QtCore.QPointF(x, y)
        for x, y in [(50, 50), (250, 50), (250, 200), (50, 200)]
    ]
    shape.close()
    shape.selected = True
    view.canvas.shapes = [shape]
    view.canvas.selected_shapes = [shape]
    original = list(shape.points)
    start = QtCore.QPointF(100, 100)
    end = start + QtCore.QPointF(80, 45)
    mouse(view, QtCore.QEvent.Type.MouseButtonPress, start)
    mouse(view, QtCore.QEvent.Type.MouseMove, end)
    mouse(view, QtCore.QEvent.Type.MouseButtonRelease, end)
    assert shape.points == original
    assert view.canvas.selected_shapes == [shape]
    assert not view.canvas.selected_shapes_copy
    ctrl = QtCore.Qt.KeyboardModifier.ControlModifier
    mouse(view, QtCore.QEvent.Type.MouseButtonPress, end, modifiers=ctrl)
    mouse(
        view,
        QtCore.QEvent.Type.MouseMove,
        end + QtCore.QPointF(20, 20),
        modifiers=ctrl,
    )
    assert len(view.canvas.selected_shapes_copy) == 1
    assert shape.points == original


def test_space_pan_keeps_unfinished_polygon_and_image_coordinates(view):
    view.canvas.mode = Canvas.CREATE
    current = Shape(shape_type="polygon")
    current.points = [QtCore.QPointF(30, 30), QtCore.QPointF(70, 50)]
    view.canvas.current = current
    original = list(current.points)
    view.canvas._space_pressed = True
    start = QtCore.QPointF(100, 100)
    end = start + QtCore.QPointF(-150, 200)
    left = QtCore.Qt.MouseButton.LeftButton
    mouse(view, QtCore.QEvent.Type.MouseButtonPress, start, left)
    mouse(view, QtCore.QEvent.Type.MouseMove, end, left)
    mouse(view, QtCore.QEvent.Type.MouseButtonRelease, end, left)
    assert view.canvas.current is current
    assert current.points == original
    assert_point(view.canvas.pan_offset, end - start)


def test_pan_interrupts_zoom_and_navigator_centers_image_coordinates(view):
    wheel(view)
    view._zoom_animation.setCurrentTime(30)
    scale = view.canvas.scale
    view.pan_canvas(QtCore.QPointF(200, -100))
    assert (
        view._zoom_animation.state() == QtCore.QAbstractAnimation.State.Stopped
    )
    assert view.canvas.scale == scale
    view.on_navigator_request(0, 0)
    center = QtCore.QPointF(
        view._canvas_scroll_area.viewport().rect().center()
    )
    assert_point(image_position(view, center), QtCore.QPointF())


def test_shift_wheel_scrolls_and_ctrl_wheel_still_zooms(view):
    scrolls = []
    view.canvas.scroll_request.connect(lambda *args: scrolls.append(args))
    wheel(view, modifiers=QtCore.Qt.KeyboardModifier.ShiftModifier)
    assert scrolls[-1] == (120, QtCore.Qt.Orientation.Vertical, 0)
    assert (
        view._zoom_animation.state() == QtCore.QAbstractAnimation.State.Stopped
    )
    wheel(view, modifiers=QtCore.Qt.KeyboardModifier.ControlModifier)
    assert (
        view._zoom_animation.state() == QtCore.QAbstractAnimation.State.Running
    )


def test_image_switch_restores_fractional_zoom_and_pan_and_fit_resets(
    tmp_path, monkeypatch, navigation_app
):
    from anylabeling import config as app_config
    from anylabeling.views.labeling.settings.schema import load_template_config
    from anylabeling.views.mainwindow import MainWindow

    app = navigation_app
    monkeypatch.setattr(app_config, "_work_directory", str(tmp_path))
    config_file = tmp_path / "config.yaml"
    config_file.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(app_config, "current_config_file", str(config_file))
    settings_class = QtCore.QSettings
    monkeypatch.setattr(
        QtCore,
        "QSettings",
        lambda *args: settings_class(
            str(tmp_path / "settings.ini"), settings_class.Format.IniFormat
        ),
    )
    window = MainWindow(app, config=copy.deepcopy(load_template_config()))
    window.resize(1400, 900)
    window.show()
    app.processEvents()
    widget = window.labeling_widget.view
    try:
        paths = [str(tmp_path / f"image_{index}.png") for index in range(2)]
        image = QtGui.QImage(800, 600, QtGui.QImage.Format.Format_RGB32)
        image.fill(QtGui.QColor("white"))
        for path in paths:
            assert image.save(path)
        assert widget.load_file(paths[0])
        app.processEvents()
        widget.set_zoom(237.125)
        widget.pan_canvas(QtCore.QPointF(245.25, -170.75))
        offset = QtCore.QPointF(widget.canvas.pan_offset)
        widget.zoom_request(120, QtCore.QPointF(100, 100))
        assert widget.load_file(paths[1])
        app.processEvents()
        assert (
            widget._zoom_animation.state()
            == QtCore.QAbstractAnimation.State.Stopped
        )
        assert widget.load_file(paths[0])
        app.processEvents()
        assert widget.canvas.scale == pytest.approx(2.37125)
        assert_point(widget.canvas.pan_offset, offset)
        widget.actions.zoom_org.trigger()
        assert widget.canvas.scale == 1.0
        widget.set_fit_window()
        assert widget.canvas.pan_offset.isNull()
        assert all(bar.value() == 0 for bar in widget.scroll_bars.values())
    finally:
        widget.dirty = False
        window.close()
        app.removeEventFilter(widget)
        window.deleteLater()
        app.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)


def test_zoom_runs_through_intermediate_frames_in_event_loop(view):
    frames = QtTest.QSignalSpy(view._zoom_animation.valueChanged)
    wheel(view)
    QtTest.QTest.qWait(160)
    assert len(frames) >= 3
    assert any(100 < frame[0] < 110 for frame in frames)
    assert view.canvas.scale == pytest.approx(1.1)


@pytest.mark.parametrize(
    "button,modifiers",
    [
        (
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.KeyboardModifier.AltModifier,
        ),
        (
            QtCore.Qt.MouseButton.MiddleButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        ),
    ],
)
def test_drawing_pan_double_press_does_not_finish_polygon(
    view, button, modifiers
):
    view.canvas.mode = Canvas.CREATE
    current = Shape(shape_type="polygon")
    current.points = [
        QtCore.QPointF(x, y)
        for x, y in [(10, 10), (80, 10), (80, 80), (10, 80)]
    ]
    view.canvas.current = current
    original = list(current.points)
    point = QtCore.QPointF(100, 100)
    mouse(
        view, QtCore.QEvent.Type.MouseButtonDblClick, point, button, modifiers
    )
    for kind in [
        QtCore.QEvent.Type.MouseMove,
        QtCore.QEvent.Type.MouseButtonRelease,
    ]:
        mouse(view, kind, point + QtCore.QPointF(100, 100), button, modifiers)
    assert view.canvas.current is current
    assert current.points == original
    assert_point(view.canvas.pan_offset, QtCore.QPointF(100, 100))


def test_right_drag_keeps_brush_active_and_click_finishes_it(view):
    view.canvas.is_brush_mode = True
    release = Mock(return_value=True)
    view.canvas._brush_mouse_release = release
    point = QtCore.QPointF(100, 100)
    mouse(view, QtCore.QEvent.Type.MouseButtonPress, point)
    mouse(view, QtCore.QEvent.Type.MouseMove, point + QtCore.QPointF(60, 40))
    mouse(
        view,
        QtCore.QEvent.Type.MouseButtonRelease,
        point + QtCore.QPointF(60, 40),
    )
    release.assert_not_called()
    mouse(view, QtCore.QEvent.Type.MouseButtonPress, point)
    mouse(view, QtCore.QEvent.Type.MouseButtonRelease, point)
    release.assert_called_once()


def test_left_drag_on_empty_canvas_pans_freely(view):
    view.set_zoom(300)
    view.canvas.pan_offset = QtCore.QPointF()
    start = QtCore.QPointF(100, 100)
    end = QtCore.QPointF(380, 250)
    origin = image_position(view, start)
    left = QtCore.Qt.MouseButton.LeftButton
    mouse(view, QtCore.QEvent.Type.MouseButtonPress, start, left)
    mouse(view, QtCore.QEvent.Type.MouseMove, end, left)
    mouse(view, QtCore.QEvent.Type.MouseButtonRelease, end, left)
    assert_point(image_position(view, end), origin)
    assert_point(view.canvas.pan_offset, end - start)


@pytest.mark.parametrize(
    "value,expected", [(-1, 10), (1, 10), (10, 10), (1000, 1000), (2000, 1000)]
)
def test_main_zoom_control_uses_cvat_range(view, value, expected):
    view.set_zoom(value)
    assert view.zoom_widget.value() == expected
    assert view.canvas.scale == expected / 100


def test_navigator_controls_use_cvat_range(navigation_app):
    navigator = NavigatorDialog()
    changed = QtTest.QSignalSpy(navigator.zoom_changed[int])
    for value, expected in [(1, 10), (10, 10), (1000, 1000), (2000, 1000)]:
        navigator.zoom_input.setText(str(value))
        navigator.on_zoom_input_changed()
        assert navigator.current_zoom == expected
        assert navigator.zoom_slider.value() == expected
        assert navigator.zoom_input.text() == str(expected)
    navigator.set_zoom_value(10)
    navigator.zoom_out()
    event = QtGui.QWheelEvent(
        QtCore.QPointF(50, 50),
        QtCore.QPointF(50, 50),
        QtCore.QPoint(),
        QtCore.QPoint(0, -120),
        QtCore.Qt.MouseButton.NoButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
        QtCore.Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    navigator.handle_wheel_zoom(event)
    assert navigator.current_zoom == 10
    navigator.set_zoom_value(1000)
    navigator.zoom_in()
    assert navigator.current_zoom == 1000
    assert all(10 <= entry[0] <= 1000 for entry in changed)
    navigator.close()
    navigator.deleteLater()
    navigation_app.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)


def selected_rectangle(view):
    shape = Shape(shape_type="rectangle")
    shape.points = [
        QtCore.QPointF(x, y)
        for x, y in [(50, 50), (150, 50), (150, 150), (50, 150)]
    ]
    shape.close()
    view.canvas.shapes = [shape]
    view.canvas.selected_shapes = [shape]
    view.canvas.auto_highlight_shape = False
    return shape


def test_rectangle_wheel_editing_is_disabled_by_default(view):
    from anylabeling.config import get_default_config

    assert not get_default_config()["canvas"]["wheel_rectangle_editing"][
        "enable"
    ]
    shape = selected_rectangle(view)
    points = list(shape.points)
    wheel(view, modifiers=QtCore.Qt.KeyboardModifier.ControlModifier)
    assert shape.points == points
    assert (
        view._zoom_animation.state() == QtCore.QAbstractAnimation.State.Running
    )


def test_plain_wheel_zooms_even_when_rectangle_editing_is_enabled(view):
    shape = selected_rectangle(view)
    points = list(shape.points)
    view.canvas.enable_wheel_rectangle_editing = True
    wheel(view)
    view._zoom_animation.setCurrentTime(90)
    assert view.canvas.scale == pytest.approx(1.1)
    assert shape.points == points
    assert not view.canvas.shapes_backups


@pytest.mark.parametrize("point", [(100, 100), (170, 100)])
@pytest.mark.parametrize("delta", [-120, 120])
def test_ctrl_wheel_edits_rectangle_and_records_undo(view, point, delta):
    shape = selected_rectangle(view)
    original = list(shape.points)
    view.canvas.store_shapes()
    view.canvas.enable_wheel_rectangle_editing = True
    changed = QtTest.QSignalSpy(view.canvas.shape_moved)
    local = (
        QtCore.QPointF(*point) + view.canvas.offset_to_center()
    ) * view.canvas.scale
    wheel(
        view,
        delta=delta,
        modifiers=QtCore.Qt.KeyboardModifier.ControlModifier,
        point=local,
    )
    assert view.canvas.scale == 1
    assert (
        view._zoom_animation.state() == QtCore.QAbstractAnimation.State.Stopped
    )
    assert len(changed) == 1
    if point[0] == 100:
        factor = 1.05 if delta > 0 else 0.95
        assert shape.bounding_rect().width() == pytest.approx(100 * factor)
        assert shape.bounding_rect().height() == pytest.approx(100 * factor)
    else:
        assert shape.bounding_rect().left() == 50
        assert shape.bounding_rect().right() == 150 + (2 if delta > 0 else -2)
    view.canvas.restore_shape()
    assert view.canvas.shapes[0].points == original


def test_locked_rectangle_ctrl_wheel_zooms_without_modifying_shape(view):
    shape = selected_rectangle(view)
    shape.locked = True
    points = list(shape.points)
    view.canvas.enable_wheel_rectangle_editing = True
    wheel(view, modifiers=QtCore.Qt.KeyboardModifier.ControlModifier)
    assert shape.points == points
    assert (
        view._zoom_animation.state() == QtCore.QAbstractAnimation.State.Running
    )


def test_brush_uses_plain_wheel_for_zoom_and_ctrl_wheel_for_radius(view):
    view.canvas.is_brush_mode = True
    radius = view.canvas.brush_radius
    wheel(view)
    view._zoom_animation.setCurrentTime(90)
    assert view.canvas.scale == pytest.approx(1.1)
    assert view.canvas.brush_radius == radius
    wheel(view, modifiers=QtCore.Qt.KeyboardModifier.ControlModifier)
    assert view.canvas.scale == pytest.approx(1.1)
    assert view.canvas.brush_radius == radius + 1


@pytest.mark.parametrize(
    "shape_type",
    [
        "polygon",
        "rectangle",
        "rotation",
        "line",
        "linestrip",
        "quadrilateral",
        "circle",
        "cuboid",
    ],
)
@pytest.mark.parametrize(
    "button,modifiers",
    [
        (
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.KeyboardModifier.AltModifier,
        ),
        (
            QtCore.Qt.MouseButton.RightButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        ),
        (
            QtCore.Qt.MouseButton.MiddleButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        ),
    ],
)
def test_drawing_pan_preserves_draft_and_resumes_at_correct_coordinates(
    view, shape_type, button, modifiers
):
    view.canvas.set_editing(False)
    view.canvas.create_mode = shape_type
    left = QtCore.Qt.MouseButton.LeftButton
    start = QtCore.QPointF(100, 100)
    end = QtCore.QPointF(320, 220)
    mouse(view, QtCore.QEvent.Type.MouseButtonPress, start, left)
    current = view.canvas.current
    original = list(current.points)
    for kind, pos in [
        (QtCore.QEvent.Type.MouseButtonPress, start),
        (QtCore.QEvent.Type.MouseMove, end),
        (QtCore.QEvent.Type.MouseButtonRelease, end),
    ]:
        mouse(view, kind, pos, button, modifiers)
    assert view.canvas.current is current
    assert current.points == original
    assert_point(view.canvas.pan_offset, end - start)
    assert not view.canvas.shapes
    assert not view.canvas.shapes_backups
    view.canvas.menus[0].exec.assert_not_called()
    view.canvas.menus[1].exec.assert_not_called()
    next_pos = QtCore.QPointF(350, 250)
    expected = image_position(view, next_pos)
    mouse(
        view,
        QtCore.QEvent.Type.MouseMove,
        next_pos,
        QtCore.Qt.MouseButton.NoButton,
    )
    assert_point(view.canvas.line.points[-1], expected)


@pytest.mark.parametrize("tool", ["brush", "magic_wand"])
def test_middle_pan_does_not_paint_or_start_magic_wand(view, tool):
    if tool == "brush":
        view.canvas.is_brush_mode = True
    else:
        view.canvas.set_editing(False)
        view.canvas.is_magic_wand_mode = True
    handlers = [
        "_brush_mouse_press",
        "_brush_mouse_move",
        "_brush_mouse_release",
        "_start_magic_wand",
        "_drag_magic_wand",
    ]
    for name in handlers:
        setattr(view.canvas, name, Mock())
    start = QtCore.QPointF(100, 100)
    end = QtCore.QPointF(200, 180)
    for kind, pos in [
        (QtCore.QEvent.Type.MouseButtonPress, start),
        (QtCore.QEvent.Type.MouseMove, end),
        (QtCore.QEvent.Type.MouseButtonRelease, end),
    ]:
        mouse(view, kind, pos, QtCore.Qt.MouseButton.MiddleButton)
    assert_point(view.canvas.pan_offset, end - start)
    for name in handlers:
        getattr(view.canvas, name).assert_not_called()


@pytest.mark.parametrize("shape_type", Shape.get_supported_shape())
def test_right_drag_pans_before_drawing_and_click_opens_menu(view, shape_type):
    view.canvas.set_editing(False)
    view.canvas.create_mode = shape_type
    point = QtCore.QPointF(100, 100)
    mouse(view, QtCore.QEvent.Type.MouseButtonPress, point)
    mouse(view, QtCore.QEvent.Type.MouseMove, point + QtCore.QPointF(100, 100))
    mouse(
        view,
        QtCore.QEvent.Type.MouseButtonRelease,
        point + QtCore.QPointF(100, 100),
    )
    assert_point(view.canvas.pan_offset, QtCore.QPointF(100, 100))
    assert view.canvas.current is None
    view.canvas.menus[0].exec.assert_not_called()
    mouse(view, QtCore.QEvent.Type.MouseButtonPress, point)
    mouse(view, QtCore.QEvent.Type.MouseButtonRelease, point)
    view.canvas.menus[0].exec.assert_called_once()
