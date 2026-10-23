# Rotatable ellipse validation

Validated on 2026-10-23 against official `CVHub520/X-AnyLabeling` main:
`673b8f1066fc5344fcc521f6dac3d3de73d76472` (v4.2.0).
The final upstream fetch returned the same commit. This report describes local
validation; it does not claim upstream CI approval.

See the [English feature guide](../../examples/rotatable_ellipse/README.md)
and [Chinese guide](../../examples/rotatable_ellipse/README_zh-CN.md) for the
native protocol, configuration, compatibility matrix and output semantics.

## Environment and comparison method

- Windows 11, build 26100; Python 3.12.10; PyQt6/Qt 6.11.0;
  pytest 9.1.1. Automated Qt checks used `QT_QPA_PLATFORM=offscreen`.
- Official and contribution checkouts used the same isolated test interpreter
  and separate user configuration/cache directories. Existing installed
  dependency packages were reused read-only; this was not a clean-machine
  dependency installation or packaged executable test.
- The official pytest options in `pyproject.toml`, including doctest collection,
  were retained. The first full baseline run aborted in native point-cloud/Qt
  code. Tests were then run in separate processes by directory, and the affected
  point-cloud/settings directories by file, with a 240-second process timeout.
- Later complete runs replace earlier results for the same test module.
  Counts below are deduplicated test identifiers, not sums of repeated runs.
  Original skip behavior was retained. No v4.0.6 test waiver was used.

## Results

| Check | Result |
| --- | --- |
| Final feature/configuration/export regression selection | 337 passed, 9 skipped, 13 subtests passed |
| Completed official baseline identifiers | 1122 passed, 79 failed, 39 skipped |
| Completed contribution identifiers | 1358 passed, 79 failed, 39 skipped |
| New failed identifiers relative to official baseline | 0 |
| Baseline identifiers missing from completed contribution results | 0 |
| Official `bash scripts/format_code.sh` | Exit 0 |
| New geometry, safe-output and parameter-bar modules: flake8 | Exit 0 using repository configuration |
| New function type-hint/docstring audit | No missing annotations, return annotations or docstrings |
| `git diff --check` | Passed |
| Translation and resource compilation | Completed using upstream scripts and Qt resource tools 6.9.3 |
| Preserved r4 executable/archive/source inventory | 1071 files, all SHA-256 values unchanged |

The full suite is **not green**. The same 79 failures are present in the
current official baseline: 67 point-cloud, 4 labeling, 3 auto-training,
2 auto-labeling, 2 widget and 1 canvas test. Observed causes include Windows
temporary-file sharing, Windows path expectations, Qt toolbar/attribute/font
behavior, point-cloud UI assumptions and an event-timer assertion. The timer
failure was reproduced in both checkouts in a controlled repeat.

Four point-cloud files did not produce complete result XML in either checkout:

| File under `tests/test_pointcloud/` | Both checkouts |
| --- | --- |
| `test_cuboids.py` | Native exit `0xC0000409` |
| `test_export_dialog.py` | Process timed out after 240 seconds |
| `test_perspective.py` | Native exit `0xC0000005` |
| `test_ui_details.py` | Native exit `0xC0000409` |

These files are incomplete checks, not passing tests or accepted xfails.
The original monolithic pytest run and the directory runs that aborted are
also retained in the local evidence package. Passing isolated tests cannot
establish that a single-process full suite succeeds.

Lint across all changed legacy modules is also not clean: the official
baseline has 292 diagnostics and the contribution has 285. The difference
contains changed complexity scores in eight already-over-limit legacy
functions (`main`, dataset preparation, CLI conversion, COCO conversion,
three GUI exporters and shape conversion). Existing star-import warnings
remain. There are no new undefined-name or redefinition diagnostics in this
comparison. This is not a claim that all touched files pass flake8.

## Reproduction

Install the normal upstream development dependencies in a suitable environment.
Use the same interpreter/dependencies for both checkouts. Run the official entry:

```bash
python -m pytest
bash scripts/format_code.sh
python scripts/compile_languages.py
```

The final focused selection was:

```bash
python -m pytest -q \
  tests/test_canvas/test_ellipse_regressions.py \
  tests/test_canvas/test_rotatable_ellipse.py \
  tests/test_config \
  tests/test_labeling/test_label_file_load.py \
  tests/test_labeling/test_label_file_save.py \
  tests/test_labeling/test_label_widget_brush_mode.py \
  tests/test_settings \
  tests/test_utils/test_ellipse_exports.py \
  tests/test_utils/test_ellipse_geometry.py \
  tests/test_utils/test_export.py \
  tests/test_utils/test_label_converter.py
```

On this Windows host the process-isolation driver, exact selected module list,
JUnit XML, stdout logs, baseline failure messages and comparison scripts are
included in the local delivery evidence. Do not interpret the focused run as
a replacement for the incomplete full-suite/platform checks above.

## Geometry and compatibility evidence

Automated checks cover all eight handles at zoom 0.25, 1 and 4, drag/two-click
creation, selection across empty canvas, live copy/paste/undo targets,
rotation-handle movement, final-axis Shift snapping, numeric commit/cancel,
one undo per edit, lock/hide guards and cancellation during tool changes.
Native save/load tests preserve four corners and unknown metadata and reject
invalid direction types, nonfinite coordinates and invalid/degenerate frames.
Tolerance tests exercise both sides of positional and orthogonality limits.

The read-only archived r4 fixture and a second file **actually saved by the
preserved r4 Windows executable** are both loaded and round-tripped. The r4
visible handle edit changed full height 104 to 115.43969665496954 pixels;
width 200 and angle 25 degrees were retained. The contribution version loaded,
saved and reopened that output without changing its four corners. Fixture
provenance and hashes are in [the fixture README](../../tests/fixtures/ellipse/README.md).
Loading did not rewrite the original file.

Sampling tests use 12, 64, 128 and 512 points, axis ratios 1, 2, 100 and 1000,
and horizontal, arbitrary, near-90-degree and near-360-degree orientations.
Unclipped area is checked against the independent analytic expression
`a*b*N*sin(2*pi/N)/2`; contour-distance checks use
`a*(1-cos(pi/N))`. Expected relative area losses are:

| Points | Area loss |
| --- | --- |
| 12 | 4.507034% |
| 64 | 0.160561% |
| 128 | 0.0401547% |
| 512 | 0.00250995% |

Clipped areas are compared with independent numerical integration of an
analytic ellipse cross section. Error is bounded using the original ellipse
area, not a potentially tiny visible cap. Nested 64/128/256/512 contours are
checked for containment and nondecreasing clipped area. Tests check valid
topology, bounds, enclosing-image cases, outside and tangent degeneracy;
the unclipped distance bound is not applied to clipped tangent boundaries.

Mixed polygon/quadrilateral/ellipse documents are exported through YOLO Seg,
COCO, VOC polygon XML, MASK and custom MOTS. Tests compare contours before
format encoding/rasterization, include an out-of-bounds quadrilateral and
verify the configured count, class/group policy, COCO union geometry,
mask priority and unchanged native source bytes. Segment data preparation
uses the same count and class mapping. Mask tests explicitly assert that
ellipse centers are foreground and control-box corners outside the ellipse
remain background.

Failures for incompatible modes, missing mapping/classes, invalid geometry,
changed source inputs, replacement permission errors and cancellation are
tested. Existing targets and unrelated directory files remain intact.
Safe replacement is per file; multi-file formats do not have group rollback.
Native conversion retains metadata and removes direction. External format
imports remain polygon imports and cannot recover native ellipse parameters.

## Visible Windows acceptance and remaining checks

Visible source-application sessions with isolated configuration verified
Chinese/English controls, eight handles and the parameter bar, resize/rotate,
copy followed by handle editing and one-step undo, Shift+E creation with
immediate editing, auto-save/reopen and the shared settings page. The r4
executable save above was also exercised visibly.

A real GUI YOLO Seg export produced four objects with 4, 128, 3 and 128
vertices and preserved the source JSON hash. Restarting the application
verified the corrected success/cancel notification: success=1, failed=0,
skipped=0, unprocessed=0. A final presentation adjustment increases affected
report popup height; it is covered by the final export regression run, but
was not separately rerun in a visible session. Screenshots in the local
delivery record the executed sessions; the feature guide includes a generic
English screenshot.

The following were **not** executed: Linux/macOS GUI or platform tests,
a clean Windows machine, a packaged EXE/build, actual model inference or
training, and an exhaustive visible manual replay of every interaction
combination (including every lock/delete/image-switch permutation). Automated
Qt event checks and visible acceptance are distinct evidence. Physical
power-loss/filesystem-failure testing was not performed; write failures were
injected in tests. No standard MOTS RLE or arbitrary polygon fitting was added.

The local contribution contains no packaging fixes, dependencies, executable,
model weights or TableSeg business feature. The two r4 JSON fixtures retain
their original provenance solely for backward-compatibility tests. No PR
was published and the CLA checkbox remains unchecked for the contributor.
