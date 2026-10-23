# r4 native ellipse fixture

`r4.json` is an unmodified copy of `examples/table_seg/桌面示例.json`
from the pre-existing X-AnyLabeling v4.0.6 TableSeg r4 source archive.
It contains a quadrilateral and an editable rotated ellipse. Its original
image reference and metadata are intentionally retained for compatibility
testing; this fixture is not a new TableSeg feature or a training dataset.

SHA-256: `e14e4eb583895f16e69ad47fcf39efa0ed2e3156fa3955190c3a4980fde06c92`.

`r4_gui_saved.json` was subsequently written by the preserved r4 Windows
executable, using a copy of that same sample in an isolated directory.
A visible handle drag changed the ellipse's full height from 104 to
115.43969665496954 pixels, retaining width 200 and angle 25 degrees.
The contribution version loaded, saved and reopened that actual r4 output
without changing its four corners. The original archived file was untouched.

SHA-256: `bee365225f1c7a1bc0e91253eec21e684201d3b029c218353f7ad25927f606e2`.
