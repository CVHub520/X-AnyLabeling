import json
import platform
import sys
import sysconfig
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import pyclipper
from onnx import TensorProto, helper
from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtWebEngineWidgets import QWebEngineView

from anylabeling.services.updates.protocol import architecture
from anylabeling.views.labeling.widgets.update_panel import UpdatePanel

output = Path(".validation-output")
output.mkdir(exist_ok=True)
left = helper.make_tensor_value_info("left", TensorProto.FLOAT, [2, 2])
right = helper.make_tensor_value_info("right", TensorProto.FLOAT, [2, 2])
result = helper.make_tensor_value_info("result", TensorProto.FLOAT, [2, 2])
model = helper.make_model(
    helper.make_graph(
        [helper.make_node("Add", ["left", "right"], ["result"])],
        "native",
        [left, right],
        [result],
    ),
    opset_imports=[helper.make_opsetid("", 13)],
)
model.ir_version = 8
session = ort.InferenceSession(
    model.SerializeToString(), providers=["CPUExecutionProvider"]
)
values = np.ones((2, 2), dtype=np.float32)
np.testing.assert_array_equal(
    session.run(None, {"left": values, "right": values})[0], values * 2
)
image = np.zeros((8, 8), dtype=np.uint8)
image[2:6, 2:6] = 255
assert cv2.ximgproc.thinning(image).shape == image.shape
assert cv2.resize(image, (4, 4)).shape == (4, 4)
clipper = pyclipper.Pyclipper()
clipper.AddPath(
    [(0, 0), (10, 0), (10, 10), (0, 10)], pyclipper.PT_SUBJECT, True
)
assert clipper.Execute(pyclipper.CT_UNION)
app = QApplication([])
parent = QWidget()
panel = UpdatePanel(parent)
panel.set_state("available", "4.1.0")
panel.show()
app.processEvents()
assert panel.grab().save(str(output / "update-panel.png"))
panel.close()
report = {
    "python": sys.version,
    "machine": platform.machine(),
    "python_platform": sysconfig.get_platform(),
    "update_architecture": architecture(),
    "opencv": cv2.__version__,
    "onnxruntime": ort.__version__,
    "providers": session.get_providers(),
    "checks": [
        "ONNX CPU inference",
        "OpenCV contrib",
        "polygon clipping",
        "Qt panel render",
        "Qt WebEngine import",
    ],
}
(output / "native-checks.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
