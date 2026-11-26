import unittest
from unittest.mock import Mock

import numpy as np
from PyQt6.QtGui import QImage

from anylabeling.services.auto_labeling.ppocr_v4 import PPOCRv4
from anylabeling.services.auto_labeling.ppocr_v5 import PPOCRv5
from anylabeling.services.auto_labeling.ppocr_v6 import PPOCRv6


class TestPPOCRShapes(unittest.TestCase):
    def test_detection_boxes_keep_their_geometry(self):
        cases = [
            ([[10, 10], [110, 10], [110, 30], [10, 30]], "rectangle"),
            (
                [[100, 100], [300, 60], [310, 110], [110, 150]],
                "quadrilateral",
            ),
            (
                [[100, 100], [300, 60], [310, 111], [110, 150]],
                "quadrilateral",
            ),
            ([[10, 10], [110, 10], [130, 30], [30, 30]], "quadrilateral"),
            ([[100, 100], [200, 50], [225, 100], [125, 150]], "quadrilateral"),
        ]
        image = QImage(400, 200, QImage.Format.Format_RGB32)
        image.fill(0)

        for model_class in (PPOCRv4, PPOCRv5, PPOCRv6):
            for points, shape_type in cases:
                with self.subTest(model=model_class.__name__, points=points):
                    model = model_class.__new__(model_class)
                    model.text_sys = Mock(
                        return_value=(
                            np.array([points], dtype=np.float32),
                            [("recognized text", 0.99)],
                            [0.99],
                            [0],
                        )
                    )

                    result = model.predict_shapes(image)

                    self.assertTrue(result.replace)
                    self.assertEqual(len(result.shapes), 1)
                    shape = result.shapes[0]
                    self.assertEqual(shape.shape_type, shape_type)
                    self.assertEqual(
                        [[p.x(), p.y()] for p in shape.points], points
                    )
                    self.assertEqual(shape.label, "text")
                    self.assertEqual(shape.description, "recognized text")
                    self.assertEqual(shape.score, 0.99)
                    self.assertEqual(shape.group_id, 0)


if __name__ == "__main__":
    unittest.main()
