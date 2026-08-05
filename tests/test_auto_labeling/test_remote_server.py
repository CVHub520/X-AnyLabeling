import tempfile
import unittest
from unittest.mock import Mock, patch

from anylabeling.services.auto_labeling.remote_server import RemoteServer


class TestRemoteServerClassFilter(unittest.TestCase):
    def setUp(self):
        with patch(
            "anylabeling.services.auto_labeling.model.get_config",
            return_value={"remote_server_settings": {}},
        ):
            self.model = RemoteServer(
                {
                    "type": "remote_server",
                    "display_name": "Remote Server",
                    "timeout": 30,
                },
                Mock(),
            )
        self.model.models_info = {
            "yolo": {
                "classes": ["person", "car", "dog"],
                "filter_classes": ["person", "dog"],
            }
        }

    def test_model_selection_loads_remote_class_metadata(self):
        self.model.set_model_id("yolo")

        self.assertEqual(self.model.classes, ["person", "car", "dog"])
        self.assertEqual(self.model.filter_classes, ["person", "dog"])

    def test_prediction_forwards_selected_classes(self):
        self.model.set_model_id("yolo")
        self.model.set_auto_labeling_filter_classes(["car"])
        response = Mock()
        response.json.return_value = {"data": {"shapes": []}}

        with tempfile.NamedTemporaryFile(suffix=".jpg") as image_file:
            image_file.write(b"image")
            image_file.flush()
            with patch(
                "anylabeling.services.auto_labeling.remote_server.requests.post",
                return_value=response,
            ) as post:
                self.model.predict_shapes(object(), image_file.name)

        params = post.call_args.kwargs["json"]["params"]
        self.assertEqual(params["filter_classes"], ["car"])

    def test_selecting_all_classes_sends_empty_filter_override(self):
        self.model.set_model_id("yolo")
        self.model.set_auto_labeling_filter_classes(self.model.classes)
        response = Mock()
        response.json.return_value = {"data": {"shapes": []}}

        with tempfile.NamedTemporaryFile(suffix=".jpg") as image_file:
            image_file.write(b"image")
            image_file.flush()
            with patch(
                "anylabeling.services.auto_labeling.remote_server.requests.post",
                return_value=response,
            ) as post:
                self.model.predict_shapes(object(), image_file.name)

        params = post.call_args.kwargs["json"]["params"]
        self.assertEqual(params["filter_classes"], [])


if __name__ == "__main__":
    unittest.main()
