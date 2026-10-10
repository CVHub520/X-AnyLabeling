"""Verify Chatbot batch label loading and annotation preservation."""

import json
import os
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

from PIL import Image

from anylabeling.app_info import __version__
from anylabeling.views.labeling.label_file import LabelFile
from anylabeling.views.labeling.widgets import chatbot_dialog


class TestChatbotBatchSave(unittest.TestCase):
    """Exercise batch saving and label loading with mocked UI components."""

    def setUp(self) -> None:
        """Isolate files and dialogs and simulate reloading the current label."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = directory.name
        self.parent = SimpleNamespace(image_list=[], filename=None)
        self.dialog = SimpleNamespace(
            batch_results={},
            batch_prompt="<image>描述图像",
            batch_lock=threading.Lock(),
            batch_complete=True,
            current_index=0,
            progress_timer=mock.Mock(),
            progress_dialog=mock.Mock(),
            parent=mock.Mock(return_value=self.parent),
            tr=lambda message: message,
            navigate_image=mock.Mock(
                side_effect=lambda index: LabelFile(
                    os.path.splitext(self.parent.filename)[0] + ".json"
                )
            ),
        )
        popup = mock.patch.object(chatbot_dialog.QMessageBox, "information")
        self.information = popup.start()
        self.addCleanup(popup.stop)

    def test_batch_without_json_creates_reloadable_labels(self) -> None:
        """Verify batch results can be reloaded without existing label files."""
        images = [
            ("原图一.jpg", "JPEG", (386, 860)),
            ("原图二.png", "PNG", (32, 24)),
        ]
        for name, image_format, size in images:
            image_path = os.path.join(self.directory, name)
            Image.new("RGB", size, "white").save(
                image_path, format=image_format
            )
            self.parent.image_list.append(image_path)
            self.dialog.batch_results[image_path] = {"content": "这是一张测试图片。"}

        # Reload through LabelFile to catch the original 'shapes' error.
        chatbot_dialog.ChatbotDialog.finish_batch_processing(self.dialog)

        for name, _, size in images:
            with self.subTest(image=name):
                image_path = os.path.join(self.directory, name)
                label_path = os.path.splitext(image_path)[0] + ".json"
                with open(label_path, encoding="utf-8") as stream:
                    data = json.load(stream)
                self.assertEqual(data["version"], __version__)
                self.assertEqual(data["flags"], {})
                self.assertFalse(data["checked"])
                self.assertEqual(data["shapes"], [])
                self.assertEqual(data["imagePath"], name)
                self.assertIsNone(data["imageData"])
                self.assertEqual(
                    (data["imageWidth"], data["imageHeight"]), size
                )
                expected_history = [
                    {
                        "role": "user",
                        "content": self.dialog.batch_prompt,
                        "image": image_path,
                    },
                    {
                        "role": "assistant",
                        "content": "这是一张测试图片。",
                        "image": None,
                    },
                ]
                self.assertEqual(data["chat_history"], expected_history)
                loaded = LabelFile(label_path)
                self.assertEqual(loaded.shapes, [])
                self.assertEqual(loaded.image_path, name)
                self.assertEqual(
                    loaded.other_data["chat_history"], expected_history
                )
                with open(image_path, "rb") as stream:
                    self.assertEqual(loaded.image_data, stream.read())

        self.dialog.navigate_image.assert_called_once_with(index=0)
        self.information.assert_called_once()

    def test_batch_preserves_existing_annotation(self) -> None:
        """Verify updating chat history preserves existing annotations."""
        image_path = os.path.join(self.directory, "image.png")
        label_path = os.path.join(self.directory, "image.json")
        Image.new("RGB", (32, 24), "white").save(image_path)
        LabelFile().save(
            filename=label_path,
            shapes=[
                {
                    "label": "object",
                    "points": [[1, 2], [3, 2], [3, 4], [1, 4]],
                    "shape_type": "rectangle",
                    "flags": {},
                    "group_id": 7,
                }
            ],
            flags={"reviewed": True},
            image_path="image.png",
            image_height=24,
            image_width=32,
            other_data={
                "checked": True,
                "description": "保留原描述",
                "custom": {"source": "manual"},
                "chat_history": [{"role": "assistant", "content": "旧回答"}],
            },
        )
        with open(label_path, encoding="utf-8") as stream:
            original = json.load(stream)
        self.parent.image_list.append(image_path)
        self.dialog.batch_results[image_path] = {"content": "新回答"}

        chatbot_dialog.ChatbotDialog.finish_batch_processing(self.dialog)

        with open(label_path, encoding="utf-8") as stream:
            saved = json.load(stream)
        self.assertEqual(
            {
                key: value
                for key, value in saved.items()
                if key != "chat_history"
            },
            {
                key: value
                for key, value in original.items()
                if key != "chat_history"
            },
        )
        loaded = LabelFile(label_path)
        self.assertEqual(len(loaded.shapes), 1)
        self.assertEqual(loaded.shapes[0].label, "object")
        self.assertEqual(loaded.shapes[0].group_id, 7)
        self.assertEqual(
            loaded.other_data["chat_history"][1]["content"], "新回答"
        )
        self.dialog.navigate_image.assert_called_once_with(index=0)


if __name__ == "__main__":
    unittest.main()
