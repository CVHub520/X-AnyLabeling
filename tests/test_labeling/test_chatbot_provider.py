import os
import tempfile
import threading
import unittest
from unittest import mock

from anylabeling.views.labeling.chatbot import provider
from anylabeling.views.labeling.chatbot import utils as chatbot_utils
from anylabeling.views.labeling.widgets import model_dropdown_widget


class TestChatbotProviderRefresh(unittest.TestCase):
    @staticmethod
    def _write_config(config_path, models_data=None):
        chatbot_utils.save_json(
            {
                "settings": {
                    "provider": "initial_provider",
                    "model_id": "initial_model",
                },
                "models_data": models_data or {},
                "supported_vision_models": [],
            },
            config_path,
        )

    def test_concurrent_provider_refreshes_preserve_both_updates(self):
        barrier = threading.Barrier(2)
        errors = []

        def get_models(base_url, api_key, timeout):
            barrier.wait(timeout=2)
            return [{"id": f"{base_url}_model"}]

        def refresh(provider_name, config_path):
            try:
                provider._refresh_models_data(
                    provider_name, provider_name, "api_key", config_path
                )
            except Exception as error:
                errors.append(error)

        with tempfile.TemporaryDirectory() as directory:
            config_path = os.path.join(directory, "models.json")
            self._write_config(config_path)
            threads = [
                threading.Thread(
                    target=refresh, args=(provider_name, config_path)
                )
                for provider_name in ("provider_a", "provider_b")
            ]

            with mock.patch.object(
                provider, "get_models_raw_data", side_effect=get_models
            ):
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(timeout=5)

            self.assertTrue(all(not thread.is_alive() for thread in threads))
            self.assertEqual(errors, [])
            total_data = chatbot_utils.load_json(config_path)
            self.assertEqual(
                set(total_data["models_data"]), {"provider_a", "provider_b"}
            )
            self.assertEqual(
                total_data["settings"]["provider"], "initial_provider"
            )

    def test_refresh_preserves_settings_and_favorites_saved_during_fetch(self):
        fetch_started = threading.Event()
        finish_fetch = threading.Event()
        errors = []

        def get_models(base_url, api_key, timeout):
            fetch_started.set()
            if not finish_fetch.wait(timeout=5):
                raise TimeoutError("model fetch was not released")
            return [{"id": "existing_model"}]

        with tempfile.TemporaryDirectory() as directory:
            config_path = os.path.join(directory, "models.json")
            self._write_config(
                config_path,
                {
                    "provider_a": {
                        "existing_model": {
                            "vision": False,
                            "selected": False,
                            "favorite": False,
                        }
                    }
                },
            )

            def refresh():
                try:
                    provider._refresh_models_data(
                        "provider_a", "base_url", "api_key", config_path
                    )
                except Exception as error:
                    errors.append(error)

            thread = threading.Thread(target=refresh)
            with mock.patch.object(
                provider, "get_models_raw_data", side_effect=get_models
            ):
                thread.start()
                self.assertTrue(fetch_started.wait(timeout=2))
                with chatbot_utils.MODELS_CONFIG_LOCK:
                    total_data = chatbot_utils.load_json(config_path)
                    total_data["settings"] = {
                        "provider": "provider_a",
                        "model_id": "existing_model",
                    }
                    total_data["models_data"]["provider_a"]["existing_model"][
                        "favorite"
                    ] = True
                    chatbot_utils.save_json(total_data, config_path)
                finish_fetch.set()
                thread.join(timeout=5)

            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])
            total_data = chatbot_utils.load_json(config_path)
            self.assertEqual(total_data["settings"]["provider"], "provider_a")
            self.assertTrue(
                total_data["models_data"]["provider_a"]["existing_model"][
                    "favorite"
                ]
            )


class TestChatbotJsonSave(unittest.TestCase):
    def test_failed_save_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = os.path.join(directory, "models.json")
            original_data = '{"original": true}\n'
            with open(config_path, "w", encoding="utf-8") as config_file:
                config_file.write(original_data)

            def fail_after_partial_write(data, config_file, **kwargs):
                config_file.write('{"partial":')
                raise OSError("simulated write failure")

            with mock.patch.object(
                chatbot_utils.json,
                "dump",
                side_effect=fail_after_partial_write,
            ):
                with self.assertRaises(OSError):
                    chatbot_utils.save_json({"updated": True}, config_path)

            with open(config_path, "r", encoding="utf-8") as config_file:
                self.assertEqual(config_file.read(), original_data)
            self.assertEqual(os.listdir(directory), ["models.json"])


class TestModelDropdownSave(unittest.TestCase):
    def test_favorite_update_preserves_refreshed_models(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = os.path.join(directory, "models.json")
            chatbot_utils.save_json(
                {
                    "settings": {
                        "provider": "provider_a",
                        "model_id": "existing_model",
                    },
                    "models_data": {
                        "provider_a": {
                            "existing_model": {
                                "vision": False,
                                "selected": True,
                                "favorite": False,
                            }
                        },
                        "provider_b": {
                            "refreshed_model": {
                                "vision": True,
                                "selected": False,
                                "favorite": False,
                            }
                        },
                    },
                    "supported_vision_models": [],
                },
                config_path,
            )
            dropdown = mock.Mock()
            dropdown.models_data = {"provider_a": {}, "provider_b": {}}

            with mock.patch.object(
                model_dropdown_widget,
                "get_models_config_path",
                return_value=config_path,
            ):
                model_dropdown_widget.ModelDropdown.save_models_data(
                    dropdown, "provider_a", "existing_model", True
                )

            total_data = chatbot_utils.load_json(config_path)
            self.assertTrue(
                total_data["models_data"]["provider_a"]["existing_model"][
                    "favorite"
                ]
            )
            self.assertIn(
                "refreshed_model", total_data["models_data"]["provider_b"]
            )


if __name__ == "__main__":
    unittest.main()
