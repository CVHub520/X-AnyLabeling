import os
import tempfile
import threading
import unittest
from unittest import mock

from anylabeling.views.labeling.chatbot import provider
from anylabeling.views.labeling.chatbot import utils as chatbot_utils
from anylabeling.views.labeling.widgets import chatbot_dialog
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

    def test_save_model_selection_updates_provider_and_clears_old_model(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = os.path.join(directory, "models.json")
            self._write_config(
                config_path,
                {
                    "ollama": {
                        "llava": {
                            "vision": True,
                            "selected": True,
                            "favorite": False,
                        }
                    }
                },
            )

            with mock.patch.object(
                provider, "get_models_config_path", return_value=config_path
            ):
                provider.save_model_selection("custom", "gpt-4o")

            total_data = chatbot_utils.load_json(config_path)
            self.assertEqual(total_data["settings"]["provider"], "custom")
            self.assertEqual(total_data["settings"]["model_id"], "gpt-4o")
            self.assertFalse(
                total_data["models_data"]["ollama"]["llava"]["selected"]
            )


class TestChatbotProviderSwitch(unittest.TestCase):
    @staticmethod
    def _field(text=""):
        field = mock.Mock()
        field.text.return_value = text
        return field

    def _dialog(self, custom_model=""):
        dialog = mock.Mock()
        dialog.providers = {
            "custom": {
                "api_address": "https://example.com/v1",
                "api_key": "secret",
                "api_key_url": None,
                "api_docs_url": None,
                "model_docs_url": None,
            }
        }
        dialog.default_provider = "ollama"
        dialog.selected_model = "llava"
        dialog.api_address = self._field()
        dialog.api_key = self._field()
        dialog.custom_model_input = self._field(custom_model)
        dialog.model_button = self._field("llava (ollama)")
        dialog.model_dropdown = mock.Mock()
        dialog.findChild.return_value = None
        return dialog

    def test_switch_to_custom_uses_manual_model_name(self):
        dialog = self._dialog("gpt-4o")

        with (
            mock.patch.object(chatbot_dialog, "get_models_data") as get_models,
            mock.patch.object(
                chatbot_dialog, "save_model_selection"
            ) as save_selection,
        ):
            chatbot_dialog.ChatbotDialog.switch_provider(dialog, "custom")

        self.assertEqual(dialog.default_provider, "custom")
        self.assertEqual(dialog.selected_model, "gpt-4o")
        dialog.custom_model_input.setVisible.assert_called_once_with(True)
        dialog.model_button.setVisible.assert_called_once_with(False)
        get_models.assert_not_called()
        save_selection.assert_called_once_with("custom", "gpt-4o")

    def test_switch_to_custom_clears_stale_model(self):
        dialog = self._dialog()

        with (
            mock.patch.object(chatbot_dialog, "get_models_data") as get_models,
            mock.patch.object(
                chatbot_dialog, "save_model_selection"
            ) as save_selection,
        ):
            chatbot_dialog.ChatbotDialog.switch_provider(dialog, "custom")

        self.assertIsNone(dialog.selected_model)
        get_models.assert_not_called()
        save_selection.assert_called_once_with("custom", None)

    def test_custom_model_name_is_persisted(self):
        dialog = mock.Mock()
        dialog.default_provider = "custom"
        dialog.providers = {"custom": {}}

        with (
            mock.patch.object(chatbot_dialog, "save_json") as save_json,
            mock.patch.object(
                chatbot_dialog,
                "get_providers_config_path",
                return_value="providers.json",
            ),
            mock.patch.object(
                chatbot_dialog, "save_model_selection"
            ) as save_selection,
        ):
            chatbot_dialog.ChatbotDialog.on_custom_model_changed(
                dialog, "  claude-3-opus  "
            )

        self.assertEqual(dialog.selected_model, "claude-3-opus")
        self.assertEqual(
            dialog.providers["custom"]["model_name"], "claude-3-opus"
        )
        save_json.assert_called_once_with(dialog.providers, "providers.json")
        save_selection.assert_called_once_with("custom", "claude-3-opus")


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
