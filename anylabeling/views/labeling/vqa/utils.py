import base64
import json
import os
import re
import requests

from PyQt5.QtCore import QThread, pyqtSignal

from anylabeling.views.labeling.chatbot.config import (
    MODELS_CONFIG_PATH,
    PROVIDERS_CONFIG_PATH,
)
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.vqa.config import REQUEST_TIMEOUT


class AIWorkerThread(QThread):
    """Worker thread for AI API calls"""

    finished = pyqtSignal(str, bool, str)  # result, success, error_message

    def __init__(
        self,
        prompt,
        current_text,
        config,
        image_path=None,
        components=None,
        parent=None,
    ):
        super().__init__()
        self.prompt = prompt
        self.current_text = current_text
        self.config = config
        self.image_path = image_path
        self.components = components or []
        self.parent = parent
        self._is_cancelled = False

    def run(self):
        try:
            if self._is_cancelled:
                return

            models_config = self.load_models_config()
            providers_config = self.load_providers_config()

            if not models_config or not providers_config:
                self.finished.emit("", False, "Configuration files not found")
                return

            settings = models_config.get("settings", {})
            provider = settings.get("provider")
            model_id = settings.get("model_id")
            max_tokens = settings.get("max_length", 2048)
            temperature = settings.get("temperature", 0.7)
            system_prompt = settings.get("system_prompt", None)

            if not provider or not model_id:
                self.finished.emit(
                    "",
                    False,
                    "Please configure model and provider in Chatbot (Ctrl+0)",
                )
                return

            provider_info = providers_config.get(provider, {})
            api_address = provider_info.get("api_address")
            api_key = provider_info.get("api_key")

            if not api_address or not api_key:
                self.finished.emit(
                    "",
                    False,
                    f"Please configure API key for {provider} in Chatbot (Ctrl+0)",
                )
                return

            if self._is_cancelled:
                return

            # Process special character references
            processed_prompt = self.process_special_references(self.prompt)

            result = self.call_openai_api(
                api_address,
                api_key,
                model_id,
                processed_prompt,
                temperature,
                max_tokens,
                system_prompt,
            )
            logger.debug(f"Completion: {result}")

            if not self._is_cancelled:
                self.finished.emit(result, True, "")

        except Exception as e:
            if not self._is_cancelled:
                self.finished.emit("", False, f"API call failed: {str(e)}")

    def process_special_references(self, prompt):
        if "@text" in prompt and self.current_text:
            prompt = prompt.replace("@text", self.current_text)

        for component in self.components:
            widget_ref = f"@widget.{component['title']}"
            if widget_ref in prompt:
                if component["type"] == "QLineEdit":
                    widget_value = component["widget"].toPlainText().strip()
                    prompt = prompt.replace(widget_ref, widget_value)

        if self.parent and self.image_path:
            label_data = self.get_label_data()
            if label_data:
                label_refs = {
                    "@label.shapes": str(label_data.get("shapes", [])),
                    "@label.imagePath": label_data.get("imagePath", ""),
                    "@label.imageHeight": str(
                        label_data.get("imageHeight", "")
                    ),
                    "@label.imageWidth": str(label_data.get("imageWidth", "")),
                    "@label.flags": str(label_data.get("flags", {})),
                }

                vqa_data = label_data.get("vqaData", {})
                for key, value in vqa_data.items():
                    label_refs[f"@label.{key}"] = str(value)

                for ref, value in label_refs.items():
                    if ref in prompt:
                        prompt = prompt.replace(ref, value)

        if "@image" in prompt:
            if not self.image_path or not os.path.exists(self.image_path):
                raise Exception("No image available for @image reference")

            # Replace @image with <image> for API compatibility
            prompt = re.sub(r"@image\s*", "<image> ", prompt).strip()

            self.has_image_reference = True
        else:
            self.has_image_reference = False

        logger.debug(f"prompt: {prompt}")

        return prompt

    def get_label_data(self):
        if not self.parent or not self.image_path:
            return None

        output_dir = getattr(self.parent, "output_dir", None)
        label_file_path = get_label_file_path(self.image_path, output_dir)

        if os.path.exists(label_file_path):
            with open(label_file_path, "r", encoding="utf-8") as f:
                return json.load(f)

    def cancel(self):
        """Cancel the operation"""
        self._is_cancelled = True

    def load_models_config(self):
        """Load models configuration"""
        try:
            if os.path.exists(MODELS_CONFIG_PATH):
                with open(MODELS_CONFIG_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load models config: {e}")
        return None

    def load_providers_config(self):
        """Load providers configuration"""
        try:
            if os.path.exists(PROVIDERS_CONFIG_PATH):
                with open(PROVIDERS_CONFIG_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load providers config: {e}")
        return None

    def call_openai_api(
        self,
        api_address,
        api_key,
        model_id,
        prompt,
        temperature,
        max_tokens,
        system_prompt,
    ):
        """Call OpenAI-compatible API with optional image support"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if hasattr(self, "has_image_reference") and self.has_image_reference:
            with open(self.image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            )
        else:
            messages.append({"role": "user", "content": prompt})

        data = {
            "model": model_id,
            "messages": messages,
            "temperature": (
                temperature / 100.0 if temperature > 2 else temperature
            ),
            "max_tokens": max_tokens,
        }

        # Ensure API address ends with correct path
        if not api_address.endswith("/"):
            api_address += "/"
        if not api_address.endswith("chat/completions"):
            api_address += "chat/completions"

        response = requests.post(
            api_address, headers=headers, json=data, timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()

        result = response.json()
        return result["choices"][0]["message"]["content"].strip()


def apply_option_mapping(value, mapping):
    """Map options to their corresponding values."""
    if isinstance(value, str):
        return mapping.get(value, value)
    elif isinstance(value, list):
        return [mapping.get(v, v) for v in value]
    return value


def value_contains_deleted_options(value, deleted_options):
    """Check if the value includes any deleted options."""
    if isinstance(value, str):
        return value in deleted_options
    elif isinstance(value, list):
        return any(v in deleted_options for v in value)
    return False


def get_default_value(comp_type, options):
    """Return the default value based on component type."""
    if comp_type == "QRadioButton" and options:
        return options[0]
    elif comp_type == "QCheckBox":
        return []
    elif comp_type == "QComboBox":
        return None
    return ""


def get_label_file_path(image_file, output_dir=None):
    """
    Get the corresponding label file path for an image file.

    Args:
        image_file (str): Path to the image file
        output_dir (str, optional): Output directory path. Defaults to None.

    Returns:
        str: Path to the corresponding JSON label file
    """
    base_name = os.path.splitext(os.path.basename(image_file))[0]
    label_filename = base_name + ".json"

    if output_dir:
        return os.path.join(output_dir, label_filename)
    else:
        image_dir = os.path.dirname(image_file)
        return os.path.join(image_dir, label_filename)


def get_real_modified_options(old_options, new_options, common_options):
    """Identify truly modified options excluding common ones."""
    modified = {}

    if len(old_options) == len(new_options):
        for i in range(len(old_options)):
            old_opt = old_options[i]
            new_opt = new_options[i]
            if old_opt != new_opt and old_opt not in common_options:
                modified[old_opt] = new_opt

    return modified
