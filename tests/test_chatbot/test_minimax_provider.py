"""Unit tests for MiniMax provider integration in the chatbot module.

These tests are self-contained and do not import the full anylabeling
package (which requires PyQt6 and many other GUI dependencies).
Instead, they parse the relevant source files directly.
"""

import ast
import os
import unittest

# Project root (relative to this test file)
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

CONFIG_PATH = os.path.join(
    PROJECT_ROOT,
    "anylabeling",
    "views",
    "labeling",
    "chatbot",
    "config.py",
)

PROVIDER_PATH = os.path.join(
    PROJECT_ROOT,
    "anylabeling",
    "views",
    "labeling",
    "chatbot",
    "provider.py",
)

RESOURCES_QRC_PATH = os.path.join(
    PROJECT_ROOT, "anylabeling", "resources", "resources.qrc"
)

ICON_PATH = os.path.join(
    PROJECT_ROOT, "anylabeling", "resources", "images", "minimax.png"
)

EN_DOCS_PATH = os.path.join(PROJECT_ROOT, "docs", "en", "chatbot.md")
ZH_DOCS_PATH = os.path.join(PROJECT_ROOT, "docs", "zh_cn", "chatbot.md")


def _extract_providers_data():
    """Extract DEFAULT_PROVIDERS_DATA from config.py using AST parsing."""
    with open(CONFIG_PATH, "r") as f:
        tree = ast.parse(f.read())

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == "DEFAULT_PROVIDERS_DATA"
                ):
                    return ast.literal_eval(node.value)
    raise RuntimeError("DEFAULT_PROVIDERS_DATA not found in config.py")


class TestMiniMaxProviderConfig(unittest.TestCase):
    """Tests for MiniMax provider configuration in config.py."""

    @classmethod
    def setUpClass(cls):
        cls.providers = _extract_providers_data()

    def test_minimax_in_default_providers(self):
        """MiniMax must be present in DEFAULT_PROVIDERS_DATA."""
        self.assertIn("minimax", self.providers)

    def test_minimax_api_address(self):
        """MiniMax API address must use the official OpenAI-compat endpoint."""
        cfg = self.providers["minimax"]
        self.assertEqual(cfg["api_address"], "https://api.minimax.io/v1")

    def test_minimax_api_key_default_none(self):
        """MiniMax API key should default to None (user must set it)."""
        cfg = self.providers["minimax"]
        self.assertIsNone(cfg["api_key"])

    def test_minimax_has_required_fields(self):
        """MiniMax config must have all 5 required fields."""
        cfg = self.providers["minimax"]
        required = [
            "api_address",
            "api_key",
            "api_key_url",
            "api_docs_url",
            "model_docs_url",
        ]
        for field in required:
            self.assertIn(field, cfg, f"Missing field: {field}")

    def test_minimax_api_key_url_not_empty(self):
        """MiniMax API key URL must be set for user guidance."""
        cfg = self.providers["minimax"]
        self.assertIsNotNone(cfg["api_key_url"])
        self.assertTrue(len(cfg["api_key_url"]) > 0)

    def test_minimax_api_docs_url_not_empty(self):
        """MiniMax API docs URL must be set."""
        cfg = self.providers["minimax"]
        self.assertIsNotNone(cfg["api_docs_url"])
        self.assertTrue(len(cfg["api_docs_url"]) > 0)

    def test_minimax_model_docs_url_not_empty(self):
        """MiniMax model docs URL must be set."""
        cfg = self.providers["minimax"]
        self.assertIsNotNone(cfg["model_docs_url"])
        self.assertTrue(len(cfg["model_docs_url"]) > 0)

    def test_minimax_config_matches_other_providers(self):
        """MiniMax config structure must match other providers."""
        minimax_keys = set(self.providers["minimax"].keys())
        for name in ["openai", "deepseek", "google"]:
            other_keys = set(self.providers[name].keys())
            self.assertEqual(
                minimax_keys, other_keys,
                f"MiniMax fields differ from {name}",
            )

    def test_minimax_api_address_uses_https(self):
        """MiniMax API address must use HTTPS."""
        cfg = self.providers["minimax"]
        self.assertTrue(cfg["api_address"].startswith("https://"))

    def test_minimax_not_anthropic_url(self):
        """MiniMax URL must not contain 'anthropic' (would trigger curl path)."""
        cfg = self.providers["minimax"]
        self.assertNotIn("anthropic", cfg["api_address"])

    def test_provider_count(self):
        """Total provider count should be 9 (including MiniMax)."""
        self.assertEqual(len(self.providers), 9)


class TestMiniMaxProviderSource(unittest.TestCase):
    """Tests for MiniMax compatibility in provider.py."""

    @classmethod
    def setUpClass(cls):
        with open(PROVIDER_PATH, "r") as f:
            cls.provider_source = f.read()

    def test_openai_client_used_for_non_anthropic(self):
        """provider.py must use OpenAI client for non-Anthropic providers."""
        self.assertIn("OpenAI(base_url=base_url", self.provider_source)

    def test_anthropic_check_is_url_based(self):
        """Anthropic special handling should be URL-based, not name-based."""
        # The check is: if "anthropic" in base_url
        # MiniMax's URL (api.minimax.io) won't match this check
        self.assertIn('"anthropic" in base_url', self.provider_source)


class TestMiniMaxResourceIcon(unittest.TestCase):
    """Tests for MiniMax provider icon resource."""

    def test_minimax_icon_file_exists(self):
        """MiniMax icon PNG must exist in resources/images/."""
        self.assertTrue(os.path.exists(ICON_PATH), "minimax.png not found")

    def test_minimax_icon_is_valid_png(self):
        """MiniMax icon must be a valid PNG file."""
        with open(ICON_PATH, "rb") as f:
            header = f.read(8)
        self.assertEqual(header[:4], b"\x89PNG")

    def test_minimax_icon_reasonable_size(self):
        """MiniMax icon should be within expected size range."""
        size = os.path.getsize(ICON_PATH)
        self.assertGreater(size, 500, "Icon too small")
        self.assertLess(size, 50000, "Icon too large")

    def test_minimax_in_qrc(self):
        """MiniMax icon must be registered in resources.qrc."""
        with open(RESOURCES_QRC_PATH, "r") as f:
            qrc_content = f.read()
        self.assertIn("images/minimax.png", qrc_content)


class TestMiniMaxDocumentation(unittest.TestCase):
    """Tests for MiniMax documentation entries."""

    def test_english_docs_mention_minimax(self):
        """English chatbot docs must include MiniMax provider row."""
        with open(EN_DOCS_PATH, "r") as f:
            content = f.read()
        self.assertIn("MiniMax", content)
        self.assertIn("platform.minimax", content)

    def test_chinese_docs_mention_minimax(self):
        """Chinese chatbot docs must include MiniMax provider row."""
        with open(ZH_DOCS_PATH, "r") as f:
            content = f.read()
        self.assertIn("MiniMax", content)
        self.assertIn("platform.minimax", content)

    def test_english_docs_minimax_in_table(self):
        """English docs MiniMax entry must be in the provider table."""
        with open(EN_DOCS_PATH, "r") as f:
            content = f.read()
        # MiniMax row should have provider name, API key link, docs link
        lines = [l for l in content.split("\n") if "MiniMax" in l]
        self.assertTrue(len(lines) >= 1, "No MiniMax line in EN docs table")
        minimax_line = lines[0]
        self.assertIn("|", minimax_line)  # Table row
        self.assertIn("Link", minimax_line)  # Has at least one link

    def test_chinese_docs_minimax_in_table(self):
        """Chinese docs MiniMax entry must be in the provider table."""
        with open(ZH_DOCS_PATH, "r") as f:
            content = f.read()
        lines = [l for l in content.split("\n") if "MiniMax" in l]
        self.assertTrue(len(lines) >= 1, "No MiniMax line in ZH docs table")
        minimax_line = lines[0]
        self.assertIn("|", minimax_line)
        self.assertIn("Link", minimax_line)


class TestMiniMaxOpenAIClientCompat(unittest.TestCase):
    """Tests for MiniMax OpenAI-compatible client instantiation."""

    def test_openai_client_with_minimax_base_url(self):
        """OpenAI client should accept MiniMax base URL without errors."""
        from openai import OpenAI

        client = OpenAI(
            base_url="https://api.minimax.io/v1",
            api_key="test-dummy-key",
            timeout=5,
        )
        self.assertIsNotNone(client)
        self.assertEqual(str(client.base_url), "https://api.minimax.io/v1/")


if __name__ == "__main__":
    unittest.main()
