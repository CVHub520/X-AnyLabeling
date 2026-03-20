"""Integration tests for MiniMax provider - requires MINIMAX_API_KEY.

Note: MiniMax does not support the /v1/models listing endpoint.
Users must manually select model IDs (e.g., MiniMax-M2.7) in the UI.
The chatbot module handles model listing failures gracefully by
returning an empty list, so this is not a problem in practice.
"""

import os
import unittest

from openai import OpenAI

MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY")
MINIMAX_BASE_URL = "https://api.minimax.io/v1"


@unittest.skipUnless(MINIMAX_API_KEY, "MINIMAX_API_KEY not set")
class TestMiniMaxAPIIntegration(unittest.TestCase):
    """Integration tests that call the real MiniMax API."""

    @classmethod
    def setUpClass(cls):
        cls.client = OpenAI(
            base_url=MINIMAX_BASE_URL,
            api_key=MINIMAX_API_KEY,
            timeout=30,
        )

    def test_chat_completion(self):
        """MiniMax chat completion should return a valid response."""
        response = self.client.chat.completions.create(
            model="MiniMax-M2.7",
            messages=[{"role": "user", "content": "Say hello in one word."}],
            temperature=1.0,
            max_tokens=10,
            stream=False,
        )
        self.assertIsNotNone(response.choices)
        self.assertTrue(len(response.choices) > 0)
        self.assertTrue(len(response.choices[0].message.content) > 0)

    def test_chat_completion_streaming(self):
        """MiniMax streaming chat completion should yield chunks."""
        response = self.client.chat.completions.create(
            model="MiniMax-M2.7",
            messages=[{"role": "user", "content": "Say hi."}],
            temperature=1.0,
            max_tokens=10,
            stream=True,
        )
        chunks = list(response)
        self.assertTrue(len(chunks) > 0, "No streaming chunks received")

    def test_chat_completion_with_system_prompt(self):
        """MiniMax should handle system prompts correctly."""
        response = self.client.chat.completions.create(
            model="MiniMax-M2.7",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say OK."},
            ],
            temperature=1.0,
            max_tokens=10,
            stream=False,
        )
        self.assertIsNotNone(response.choices)
        self.assertTrue(len(response.choices) > 0)


if __name__ == "__main__":
    unittest.main()
