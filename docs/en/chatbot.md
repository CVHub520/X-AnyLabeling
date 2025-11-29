# Overview

The `X-AnyLabeling` **Chatbot** is an integrated AI assistant that allows users to interact directly with Large Language Models (LLMs) within their labeling workflow. 

This feature enables you to engage in chat conversations using natural language, batch process image-text question-answering data, and supports one-click import/export of multimodal image data in the [ShareGPT](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README.md#multi-modal-image-dataset) format (based on single-turn or multi-turn conversations) for direct use in fine-tuning frameworks like [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).

<video src="https://github.com/user-attachments/assets/c97b943a-71e6-470c-bb73-b4c8d299687f" width="100%" controls>
</video>


# Getting Started

## Accessing the Chatbot

To open the Chatbot, click the Chatbot icon in the left toolbar of X-AnyLabeling or use the following keyboard shortcut to quickly launch the Chatbot interface:

- Windows/Linux: `Ctrl` + `1`
- macOS: `⌘` + `1`

## Initial Setup

On the first launch, you need to configure the necessary API credentials and models in the right panel of the `Chatbot` window.

# User Interface

The Chatbot features a three-panel interface designed for streamlined interaction.

<video src="https://github.com/user-attachments/assets/41c30839-4b49-4de2-8252-fe856956daa7" width="100%" controls>
</video>

## Left Panel - Model Providers

| Provider   | API Key                                                       | API Docs                                                      | Model Docs                                                                  |
| :--------- | :------------------------------------------------------------ | :------------------------------------------------------------ | :-------------------------------------------------------------------------- |
| Anthropic  | [Link](https://console.anthropic.com/settings/keys)           | [Link](https://docs.anthropic.com/en/docs)                    | [Link](https://docs.anthropic.com/en/docs/about-claude/models/all-models)   |
| DeepSeek   | [Link](https://platform.deepseek.com/api_keys)                | [Link](https://platform.deepseek.com/docs)                    | [Link](https://platform.deepseek.com/models)                                |
| Google AI  | [Link](https://aistudio.google.com/app/apikey)                | [Link](https://ai.google.dev/gemini-api/docs)                 | [Link](https://ai.google.dev/gemini-api/docs/models)                        |
| Ollama     | -                                                             | [Link](https://github.com/ollama/ollama/blob/main/docs/api.md) | [Link](https://ollama.com/search)                                           |
| OpenAI     | [Link](https://platform.openai.com/api-keys)                  | [Link](https://platform.openai.com/docs)                    | [Link](https://platform.openai.com/docs/models)                             |
| OpenRouter | [Link](https://openrouter.ai/settings/keys)                   | [Link](https://openrouter.ai/docs/quick-start)                | [Link](https://openrouter.ai/models)                                        |
| Qwen       | [Link](https://bailian.console.aliyun.com/?apiKey=1#/api-key) | [Link](https://help.aliyun.com/document_detail/2590237.html)   | [Link](https://help.aliyun.com/zh/model-studio/developer-reference/what-is-qwen-llm) |

> [!NOTE]
> The Custom provider supports configuring any custom endpoint that is compatible with the OpenAI API format. After selecting a provider, the model list will only display relevant models and favorites for the current provider.

## Middle Panel - Chat Interface

- **Chat Window**: View your conversation history with the AI. Supports copy, edit, delete, and rerun functions.
- **Message Input**: Enter your questions or instructions. Includes a one-click option to clear the current conversation history.
- **Special Command**: After importing an image, use the `@image` prompt to include the current image in your query.

## Right Panel - Image Preview and Settings

- **Image Preview**: Displays the current image.
- **Function Components**:
  - **Image Navigation**: Switch between the previous and next images.
  - **Image Import**: Import a single image file or an entire directory.
  - **Data Export**: Export annotation results.
  - **Batch Processing**: Run processing tasks on multiple images.
- **Backend Settings**: Configure API endpoints, keys, and select models.
- **Generation Parameters**: Input system prompts, temperature settings, and maximum output length.

# Key Features

## Visual Question Answering (VQA)

<video src="https://github.com/user-attachments/assets/1119fa89-b885-4d4f-ad76-499a74aa81eb" width="100%" controls>
</video>

Ask questions about the current image to efficiently generate single-turn or multi-turn image-text dialogues:

> @image Please describe this image.

## Batch Image Processing

<video src="https://github.com/user-attachments/assets/4ec36aaa-2f0b-442f-9315-cc6ab1d1e0c2" width="100%" controls>
</video>

Process multiple images with the same prompt to speed up your workflow:

1. Load an image folder.
2. Click the "Run All Images" button.
3. Enter the prompt to apply to all images.
4. You can set the concurrency level to control processing speed (default is 80% of CPU cores, maximum is 95%).

## Dataset Import/Export

<video src="https://github.com/user-attachments/assets/8dc44d1c-0317-4b00-9967-105274dee59f" width="100%" controls>
</video>

Supports one-click export of multimodal image data in the ShareGPT format.

# Other

## Configuration Files

The Chatbot stores its configuration in the following location within the user's home directory:

```
~/.xanylabeling_data/chatbot/
```

This includes:
- `models.json`: Contains user preferences and model configurations.
- `providers.json`: API provider settings.

## Keyboard Shortcuts

- `Ctrl`/`⌘`+`Enter`: Send message
- `Enter`: Add a new line in the message input

## Notes

- The Chatbot feature is currently in beta and may be updated in future versions.
- Local models via Ollama can be used without an internet connection and theoretically support any model that adheres to the OpenAI-compatible API standard.
