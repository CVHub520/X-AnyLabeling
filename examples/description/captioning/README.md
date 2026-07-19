# Image Captioning Example

## Overview

X-AnyLabeling supports image-captioning workflows through its Chatbot and visual question answering tools. Both tools can send the current image to a vision-language model and store or export the generated description for review.

## Workflow

1. Configure a vision-capable model in the [Chatbot](../../../docs/en/chatbot.md).
2. Load an image directory in X-AnyLabeling.
3. Use `@image` with a prompt such as `Describe this image in one concise sentence.`
4. Review the generated caption before saving or exporting it.

For structured datasets with configurable fields, use the [VQA tool](../../../docs/en/vqa.md). For batch conversations in ShareGPT format, use the Chatbot's batch-processing and export features.

Always review generated captions before using them as training data,
especially descriptions of small objects, text, spatial relationships, and
domain-specific terminology.
