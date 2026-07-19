# 概述

X-AnyLabeling 的聊天机器人是集成在标注流程中的 AI 助手，可与大语言模型（LLM）进行自然语言交互、批量处理图文问答数据，并导入或导出单轮及多轮对话数据。导出的多模态数据采用 [ShareGPT](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README_zh.md#%E5%A4%9A%E6%A8%A1%E6%80%81%E5%9B%BE%E5%83%8F%E6%95%B0%E6%8D%AE%E9%9B%86-1) 格式，可用于 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 等大模型微调框架。

<video src="https://github.com/user-attachments/assets/c97b943a-71e6-470c-bb73-b4c8d299687f" width="100%" controls>
</video>


# 入门

## 访问聊天机器人

要打开聊天机器人，请点击 X-AnyLabeling 中左侧工具栏的聊天机器人图标或使用如下快捷键快速启动聊天机器人对话界面。

- Windows/Linux: `Ctrl` + `1`
- macOS: `⌘` + `1`

## 初始设置

首次启动时，您需要在 `Chatbot` 聊天窗口的右侧面板中配置必要的 API 凭据和模型。

# 用户界面

该聊天机器人具有一个三面板界面，旨在简化交互。

<video src="https://github.com/user-attachments/assets/41c30839-4b49-4de2-8252-fe856956daa7" width="100%" controls>
</video>

## 左侧面板：模型提供商

| 提供商 | API 密钥 | API 文档 | 模型文档 |
| :--- | :--- | :--- | :--- |
| Anthropic | [链接](https://console.anthropic.com/settings/keys) | [链接](https://docs.anthropic.com/en/docs) | [链接](https://docs.anthropic.com/en/docs/about-claude/models/all-models) |
| DeepSeek   | [链接](https://platform.deepseek.com/api_keys)         | [链接](https://platform.deepseek.com/docs)                  | [链接](https://platform.deepseek.com/models)                                |
| Google AI  | [链接](https://aistudio.google.com/app/apikey)         | [链接](https://ai.google.dev/gemini-api/docs)               | [链接](https://ai.google.dev/gemini-api/docs/models)                        |
| Ollama     | -                                                             | [链接](https://github.com/ollama/ollama/blob/main/docs/api.md) | [链接](https://ollama.com/search)                                           |
| OpenAI     | [链接](https://platform.openai.com/api-keys)           | [链接](https://platform.openai.com/docs)                    | [链接](https://platform.openai.com/docs/models)                             |
| OpenRouter | [链接](https://openrouter.ai/settings/keys)            | [链接](https://openrouter.ai/docs/quick-start)              | [链接](https://openrouter.ai/models)                                        |
| Qwen       | [链接](https://bailian.console.aliyun.com/?apiKey=1#/api-key) | [链接](https://help.aliyun.com/document_detail/2590237.html) | [链接](https://help.aliyun.com/zh/model-studio/developer-reference/what-is-qwen-llm) |

> [!NOTE]
> Custom 提供商支持配置与 OpenAI API 兼容的自定义端点。切换提供商后，模型列表会自动更新为当前提供商的可用模型。

## 中间面板：聊天界面

- 聊天窗口：查看您与 AI 的对话历史，支持复制、编辑、删除、重新运行功能
- 消息输入：输入您的问题或指令，支持一键清除当前对话历史记录
- 特殊命令：导入图像后，使用 `@image` 提示包含当前图像

## 右侧面板：图像预览与设置

- **图像预览**：显示当前图像。
- **功能组件**：切换图像、导入单个图像或目录、导出标注结果以及批量处理图像。
- **后端设置**：配置 API 端点、密钥和模型。
- **生成参数**：设置系统提示词、温度和最大输出长度。

# 关键特性

## 视觉问答

<video src="https://github.com/user-attachments/assets/1119fa89-b885-4d4f-ad76-499a74aa81eb" width="100%" controls>
</video>

针对当前图像提问，生成单轮或多轮图文对话：

> @image 请描述这张图像。

## 图像批量处理

<video src="https://github.com/user-attachments/assets/4ec36aaa-2f0b-442f-9315-cc6ab1d1e0c2" width="100%" controls>
</video>

使用相同的提示处理多张图片以加快工作流程：

1. 加载一个图片文件夹
2. 点击“运行所有图片”按钮
3. 输入要应用于所有图片的提示
4. 可设置并发数以控制处理速度（默认值为 CPU 核心数的 80%，最大值为 95%）

## 数据集导入/导出

<video src="https://github.com/user-attachments/assets/8dc44d1c-0317-4b00-9967-105274dee59f" width="100%" controls>
</video>

支持导入和导出 ShareGPT 格式的多模态图像数据。

# 其他

## 配置文件

聊天机器人将其配置存储在用户目录下的如下位置：

```
~/xanylabeling_data/chatbot/
```

其中包括：

- `models.json`：包含用户偏好设置和模型配置
- `providers.json`：API 提供商设置
