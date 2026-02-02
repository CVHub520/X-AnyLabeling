# PaddleOCR-VL-1.5 Example

<video src="https://github.com/user-attachments/assets/493183fd-6cbe-45fb-9808-ec2b0af7a0f9" width="100%" controls>
</video>

## Introduction

[PaddleOCR-VL-1.5](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5) is a unified Vision-Language OCR model that supports multiple document understanding tasks through a single model architecture. Built upon powerful vision-language foundations, it can handle diverse OCR scenarios including text recognition, table extraction, formula recognition, chart understanding, seal recognition, and text spotting with bounding boxes.

Here, we will show you how to use PaddleOCR-VL-1.5 on X-AnyLabeling to perform various OCR and document understanding tasks.

Let's get started!

## Supported Tasks

PaddleOCR-VL-1.5 supports six distinct tasks:

| Task | Description | Output |
|------|-------------|--------|
| **OCR** | Optical Character Recognition for text extraction | Text content |
| **Table Recognition** | Extract table structure and content | HTML/Markdown table |
| **Formula Recognition** | Recognize mathematical formulas | LaTeX format |
| **Chart Recognition** | Extract information from charts and graphs | Structured data |
| **Text Spotting** | Detect and recognize text with bounding boxes | Polygon shapes with text |
| **Seal Recognition** | Recognize seal stamps and chop marks | Text content |

## Installation

You'll need to get X-AnyLabeling-Server up and running first. Check out the [installation guide](https://github.com/CVHub520/X-AnyLabeling-Server) for the details. Make sure you're running at least **v0.0.7** of the server and **v3.3.9** of the X-AnyLabeling client, otherwise you might run into compatibility issues.

> [!IMPORTANT]
> `PaddleOCR-VL-1.5` requires `transformers>=5.0.0`. Install it with:
> ```bash
> python -m pip install "transformers>=5.0.0"
> ```
> For more details, see the [official model page](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5).

> [!TIP]
> We highly recommend installing [flash-attn](https://github.com/Dao-AILab/flash-attention) to boost performance and reduce memory usage:
> ```bash
> pip install flash-attn --no-build-isolation
> ```

Once that's done, head over to `configs/models.yaml` and enable `paddleocr_vl_1_5`. There's an [example config](https://github.com/CVHub520/X-AnyLabeling-Server/blob/main/configs/models.yaml) you can reference if you're not sure how to set it up.

You can tweak the settings in [paddleocr_vl_1_5.yaml](https://github.com/CVHub520/X-AnyLabeling-Server/blob/main/configs/auto_labeling/paddleocr_vl_1_5.yaml) to fit your needs.

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_path` | `PaddlePaddle/PaddleOCR-VL-1.5` | HuggingFace model path |
| `device` | `cuda:0` | Device for inference |
| `torch_dtype` | `bfloat16` | Model precision |
| `max_new_tokens` | `512` | Maximum tokens for generation |
| `max_pixels` | `1605632` | Maximum pixels for text tasks (1280×28×28) |
| `spotting_max_pixels` | `1605632` | Maximum pixels for spotting task (2048×28×28) |
| `spotting_upscale_threshold` | `1500` | Threshold for image upscaling in spotting |

> [!NOTE]
> If inference times out, try adjusting `max_new_tokens`, `max_pixels`, `spotting_max_pixels`, and `spotting_upscale_threshold` based on your GPU memory.

## Getting Started

Launch the X-AnyLabeling client, press `Ctrl+A` or click the `AI` button in the left menu bar to open the auto-labeling panel. In the model dropdown list, select `Remote-Server`, then choose `PaddleOCR-VL-1.5`.

### OCR (Text Recognition)

The **OCR** task extracts text content from images.

**Usage:**
1. Select the "OCR" task from the task dropdown
2. Click the "Run" button to extract text
3. The recognized text will be displayed in the description field

### Table Recognition

The **Table Recognition** task extracts table structure and content from document images.

**Usage:**
1. Select the "Table Recognition" task from the task dropdown
2. Click the "Run" button to extract table content
3. The result will be formatted as HTML/Markdown table structure

### Formula Recognition

The **Formula Recognition** task recognizes mathematical formulas and converts them to LaTeX format.

**Usage:**
1. Select the "Formula Recognition" task from the task dropdown
2. Click the "Run" button to recognize formulas
3. The result will be in LaTeX format

### Chart Recognition

The **Chart Recognition** task extracts information from charts and graphs.

**Usage:**
1. Select the "Chart Recognition" task from the task dropdown
2. Click the "Run" button to analyze the chart
3. The extracted data will be displayed in structured format

### Text Spotting

The **Text Spotting** task detects text regions and recognizes their content with polygon bounding boxes.

**Usage:**
1. Select the "Text Spotting" task from the task dropdown
2. Click the "Run" button to detect and recognize text
3. Polygon shapes with recognized text will be created on the canvas

> [!TIP]
> For small images (width and height both less than 1500 pixels), the model automatically upscales the image by 2x for better detection accuracy. You can adjust this threshold via `spotting_upscale_threshold`.

### Seal Recognition

The **Seal Recognition** task recognizes text from seal stamps and chop marks.

**Usage:**
1. Select the "Seal Recognition" task from the task dropdown
2. Click the "Run" button to recognize seal text
3. The recognized text will be displayed in the description field

> [!TIP]
> All tasks support batch processing. You can run inference on the entire dataset with a single click using `Ctrl+M` or the batch processing feature in X-AnyLabeling.

## Related Models

- [PP-DocLayoutV3](../document_layout_analysis/README.md#pp-doclayoutv3): Document layout analysis model that works well with PaddleOCR-VL-1.5
