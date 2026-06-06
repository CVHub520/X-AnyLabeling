# LocateAnything: Open-Vocabulary Grounding in X-AnyLabeling

## Overview

[LocateAnything](https://huggingface.co/nvidia/LocateAnything-3B) is a vision-language grounding model that returns boxes and points using normalized coordinates. X-AnyLabeling integrates it through X-AnyLabeling-Server, so inference runs in a dedicated server environment and the client receives standard annotation shapes.

LocateAnything supports several remote tasks:

| Task | Input | Output |
| --- | --- | --- |
| Detection | Object categories such as `person, car, bicycle` | Rectangle boxes |
| Grounding | A natural-language phrase such as `people wearing red shirts` | Rectangle boxes |
| Pointing | A phrase such as `the traffic light` | Point shapes |
| Text Detection | No text prompt | Text boxes with recognized text stored in shape descriptions |

## 🎬 Visual Demo

<table>
<tr>
<td width="60.4%" align="center" valign="top">
<video src="https://github.com/user-attachments/assets/814e042c-baf4-41ba-b7c9-655e909f82d6" autoplay loop muted playsinline controls width="100%"></video>

<b>Dense Object Detection</b><br>
<sub>LocateAnything performs diverse localization tasks under a unified VLM — document understanding, GUI grounding, dense object detection, and OCR.</sub>
</td>
<td width="39.6%" align="center" valign="top">
<video src="https://github.com/user-attachments/assets/154b5f61-e26e-451b-9518-88c63d437cc4" autoplay loop muted playsinline controls width="100%"></video>

<b>Fast Decoding Speed</b><br>
<sub>Parallel Box Decoding (PBD) vs. Quantized Coordinate Decoding — PBD predicts each bounding box atomically in a single forward pass for substantially faster throughput.</sub>
</td>
</tr>
</table>

## Server-side

### Installation

Create or reuse a dedicated environment for LocateAnything. Do not install it through the server `[all]` extra, because LocateAnything requires `transformers>=4.50,<5`, while some other server models require Transformers 5.x.

```bash
conda activate locateanything
cd /path/to/X-AnyLabeling-Server

pip install --upgrade uv
uv pip install -e .[locateanything]
```

Make sure PyTorch is installed with the CUDA version that matches your machine. The server extra installs the X-AnyLabeling-Server core dependencies plus:

```text
transformers>=4.50,<5
peft
```

### Enable the Model

In X-AnyLabeling-Server, enable LocateAnything in `configs/models.yaml`:

```yaml
enabled_models:
  - locateanything
```

The model config is located at:

```text
configs/auto_labeling/locateanything.yaml
```

Common parameters:

| Parameter | Default | Description |
| --- | --- | --- |
| `model_path` | `nvidia/LocateAnything-3B` | Hugging Face model id or local model path |
| `device` | `cuda` | PyTorch device, for example `cuda`, `cuda:0`, `cuda:1`, or `cpu` |
| `dtype` | `bfloat16` | Inference dtype |
| `max_image_size` | `1024` | Resize the long edge before inference |
| `max_new_tokens` | `512` | Maximum generation length |
| `generation_mode` | `hybrid` | Supported values: `fast`, `slow`, `hybrid` |
| `temperature` | `0.7` | Sampling temperature used by the original worker default |
| `verbose` | `false` | Print generation statistics in server logs |

Start the server:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Usage

Launch X-AnyLabeling, press `Ctrl+A` or click the `AI` button in the left menu bar to open the auto-labeling panel. In the model dropdown list, select `Remote-Server`, then choose `LocateAnything`.

### Detection

Use this mode when you want to find one or more object categories.

1. Select **Detection** in the task dropdown.
2. Enter object names in the text field, for example `person`, `car`, or `bicycle`.
3. Separate multiple categories with periods: `person.car.bicycle`.
4. Click **Send** to run inference.

> [!NOTE]
> LocateAnything may produce weaker results when detecting multiple target categories in one prompt. For practical annotation work, single-target detection is recommended when accuracy matters.
>
> The default `max_image_size` is set to `1024` as a conservative value for GPUs with limited memory, such as an RTX 3060 with about 11 GB VRAM. Increasing the inference resolution can improve detail, but when the visual tokens approach the GPU memory limit, prefill latency can increase sharply. In local tests, prefill time increased from roughly 1-2 seconds to about 100+ seconds at a much larger resolution. Tune `max_image_size` according to your GPU memory.
>
> If you request many targets or expect many output boxes, increase `max_new_tokens` accordingly.

### Grounding

Use this mode when the target is better described as a phrase.

1. Select **Grounding** in the task dropdown.
2. Enter a phrase such as `people wearing red shirts`.
3. Click **Send** to run inference.

### Pointing

Use this mode when you want point annotations instead of boxes.

1. Select **Pointing** in the task dropdown.
2. Enter a phrase such as `the search button` or `the traffic light`.
3. Click **Send** to run inference.

### Text Detection

Use this mode to detect scene text.

1. Select **Text Detection** in the task dropdown.
2. Click **Run**.

Text detection returns rectangle shapes with label `object`. Recognized text is stored in each shape's `description` field.

## Notes

- The model outputs coordinates as integers in `[0, 1000]`; the server maps them back to the original image size.
- Reducing `max_image_size` can greatly reduce prefill latency and GPU memory usage.
- Keep LocateAnything in a separate server environment from models that require Transformers 5.x.
