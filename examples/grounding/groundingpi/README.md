# GroundingPI

## Overview

[GroundingPI](https://github.com/groundingpi/GroundingPI) is a 4B-parameter grounding foundation model designed to provide spatial perception for physical intelligence. It combines a visual encoder with a language backbone, using staged training on visual primitives and an autoregressive spatial-token format to represent object locations. The model localizes targets from category names, natural-language descriptions, or reference boxes, producing bounding boxes or points for detection, referring-expression grounding, point localization, and visual prompting.

| Task | Input | Output |
| --- | --- | --- |
| Detection | Categories, for example `dog.cup.lamp` | Rectangle boxes with category labels |
| Grounding | A description, for example `the dog on the sofa` | Rectangle boxes |
| Pointing | A description, for example `the dog on the sofa` | Point annotations |
| Visual Prompting | One or more positive reference boxes | Rectangle boxes around similar objects |

## Installation

Use a dedicated Server environment. GroundingPI requires the upstream **Transformers 5.7.0 fork**, installed directly from its source archive below. The Server already includes the prompt and response protocol code; the checkpoint supplies the model implementation.

From the X-AnyLabeling-Server repository:

```bash
python3 -m venv .venv-groundingpi
source .venv-groundingpi/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision
python -m pip install -e . accelerate
python -m pip install "https://raw.githubusercontent.com/groundingpi/GroundingPI/090932430bbbeea004cfa461767ede7f563c0f3a/third_party/transformers.tar.gz#sha256=3b9d4b20efc92041d3d270be1b875a760f7ed2651ba01be59286d1e0ad6d4f25"
```

Select a PyTorch/Torchvision build compatible with your hardware when using CUDA. The Transformers archive is pinned to the required fork; installing a package with the same version number from PyPI does not identify this fork.

Enable GroundingPI in the Server's `configs/models.yaml`:

```yaml
enabled_models:
  - groundingpi
```

Edit `configs/auto_labeling/groundingpi.yaml` as needed. The default local backend uses:

```yaml
params:
  model_path: "GroundingPI/GroundingPI"
  backend: "transformers"
  device: "cuda"
  dtype: "bfloat16"
  attn_implementation: "sdpa"
  task: "detection"
  max_new_tokens: 4096
  max_model_len: 16384
```

`model_path` accepts the [Hugging Face model ID](https://huggingface.co/GroundingPI/GroundingPI) or a local directory containing the complete checkpoint, processor, tokenizer, and custom model code. The Hub model is downloaded on first load.

Start the Server in the same environment:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Check `/v1/models` and confirm that `groundingpi` loaded successfully before connecting the client.

To use an existing upstream OpenAI-compatible inference service instead, replace the local backend parameters with:

```yaml
params:
  backend: "openai"
  api_base: "http://127.0.0.1:8001/v1"
  model_name: "groundingpi"
  api_key: ""
  timeout: 120
  task: "detection"
  max_new_tokens: 4096
```

This backend sends requests to an already running OpenAI-compatible service. See the upstream [inference guide](https://github.com/groundingpi/GroundingPI/blob/090932430bbbeea004cfa461767ede7f563c0f3a/docs/INFERENCE.md) for deployment.

## Usage

Open an image in X-AnyLabeling, press `Ctrl+A` to open the AI panel, select **Remote-Server**, enter the Server URL, and choose **GroundingPI**.

### Detection

Select **Detection**, enter categories such as `dog.cup.lamp`, and click **Send**. Commas, periods, newlines, and mixed separators are supported; repeated categories are removed while preserving their order.

![Detection with category prompts](.data/detection.jpg)

### Grounding

Select **Grounding**, enter `the dog on the sofa`, and click **Send**. The description is kept as one phrase, including its internal punctuation.

![Grounding with a description](.data/grounding.jpg)

### Pointing

Select **Pointing**, enter `the dog on the sofa`, and click **Send**. This task returns point annotations instead of boxes.

![Pointing with a description](.data/pointing.jpg)

### Visual Prompting

Select **Visual Prompting**, click **+Rect**, and draw a positive reference box around an example object. Click **Run Rect** to find similar objects, then click **Finish** and enter a label, such as `panda`, to name the results. **Clear** removes prompt marks and unfinished predictions while keeping finished annotations.

![Visual prompting with a positive reference box](.data/visual_prompting.jpg)

The screenshots use the Server's `assets/demo/images/demo.jpg` and `assets/demo/images/demo_pandas.jpg`.

For all tasks, **Replace (On)** replaces existing annotations. Click it to switch to **Preserve (On)** and keep existing annotations when applying new results.

Detection, Grounding, and Pointing support text-prompt batch processing: open an image folder, choose the task, click **Auto Run**, select the image range, and enter the prompt. Visual Prompting requires per-image reference boxes and disables batch processing.

## Notes

- GroundingPI generates coordinates on a 0–999 grid; the Server restores them to the original image size. It does not provide confidence scores or confidence/IoU controls.
- The model can return false positives for absent targets. In the demo image, an `elephant` query returned a box even though no elephant was present, in both upstream inference and the adapter. Review predictions before accepting annotations.
- Negative reference boxes, point prompts, segmentation, and video are not exposed by this adapter. Truncated or malformed model responses are rejected rather than applied as partial annotations.
