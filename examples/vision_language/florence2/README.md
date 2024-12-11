# Florence 2 Example

## Introduction

[Florence-2](https://arxiv.org/abs/2311.06242) is a novel vision foundation model with a unified, prompt-based representation for a variety of computer vision and vision-language tasks, developed by Microsoft.

Here, we will show you how to use Florence-2 on X-AnyLabeling to perform various vision tasks.

Let's get started!


## Installation

Before you begin, make sure you have the following prerequisites installed:

**Step 0:** Download and install Miniconda from the [official website](https://docs.anaconda.com/miniconda/).

**Step 1:** Create a new Conda environment with Python version `3.10` or higher, and activate it:

```bash
conda create -n x-anylabeling-transformers python=3.10 -y
conda activate x-anylabeling-transformers
```

**Step 2:** Install required dependencies.

First, follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies.

Then, install the `transformers` package via:

```bash
pip install transformers
```

Finally, you can back to the installation guide ([简体中文](../../docs/zh_cn/get_started.md) | [English](../../docs/en/get_started.md)) to install the necessary dependencies for X-AnyLabeling (v2.5.0+):


## Getting Started

> [!TIP]
> We recommend that you download the model before loading it.

You can download the model from [HuggingFace](https://huggingface.co/microsoft/Florence-2-large-ft) via:

```python3
import torch
from transformers import AutoProcessor, AutoModelForCausalLM 

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large-ft", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large-ft", trust_remote_code=True)
```

The model will be automatically downloaded and cached in the `transformers` package directory.

If you have an unstable network connection, you can:
1. Manually download the model files
2. Update the `model_path` parameter in the [configuration file](../../../anylabeling/configs/auto_labeling/florence2_large_ft.yaml)


### Image-level Captioning Tasks

The image-level captioning task is including three sub-tasks:
- Caption: Generate a concise caption for the image
- Detailed caption: Generate a detailed caption for the image
- More detailed caption: Generate a more detailed caption for the image

<video src="https://github.com/user-attachments/assets/a86ede3c-513d-4cd5-abf3-46a66e718124" width="100%" controls>
</video>

### Region-level Tasks

The region-level tasks are including:

- Object detection

![Florence2-od](https://github.com/user-attachments/assets/f9362c85-490a-4516-8aa1-b1e7240866b2)

- Region proposal

![Florence2-region-proposal](https://github.com/user-attachments/assets/86d83c52-8544-4d44-8d81-29ff35d0eeac)

- Dense region caption

![Florence2-dense-region-caption](https://github.com/user-attachments/assets/c3b1335d-6963-4092-b059-aeb6179c05c1)

> [!NOTE]
> The following tasks require additional box input.

- Region to category: Assign a category to the region
- Region to description: Generate a description for the region
- Region to segmentation: Generate a segmentation mask for the region

<video src="https://github.com/user-attachments/assets/4c35f3b3-a012-4abb-939e-8db61d6d797c" width="100%" controls>
</video>


### Phrase Grounding & OVD

> [!NOTE]
> Both phrase grounding and open vocabulary detection tasks require additional text input.

- Caption to parse grounding

<video src="https://github.com/user-attachments/assets/23f74170-226a-4fa9-ac6c-4cfd98ac8e98" width="100%" controls>
</video>

- Referring expression segmentation

<video src="https://github.com/user-attachments/assets/29f07d7d-bc42-42b1-a321-a2ef8a15ffa6" width="100%" controls>
</video>

- Open vocabulary detection

![Florence2-ovd](https://github.com/user-attachments/assets/269db58b-15d3-4c58-bddf-3cce253b580a)


### Optical Character Recognition

- OCR

![Florence2-ocr](https://github.com/user-attachments/assets/baf2240f-8331-45a2-a452-bef5ceae0c5e)

- OCR with region

![Florence2-ocr-with-region](https://github.com/user-attachments/assets/4ca2fb60-7df5-4a29-8a8d-aa93118bda6a)