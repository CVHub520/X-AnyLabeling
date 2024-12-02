# Object Detection Example

## Introduction

Object detection is a computer vision solution that identifies objects, and their locations, in an image.

<img src=".data/annotated_hbb_task.png" width="100%" />

## Basic Usage

Here's how to set up for the object detection job:

- Start by adding the image files.
- Then, tap the `rectangle` button on the left menu or press the `R` key to quickly create a rectangle shape.
- Finally, type in the matching name in the label dialog.

## Advanced Usage

<img src=".data/upn.jpg" width="100%" />

Let's take the [Universal Proposal Network](https://arxiv.org/pdf/2411.18363) (UPN) model as an example to demonstrate advanced usage, which adopts a dual-granularity prompt tuning strategy to generate comprehensive proposals for objects at both instance and part levels:

- `fine_grained_prompt`: For detecting detailed object parts and subtle differences between similar objects. This mode excels at identifying specific features like facial characteristics or distinguishing between similar species.
- `coarse_grained_prompt`: For detecting broad object categories and major scene elements. This mode focuses on identifying general objects like people, vehicles, or buildings without detailed sub-categorization.


Before you begin, make sure you have the following prerequisites installed:

**Step 0:** Download and install Miniconda from the [official website](https://docs.anaconda.com/miniconda/).

**Step 1:** Create a new Conda environment with Python version `3.9` or higher, and activate it:

```bash
conda create -n x-anylabeling-upn python=3.9 -y
conda activate x-anylabeling-upn
```

You'll need to install Pytorch first. Follow the instructions [here](https://pytorch.org/get-started/locally/) to install related dependencies.

Afterward, you can install ChatRex on a GPU-enabled machine using:

```bash
git clone https://github.com/IDEA-Research/ChatRex.git
cd ChatRex
pip install -v -e .
# install deformable attention for universal proposal network
cd chatrex/upn/ops
pip install -v -e .
# Back to the project root directory
cd -
```

Finally, install the necessary dependencies for X-AnyLabeling (v2.5.0+):

```bash
cd ..
git clone https://github.com/CVHub520/X-AnyLabeling
cd X-AnyLabeling
```

Now, you can back to the installation guide ([简体中文](../../../docs/zh_cn/get_started.md) | [English](../../../docs/en/get_started.md)) to install the remaining dependencies.


Here's how to set up for the UPN job:

1. Import your image (`Ctrl+I`) or video (`Ctrl+O`) file into X-AnyLabeling
2. Select and load the `Universal Proposal Network (IDEA)` model from the model list
3. Click `Run (i)` to start processing. After verifying the results are satisfactory, use `Ctrl+M` to batch process all images

Additionally, you can adjust the following parameters to filter detection results directly from the GUI:

- Detection Mode: Switch between `Coarse Grained` and `Fine Grained` modes using the dropdown menu next to the model selection
- Confidence Threshold: Adjust the confidence score (0-1) using the "Confidence" spinner control
- IoU Threshold: Control the Non-Maximum Suppression (NMS) threshold (0-1) using the "IoU" spinner control